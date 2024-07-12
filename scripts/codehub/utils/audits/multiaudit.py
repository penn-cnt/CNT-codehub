import os
import time
import pickle
import argparse
import subprocess
import numpy as np
import pandas as PD
from sys import exit
from tqdm import tqdm
import multiprocessing
from pathlib import Path
from datetime import datetime

class audit:

    def __init__(self,search_root,outdir,ostype,cmd_path,audit_history,username,systemname):

        # Save the inputs to class instance
        self.rootdir       = search_root
        self.outdir        = outdir
        self.os            = ostype
        self.cmd_path      = cmd_path
        self.systemname    = systemname

        if not os.path.exists(self.outdir):
            os.system(f"mkdir -p {self.outdir}")

        # Define the delimter we will use for folder breaks
        self.delimiter     = '.'

        # Some basic cleanup to ensure we have trailing directory slashes
        if self.rootdir[-1] != '/':
            self.rootdir = self.rootdir+'/'
        if self.outdir[-1] != '/':
            self.outdir = self.outdir+'/'

        # Get the audit history
        if audit_history != None:
            self.audit_history = audit_history
        else:
            self.audit_history = self.outdir+f"audit_history_{systemname}_{username}.csv"

        # Create a temporary file that stores all of the input paths for the given root directory. This speeds up runs if testing/restarting.
        fname           = f"{self.rootdir.replace('/',self.delimiter)}inputs"
        if fname[0] == self.delimiter:
            fname = fname[1:]
        self.input_file = f"{self.outdir}{fname}"

        # Output audit location
        self.audit_data = self.outdir+f"audit_data_{systemname}_{username}.csv"

        # Lock file location for staggered history update
        self.lock_file = self.outdir+f"audit_history_{username}.lock"

    def argcheck(self):
        """
        Make sure the user provided the correct inputs to run the script. Looks for root directory to search through, and a correct filesystem type.
        """

        # Read in the user provided root directory and make sure it exists
        if not os.path.exists(self.rootdir):
            raise NameError("No directory found matching that name.")
                
        # Make sure the os archiecture is allowed
        if self.os.lower() not in ['unix','windows']:
            raise NameError("Please specify 'unix' or 'windows' back-end.")
        
    def read_cmd(self):
        """
        Read in the filesystem specific command needed to run the audit.
        """

        # Read in the relevant config
        fp              = open(self.cmd_path)
        self.cmd_master = fp.read()
        fp.close()

    def define_inputs(self):

        # Locally scoped function to travese directories to get finer granularity input list
        def get_all_subdirectories(directory):
            """Recursively get all subdirectories."""
            subdirectories = []
            for entry in os.scandir(directory):
                if entry.is_dir():
                    subdirectories.append(entry.path)
                    subdirectories.extend(get_all_subdirectories(entry.path))
            return subdirectories

        # Define the directories in the root directory
        if os.path.exists(self.input_file):
            self.folders = pickle.load(open(self.input_file,'rb'))
        else:
            print("Getting all the subdirectories to search. This may take awhile.")
            folders = get_all_subdirectories(self.rootdir)
            self.folders = np.sort(folders)
            pickle.dump(self.folders,open(self.input_file,"wb"))

        # Read in the audit history
        if os.path.exists(self.audit_history):
            self.history = PD.read_csv(self.audit_history)
        else:
            self.history = PD.DataFrame(columns=['directory_path','mjd'])
            self.history.to_csv(self.audit_history,index=False)

        # Check for already completed entries
        self.input_paths  = []
        self.output_names = []
        raw_output        = np.arange(len(self.folders))
        completed_folders = self.history['directory_path'].values
        for ii,ifolder in enumerate(self.folders):
            alt_path = str(ifolder) + '/'
            if ifolder not in completed_folders and alt_path not in completed_folders:
                self.input_paths.append(str(ifolder))
                self.output_names.append(f"{raw_output[ii]:09}.audit")
            else:
                print(f"Skipping {ifolder}.")
                pass
        self.input_paths.append(str(self.rootdir))
        self.output_names.append(f"{len(self.folders):09}.audit")

        # Clean up the paths as needed
        for idx in range(len(self.input_paths)):
            if self.input_paths[idx][-1] != '/':
                self.input_paths[idx] += '/'

        # Fix the typing of the outputs to allow better indexing
        self.input_paths  = np.array(self.input_paths)
        self.output_names = np.array(self.output_names)

    def clean_audit(self):
        """
        Clean up the audit outputs to one csv file.
        """

        # Make or load the audit data file as needed
        if not os.path.exists(self.audit_data):
            fp = open(self.audit_data,'w')
            fp.write('path,md5,size-(MB),last-modified-date,owner\n')
        else:
            fp = open(self.audit_data,'a')

        # Add the root directory to the search path, then loop
        folder_list = [str(ifolder) for ifolder in self.folders]
        folder_list.append(str(self.rootdir))
        for idx in range(len(folder_list)+1):

            # Get the expected target output file for this folder
            self.outname   = f"{idx:09}.audit"
            self.outname = f"{self.outdir}{self.outname}"

            # If file exists, add it to the data file then remove
            if os.path.exists(self.outname):
                
                # Read in and clean the raw audit data
                fp2     = open(self.outname,'r')
                rawdata = np.array(fp2.readlines())
                rawdata = rawdata.reshape((-1,4))
                for irow in rawdata:
                    irow[0]    = f"{' '.join(irow[0].split()[:-1])},{irow[0].split()[-1]}"
                    irow[0]   += '\n'
                    rawstring  = ' '.join(irow)
                    clean_data = rawstring.replace('\n',',').replace(' MB','')
                    if clean_data[-1] == ',':
                        clean_data = clean_data[:-1]
                    fp.write(f"{clean_data}\n")
                    rawdata = fp2.readline()
                fp2.close()
                os.remove(self.outname)
        fp.close()

        # Remove the inputs file
        if os.path.exists(self.input_file):
            os.remove(self.input_file)

    #############################
    ###### Audit functions ######
    #############################

    def audit_handler(self,os,ncpu):

        # Break up the inputs across the cpus
        index_arr   = np.arange(len(self.input_paths))
        subset_size = len(self.input_paths) // ncpu
        while subset_size == 0:
            ncpu       -= 1
            subset_size = len(self.input_paths) // ncpu
        print(f"Indexing solution found with {ncpu:02} cpus for {len(self.input_paths)} folders.")
        index_subsets = [index_arr[i:i + subset_size] for i in range(0, len(self.input_paths), subset_size)]

        # Handle leftovers
        if len(index_subsets) > ncpu:
            arr_ncpu  = index_subsets[ncpu-1]
            arr_ncpu1 = index_subsets[ncpu]

            index_subsets[ncpu-1] = np.concatenate((arr_ncpu,arr_ncpu1), axis=0)
            index_subsets.pop(-1)

        # Add a sempahore to allow orderly file access (to mimic multiprocesing for ease of argument definition)
        semaphore = multiprocessing.Semaphore(1)

        # Create a barrier for synchronization
        barrier = multiprocessing.Barrier(args.ncpu)

        # Setup an output object
        manager     = multiprocessing.Manager()
        return_dict = manager.dict()

        # Check for the right operating system logic
        if os.lower() == 'unix':

            processes = []
            for worker_id,data_chunk in enumerate(index_subsets):
                indata = (self.input_paths[data_chunk],self.output_names[data_chunk],worker_id)
                process = multiprocessing.Process(target=self.perform_audit_linux, args=(indata,semaphore,barrier,return_dict))
                processes.append(process)
                process.start()

            # Wait for all processes to complete
            for process in processes:
                process.join()
        
        # Add the results to the audit history and save
        self.history = PD.read_csv(self.audit_history)
        for iresult in return_dict.values():
            results_DF   = PD.DataFrame(iresult,columns=self.history.columns)
            self.history = PD.concat([self.history,results_DF],ignore_index=True)
        self.history.to_csv(self.audit_history,index=False)

    def perform_audit_linux(self,args,semaphore,barrier,return_dict):
        """
        Perform a data audit on a linux/unix filesystem. 
        """

        # Unpack arguments
        inpaths,outpaths,worker_number = args

        # Output object
        output = []

        # Make a reference time so we know when to make a backup
        start_time = time.time()

        # Loop over the folders to audit
        for idx,ifolder in tqdm(enumerate(inpaths), desc='Audit: ', total=len(inpaths), position=worker_number, disable=False, leave=False):

            # Save the input string to a different name in case of modifications
            instr = ifolder

            # Modify the input directory name to make an output filename
            self.outname = f"{self.outdir}{outpaths[idx]}"

            # Update the cmd string for this case
            cmd = self.cmd_master.replace("INDIR_SUBSTR",instr)
            cmd = cmd.replace("OUTDIR_SUBSTR",self.outname)

            # Run command
            try:
                subprocess.run(cmd, shell=True, timeout=2*60)

                # Update audit history
                output.append([ifolder,datetime.now().timestamp()])
            except:
                pass

            # Update the audit history ocassionally to speed up subsequent loads
            current_time = time.time()
            dt           = (current_time-start_time)
            stagger_time = 60+5*worker_number 
            
            if dt > stagger_time and len(output)>0:

                with semaphore:

                    # Read in the latest audit history
                    self.history = PD.read_csv(self.audit_history)

                    # Write the current history for this process to the audit history file
                    results_DF   = PD.DataFrame(np.array(output),columns=self.history.columns)
                    self.history = PD.concat([self.history,results_DF],ignore_index=True)
                    self.history.to_csv(self.audit_history,index=False)

                    # Update the current process's outputs and stagger time
                    start_time = time.time()
                    output     = []                        

        return_dict[worker_number] = np.array(output)

if __name__ == '__main__':
    
    # Define the system choices
    systemchoices = ['leif','bsc','pioneer','cnt1','cntfs']

    # Command line options needed to obtain data.
    parser = argparse.ArgumentParser(description="iEEG to bids conversion tool.")
    parser.add_argument("--search_root", type=str, required=True, help="Root directory to recursively audit down from.")
    parser.add_argument("--outdir", type=str, required=True, help="Path to output directory to store the audit.")
    parser.add_argument("--os", type=str, default='unix', help="OS architecture. Allowed Arguments='unix' or 'windows'.")
    parser.add_argument("--cmd_path", type=str, default='config/audit.md5sum.linux', help="Path to command string to execute.")
    parser.add_argument("--audit_history", type=str, help="Path to the audit history.")
    parser.add_argument("--merge", action='store_true', default=False, help="Merge outputs to final audit file.")
    parser.add_argument("--username", type=str, default='main', help="Username for data audit.")
    parser.add_argument("--ncpu", type=int, default=1, help="Multiprocessing. Number of cpus to use.")
    parser.add_argument("--system", type=str, required=True, choices=systemchoices, help="System name.")
    args = parser.parse_args()

    # Run through the audit
    AH = audit(args.search_root,args.outdir,args.os,args.cmd_path,args.audit_history,args.username,args.system)
    AH.argcheck()
    AH.read_cmd()
    AH.define_inputs()
    AH.audit_handler(args.os,args.ncpu)
    if args.merge:
        print(f"Merging data. This may take awhile.")
        AH.clean_audit()
    
