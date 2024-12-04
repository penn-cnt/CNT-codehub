# General libraries
import re
import os
import numpy as  np
import pandas as PD
from pyedflib.highlevel import read_edf_header
from sklearn.model_selection import train_test_split

# Component imports
from components.curation.public.data_loader import *

class data_curation:
    """
    Class devoted to loading in raw data into the shared class instance.

    New functions should make use of the specific raw data handler for their dataset.
    """

    def __init__(self,args,files,start_times,end_times):
        self.args        = args
        self.files       = files
        self.start_times = start_times
        self.end_times   = end_times

    def test_input_data(self):
        """
        Create a cache in/cache out list of valid datasets to avoid errors and speed up subsequent data loads.
        """
        
        # Get the pathing to the excluded data
        if self.args.exclude == None:
            exclude_path = self.args.outdir+"excluded.txt"
        else:
            exclude_path = self.args.exclude

        # Get the files to use and which to save
        good_index = []
        bad_index  = []
        if os.path.exists(exclude_path):
            excluded_files = PD.read_csv(exclude_path)['file'].values
            for idx,ifile in enumerate(self.files):
                if ifile not in excluded_files:
                    good_index.append(idx)
        else:
            if not self.args.silent:
                print("Creating cache of useable data. This may take awhile.")

            # Confirm that data can be read in properly
            excluded_files = []
            for idx,ifile in enumerate(self.files):

                # Get the load type
                ftype = self.args.datatype.lower()
                if ftype == 'mix':
                    ftype = ifile.split('.')[-1]

                # Use the load type and perform a load test
                if ftype == 'edf':
                    DLT  = data_loader_test()
                    flag = DLT.test_logic(ifile,ftype)
                else:
                    flag = (True,)

                # Store the files that pass and fail, including error if it fails
                if flag[0]:
                    good_index.append(idx)
                else:
                    excluded_files.append([ifile,flag[1]])

            # Create the output failure dataframe for future use with error for the user to look at, then save.
            excluded_df = PD.DataFrame(excluded_files,columns=['file','error'])
            if not self.args.debug:
                excluded_df.to_csv(exclude_path,index=False)
        
        # Save new file info
        self.files       = self.files[good_index]
        self.start_times = self.start_times[good_index]
        self.end_times   = self.end_times[good_index]

    def stratified_resample_index(self,arr,strat,window_size=100):
        """
        Provide a stratified series of data window. This way we can maintain class balance for a limited data load, and also allow offsets and windowed number of files.

        Args:
            arr (_type_): Array to stratify by index.
            strat (_type_): The stratification/class array
            window_size (int, optional): Window size to force class balance within. Defaults to 100.

        Returns:
            array: Array of indices that maintain a windowed stratification.
        """

        # Make an index array to allow subscripts
        nvals   = len(arr)
        indices = np.arange(nvals)

        # If we have more entries than our window size, enforce stratification
        try:
            if nvals > window_size:
                sorted_index,remain_index = train_test_split(indices,train_size=window_size,stratify=strat,random_state=42)
                while remain_index.size > window_size:
                    current_index,remain_index = train_test_split(remain_index,train_size=window_size,stratify=strat[remain_index],random_state=42)
                    sorted_index               = np.concatenate((sorted_index,current_index))
                sorted_index = np.concatenate((sorted_index,remain_index))
            else:
                sorted_index = indices.copy()
        except:
            sorted_index = indices.copy()

        return sorted_index

    def overlapping_start_times(self, start, end, step, overlap_frac):

        # Define tracking variables
        current_time = start
        start_times  = []
        end_times    = []

        # Sanity check on step and overlap sizes
        if overlap_frac >= 1:
            raise ValueError("--t_overlap must be smaller than --t_window.")
        else:
            overlap = overlap_frac*step

        # Loop over the time range using the start, end, and step values. But then backup by windowed overlap as need
        while current_time <= end:
            start_times.append(current_time)
            if (current_time + step) < end:
                end_times.append(current_time+step)
            else:
                end_times.append(end)
            current_time = current_time + step - overlap
        start_times = np.array(start_times)
        end_times   = np.array(end_times)

        # Find edge cases where taking large steps with small offsets means multiple slices that reach the end time
        limiting_index = np.argwhere(end_times>=end).min()+1

        return start_times[:limiting_index],end_times[:limiting_index]

    def create_time_windows(self):

        # If using a sliding time window, duplicate inputs with the correct inputs
        if self.args.t_window != None:
            new_files  = []
            new_start  = []
            new_end    = []
            new_refwin = []
            for ifile in self.files:

                # Read in just the header to get duration
                t_end = self.args.t_end.copy()
                for idx,ival in enumerate(t_end):
                    if ival == -1:

                        # Determine how we are going to grab the duration
                        dtype = self.args.datatype.lower()
                        if dtype == 'mix':
                            dtype = ifile.split('.')[-1].lower()
                        
                        if dtype == 'edf':
                            header = read_edf_header(ifile)
                            t_end[idx] = header['Duration']
                        elif dtype == 'pickle':
                            idict      = pickle.load(open(ifile,'rb'))
                            t_end[idx] = idict['data'].shape[0]/idict['samp_freq']

                # Get the start time for the windows
                t_start = self.args.t_start

                # Calculate the correct step if using -1 flags to denote rest of file
                t_window = self.args.t_window.copy()
                for idx,ival in enumerate(t_window):
                    if ival == -1:
                        t_window[idx] = t_end[idx]-t_start[idx]

                # Step through the different window sizes to get the new file list, using overlaps
                for ii,iwindow in enumerate(t_window):
                    
                    # Get the list of windows start and end times
                    windowed_start, windowed_end = self.overlapping_start_times(t_start[ii],t_end[ii],iwindow,self.args.t_overlap[ii])

                    # Loop over the new entries and tile the input lists as needed
                    for idx,istart in enumerate(windowed_start):
                        if (windowed_end[idx]-istart)==iwindow:
                            new_files.append(ifile)
                            new_start.append(istart)
                            new_end.append(windowed_end[idx])
                            new_refwin.append(self.args.t_window[ii])
            self.files       = new_files
            self.start_times = new_start
            self.end_times   = new_end
            self.ref_win     = new_refwin 

    def limit_data_volume(self):
        """
        Apply stratification and data limits.
        """

        # Get the stratification array
        self.stratifier_logic()

        # Get the stratified indices
        sorted_index = self.stratified_resample_index(self.stratification_array,self.stratification_array)

        # Apply the sorting index
        self.files       = self.files[sorted_index]
        self.start_times = self.start_times[sorted_index]
        self.end_times   = self.end_times[sorted_index]

        # Apply any file offset as needed
        self.files       = self.files[self.args.n_offset:]
        self.start_times = self.start_times[self.args.n_offset:]
        self.end_times   = self.end_times[self.args.n_offset:]

        # Limit file length as needed
        if self.args.n_input > 0:
            self.files       = self.files[:self.args.n_input]
            self.start_times = self.start_times[:self.args.n_input]
            self.end_times   = self.end_times[:self.args.n_input]

        # Sort the results to ensure data loads in order
        sorted_index     = np.argsort(self.files)
        self.files       = self.files[sorted_index]
        self.start_times = self.start_times[sorted_index]
        self.end_times   = self.end_times[sorted_index]

    def get_dataload(self):
        
        self.test_input_data()
        self.limit_data_volume()
        self.create_time_windows()
        return self.files,self.start_times,self.end_times,self.ref_win

    ########################################################
    ##### Functions for different stratification types. ####
    ########################################################

    def stratifier_logic(self,strat_type='bids_subject'):
        
        if strat_type == 'bids_subject':
            self.stratifier_BIDS_subject_count()

    def stratifier_BIDS_subject_count(self):
        """
        Calculate the approximate number of subjects loaded into this analysis.
        """

        self.stratification_array = []
        for ifile in self.files:
            regex_match = re.match(r"(\D+)(\d+)", ifile)
            self.stratification_array.append(int(regex_match.group(2)))
        subcnt = np.unique(self.stratification_array).size
        if not self.args.silent:
            print(f"Assuming BIDS data, approximately {subcnt:04d} subjects loaded.")
