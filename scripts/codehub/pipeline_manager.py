# Set the random seed
import random as rnd
rnd.seed(42)

# Libraries to help path complete raw inputs
from pathlib import Path
from prompt_toolkit import prompt
from prompt_toolkit.completion import PathCompleter

# General libraries
import re
import os
import sys
import glob
import uuid
import yaml
import time
import argparse
import datetime
import numpy as np
import pandas as PD
from tqdm import tqdm
import multiprocessing

# Core imports
from components.core.internal.target_loader import *
from components.core.internal.output_manager import *
from components.core.internal.dataframe_manager import *

# Curation imports
from components.curation.public.data_loader import *
from components.curation.internal.data_curation import *

# Feature imports
from components.features.public.features import *

# Metadata imports
from components.metadata.public.metadata_handler import *

# Validation imports
from components.validation.public.data_viability import *

# Workflow imports
from components.workflows.public.preprocessing import *
from components.workflows.public.channel_clean import *
from components.workflows.public.channel_mapping import *
from components.workflows.public.channel_montage import *
from components.workflows.public.project_handler import *

# Import the configuration maker
from configs.makeconfigs import *

class data_manager(project_handlers, metadata_handler, data_loader, channel_mapping, dataframe_manager, channel_clean, channel_montage, output_manager, data_viability, target_loader):

    def __init__(self, input_params, args, timestamp, worker_number, barrier):
        """
        Initialize parent class for data loading.
        Store pathing for different data type loads.

        Args:
            infile (str): path to datafile that needs to be loaded
        """

        # Make args visible across inheritance
        self.infiles       = input_params[:,0]
        self.start_times   = input_params[:,1].astype('float')
        self.end_times     = input_params[:,2].astype('float')
        self.ref_windows   = input_params[:,3]
        self.args          = args
        self.unique_id     = uuid.uuid4()
        self.bar_frmt      = '{l_bar}{bar}| {n_fmt}/{total_fmt}|'
        self.timestamp     = timestamp
        self.worker_number = worker_number
        self.barrier       = barrier

        # Create the metalevel container
        metadata_handler.__init__(self)
 
        # Initialize the output list so it can be updated with each file
        output_manager.__init__(self)
        
        ##############################################
        ##### Start the actual project workflows #####
        ##############################################

        # File management
        project_handlers.file_manager(self)

        # Select valid data slices
        data_viability.__init__(self)

        # Pass to feature selection managers
        self.feature_manager()

        # Associate targets if requested
        self.target_manager()

        # In the case that all of the data is removed, skip write step
        if len(self.metadata.keys()) > 0:
            
            # Save the results
            output_manager.save_features(self)

            if self.args.clean_save:
                output_manager.save_output_list(self)

    def feature_manager(self):
        """
        Kick off function for feature extraction if requested by the user.
        Also handles multithreading and screen appearance for verbose options. (tqdm with multithreading requires management of terminal whitespace)
        """

        if not self.args.no_feature_flag:
            if self.args.multithread:
                self.barrier.wait()

                # Add a wait for proper progress bars
                time.sleep(self.worker_number)

                # Clean up the screen
                if self.worker_number == 0:
                    sys.stdout.write("\033[H")
                    sys.stdout.flush()

            # In the case that all of the data is removed, skip the feature step
            if len(self.metadata.keys()) > 0:
                features.__init__(self)

    def target_manager(self):
        """
        Kick off function for loading targets.
        """

        if self.args.targets:
            if not self.args.no_feature_flag:
                for ikey in self.metadata.keys():
                    ifile   = self.metadata[ikey]['file']
                    target_loader.load_targets(self,ifile,'bids','target')

class CustomFormatter(argparse.HelpFormatter):
    """
    Custom formatting class to get a better argument parser help output.
    """

    def _split_lines(self, text, width):
        if text.startswith("R|"):
            return text[2:].splitlines()
        return super()._split_lines(text, width)

############################
##### Helper Functions #####
############################

def make_help_str(idict):
    """
    Make a well-formated help string for the possible keyword mappings

    Args:
        idict (dict): Dictionary containing the allowed keywords values and their explanation.

    Returns:
        str: Formatted help string
    """

    return "\n".join([f"{key:15}: {value}" for key, value in idict.items()])

def parse_list(input_str):
    """
    Helper function to allow list inputs to argparse using a space or comma

    Args:
        input_str (str): Users inputted string

    Returns:
        list: Input argument list as python list
    """

    # Split the input using either spaces or commas as separators
    values = input_str.replace(',', ' ').split()
    return [float(value) for value in values]

def start_analysis(data_chunk,args,timestamp,worker_id,barrier):
    """
    Helper function to allow for easy multiprocessing initialization.
    """

    DM = data_manager(data_chunk,args,timestamp,worker_id,barrier)

def merge_outputs(args,timestamp):
    """
    If requested, automatically merge outputs to just one file.
    """

    # Find the various filepaths
    metadata_files   = np.sort(glob.glob(f"{args.outdir}/{timestamp}*meta.pickle"))
    feature_files    = np.sort(glob.glob(f"{args.outdir}/{timestamp}*features.pickle"))
    featurecmd_files = np.sort(glob.glob(f"{args.outdir}/{timestamp}*fconfigs.pickle"))
    data_list        = np.sort(glob.glob(f"{args.outdir}/{timestamp}*data.pickle"))

    # Make a merged downcasted feature file
    if len(feature_files) > 0:
        for idx,ifile in enumerate(feature_files):
            
            # Read in the dataframe
            iDF = PD.read_pickle(ifile)
            
            # Attempt downcasting as much as possible
            for icol in iDF.columns:
                itype = iDF[icol].dtype
                try:
                    iDF[icol] = PD.to_numeric(iDF[icol],downcast='integer')
                    if iDF[icol].dtype == itype:
                        iDF[icol] = PD.to_numeric(iDF[icol],downcast='float')
                except ValueError:
                    pass

            # Merge the outputs to one final file
            if idx == 0:
                output_DF = iDF.copy()
            else:
                output_DF = PD.concat((output_DF,iDF))
        
        # Make the new output and only remove files once things were confirmed to work
        output_DF.to_pickle(f"{args.outdir}/{timestamp}_features.pickle")
        for ifile in feature_files:os.remove(ifile)

    # Clean up the feature config files (if present)
    if len(featurecmd_files) > 0:
        os.system(f"cp {featurecmd_files[0]} {args.outdir}/{timestamp}_fconfigs.pickle")
        for ifile in featurecmd_files:os.remove(ifile)
        
    # Clean up the meta files as needed
    if len(metadata_files) > 0:
        for idx,ifile in enumerate(metadata_files):
            imeta = pickle.load(open(ifile,"rb"))
            if idx == 0:
                metadata = imeta.copy()
            else:
                masterkeys = list(metadata.keys())
                newkeys    = list(imeta.keys())
                offset     = max(masterkeys)+1
                for ikey in newkeys:
                    imeta[ikey+offset] = imeta.pop(ikey)
                metadata = {**metadata,**imeta}
        pickle.dump(metadata,open(f"{args.outdir}/{timestamp}_meta.pickle","wb"))
        for ifile in metadata_files:os.remove(ifile)

    # Clean up the raw data files as needed
    if len(data_list) > 0:
        for idx,ifile in enumerate(data_list):
            idata = pickle.load(open(ifile,"rb"))
            if idx == 0:
                data = idata.copy()
            else:
                data.extend(idata)
        pickle.dump(data,open(f"{args.outdir}/{timestamp}_data.pickle","wb"))
        for ifile in data_list:os.remove(ifile)


def argument_handler(argument_dir='./',require_flag=True):

    # Read in the allowed arguments
    raw_args  = yaml.safe_load(open(f"{argument_dir}allowed_arguments.yaml","r"))
    for key, inner_dict in raw_args.items():
        globals()[key] = inner_dict

    # Make a useful help string for each keyword
    allowed_project_help   = make_help_str(allowed_project_args)
    allowed_datatype_help  = make_help_str(allowed_datatypes)
    allowed_clean_help     = make_help_str(allowed_clean_args)
    allowed_channel_help   = make_help_str(allowed_channel_args)
    allowed_montage_help   = make_help_str(allowed_montage_args)
    allowed_input_help     = make_help_str(allowed_input_args)
    allowed_viability_help = make_help_str(allowed_viability_args)
    allowed_majoraxis_help = make_help_str(allowed_majoraxis_args)

    # Command line options needed to obtain data.
    parser = argparse.ArgumentParser(description="Simplified data merging tool.", formatter_class=CustomFormatter)

    datamerge_group = parser.add_argument_group('Data Merging Options')
    datamerge_group.add_argument("--input", type=str, choices=list(allowed_input_args.keys()), default="GLOB", help=f"R|Choose an option:\n{allowed_input_help}")
    datamerge_group.add_argument("--n_input", type=int, default=0, help=f"Limit number of files read in. Useful for testing or working in batches. (0=all)")
    datamerge_group.add_argument("--n_offset", type=int, default=0, help=f"Offset the files read in. Useful for testing or working in batch.")
    datamerge_group.add_argument("--project", type=str, choices=list(allowed_project_args.keys()), default="SCALP_BASIC", help=f"R|Choose an option:\n{allowed_project_help}")
    datamerge_group.add_argument("--multithread", action='store_true', default=False, help="Multithread flag.")
    datamerge_group.add_argument("--ncpu", type=int, default=1, help="Number of CPUs to use if multithread.")

    datachunk_group = parser.add_argument_group('Data Chunking Options')
    datachunk_group.add_argument("--t_start", type=parse_list, default=[0], help="Time in seconds to start data collection.")
    datachunk_group.add_argument("--t_end", type=parse_list, default=[-1], help="Time in seconds to end data collection. (-1 represents the end of the file.)")
    datachunk_group.add_argument("--t_window", type=parse_list, default=[-1], help="List of window sizes, effectively setting multiple t_start and t_end for a single file.")
    datachunk_group.add_argument("--t_overlap", type=parse_list, default=[0], help="If you want overlapping time windows, this is the fraction of t_window overlapping.")

    ssh_group = parser.add_argument_group('SSH Data Loading Options')
    ssh_group.add_argument("--ssh_host", type=str, help="If loading data via ssh tunnel, this is the host ssh connection string.")
    ssh_group.add_argument("--ssh_username", type=str, help="If loading data via ssh tunnel, this is the host ssh username to log in as.")

    datatype_group = parser.add_argument_group('Input datatype Options')
    datatype_group.add_argument("--datatype", type=str, default='EDF', choices=list(allowed_datatypes.keys()), help=f"R|Choose an option:\n{allowed_datatype_help}")

    channel_group = parser.add_argument_group('Channel cleaning Options')
    channel_group.add_argument("--channel_clean", type=str,  choices=list(allowed_clean_args.keys()), default="HUP", help=f"R|Choose an option:\n{allowed_clean_help}")

    channel_group = parser.add_argument_group('Channel label Options')
    channel_group.add_argument("--channel_list", type=str,  choices=list(allowed_channel_args.keys()), default="HUP1020", help=f"R|Choose an option:\n{allowed_channel_help}")

    montage_group = parser.add_argument_group('Montage Options')
    montage_group.add_argument("--montage", type=str,  choices=list(allowed_montage_args.keys()), default="HUP1020", help=f"R|Choose an option:\n{allowed_montage_help}")

    orientation_group = parser.add_argument_group('Orientation Options')
    orientation_group.add_argument("--orientation", type=str,  choices=list(allowed_majoraxis_args.keys()), default="column", help=f"R|Choose an option:\n{allowed_majoraxis_help}")

    viability_group = parser.add_argument_group('Data viability Options')
    viability_group.add_argument("--viability", type=str,  choices=list(allowed_viability_args.keys()), default="None", help=f"R|Choose an option:\n{allowed_viability_help}")
    viability_group.add_argument("--interp", action='store_true', default=False, help="Interpolate over NaN values of sequence length equal to n_interp.")
    viability_group.add_argument("--n_interp", type=int,  default=1, help="Number of contiguous NaN values that can be interpolated over should the interp option be used.")

    preprocessing_group = parser.add_argument_group('Preprocessing Options')
    preprocessing_group.add_argument("--no_preprocess_flag", action='store_true', default=False, help="Do not run preprocessing on data.")
    preprocessing_group.add_argument("--preprocess_file", type=str,  help="Path to preprocessing YAML file. If not provided, code will walk user through generation of a pipeline.")

    feature_group = parser.add_argument_group('Feature Extraction Options')
    feature_group.add_argument("--no_feature_flag", action='store_true', default=False, help="Do not run feature extraction on data.")
    feature_group.add_argument("--feature_file", type=str,  help="Path to preprocessing YAML file. If not provided, code will walk user through generation of a pipeline.")

    target_group = parser.add_argument_group('Target Association Options')
    target_group.add_argument("--targets", action='store_true', default=False, help="Join target data with the final dataframe")

    output_group = parser.add_argument_group('Output Options')
    output_group.add_argument("--outdir", type=str,  required=require_flag, help="Output directory.") 
    output_group.add_argument("--exclude", type=str,  help="Exclude file. If any of the requested data is bad, the path and error gets dumped here. \
                              Also allows for skipping on subsequent loads. Default=outdir+excluded.txt (In Dev. Just gets initial load fails.)") 
    output_group.add_argument("--nomerge", action='store_true', default=False, help="Do not merge the outputs from multiprocessing into one final set of files.")
    output_group.add_argument("--clean_save", action='store_true', default=False, help="Save cleaned up raw data. Mostly useful if you need time series and not just features.")

    misc_group = parser.add_argument_group('Misc Options')
    misc_group.add_argument("--nfreq_window", type=int, default=8, help="Optional. Minimum number of samples required to send to preprocessing and feature extraction.")
    misc_group.add_argument("--input_str", type=str, help="Optional. If glob input, wildcard path. If csv/manual, filepath to input csv/raw data.")
    misc_group.add_argument("--silent", action='store_true', default=False, help="Silent mode.")
    misc_group.add_argument("--debug", action='store_true', default=False, help="Debug mode. If set, does not save results. Useful for testing code.")
    misc_group.add_argument("--trace", action='store_true', default=False, help="Trace data through the code. If selected, any user function that looks for trace can return extra information (i.e. intermediate calculations) to the metadata object.")
    args = parser.parse_args()

    # Make sure the output directory has a trailing /
    if require_flag:
        if args.outdir[-1] != '/':
            args.outdir += '/'

    # Make sure the times are properly defined
    if (np.array(args.t_start)).any() < 0: raise ValueError("t_start must be greater than 0.")
    if ((np.array(args.t_end)<=0)*(np.array(args.t_end)!=-1)).any(): raise ValueError("t_end must be greater than 0 or -1 (denoting end of file).")
    if ((np.array(args.t_window)<=0)*(np.array(args.t_window)!=-1)).any(): raise ValueError("t_window must be greater than 0 or -1 (denoting end of file).")
    if (np.array(args.t_overlap)<0).any() or (np.array(args.t_overlap)>1).any(): raise ValueError("t_overlap must be between 0 and 1.")

    # Make sure if the user passed a list entry for time, all the time entries are lists
    itr = iter([args.t_start,args.t_end,args.t_window,args.t_overlap])
    if not all(len(l) == len(next(itr)) for l in itr): raise IndexError("Please provide equal number of entries for t_start, t_end, t_window, and t_overlap")

    # Help info if needed to be passed back as an object and not string
    help_info    = {}
    type_info    = {}
    default_info = {}

    for action in parser._get_optional_actions():
        default_val               = action.default
        type_val                  = action.type
        default_info[action.dest] = default_val
        help_info[action.dest]    = action.help
        if type_val != None:
            type_info[action.dest] = type_val
        else:
            if  type(default_val) == bool:
                type_info[action.dest] = bool
            else:
                type_info[action.dest] = str
    return args,(help_info,type_info,default_info,raw_args)

if __name__ == "__main__":

    # Get the argument handler
    args,_ = argument_handler()

    # Make the output directory as needed
    if not os.path.exists(args.outdir) and not args.debug:
        print("Output directory does not exist. Make directory at %s (Y/y)?" %(args.outdir))
        user_input = input("Response: ")
        if user_input.lower() == 'y':
            os.system("mkdir -p %s" %(args.outdir))

    # Set the input file list
    if args.input == 'CSV':
        
        if args.input_str == None:
            # Tab completion enabled input
            completer = PathCompleter()
            print("Using CSV input. Enter a three column csv file with filepath,starttime,endtime.")
            print("If not starttime or endtime provided, defaults to argument inputs. Use --help for more information.")
            file_path = prompt("Please enter path to input file csv: ", completer=completer)
        else:
            file_path = args.input_str

        # Read in csv file
        input_csv = PD.read_csv(file_path)

        # Clean up any missing values using the user provided values
        input_DF = PD.DataFrame()
        for idx in range(len(args.t_start)):
            iDF               = input_csv.copy()
            iDF['start_time'] = iDF['start_time'].fillna(args.t_start[idx])
            iDF['end_time']   = iDF['end_time'].fillna(args.t_end[idx])
            input_DF          = PD.concat((input_DF,iDF))
        input_DF = input_DF.drop_duplicates().sort_values(by=['filepath'])

        # Grab the distinct arrays 
        files       = input_DF['filepath'].values
        start_times = input_DF['start_time'].values
        end_times   = input_DF['end_time'].values
    elif args.input == 'GLOB':

        if args.input_str == None:
            # Tab completion enabled input
            completer = PathCompleter()
            file_path = prompt("Please enter (wildcard enabled) path to input files: ", completer=completer)
        else:
            file_path = args.input_str
        files = glob.glob(file_path)

        # Make sure we were handed a good filepath
        if len(files) == 0:
            raise IndexError("No data found with that search. Cannot iterate over a null file list.")

        # Create start and end times array
        start_times = np.array([])
        end_times   = np.array([])
        for idx,ival in enumerate(args.t_start):
            istart      = ival*np.ones(len(files))
            iend        = args.t_end[idx]*np.ones(len(files))
            start_times = np.concatenate((start_times,istart))
            end_times   = np.concatenate((end_times,iend))
        files = np.tile(files,len(args.t_start))
    else:
        # Tab completion enabled input
        completer = PathCompleter()
        file_path = prompt("Please enter path to input file: ", completer=completer)
        files     = [file_path]

        # Create start and end times array
        start_times = np.array([])
        end_times   = np.array([])
        for idx,ival in enumerate(args.t_start):
            istart      = ival*np.ones(len(files))
            iend        = args.t_end[idx]*np.ones(len(files))
            start_times = np.concatenate((start_times,istart))
            end_times   = np.concatenate((end_times,iend))
        files = np.tile(files,len(args.t_start))

    # Cast the inputs as arrays
    files       = np.array(files)
    start_times = np.array(start_times)
    end_times   = np.array(end_times)

    # Curate the data inputs to get a valid (sub)set that maintains stratification of subjects
    DC = data_curation(args,files,start_times,end_times)
    files, start_times, end_times,ref_windows = DC.get_dataload()

    # Make configuration files as needed
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
    if args.preprocess_file == None and not args.no_preprocess_flag:
        from components.workflows.public import preprocessing
        dirpath              = args.outdir+"configs/"
        os.system("mkdir -p %s" %(dirpath))
        args.preprocess_file = dirpath+"preprocessing_"+timestamp+".yaml"
        config_handler       = make_config(preprocessing,args.preprocess_file)
        config_handler.create_config()
    if args.feature_file == None and not args.no_feature_flag:
        from components.features.public import features
        dirpath           = args.outdir+"configs/"
        os.system("mkdir -p %s" %(dirpath))
        args.feature_file = dirpath+"features_"+timestamp+".yaml"
        config_handler    = make_config(features,args.feature_file)
        config_handler.create_config()

    # Multithread options
    input_parameters = np.column_stack((files, start_times, end_times,ref_windows))
    if args.multithread:

        # Calculate the size of each subset based on the number of processes
        subset_size  = input_parameters.shape[0] // args.ncpu
        list_subsets = [input_parameters[i:i + subset_size] for i in range(0, subset_size*args.ncpu, subset_size)]

        # Handle leftovers
        remainder = list_subsets[args.ncpu*subset_size:]
        for idx,ival in enumerate(remainder):
            list_subsets[idx] = np.concatenate((list_subsets[idx],np.array([ival])))

        # Create a barrier for synchronization
        barrier = multiprocessing.Barrier(args.ncpu)

        # Create processes and start workers
        processes = []
        for worker_id, data_chunk in enumerate(list_subsets):
            process = multiprocessing.Process(target=start_analysis, args=(data_chunk,args,timestamp,worker_id,barrier))
            processes.append(process)
            process.start()
        
        # Wait for all processes to complete
        for process in processes:
            process.join()
    else:
        # Run a non parallel version.
        start_analysis(input_parameters, args, timestamp, 0, None)
    
    # Perform merge if requested
    if not args.nomerge:
        merge_outputs(args,timestamp)