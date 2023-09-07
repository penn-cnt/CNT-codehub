# Libraries to help path complete raw inputs
from pathlib import Path
from prompt_toolkit import prompt
from prompt_toolkit.completion import PathCompleter

# General libraries
import glob
import time
import resource
import argparse
import numpy as np
import pandas as PD
from sys import exit

# Import the classes
from modules.data_loader import *
from modules.channel_mapping import *
from modules.dataframe_manager import *
from modules.channel_clean import *
from modules.channel_montage import *
from modules.tensor_manager import *
from modules.data_viability import *

class data_manager(data_loader, channel_mapping, dataframe_manager, channel_clean, channel_montage, tensor_manager, data_viability):

    def __init__(self, infiles, args):
        """
        Initialize parent class for data loading.
        Store pathing for different data type loads.

        Args:
            infile (str): path to datafile that needs to be loaded
        """

        # Make args visible across inheritance
        self.args = args

        # Initialize the tensor list so it can be updated with each file
        tensor_manager.__init__(self)

        # Loop over files to read and store each ones data
        filecnt = len(infiles)
        for ifile in infiles:
            
            # Save current file
            self.infile = ifile
            
            # Case statement the workflow
            if self.args.dtype == 'EDF':
                try:
                    self.edf_handler()
                except OSError:
                    filecnt -= 1
        tensor_manager.create_tensor(self)
        print("Processed %04d files." %(filecnt))

    def edf_handler(self):
        """
        Run pipeline to load EDF data.
        """

        # Import data into memory
        data_loader.load_edf(self)

        # Clean the channel names
        channel_clean.__init__(self)

        # Get the correct channels for this merger
        channel_mapping.__init__(self,self.args.channel_list)

        # Create the dataframe for the object with the cleaned labels
        dataframe_manager.__init__(self)
        dataframe_manager.column_subsection(self,self.channel_map_out)

        # Put the data into a specific montage
        channel_montage.__init__(self)

        # Clean up the data before going to the tensor
        data_viability.__init__(self)

        # Update the tensor list
        tensor_manager.update_tensor_list(self,self.montaged_dataframe.values)

class CustomFormatter(argparse.HelpFormatter):
    """
    Custom formatting class to get a better argument parser help output.
    """

    def _split_lines(self, text, width):
        if text.startswith("R|"):
            return text[2:].splitlines()
        return super()._split_lines(text, width)

def make_help_str(idict):
    """
    Make a well-formated help string for the possible keyword mappings

    Args:
        idict (dict): Dictionary containing the allowed keywords values and their explanation.

    Returns:
        str: Formatted help string
    """

    return "\n".join([f"{key:15}: {value}" for key, value in idict.items()])

def input_with_tab_completion(prompt):
    def complete(text, state):
        return (file for file in readline.get_completions() if file.startswith(text))

    readline.set_completer(complete)
    readline.parse_and_bind("tab: complete")

    return input(prompt)

if __name__ == "__main__":

    # Define the allowed keywords a user can input
    allowed_input_args     = {'CSV' : 'Use a comma separated file of files to read in. (default)',
                              'MANUAL' : "Manually enter filepaths.",
                              'GLOB' : 'Use Python glob to select all files that follow a user inputted pattern.'}
    allowed_dtype_args     = {'EDF': "EDF file formats. (default)"}
    allowed_channel_args   = {'HUP1020': "Channels associated with a 10-20 montage performed at HUP.",
                              'RAW': "Use all possible channels. Warning, channels may not match across different datasets."}
    allowed_montage_args   = {'HUP1020': "Use a 10-20 montage.",
                              'COMMON_AVERAGE': "Use a common average montage."}
    allowed_viability_args = {'VIABLE_DATA': "Drop datasets that contain a NaN column. (default)",
                              'VIABLE_COLUMNS': "Use the minimum cross section of columns across all datasets that contain no NaNs."}
    
    # Make a useful help string for each keyword
    allowed_input_help     = make_help_str(allowed_input_args)
    allowed_dtype_help     = make_help_str(allowed_dtype_args)
    allowed_channel_help   = make_help_str(allowed_channel_args)
    allowed_montage_help   = make_help_str(allowed_montage_args)
    allowed_viability_help = make_help_str(allowed_viability_args)

    # Command line options needed to obtain data.
    parser = argparse.ArgumentParser(description="Simplified data merging tool.", formatter_class=CustomFormatter)
    parser.add_argument("--input", choices=list(allowed_input_args.keys()), default="CSV", help=f"R|Choose an option:\n{allowed_input_help}")
    parser.add_argument("--dtype", choices=list(allowed_dtype_args.keys()), default="EDF", help=f"R|Choose an option:\n{allowed_dtype_help}")
    parser.add_argument("--channel_list", choices=list(allowed_channel_args.keys()), default="HUP1020", help=f"R|Choose an option:\n{allowed_channel_help}")
    parser.add_argument("--montage", choices=list(allowed_montage_args.keys()), default="HUP1020", help=f"R|Choose an option:\n{allowed_montage_help}")
    parser.add_argument("--viability", choices=list(allowed_viability_args.keys()), default="VIABLE_DATA", help=f"R|Choose an option:\n{allowed_viability_help}")
    parser.add_argument("--interp", action='store_true', default=False, help="Interpolate over NaN values of sequence length equal to n_interp.")
    parser.add_argument("--n_interp", default=1, help="Number of contiguous NaN values that can be interpolated over should the interp option be used.")
    args = parser.parse_args()

    # For testing purposes
    start = time.time()

    # Set the input file list
    if args.input == 'CSV':
        
        # Tab completion enabled input
        completer = PathCompleter()
        file_path = prompt("Please enter path to input file csv: ", completer=completer)

        # Due to the different ways paths can be inputted, using a filepointer to clean each entry best we can
        fp    = open(file_path,'r')
        files = []
        data  = fp.readline()
        while data:
            clean_data = data.replace('\n', '')
            clean_data = clean_data.split(',')
            for ival in clean_data:
                if ival != '':
                    files.append(ival)
            data = fp.readline()
    elif args.input == 'GLOB':

        # Tab completion enabled input
        completer = PathCompleter()
        #file_path = prompt("Please enter (wildcard enabled) path to input files: ", completer=completer)
        file_path = "/Users/bjprager/Documents/GitHub/SCALP_CONCAT-EEG/user_data/sample_data/edf/teug/a*/*edf"
        files     = glob.glob(file_path)

    # Load the parent class
    DM = data_manager(files, args)

    # For testing purposes
    print("Time taken in seconds: %f" %(time.time()-start))
    max_mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    print("Max memory usage in GB: ~%f" %(max_mem/1e9))