# Set the random seed
import random as rnd
rnd.seed(42)

# Basic Python Imports
import glob
import argparse
import pandas as PD
import pylab as PLT

# Local Imports
from components.internal.plot_handler import *

class CustomFormatter(argparse.HelpFormatter):
    """
    Custom formatting class to get a better argument parser help output.
    """

    def _split_lines(self, text, width):
        if text.startswith("R|"):
            return text[2:].splitlines()
        return super()._split_lines(text, width)

#####################
#### Helper Fncs ####
#####################

def make_help_str(idict):
    """
    Make a well-formated help string for the possible keyword mappings

    Args:
        idict (dict): Dictionary containing the allowed keywords values and their explanation.

    Returns:
        str: Formatted help string
    """

    return "\n".join([f"{key:15}: {value}" for key, value in idict.items()])

if __name__ == '__main__':

    # Command line options needed to obtain data.
    parser = argparse.ArgumentParser(description="Simplified data merging tool.", formatter_class=CustomFormatter)

    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument("--infile", type=str, help="Single input file to plot from cli.")
    input_group.add_argument("--wildcard", type=str, help="Wildcard enabled path to plot multiple datasets.")
    input_group.add_argument("--file", type=str, help="Filepath to txt or csv of input files.")

    dtype_group = parser.add_argument_group('Datatype options')
    dtype_group.add_argument("--pickle_load", action='store_true', default=False, help="Load from pickledata. Accepts pickled tuple/list of dataframe/fs or just dataframe. If only dataframe, must provide --fs sampling frequency.")
    dtype_group.add_argument("--fs", type=float, help="Sampling frequency.")

    output_group = parser.add_argument_group('Output options')
    output_group.add_argument("--outfile", default='./edf_viewer_annotations.csv', type=str, help="Output filepath if predicting sleep/spikes/etc.")

    prep_group = parser.add_argument_group('Data preparation options')
    prep_group.add_argument("--chcln", type=str, default="hup", help="Channel cleaning option")
    prep_group.add_argument("--chmap", type=str, default="hup1020", help="Channel mapping option. 'None' to skip.")
    prep_group.add_argument("--montage", type=str, default="hup1020", help="Channel montage option 'None' to skip.")

    time_group = parser.add_mutually_exclusive_group()
    time_group.add_argument("--t0", type=float, help="Start time to plot from in seconds.")
    time_group.add_argument("--t0_frac", action='store_true', default=False, help="Flag. Start time in fraction of total data.")

    duration_group = parser.add_mutually_exclusive_group()
    duration_group.add_argument("--dur", type=float, default=10, help="Duration to plot in seconds.")
    duration_group.add_argument("--dur_frac", action='store_true', default=False, help="Flag. Duration is interpreted as a fraction of total data.")

    misc_group = parser.add_argument_group('Misc options')
    misc_group.add_argument("--winfrac", type=float, default=0.9, help="Fraction of the window for the plot.")
    misc_group.add_argument("--nstride", type=int, default=8, help="Stride factor for plotting.")
    misc_group.add_argument("--debug", action='store_true', default=False, help="Debug mode. Save no outputs.")
    args = parser.parse_args()

    # Clean up some argument types
    args.chmap = None if args.chmap == 'None' else args.chmap
    args.montage = None if args.montage == 'None' else args.montage

    # Create the file list to read in
    if args.infile != None:
        files = [args.infile]
    elif args.wildcard != None:
        files = glob.glob(args.wildcard)
    elif args.file != None:
        files = PD.read_csv(args.file,usecols=[0],names=['files']).values.flatten()

    # Alert user if there are no eligible files
    if len(files) == 0:
        print("No files found matching your criteria.")
    
    # Iterate over the data and create the relevant plots
    tight_layout_dict = None
    for ifile in files:
        DV                = data_viewer(ifile,args,tight_layout_dict)
        tight_layout_dict = DV.workflow()
        PLT.show()
