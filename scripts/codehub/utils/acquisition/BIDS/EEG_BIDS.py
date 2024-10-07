import os
import argparse
import pandas as PD
from sys import exit
from prettytable import PrettyTable,ALL

# Local import
from components.internal.BIDS_handler import *
from components.public.edf_handler import edf_handler
from components.public.iEEG_handler import ieeg_handler
from components.public.jar_handler import jar_handler

# MNE is very chatty. Turn off some warnings.
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)

def print_examples():
        
        # Read in the sample time csv
        script_dir  = '/'.join(os.path.abspath(__file__).split('/')[:-1])
        example_csv = PD.read_csv(f"{script_dir}/samples/sample_times.csv")
        
        # Initialize a pretty table for easy reading
        table = PrettyTable(hrules=ALL)
        table.field_names = example_csv.columns
        for irow in example_csv.index:
            iDF           = example_csv.loc[irow]
            formatted_row = [iDF[icol] for icol in example_csv.columns]
            table.add_row(formatted_row)
        table.align['path'] = 'l'
        print("Sample inputs that explicitly set the download times.")
        print(table)

        # Read in the sample annotation csv
        script_dir  = '/'.join(os.path.abspath(__file__).split('/')[:-1])
        example_csv = PD.read_csv(f"{script_dir}/samples/sample_annot.csv")
        
        # Initialize a pretty table for easy reading
        table = PrettyTable(hrules=ALL)
        table.field_names = example_csv.columns
        for irow in example_csv.index:
            iDF           = example_csv.loc[irow]
            formatted_row = [iDF[icol] for icol in example_csv.columns]
            table.add_row(formatted_row)
        table.align['path'] = 'l'
        print("Sample inputs that use annotations.")
        print(table)

def ieeg(args):
    IH = ieeg_handler(args)
    IH.workflow()

def raw_edf(args):
    EH = edf_handler(args)
    EH.workflow()

def read_jar(args):
    JH = jar_handler(args)
    JH.workflow()

if __name__ == '__main__':

    # Command line options needed to obtain data.
    parser = argparse.ArgumentParser(description="Make an EEG BIDS dataset from various sources. Also manages helper scripts for the CNT.")

    source_group = parser.add_argument_group('Data source options')
    source_option_group = source_group.add_mutually_exclusive_group(required=True)
    source_option_group.add_argument("--ieeg", action='store_true', default=False, help="iEEG data pull.")
    source_option_group.add_argument("--edf", action='store_true', default=False, help="Raw edf data pull.")
    source_option_group.add_argument("--jar", action='store_true', default=False, help="Convert jar file to EDF Bids.")

    data_group = parser.add_argument_group('Data configuration options')
    data_group.add_argument("--bids_root", type=str, required=True, default=None, help="Output directory to store BIDS data.")
    data_group.add_argument("--data_record", type=str, default='subject_map.csv', help="Filename for data record. Outputs to bids_root.")

    ieeg_group = parser.add_argument_group('iEEG connection options')
    ieeg_group.add_argument("--username", type=str, help="Username for iEEG.org.")
    ieeg_group.add_argument("--input_csv", type=str, help="CSV file with the relevant filenames, start times, durations, and keywords. For an example, use the --example_input flag.")
    ieeg_group.add_argument("--dataset", type=str, help="iEEG.org Dataset name. Useful if downloading just one dataset,")
    ieeg_group.add_argument("--start", type=float, help="Start time of clip in usec. Useful if downloading just one dataset,")
    ieeg_group.add_argument("--duration", type=float, help="Duration of clip in usec. Useful if downloading just one dataset,")
    ieeg_group.add_argument("--failure_file", default='./failed_ieeg_calls.csv', type=str, help="CSV containing failed iEEG calls.")    
    ieeg_group.add_argument("--annotations", action='store_true', default=False, help="Download by annotation layers. Defaults to scalp layer names.")
    ieeg_group.add_argument("--time_layer", type=str, default='EEG clip times', help="Annotation layer name for clip times.")
    ieeg_group.add_argument("--annot_layer", type=str, default='Imported Natus ENT annotations', help="Annotation layer name for annotation strings.")
    ieeg_group.add_argument("--timeout", type=int, default=60, help="Timeout interval for ieeg.org calls")

    bids_group = parser.add_argument_group('BIDS keyword options')
    bids_group.add_argument("--uid_number", type=str, help="Unique identifier string to use when not referencing a input_csv file. Only used for single data pulls. Can be used to map the same patient across different datasets to something like an MRN behind clinical firewalls.")
    bids_group.add_argument("--subject_number", type=str, help="Subject string to use when not referencing a input_csv file. Only used for single data pulls.")
    bids_group.add_argument("--session", type=str, help="Session string to use when not referencing a input_csv file. Only used for single data pulls.")
    bids_group.add_argument("--run", type=str, help="Run string to use when not referencing a input_csv file. Only used for single data pulls.")
    bids_group.add_argument("--task", type=str, default='rest', help="Task string to use when not referencing a input_csv file value. Used to populate all entries if not explicitly set.")

    multithread_group = parser.add_argument_group('Multithreading Options')
    multithread_group.add_argument("--multithread", action='store_true', default=False, help="Multithreaded download.")
    multithread_group.add_argument("--ncpu", default=1, type=int, help="Number of CPUs to use when downloading.")
    multithread_group.add_argument("--writeout_frequency", default=10, type=int, help="How many files to download before writing out results and continuing downloads. Too frequent can result in a large slowdown. But for buggy iEEG pulls, frequent saves save progress.")

    misc_group = parser.add_argument_group('Misc options')
    misc_group.add_argument("--include_annotation", action='store_true', default=False, help="If downloading by time, include annotations/events file. Defaults to scalp layer names.")
    misc_group.add_argument("--target", type=str, help="Target to associate with the data. (i.e. PNES/EPILEPSY/etc.)")
    misc_group.add_argument("--example_input", action='store_true', default=False, help="Show example input file structure.")
    misc_group.add_argument("--backend", type=str, default='MNE', help="Backend data handler.")
    misc_group.add_argument("--ch_type", default=None, type=str, help="Manual set of channel type if not matched by known patterns. (i.e. 'seeg' for intracranial data)")
    misc_group.add_argument("--debug", action='store_true', default=False, help="Debug tools. Mainly removes files after generation.")
    args = parser.parse_args()

    # If the user wants an example input file, print it then close application
    if args.example_input:
        print_examples()
        exit()

    # Basic clean-up
    if args.bids_root[-1] != '/': args.bids_root+='/'

    # Main Logic
    if args.ieeg:
        ieeg(args)
    elif args.edf:
        raw_edf(args)
    elif args.jar:
        read_jar(args)
    else:
        print("Please select at least one source from the source group. (--help for all options.)")
