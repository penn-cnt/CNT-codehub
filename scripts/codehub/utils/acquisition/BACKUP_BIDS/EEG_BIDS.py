import glob
import argparse
import numpy as np
import pandas as PD
from os import path
from sys import exit
from time import sleep

# Locate import
from modules.EDF_handler import EDF_handler
from modules.iEEG_handler import ieeg_handler

# For testing, mute mne future warning
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)

if __name__ == '__main__':

    # Command line options needed to obtain data.
    parser = argparse.ArgumentParser(description="iEEG to bids conversion tool.")

    source_group = parser.add_mutually_exclusive_group()
    source_group.add_argument("--ieeg", action='store_true', default=False, help="Obtain data from iEEG.org.")
    source_group.add_argument("--edf", action='store_true', default=False, help="Convert local EDF data.")

    edf_group = parser.add_argument_group('EDF input options')
    edf_group.add_argument("--edf_path", type=str, help="Path to the EDF file.")

    ieeg_group = parser.add_argument_group('iEEG connection options')
    ieeg_group.add_argument("--username", type=str, help="Username for iEEG.org.")
    ieeg_group.add_argument("--password", type=str, help="Password for iEEG.org.")
    ieeg_group.add_argument("--dataset", type=str, help="iEEG.org Dataset name")
    ieeg_group.add_argument("--start", type=float, help="Start time of clip")
    ieeg_group.add_argument("--duration", type=float, help="Duration of clip")
    ieeg_group.add_argument("--failure_file", default='./failed_ieeg_calls.csv', type=str, help="CSV containing failed iEEG calls.")

    bids_group = parser.add_argument_group('BIDS options')
    bids_group.add_argument("--bidsroot", type=str, required=True, help="Bids Root Directory.")
    bids_group.add_argument("--session", type=str, required=True, help="Base string session keyword for BIDS. (i.e. 'preimplant')")
    bids_group.add_argument("--session_num", type=int, default=-1, help="Session number for this file.")
    bids_group.add_argument("--run_num", type=int, default=-1, help="Run number for this file.")
    bids_group.add_argument("--subject_prefix", type=str, help="Prefix to add to a subject number i.e. 'HUP'->sub-HUP00001")

    other_group = parser.add_argument_group('Other options')
    other_group.add_argument("--inputs_file", type=str, help="Optional file of input datasets to (download and) turn into BIDS.")
    other_group.add_argument("--subject_file", type=str, default='subject_map.csv', help="File mapping subject id to ieeg file. (Defaults to bidroot+'subject_map.csv)")
    other_group.add_argument("--uid", default=0, type=int, help="Unique patient identifier for single ieeg calls. This is to map patients across different admissions. See sample subject_map.csv file for an example.")
    other_group.add_argument("--subnum", default=0, type=int, help="Subject number to use for the bids folders.")
    other_group.add_argument("--target", default=None, type=str, help="Target value to associate with single subject inputs. (i.e. epilepsy vs. pnes)")
    other_group.add_argument("--ch_type", default=None, type=str, help="Manual set of channel type if not matched by known patterns. (i.e. 'seeg' for intracranial data)")
    other_group.add_argument("--multithread", action='store_true', default=False, help="Multithreaded download.")
    other_group.add_argument("--ncpu", default=1, type=int, help="Number of CPUs to use when downloading.")
    other_group.add_argument("--debug", action='store_true', default=False, help="Print error messages.")
    other_group.add_argument("--silent", action='store_true', default=False, help="Silence progress messages.")
    other_group.add_argument("--skipzero", action='store_true', default=False, help="Time zero is almost always garbage data. This flag skips it. Good for quickly going back over data to catch missed iEEG calls.")

    selection_group = parser.add_mutually_exclusive_group()
    selection_group.add_argument("--cli", action='store_true', default=False, help="Use start and duration from this CLI.")
    selection_group.add_argument("--annotations", action='store_true', default=False, help="CSV file with de-identified unique patient id, ieeg filename, and targets (optional). Format:[uid,ieeg_filename,target]")
    args = parser.parse_args()

    # Clean up directory structure
    if args.bidsroot[-1] != '/':
        args.bidsroot += '/'

    # Input data array generation
    incols = ['orig_filename','uid','subject_number','session_number','run_number','start','duration','target']
    if args.cli:
        input_data  = PD.DataFrame([[args.dataset,args.uid,args.subnum,args.session_num,args.run_num,args.start,args.duration,args.target]],columns=incols)
    elif args.annotations:
        input_data  = PD.DataFrame([[args.dataset,args.uid,args.subnum,args.session_num,args.run_num,-1,-1,args.target]],columns=incols)
    
    # Use input file if provided. Cleanup if missing columns
    if args.inputs_file != None:
        input_data = PD.read_csv(args.inputs_file)
        col_diff   = np.setdiff1d(incols,input_data.columns)
        for icol in col_diff:
            input_data[icol] = -1

    # Make sure we have numeric types
    for icol in input_data.columns:
        try:
            input_data[icol] = PD.to_numeric(input_data[icol],downcast='float')
        except ValueError:
            pass

    # If iEEG.org, pass inputs to that handler to get the data
    if args.ieeg:
        IH = ieeg_handler(args,input_data)
        if not args.multithread:
            IH.single_pull()
        else:
            IH.multicore_pull()
    elif args.edf:
        EH = EDF_handler(args,input_data)
        EH.save_data()

    # Make a bids ignore file
    fp = open(args.bidsroot+'.bidsignore','w')
    fp.write('%s\n' %(args.subject_file))
    fp.write('**targets**pickle')
    fp.close()