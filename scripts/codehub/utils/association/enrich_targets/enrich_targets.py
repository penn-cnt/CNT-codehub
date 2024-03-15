import os
import pickle
import argparse
import numpy as np
import pandas as PD
from sys import exit

class CustomFormatter(argparse.HelpFormatter):
    """
    Custom formatting class to get a better argument parser help output.
    """

    def _split_lines(self, text, width):
        if text.startswith("R|"):
            return text[2:].splitlines()
        return super()._split_lines(text, width)

class data_reader:

    def __init__(self,infile):
        
        # Create class-wide variables
        self.infile = infile

    def enrichment_keypair(self):
        
        # Read in the key-pair
        fp   = open(self.infile,'r')
        data = fp.readline()
        fp.close()

        # Clean up the string and break it up
        data       = data.replace('\n','')
        data_array = data.split(',')

        # Make the new output target dict 
        output = {}
        output[data_array[0]] = data_array[1]

        return output

    def TUEG_dt(self):
        
        # Read in the tsv using pandas so we can just skip rows and assign column headers
        DF     = PD.read_csv(self.infile,skiprows=2,delimiter=' ', names=['t0','t1','tag','prob'])
        
        # Create an output dictionary that will be merged with the current targets
        output = {}
        output['TUEG_dt_t0']  = '_'.join([f"{ival:.1f}" for ival in DF['t0'].values])
        output['TUEG_dt_t1']  = '_'.join([f"{ival:.1f}" for ival in DF['t1'].values])
        output['TUEG_dt_tag'] = '_'.join(DF['tag'].values)
        
        return output

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

    # Define the allowed enrichment information
    allowed_enrichment_types                = {}
    allowed_enrichment_types['TUEG_TSV_dt'] = 'Read in a TUEG tsv file and assign the target variable to a time window.'
    allowed_enrichment_types['keypair']     = 'Add a keypair to the targets'
    allowed_enrichment_help                 = make_help_str(allowed_enrichment_types)

    # Command line options needed to obtain data.
    parser = argparse.ArgumentParser(description="iEEG to bids conversion tool.", formatter_class=CustomFormatter)
    parser.add_argument("--target_file", type=str, help="Path to target file to enrich.")
    parser.add_argument("--enrichment_type", type=str, choices=list(allowed_enrichment_types.keys()), default="TUEG_TSV_dt", help=f"R|Choose an option:\n{allowed_enrichment_help}")

    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument("--enrichment_map", type=str, help="Csv file with enrichment info. Columns:[path_to_datafile_that_enriches_target,path_to_target_file,enrichment_type].")
    input_group.add_argument("--enrichment_file", type=str, help="Path to datafile that will enrich target.")
    args = parser.parse_args()

    # Check which type of input format we are working with, and create relevant work list
    if args.enrichment_map != None and args.enrichment_file == None:
        enrichment_df    = PD.read_csv(args.enrichment_map)
        enrichment_files = enrichment_df['enrichment_files'].values 
        target_files     = enrichment_df['target_files'].values
        enrichment_types = enrichment_df['enrichment_types'].values
    elif args.enrichment_map == None and args.enrichment_file != None:
        # Store the enrichment data
        enrichment_files = [args.enrichment_file]

        # Check for target file data and store if provided, or warn user
        if args.target_file != None:
            target_files = [args.target_file]
        else:
            raise FileNotFoundError("Please provide a target file to enrich using the --target_file keyword.")

        # Check for enrichment type and store if provided, or warn user
        if args.target_file != None:
            enrichment_types = [args.enrichment_type]
        else:
            raise NameError("Please provide an enrichment type using the --enrichment_type flag. Using --help will show allowed enrichment types.")

    else:
        raise FileNotFoundError("Please provide an enrichment file or an enrichment map file.")
    
    # Loop over all of the target files to enrich
    for idx,ifile in enumerate(target_files):

        # Confirm if the target file exists, read in or create as needed
        if os.path.exists(ifile):
            targets = pickle.load(open(ifile,"rb"))
        else:
            targets = {}

        # Initialize the data reading class
        DR = data_reader(enrichment_files[idx])

        # Apply the right enrichment logic to get out data
        if args.enrichment_type == 'TUEG_TSV_dt':
            additional_targets = DR.TUEG_dt()
        elif args.enrichment_type == 'keypair':
            additional_targets = DR.enrichment_keypair()
        new_targets = {**targets,**additional_targets}

        # Save the new target file
        pickle.dump(new_targets,open(ifile,"wb"))
