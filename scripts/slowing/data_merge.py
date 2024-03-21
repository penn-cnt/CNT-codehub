import yaml
import numpy as np
import pandas as PD
from glob import glob
from sys import argv,exit

def TUEG_SLOW_STRING(t_start,t_end,t0,t1,tag):



if __name__ == '__main__':

    # Get the root directory to search for data within
    rootdir = argv[1]
    if rootdir[-1] != '/': rootdir += '/'

    # Find all the feature dataframes
    files = glob(f"{rootdir}*features*pickle")
    
    # Read in the mapping file from yaml
    yaml_dict = yaml.safe_load(open(argv[2],'r'))

    print(yaml_dict['target']['sources'])
    exit()

    # Make the output dataframe object
    output = PD.DataFrame(columns=yaml_dict.keys())
    
    # Loop over input files and read in their data
    for ifile in files:
        
        # Read in the data
        iDF = PD.read_pickle(ifile)

        # Handle some of the unique logic cases for cleanup
        ### Temple Data
        if 'TUEG_dt_tag' in yaml_dict['target']['sources']:
            print("hi")