import yaml
import numpy as np
import pandas as PD
from sys import argv
from glob import glob

if __name__ == '__main__':

    # Get the root directory to search for data within
    rootdir = argv[1]
    if rootdir[-1] != '/': rootdir += '/'

    # Find all the feature dataframes
    files = glob(f"{rootdir}*features*pickle")
    
    # Read in the mapping file from yaml
    yaml_dict = yaml.safe_load(open(argv[2],'r'))

    # Make the output dataframe object
    output = PD.DataFrame(columns=yaml_dict.keys())
    print(output)