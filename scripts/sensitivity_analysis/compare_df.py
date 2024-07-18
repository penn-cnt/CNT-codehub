import glob
import pickle
import argparse
import numpy as np
import pandas as PD
from os import path
from sys import argv

def update_data_dict(ifile,data_dict):

    # Define the values to index by. This allows sorting and to find channel names
    blacklist = ['file', 't_start', 't_end', 'dt', 'method', 'tag','uid','target','annotation']
    droplist  = ['t_end','uid','target','annotation']
    indexlist = list(np.sort(np.setdiff1d(blacklist,droplist)))

    iDF              = PD.read_pickle(ifile)
    channels         = np.setdiff1d(iDF.columns,blacklist)
    iDF              = iDF.drop(droplist,axis=1)
    data_dict[ifile] = iDF.set_index(indexlist)
    return data_dict,channels

if __name__=='__main__':

    parser = argparse.ArgumentParser(description="Simplified data merging tool.")
    parser.add_argument("--searchpath", type=str, required=True, help='Search path for files to compare.')
    args = parser.parse_args()

    # Find the filelist to read in
    filelist = glob.glob(args.searchpath)
    
    # Create data dict if needed
    outpath = "dcomp.pickle"
    if not path.exists(outpath):
        # Loop over the files and read them in
        data_dict = {}
        for ifile in filelist:
            print(ifile)
            data_dict, channels = update_data_dict(ifile,data_dict)
        pickle.dump((data_dict,channels),open(outpath,"wb"))
    else:
        data_dict, channels = pickle.load(open(outpath,"rb"))

    # Go through indices and compare
    keys        = list(data_dict.keys())
    index_lists = [data_dict[ikey].index for ikey in keys]