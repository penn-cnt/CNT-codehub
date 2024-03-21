import yaml
import numpy as np
import pandas as PD
from glob import glob
from sys import argv,exit

def TUEG_SLOW_STRING(t_start,t_end,t0,t1,tag):

    # Loop over the arrays
    output = []
    for irow in range(len(t_start)):
        # Break up the temple strings
        try:
            t0_array  = t0[irow].split('_')
            t1_array  = t1[irow].split('_')
            tag_array = tag[irow].split('_') 
        except:
            print(t0[irow])
            print(t1[irow])
            print(tag[irow])
            exit()

        # Make arrays to see if there is any overlap (easier than a bunch of logic gates)
        tagflag     = True
        time_window = np.around(np.arange(t_start[irow],t_end[irow],0.1),1)
        for ii in range(len(t0_array)):
            tag_window = np.around(np.arange(float(t0_array[ii]),float(t1_array[ii]),0.1),1)
            if np.intersect1d(time_window,tag_window).size > 0:
                output.append(tag_array[ii])
                flag = False
        if tagflag:
            output.append("INTERSLOW")
    return output


if __name__ == '__main__':

    # Get the root directory to search for data within
    rootdir = argv[1]
    if rootdir[-1] != '/': rootdir += '/'

    # Find all the feature dataframes
    files = glob(f"{rootdir}*features*pickle")
    
    # Read in the mapping file from yaml
    yaml_dict = yaml.safe_load(open(argv[2],'r'))

    # Get the target sources
    target_sources = yaml_dict['target']['sources']

    # Make the output dataframe object
    output = PD.DataFrame(columns=yaml_dict.keys())
    
    # Loop over input files and read in their data
    for ifile in files:
        
        # Read in the data
        iDF           = PD.read_pickle(ifile)
        iDF['target'] = np.nan

        # Handle some of the unique logic cases for cleanup
        ### Temple Data
        varname = 'TUEG_dt_tag'
        if varname in target_sources and varname in iDF.columns:
            mask    = (iDF[varname].values!=None)
            t_start = iDF['t_start'].astype('float').values[mask]
            t_end   = iDF['t_end'].astype('float').values[mask]
            t0      = iDF['TUEG_dt_t0'].values[mask]
            t1      = iDF['TUEG_dt_t1'].values[mask]
            tags    = iDF[varname].values
            targets = TUEG_SLOW_STRING(t_start,t_end,t0,t1,tags)
            iDF['target'].iloc[mask] = targets
            
            print(iDF)
            exit()