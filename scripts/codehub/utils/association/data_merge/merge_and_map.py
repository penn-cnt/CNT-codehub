import glob
import pickle
import argparse
import numpy as np
import pandas as PD

def merge_data(searchpath,map_dict):

    # Get the columns to map
    map_keys = list(map_dict.keys())

    # Read in the files in order, and apply mapping, storing to the output
    for idx,ifile in enumerate(searchpath):

        # Special condition for first time load, otherwise append
        if idx == 0:
            
            # Read in data to final variable name
            new_df = PD.read_pickle(ifile)
            
            # Apply any needed mapping
            if len(map_keys) > 0:
                for ikey in map_keys:
                    new_df[ikey] = new_df[ikey].apply(lambda x: map_dict[ikey][x])
        else:
            # Read in data to temporary namespace
            idf = PD.read_pickle(ifile)

            # Apply any needed mapping
            if len(map_keys) > 0:
                for ikey in map_keys:
                    idf[ikey] = idf[ikey].apply(lambda x: map_dict[ikey][x])

            # Append results to final variable name
            new_df = PD.concat((new_df,idf))
    return new_df

def create_map(searchpath,cols):
    
    # Make the data dictionary to store info for each column that needs to be mapped
    data_dict = {}
    for icol in cols:
        data_dict[icol] = []

    # Read in the files in order, and grab just the mapping columns to reduce memory usage
    print("Generating a unique mapping for each column. This may take awhile.")
    for ifile in searchpath:
        iDF = PD.read_pickle(ifile)[cols]
        for icol in cols:
            vals = list(iDF[icol].unique())
            data_dict[icol].extend(vals)

    # Create the mapping dictionary
    map_dict = {}
    for icol in cols:

        # Create the mapping dictionary
        uvals   = np.unique(data_dict[icol])
        newvals = np.arange(uvals.size)
        udict   = dict(zip(uvals.ravel(),newvals.ravel()))

        # Save the mapping
        map_dict[icol] = udict

    return map_dict


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
    return [value for value in values]

if __name__ == '__main__':
    """
    Create a mapping file from EPIPY output for given columns, downcast to mapped varaibles, and merge the files together.
    Reduces size in memory of the dataframe.
    """

    # Argument parsing
    parser = argparse.ArgumentParser(description="Simplified data merging tool.")
    parser.add_argument("--searchpath", type=str, required=True, help='Search path for files to map and downcast. Wildcard enabled.')
    parser.add_argument("--cols", type=parse_list, help="Comma separated list of columsn to map and downcast.")
    parser.add_argument("--outfile_data", default="merged_data.pickle", type=str, help='Output filename for the new merged and downcasted')
    parser.add_argument("--outfile_map", default="merged_map.pickle", type=str, help='Output filename for the mapped data column dictionaries')
    parser.add_argument("--mapping_file", default=None, type=str, help="Optional filepath to an exisitng mapping file. Useful if you are doing sensitivity analysis or reprocessing the same dataset.")
    args = parser.parse_args()

    # Read in the searchpath and create a filelist
    filelist = glob.glob(args.searchpath)

    # Get the list of columns to map, if any
    if args.mapping_file == None:
        if len(args.cols) > 0:
            map_dict = create_map(filelist,args.cols)
            pickle.dump(map_dict,open(args.outfile_map,"wb"))
        else:
            map_dict = {}
    else:
        print("Using existing mapping file.")
        map_dict = pickle.load(open(args.mapping_file,"rb"))
    
    # Merge the files
    out_DF = merge_data(filelist,map_dict)

    # Save the results
    out_DF.to_pickle(args.outfile_data)

