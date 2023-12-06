import pickle
import argparse
import numpy as np
import pandas as PD
from os import path
from sys import exit
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import StandardScaler

def vectorize_awake_sleep(annotations):

    # Loop logic gate for the sleep awake annotations
    output = []
    for idx,iann in enumerate(annotations):
        ival = iann.lower()
        if 'wake' in ival or 'awake' in ival or 'pdr' in ival:
            output.append(1)
        elif 'sleep' in ival or 'spindle' in ival or 'k complex' in ival or 'sws' in ival:
            output.append(2)
        else:
            output.append(0)
    output = np.array(output).reshape((-1,1))

    # Get the one hot encoded results
    encoder = LabelBinarizer()
    one_hot_encoded = encoder.fit_transform(output)
    return PD.DataFrame(one_hot_encoded,columns=['Unknown Sleep State','Awake','Sleep'])

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
    return [int(value) for value in values]

if __name__ == '__main__':

    # Argument parsing
    parser = argparse.ArgumentParser(description="Simplified data merging tool.")
    parser.add_argument("--infile", type=str, help='Input data file')
    parser.add_argument("--plotdir", type=str, help='Output plot directory')
    parser.add_argument("--datadir", type=str, help='Output data directory')
    parser.add_argument("--processed_file", type=str, help=r"Filename for processed data. Defaults to {datadir}/preprocessed_data.pickle")

    dataprep_group = parser.add_argument_group('Data Prep Options')
    dataprep_group.add_argument("--exclude_target_list", type=parse_list, default=[2,3], help="Values in the target vector to exclude.")
    dataprep_group.add_argument("--log_transform", action='store_true', default=False, help="Take a log-10 transform of the input vectors to reduce scale.")
    dataprep_group.add_argument("--annotation_sleep", action='store_true', default=False, help="Use annotations to determine sleep-state if possible.")
    args = parser.parse_args()

    # Ensure that the directory strings have proper trailing character
    if args.plotdir[-1] != '/':
        args.plotdir += '/'
    if args.datadir[-1] != '/':
        args.datadir += '/'

    # Create a default output data path if needed
    if args.processed_file == None:
        args.processed_file = f"{args.datadir}/preprocessed_data.pickle"

    # Check if the scaled data exists
    if not path.exists(args.processed_file):

        # Read in the data
        rawdata = PD.read_pickle(args.infile)
        rawdata = rawdata.loc[~rawdata.target.isin(args.exclude_target_list)]
        
        # Get the channel names
        channels = []
        for icol in rawdata.columns:
            if icol not in ['uid','target','annotation','tag']:
                channels.append(icol)

        # Add in sleep wake information if available
        if args.annotation_sleep:
            categorical_df = vectorize_awake_sleep(rawdata['annotation'].values)
            for icol in categorical_df.columns:
                rawdata[icol] = categorical_df[icol].values

        # Break out the exogenous and endogenous portions
        raw_Y = rawdata.target.values
        raw_X = rawdata.drop(['target','annotation'],axis=1)

        # Try a log transformation on the data
        if args.log_transform:
            for ichan in channels:
                raw_X[ichan] = np.log10(raw_X[ichan].values)

        # Normalize the input vectors using StandardScaler
        scaler              = StandardScaler()
        scaled_channel_data = scaler.fit_transform(raw_X[channels])
        for idx,ichan in enumerate(channels):
            raw_X[ichan] = scaled_channel_data[:,idx]
        
        # Save the results for future data loads
        pickle.dump((raw_X,raw_Y, args), open(args.processed_file,"wb"))
    else:
        raw_X, raw_Y, creation_args = pickle.load(open(args.processed_file,"rb"))