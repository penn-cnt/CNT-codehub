# Set the random seed
import random as rnd
rnd.seed(42)

import os
import pickle
import argparse
import numpy as np
import pandas as PD
import pylab as PLT
from sys import argv
from tqdm import tqdm
import seaborn as sns
from sklearn.metrics import roc_auc_score
from sklearn.decomposition import PCA,SparsePCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler,StandardScaler,RobustScaler

# Models
from sklearn.linear_model import LogisticRegression

# Project imports
from models.lr import *
from cleaning.data_clean import *

def lr_handler(args,vectors,holdout):

    # Run tests as needed
    if not os.path.exists(args.model_output):
        # Run a logistic regression model
        ntests     = 25
        lr_results = {'acc_train':{},'acc_test':{},'auc':{},'auc_holdout':{},'clf':{},'sratios':{}}
        stypes     = ['standard']
        for itype in stypes:
            
            # Let user know the current scaler type
            print(f"Testing the {itype} scaler.")

            # Add in the scaler types
            for ikey in list(lr_results.keys()):
                lr_results[ikey][itype] = []

            for idx in tqdm(range(ntests), total=ntests):

                # Run the logistic regression model
                LRH = LR_handler(vectors,holdout,args.no_strat)
                LRH.data_scale(stype=itype)
                LRH.data_split()
                acc_train,acc_test,auc,hold_auc,clf,sratios = LRH.simple_LR()

                print(acc_train,acc_test,auc,hold_auc)

def reshape_dataframe(df,channels):
    reshaped_data = {}
    for file_id in df['file'].unique():
        print(f"Working on file {file_id}.")
        for time_id in df['t_start'].unique():
            for group_id in df['tag'].unique():
                subset = df[(df['file'] == file_id) & (df['t_start'] == time_id) & (df['tag'] == group_id)]
                for col in channels:
                    reshaped_data[f"{col}_{file_id}_{time_id}_{group_id}"] = subset[col].values
    return PD.DataFrame(reshaped_data)

if __name__ == '__main__':
    
    # Command line options needed to obtain data.
    parser = argparse.ArgumentParser(description="iEEG to bids conversion tool.")
    parser.add_argument("--lr_file", type=str, required=True, help="Filepath to slowing lr fitter.")
    parser.add_argument("--dataset", type=str, required=True, help="Output Dataset Filepath for initial cleanup.")
    parser.add_argument("--model_file", type=str, default="../../user_data/derivative/slowing/032924/merged_model.pickle", help="Merged Model file path.")
    parser.add_argument("--map_file", type=str, default="../../user_data/derivative/slowing/032924/merged_map.pickle", help="Merged map file path.")
    parser.add_argument("--model_output", type=str, required=True, help="Output path for logistic regression fitting.")
    parser.add_argument("--pca", action='store_true', default=False, help="Apply PCA first.")
    parser.add_argument("--npca", type=int, default=200, help="Number of PCA components")
    parser.add_argument("--ptitle", type=str, default='', help="Plot title.")
    parser.add_argument("--drop_peak", action='store_true', default=False, help="Dont use peak info.")
    parser.add_argument("--no_strat", action='store_true', default=False, help="Dont use peak info.")
    args = parser.parse_args()

    # Create dataset as needed
    if not os.path.exists(args.dataset):
        # Read in the data
        DF      = PD.read_pickle(args.model_file).reset_index(drop=True)
        mapping = pickle.load(open(args.map_file,'rb'))
        print(f"Original data of shape {DF.shape}.")

        # Clean the data
        CC = CLEAN_CLASS(DF)
        DF = CC.clean_data()
        print(f"Cleaned data of shape {DF.shape}.")

        # Prepare the data
        DP = DATA_PREP(DF)
        DP.get_channels()
        DF,channels = DP.return_data()

        # Loop over columns and downcast
        for icol in DF.columns:
            try:
                DF[icol] = PD.to_numeric(DF[icol],downcast='integer')
                DF[icol] = PD.to_numeric(DF[icol],downcast='float')
            except ValueError:
                pass

        # Drop duplictes
        DF = DF.drop_duplicates(ignore_index=True)

        # Save the output
        pickle.dump((DF,channels),open(args.dataset,"wb"))
    else:
        mapping      = pickle.load(open(args.map_file,'rb'))
        DF, channels = PD.read_pickle(args.dataset)

    # Make the mapping dictionaries
    tag_map  = list(mapping['tag'])
    tag_dict = dict(zip(np.arange(len(tag_map)).ravel(),np.array(tag_map).ravel()))

    # Create the vectors
    MI                  = make_inputs(DF,tag_dict)
    rawvectors,channels = MI.vector_creation()
    targets             = rawvectors['target'] 

    # Apply PCA
    if args.pca and not os.path.exists(args.model_output):
        print("Applying PCA...")
        pca_enc              = PCA(n_components=args.npca, svd_solver='full')
        rawvectors_trans     = pca_enc.fit_transform(rawvectors[channels].values)
        channels             = [f"{ii:03d}" for ii in range(rawvectors_trans.shape[1])]
        rawvectors           = PD.DataFrame(rawvectors_trans,columns=channels)
        rawvectors['target'] = targets
        print("Finished PCA...")

    # Create a hold out dataset
    vectors, holdout = train_test_split(rawvectors,test_size=0.2,stratify=targets)

    # Get the predictions
    #clf = pickle.load(open(args.lr_file,"rb"))['clf']['standard']

    # Run the LR
    lr_handler(args,vectors,holdout)
