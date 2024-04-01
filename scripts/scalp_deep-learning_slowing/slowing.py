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
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler,StandardScaler,RobustScaler

# Models
from sklearn.linear_model import LogisticRegression

# Project imports
from models.lr import *
from models.mlp import *
from cleaning.data_clean import *

def NN_handler(args,vectors,holdout):

    NN = mlp(vectors,holdout)
    NN.data_scale(stype='standard')
    NN.data_split()
    NN.run_network()

def lr_handler(args,vectors,holdout):

    # Run tests as needed
    if not os.path.exists(args.lr_output):
        # Run a logistic regression model
        ntests     = 25
        lr_results = {'acc_train':{},'acc_test':{},'auc':{},'auc_holdout':{},'clf':{}}
        stypes     = ['minmax','standard','robust']
        for itype in stypes:
            
            # Let user know the current scaler type
            print(f"Testing the {itype} scaler.")

            # Add in the scaler types
            for ikey in list(lr_results.keys()):
                lr_results[ikey][itype] = []

            for idx in tqdm(range(ntests), total=ntests):

                # Run the logistic regression model
                LRH = LR_handler(vectors,holdout)
                LRH.data_scale(stype=itype)
                LRH.data_split()
                acc_train,acc_test,auc,hold_auc,clf = LRH.simple_LR()

                # Save the results
                lr_results['acc_train'][itype].append(acc_train)
                lr_results['acc_test'][itype].append(acc_test)
                lr_results['auc'][itype].append(auc)
                lr_results['auc_holdout'][itype].append(hold_auc)
                lr_results['clf'][itype].append(clf)

                # Make sure we randomize the training data
                vectors.sample(frac=1)

        pickle.dump(lr_results,open(args.model_output,"wb"))
        pickle.dump(holdout,open("holdouts.pickle","wb"))
    else:
        lr_results = pickle.load(open(args.model_output,'rb'))
        ntests     = len(lr_results['auc'][list(lr_results['auc'].keys())[0]])

    # Make the outcome dataframe
    LRDF         = PD.DataFrame(columns=['solver','auc','flag'])
    LRDF2        = PD.DataFrame(columns=['solver','auc','flag'])
    svals        = np.repeat(list(lr_results['auc'].keys()),ntests)
    aucs         = np.array(list(lr_results['auc'].values())).flatten()
    aucs_holdout = np.array(list(lr_results['auc_holdout'].values())).flatten()
    accs         = np.array(list(lr_results['acc_test'].values())).flatten()
    LRDF['solver']   = svals
    LRDF['auc']      = aucs
    LRDF['dataset']  = 'testdata'
    LRDF['acc']      = accs
    LRDF2['solver']  = svals
    LRDF2['auc']     = aucs_holdout
    LRDF2['dataset'] = 'holdouts'
    LRDF_COMB = PD.concat((LRDF,LRDF2))

    # Plot the results
    fig = PLT.figure(dpi=100.,figsize=(5.,5.))
    ax1 = fig.add_subplot(111)
    sns.boxplot(data=LRDF_COMB, x="solver", y="auc", fill=False, hue="dataset",ax=ax1)
    ax1.set_title(f"{args.ptitle}",fontsize=14)
    PLT.show()

    fig = PLT.figure(dpi=100.,figsize=(5.,5.))
    ax1 = fig.add_subplot(111)
    sns.boxplot(data=LRDF, x="solver", y="acc", fill=False,ax=ax1, hue='dataset')
    ax1.set_title(f"{args.ptitle}",fontsize=14)
    PLT.show()

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
    parser.add_argument("--dataset", type=str, required=True, help="Output Dataset Filepath for initial cleanup.")
    parser.add_argument("--model_file", type=str, default="../../user_data/derivative/slowing/032924/merged_model.pickle", help="Merged Model file path.")
    parser.add_argument("--map_file", type=str, default="../../user_data/derivative/slowing/032924/merged_map.pickle", help="Merged map file path.")
    parser.add_argument("--model_output", type=str, required=True, help="Output path for logistic regression fitting.")
    parser.add_argument("--pca", action='store_true', default=False, help="Apply PCA first.")
    parser.add_argument("--npca", type=int, default=200, help="Number of PCA components")
    parser.add_argument("--ptitle", type=str, default='', help="Plot title.")
    parser.add_argument("--drop_peak", action='store_true', default=False, help="Dont use peak info.")
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
        DP.update_targets(mapping)
        if not args.drop_peak:
            DP.peak_freqs()
        else:
            DP.drop_peak()
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

    # Run the LR
    #lr_handler(args,vectors,holdout)

    # Run the NN
    NN_handler(args,vectors,holdout)