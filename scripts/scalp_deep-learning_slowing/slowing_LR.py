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

class CLEAN_CLASS:

    def __init__(self,rawdata):
        self.rawdata = rawdata

    def clean_data(self):
        self.drop_all_NaN_rows()
        self.clean_partial_NaN()
        return self.data

    def drop_all_NaN_rows(self):
        self.data = self.rawdata.dropna(how='all')

    def clean_partial_NaN(self,fillval=-1):
        self.data = self.data.fillna(value=fillval)
        self.data = self.data.replace(to_replace='nan',value=-1)

class DATA_PREP:

    def __init__(self,DF):
        self.DF = DF

    def return_data(self):
        return self.DF,self.channels

    def get_channels(self):

        # Get the channel columns
        nonchan       = ['file','t_start','tag','TUEG']
        self.channels = np.setdiff1d(self.DF.columns,nonchan)

    def update_targets(self,mapping):

        # Make a target dictionary
        target_map                   = list(mapping['TUEG'])
        target_map[(target_map=='')] = 'interslow'
        target_dict                  = dict(zip(np.arange(len(target_map)).ravel(),np.array(target_map).ravel()))

        # Inform user of class balance
        targets     = self.DF['TUEG'].values
        uvals,ucnts = np.unique(targets,return_counts=True)
        for idx,ival in enumerate(uvals):
            print(f"Target {target_dict[ival]:15} is ~{ucnts[idx]/ucnts.sum():.3f}% of the data.")

        # Find the indices to keep and drop
        keep_targets = ['interslow','slow','noslow']
        keep_inds    = []
        conversion   = {}
        for k, v in target_dict.items():
            print(k,v)
            if v in keep_targets:
                keep_inds.append(k)
                if v == 'noslow':conversion[k] = 0
                if v == 'interslow':conversion[k] = 1
                if v == 'slow':conversion[k] = 1
        self.DF         = self.DF.loc[self.DF.TUEG.isin(keep_inds)]
        self.DF['TUEG'] = self.DF['TUEG'].map(conversion)
        print(f"New data shape is {self.DF.shape}.")

        # Loop over columns and downcast
        for icol in DF.columns:
            try:
                self.DF[icol] = PD.to_numeric(self.DF[icol],downcast='integer')
                self.DF[icol] = PD.to_numeric(self.DF[icol],downcast='float')
            except ValueError:
                pass

    def peak_freqs(self):

        print("Creating a peak frequency (# of peaks/# number channels)")
        iDF     = self.DF.loc[self.DF.tag==6][self.channels]
        indices = list(iDF.index)
        pfreq   = (iDF.values>0).sum(axis=1)/iDF.shape[1]
        for ichannel in self.channels:
            self.DF.loc[indices,ichannel] = pfreq

    def drop_peak(self):
        self.DF = self.DF.loc[self.DF.tag!=6]

class make_inputs:

    def __init__(self,DF,tag_dict):
        self.DF       = DF
        self.tag_dict = tag_dict

    def frequency_squeeze(self,squeeze_factor=5):

        # Apply any aggregations
        farr           = np.array(list(tag_dict.keys()))
        farr           = farr[(farr<100)]
        farr           = farr.reshape((-1,squeeze_factor))
        toffset        = max(tag_dict.keys())+1
        new_tag_dict   = dict(zip(np.arange(farr.shape[0])+toffset,farr))
        orig_shape     = self.DF.shape
        for itag,iarr in new_tag_dict.items():
            iDF        = self.DF.loc[self.DF.tag.isin(iarr)]
            jDF        = iDF.groupby(['file','t_start'],as_index=False).mean()
            jDF['tag'] = itag
            self.DF    = self.DF.loc[~self.DF.tag.isin(iarr)]
            self.DF    = PD.concat((self.DF,jDF))
        print(f"After frequency squeeze, dataframe went from {orig_shape[0]} rows to {self.DF.shape[0]} rows.")

    def vector_creation(self):

        # Attempt to make the vector inputs
        self.DF = self.DF.drop_duplicates(subset=['file','t_start','tag'])
        vectors = self.DF.set_index(['file', 't_start', 'tag']).unstack(level=-1).unstack(level=-1)
        newcols = []
        mask    = []
        for col in vectors.columns:
            newcols.append('{}_{}_{}'.format(*col))
            if col[0] == 'TUEG':
                mask.append(True)
            else:
                mask.append(False)
        newcols = np.array(newcols)
        vectors.columns = newcols
        vectors.reset_index(inplace=True)
        vectors.set_index('file',inplace=True)
        
        # We dont need expanded target columns, squeeze those in
        targets = [np.unique(iarr)[0] for iarr in vectors[newcols[mask]].values]
        vectors = vectors.drop(newcols[mask],axis=1)
        vectors['target'] = targets

        # Replace NaNs with a more useful value
        vectors = vectors.fillna(value=-1)

        return vectors,newcols[np.invert(mask)]

class LR_handler:

    def __init__(self,vectors,holdout):
        self.incols  = vectors.columns
        self.vectors = vectors
        self.X_cols  = self.incols[self.incols!='target']
        self.Y_cols  = self.incols[self.incols=='target']
        self.X       = vectors[self.X_cols].values
        self.Y       = vectors[self.Y_cols].values
        self.hold_X  = holdout[self.X_cols].values
        self.hold_Y  = holdout[self.Y_cols].values

    def data_scale(self,stype='minmax'):

        if stype.lower() not in ['minmax','standard','robust']:
            print("Could not find {stype} scaler. Defaulting to Standard")
            stype = 'standard'

        if stype.lower() == 'minmax':
            scaler = MinMaxScaler()
        elif stype.lower() == 'standard':
            scaler = StandardScaler()
        elif stype.lower() == 'robust':
            scaler = RobustScaler()

        scaler.fit(self.X)
        self.X_scaled = scaler.transform(self.X)

        # Get the holdout fit
        self.hold_X_scaled = scaler.transform(self.hold_X)

    def data_split(self):

        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(self.X_scaled, self.Y, stratify=self.Y, test_size=0.3)
        
    def simple_LR(self):

        # Fit a LR model
        clf               = LogisticRegression(max_iter=500,solver='liblinear').fit(self.X_train, self.Y_train.flatten())
        self.Y_train_pred = clf.predict(self.X_train).flatten()
        self.Y_test_pred  = clf.predict(self.X_test).flatten()
        
        # Flatten arrays for ease
        self.Y_train = self.Y_train.flatten()
        self.Y_test  = self.Y_test.flatten()

        # Compare results
        acc_train = (self.Y_train==self.Y_train_pred).sum()/self.X_train.shape[0]
        acc_test  = (self.Y_test==self.Y_test_pred).sum()/self.X_test.shape[0]

        # Get the AUC
        auc = roc_auc_score(self.Y_test,self.Y_test_pred)

        # Get the holdout prediction
        self.hold_Y_pred = clf.predict(self.hold_X_scaled).flatten()
        hold_auc    = roc_auc_score(self.hold_Y,self.hold_Y_pred)

        return acc_train,acc_test,auc,hold_auc,clf
    
    def return_values(self):

        return self.Y_train,self.Y_train_pred,self.Y_test,self.Y_test_pred

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
    parser.add_argument("--dataset", type=str, default='../../user_data/derivative/slowing/032924/dataset.pickle', help="Output Dataset Filepath for initial cleanup.")
    parser.add_argument("--model_file", type=str, default="../../user_data/derivative/slowing/032924/merged_model.pickle", help="Merged Model file path.")
    parser.add_argument("--map_file", type=str, default="../../user_data/derivative/slowing/032924/merged_map.pickle", help="Merged map file path.")
    parser.add_argument("--lr_output", type=str, required=True, help="Output path for logistic regression fitting.")
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
    if args.pca and not os.path.exists(args.lr_output):
        print("Applying PCA...")
        pca_enc              = PCA(n_components=args.npca, svd_solver='full')
        rawvectors_trans     = pca_enc.fit_transform(rawvectors[channels].values)
        channels             = [f"{ii:03d}" for ii in range(rawvectors_trans.shape[1])]
        rawvectors           = PD.DataFrame(rawvectors_trans,columns=channels)
        rawvectors['target'] = targets
        print("Finished PCA...")

    # Create a hold out dataset
    vectors, holdout = train_test_split(rawvectors,test_size=0.2,stratify=targets)

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

        pickle.dump(lr_results,open(args.lr_output,"wb"))
        pickle.dump(holdout,open("holdouts.pickle","wb"))
    else:
        lr_results = pickle.load(open(args.lr_output,'rb'))
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
