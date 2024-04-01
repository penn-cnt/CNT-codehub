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