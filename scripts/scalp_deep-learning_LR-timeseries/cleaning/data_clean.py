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
        self.clean_partial_NaN_None()
        return self.data

    def drop_all_NaN_rows(self):
        self.data = self.rawdata.dropna(how='all')

    def clean_partial_NaN_None(self,fillval=-1):
        self.data = self.data.fillna(value=fillval)
        self.data = self.data.replace(to_replace='nan',value=-1)
        self.data = self.data.replace(to_replace='None',value=-1)

class DATA_PREP:

    def __init__(self,DF):
        self.DF = DF

    def return_data(self):
        return self.DF,self.channels

    def get_channels(self):

        # Get the channel columns
        nonchan       = ['file','t_start','tag','TUEG']
        self.channels = np.setdiff1d(self.DF.columns,nonchan)

    def update_TUEG_targets(self,mapping):

        # Make a target dictionary
        target_map                   = np.array(list(mapping['TUEG'])).astype('<U16')
        target_map[(target_map=='')] = 'interslow'
        target_dict                  = dict(zip(np.arange(len(target_map)).ravel(),np.array(target_map).ravel()))

        # Inform user of class balance
        targets     = self.DF['TUEG'].values
        uvals,ucnts = np.unique(targets,return_counts=True)

        # Find the indices to keep and drop
        keep_targets = ['interslow','slow','noslow']
        keep_inds    = []
        conversion   = {}
        for k, v in target_dict.items():
            if v in keep_targets:
                keep_inds.append(k)
                if v == 'noslow':conversion[k] = 0
                if v == 'interslow':conversion[k] = 1
                if v == 'slow':conversion[k] = 1
        self.DF               = self.DF.loc[self.DF.TUEG.isin(keep_inds)]
        self.DF.loc[:,'TUEG'] = self.DF['TUEG'].map(conversion)

        # Loop over columns and downcast
        for icol in self.DF.columns:
            try:
                self.DF.loc[:,icol] = PD.to_numeric(self.DF[icol],downcast='integer')
                self.DF.loc[:,icol] = PD.to_numeric(self.DF[icol],downcast='float')
            except ValueError:
                pass

class make_inputs:

    def __init__(self,DF):
        self.DF = DF

    def vector_creation(self):

        # Clean up the tueg label if present
        if 'TUEG' in self.DF.columns:
            self.DF = self.DF.rename(columns={"TUEG": "target"})

        # Get the tags to iterate over
        all_tags = self.DF.tag.unique()

        # Loop over tag data
        new_DF = self.DF.loc[self.DF.tag==all_tags[0]].drop(['tag'],axis=1)
        for idx,itag in enumerate(all_tags[1:]):
            iDF = self.DF.loc[self.DF.tag==itag].drop(['tag'],axis=1)

            if idx == 0:
                new_DF = PD.merge(new_DF, iDF, how='outer', on=['file','t_start','target'], suffixes=[f"_0",f"_{itag}"])
            else:
                new_DF = PD.merge(new_DF, iDF, how='outer', on=['file','t_start','target'], suffixes=["",f"_{itag}"])

        # Get the channels
        channels = np.setdiff1d(new_DF.columns,['file','t_start','target'])

        return new_DF,channels