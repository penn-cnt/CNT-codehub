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

class LR_handler:

    def __init__(self,vectors,holdout,sflag):
        self.incols  = vectors.columns
        self.vectors = vectors
        self.X_cols  = self.incols[self.incols!='target']
        self.Y_cols  = self.incols[self.incols=='target']
        self.X       = vectors[self.X_cols].values
        self.Y       = vectors[self.Y_cols].values
        self.hold_X  = holdout[self.X_cols].values
        self.hold_Y  = holdout[self.Y_cols].values
        self.sflag   = sflag

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

        if not self.sflag:
            self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(self.X_scaled, self.Y, stratify=self.Y, test_size=0.3)
        else:
            self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(self.X_scaled, self.Y, test_size=0.3)
        self.svals_train,self.sratios_train = np.unique(self.Y_train,return_counts=True)
        self.svals_test,self.sratios_test   = np.unique(self.Y_test,return_counts=True)

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
        hold_auc         = roc_auc_score(self.hold_Y,self.hold_Y_pred)

        return acc_train,acc_test,auc,hold_auc,clf,(self.sratios_train,self.sratios_test)
    
    def return_values(self):

        return self.Y_train,self.Y_train_pred,self.Y_test,self.Y_test_pred