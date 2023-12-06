import os
import glob
import argparse
import numpy as np
import pandas as PD
from sys import exit,path
from pyedflib import highlevel
import matplotlib.pyplot as PLT
from scipy.signal import find_peaks
import matplotlib.colors as mcolors
from prompt_toolkit.completion import PathCompleter

from modules.alpha_delta import *

class data_loader():

    def __init__(self,infile, outdir):

        # Define any hard-coded variables here, or save passed arguments to class
        self.outdir    = outdir
        self.alpha_str = '[8.0,12.0]'
        self.delta_str = '[1.0,4.0]'

        # Read in the input dataset and get channel names
        self.data = PD.read_pickle(infile)
        self.get_channels()

        # Create output directory structure as needed
        self.plotdir   = self.outdir+"PLOTS/"
        if not os.path.exists(self.plotdir):
            os.system("mkdir -p %s" %(self.plotdir))

        # Define the colorset
        color_names      = np.array(list(mcolors.BASE_COLORS.keys()))
        self.color_names = []
        for icolor in color_names:
            if icolor not in ['k','w','y']:
                self.color_names.append(icolor)
        self.color_names = np.array(self.color_names)
    
    def get_channels(self):
        """
        Determine channel names.

        Returns:
            List of channel names
        """

        black_list    = ['file','t_start','t_end','dt','method','tag','uid','target','annotation','sleep','awake']
        self.channels = []
        for icol in self.data.columns:
            if icol not in black_list:
                self.channels.append(icol)
        return self.channels

    def get_state(self):

        annots = self.data.annotation.values
        uannot = self.data.annotation.unique()
        sleep  = np.zeros(annots.size)
        awake  = sleep.copy()
        for iannot in uannot:
            if iannot != None:
                ann = iannot.lower()
                if 'wake' in ann or 'awake' in ann or 'pdr' in ann:
                    inds = (annots==iannot)
                    awake[inds]=1
                if 'sleep' in ann or 'spindle' in ann or 'k complex' in ann or 'sws' in ann:
                    inds = (annots==iannot)
                    sleep[inds]=1
        self.data['sleep'] = sleep
        self.data['awake'] = awake

    def state_split(self):

        self.sleep_data = self.data.loc[(self.data.sleep==1)]
        self.awake_data = self.data.loc[(self.data.awake==1)]
        return self.sleep_data,self.awake_data

    def recast(self):
        
        for icol in self.data.columns:
            try:
                self.data[icol]=self.data[icol].astype('float')
            except:
                pass

    def load_data(self,sleep_path,awake_path):

        self.sleep_data = PD.read_pickle(sleep_path)
        self.awake_data = PD.read_pickle(awake_path)
        return self.sleep_data,self.awake_data

    def call_stats(self,mid_frac,duration):

        data_analysis.plot_handler(self,mid_frac,duration)

if __name__ == '__main__':

    # Command line options needed to obtain data.
    parser = argparse.ArgumentParser(description="Analysis tool for sleep/awake data.")
    parser.add_argument("--infile", help="Path to input pickle file. Either individual outputs from SCALP_CONCAT_EEG or a merged pickle.")
    parser.add_argument("--outdir", default="./", help="Path to output directory.")
    parser.add_argument("--awakefile", help="Filepath to a csv with outputs from SCALP_CONCAT_EEG that have already been sliced against awake annotations.")
    parser.add_argument("--sleepfile", help="Filepath to a csv with outputs from SCALP_CONCAT_EEG that have already been sliced against sleep annotations.")
    parser.add_argument("--mid_frac", default=0.5, help="Filepath to a csv with outputs from SCALP_CONCAT_EEG that have already been sliced against sleep annotations.")
    parser.add_argument("--duration", default=30, help="Filepath to a csv with outputs from SCALP_CONCAT_EEG that have already been sliced against sleep annotations.")
    args = parser.parse_args()

    # Save or load the sleep and awake dataframes as needed
    sleep_path = args.outdir+"sleep.pickle"
    awake_path = args.outdir+"awake.pickle"
    if not os.path.exists(sleep_path) or not os.path.exists(awake_path):
        # Initial class load and some meta variables
        DL       = data_loader(args.infile,args.outdir)
        channels = DL.get_channels()
        DL.get_state()
        DL.recast()
        DF_sleep,DF_awake = DL.state_split()
        DF_sleep.to_pickle(sleep_path)
        DF_awake.to_pickle(awake_path)
    else:
        # Initial class load and some meta variables
        DL       = data_loader(sleep_path,args.outdir)
        channels = DL.get_channels()
        DF_sleep,DF_awake = DL.load_data(sleep_path,awake_path)

    # Get the alpha delta
    AD = alpha_delta(DF_sleep,DF_awake)
    AD.get_indices()
    AD.get_alpha_delta()
    AD.plot_alpha_delta()
    AD.plot_alpha()
    AD.plot_delta()