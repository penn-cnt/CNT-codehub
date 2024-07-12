import glob
import numpy as np
import pandas as PD
from sys import exit
from sklearn.metrics import mean_squared_error as MSE

class reference_to_file:
    """
    This class allows data rejection based on how a clip is behaving in comparison to the whole files worth of data.

    The code expects a feature dataframe with clip data (t_end-t_start<file_duration) and rows with (t_end-t_start==duration).
    This can be accomplished by running the EPIPY pipeline_manager with t_window=<clip_duration_of_choice>,-1, where -1 is a flag
    that denotes the code should function on the whole file.
    """

    def __init__(self,fpath):
        """
        Initialize class and read in the raw data.

        Args:
            fpath (str): path to the merged featue dataframe pickle from EPIPY
        """

        self.rawdata  = PD.read_pickle(fpath)

    def get_channels(self,blacklist=None):
        """
        Get

        Args:
            blacklist (string array, optional): List of columns to exclude from finding channels. If None, uses typical EPIPY output to get the remaining column names as channels.
        """

        if blacklist == None:
            blacklist = ['file', 't_start', 't_end', 't_window', 'method', 'tag','uid', 'target', 'annotation']
        self.channels = np.setdiff1d(self.rawdata.columns,blacklist)

    def reference_split(self):
        """
        For this class, search for entries with t_window==-1 for whole file info, with the rest being clip info.
        """

        self.clip_data = self.rawdata.loc[self.rawdata.t_window!=-1]
        self.file_data = self.rawdata.loc[self.rawdata.t_window==-1]

    def quality_check(self):
        """
        Ensures that there is a reference measurement for every clip.

        Raises:
            IndexError: Raises an error if there isn't a matching index for every clip/file pair.
        """

        clip_ufiles = np.sort(self.clip_data['file'].unique())
        file_ufiles = np.sort(self.file_data['file'].unique())
        if not (clip_ufiles==file_ufiles).all():
            print("Missing some reference data for some clips. This may be due issues when processing the data, resulting in data loss.")
            raise IndexError("Cannot create proper clip references for rejection.")
        
    def merge_data(self):

        self.merged_data = self.clip_data.merge(self.file_data, on=['file', 'method', 'tag', 'uid', 'target', 'annotation'], suffixes=('', '_file'))
        self.merged_data = self.merged_data.drop(['t_start_file', 't_end_file', 't_window_file'], axis=1)
        
        # Make a mask column for rejections
        self.merged_data['mask'] = 1

    def return_data(self):
        return self.rejection_DF

    #######################################################################################
    ###### Functions that interact with the user for cleaning and applying criteria. ######
    #######################################################################################

    def pipeline(self):
        self.get_channels()
        self.reference_split()
        self.quality_check()
        self.merge_data()
        self.marsh_rejection()

    def marsh_rejection(self,amplitude_cutoff=500, rms_cutoff=2, ll_cutoff=2):
        
        # Grab a dataslice of rms and stdev
        rms_df    = self.merged_data.loc[self.merged_data.method=='rms'].drop_duplicates(subset=['file']).reset_index(drop=True)
        stdev_df  = self.merged_data.loc[self.merged_data.method=='stdev'].drop_duplicates(subset=['file']).reset_index(drop=True)
        ref_files = rms_df.file.unique()

        # Loop over channel to find the cutoff values
        self.rejection_DF = self.merged_data.loc[self.merged_data.method.isin(['rms','stdev'])]
        for ichannel in self.channels:
            for ifile in ref_files:

                # Get the current cutoff
                center = rms_df.loc[rms_df.file==ifile][f"{ichannel}_file"].values
                offset = 2*stdev_df.loc[rms_df.file==ifile][f"{ichannel}_file"].values
                cutoff = center+offset
                
                # Adjust the rejection mask
                self.rejection_DF.loc[(self.rejection_DF['file'] == ifile) & (self.rejection_DF[ichannel] > cutoff[0])]['mask'] = 0

if __name__ == '__main__':

    fpath      = glob.glob("/Users/bjprager/Documents/GitHub/CNT-codehub/user_data/EPIPY_TESTING/outputs/*features*pickle")[0]
    RF         = reference_to_file(fpath)
    RF.pipeline()
    DF_rejects = RF.return_data()
    print(DF_rejects['mask'].unique())