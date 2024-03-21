# General libraries
import torch
import pickle
import datetime
import numpy as np
import pandas as PD

class output_manager:
    """
    Manages various output functionality.
    """

    def __init__(self):
        """
        Initialize containers that store the results that need to be fed-forward.
        """

        self.output_list = []
        self.output_meta = []

    def update_output_list(self,data):
        """
        Add elements to the container of objects we want to analyze.

        Args:
            data (array): Data array for a given dataset/timeslice.
            meta (dict): Dictionary with important metadata about the data arrays.
        """

        self.output_list.append(data)
        self.output_meta.append(self.file_cntr)

    def create_tensor(self):
        """
        Create a pytorch tensor input.
        """

        # Create the tensor
        self.input_tensor_dataset = [torch.utils.data.DataLoader(dataset) for dataset in self.output_list]

    def save_features(self):
        """
        Save the feature dataframe
        """

        if not self.args.debug and not self.args.no_feature_flag:
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
            pickle.dump(self.metadata,open("%s/%s_meta_%s.pickle" %(self.args.outdir,timestamp,self.unique_id),"wb"))
            pickle.dump(self.feature_df,open("%s/%s_features_%s.pickle" %(self.args.outdir,timestamp,self.unique_id),"wb"))
            pickle.dump(self.feature_commands,open("%s/%s_fconfigs_%s.pickle" %(self.args.outdir,timestamp,self.unique_id),"wb"))

    def save_output_list(self):
        """
        Save the container of data and metadata to ourput directory.
        """

        if not self.args.debug:
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
            pickle.dump(self.output_list,open("%s/%s_data_%s.pickle" %(self.args.outdir,timestamp,self.unique_id),"wb"))
            pickle.dump(self.metadata,open("%s/%s_meta_%s.pickle" %(self.args.outdir,timestamp,self.unique_id),"wb"))