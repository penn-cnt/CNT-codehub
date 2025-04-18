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

            # Pickled objects
            fp1 = open("%s/%s_%s_meta.pickle" %(self.args.outdir,self.timestamp,self.unique_id),"wb")
            fp2 = open("%s/%s_%s_fconfigs.pickle" %(self.args.outdir,self.timestamp,self.unique_id),"wb")
            fp3 = open("%s/%s_%s_features.pickle" %(self.args.outdir,self.timestamp,self.unique_id),"wb")
            pickle.dump(self.metadata,fp1)
            pickle.dump(self.feature_commands,fp2)
            pickle.dump(self.feature_df,fp3)
            fp1.close()
            fp2.close()
            fp3.close()

            # CSV object
            #self.feature_df.to_csv("%s/%s_%s_features.pickle" %(self.args.outdir,self.timestamp,self.unique_id),index=False)

    def save_output_list(self):
        """
        Save the container of data and metadata to ourput directory.
        """

        if not self.args.debug:
            fp1 = open("%s/%s_%s_data.pickle" %(self.args.outdir,self.timestamp,self.unique_id),"wb")
            fp2 = open("%s/%s_%s_meta.pickle" %(self.args.outdir,self.timestamp,self.unique_id),"wb")
            pickle.dump(self.output_list,fp1)
            pickle.dump(self.metadata,fp2)
            fp1.close()
            fp2.close()