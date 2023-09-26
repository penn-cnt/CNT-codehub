# General libraries
import torch
import pickle
import datetime
import numpy as np
import pandas as PD


# Import the classes
from .data_loader import *
from .channel_mapping import *
from .dataframe_manager import *
from .channel_clean import *
from .channel_montage import *
from .data_viability import *

class output_manager:

    def __init__(self):
        """
        Initialize containers that store the results that need to be fed-forward.
        """

        self.output_list = []
        self.output_meta = []

    def update_output_list(self,data,meta):
        """
        Add elements to the container of objects we want to analyze.

        Args:
            data (array): Data array for a given dataset/timeslice.
            meta (dict): Dictionary with important metadata about the data arrays.
        """

        self.output_list.append(data)
        self.output_meta.append(meta)

    def save_output_list(self):
        """
        Save the container of data and metadata to ourput directory.
        """

        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
        pickle.dump(self.output_list,open("%s/%s_data.pickle" %(self.args.outdir,timestamp),"wb"))
        pickle.dump(self.output_meta,open("%s/%s_meta.pickle" %(self.args.outdir,timestamp),"wb"))

    def create_tensor(self):
        """
        Create a pytorch tensor input.
        """

        # Create the tensor
        self.input_tensor_dataset = [torch.utils.data.DataLoader(dataset) for dataset in self.output_list]