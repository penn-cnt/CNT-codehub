import numpy as np

class metadata_handler:

    def __init__(self):

        # Initialize the metadata dictionary
        self.metadata = {}

    def update_metadata(self,inputs):
        """
        Update the metadata dictionary with new dictionary. This is useful if rejecting datasets.
        """

        self.metadata = inputs.copy()

    def highlevel_info(self):
        """
        Store high level information about the current dataset and the timeslices extracted from it
        """

        # Because we are using built-in lists to store each data transform, we want to index the dictionary to find the right metadata
        self.metadata[self.file_cntr] = {}

        # Other high level dataset info
        self.metadata[self.file_cntr]['file']    = self.infile
        self.metadata[self.file_cntr]['t_start'] = self.t_start
        self.metadata[self.file_cntr]['t_end']   = self.t_end
        self.metadata[self.file_cntr]['history'] = self.args

    def set_ref_window(self):
        self.metadata[self.file_cntr]['t_window'] = self.t_window

    def set_channels(self,inputs):

        self.metadata[self.file_cntr]['channels'] = inputs

    def set_montage_channels(self,inputs):

        self.metadata[self.file_cntr]['montage_channels'] = inputs

    def set_sampling_frequency(self,inputs):

        self.metadata[self.file_cntr]['fs'] = inputs

    def set_target_file(self,inputs):

        self.metadata[self.file_cntr]['target_file'] = inputs
    
    def add_metadata(self,file_cntr,key,values):
        """
        Add to the metadata dictionary for a non-specific keyword. (i.e. Adding extra info for a users personal preprocessing/featute logic.)
        """

        if key in self.metadata[file_cntr].keys():
            if type(values) == dict and type(self.metadata[file_cntr][key]) == dict:
                olddict = self.metadata[file_cntr][key]
                values  = {**olddict,**values}
        self.metadata[file_cntr][key] = values