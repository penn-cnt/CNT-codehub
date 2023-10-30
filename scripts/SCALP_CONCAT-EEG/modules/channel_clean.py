# General libraries
import re
import numpy as np
import pandas as PD

# Import the classes
from .metadata_handler import *
from .target_loader import *
from .data_loader import *
from .channel_mapping import *
from .dataframe_manager import *
from .channel_montage import *
from .output_manager import *
from .data_viability import *

class channel_clean:
    """
    Class devoted to cleaning different channel naming conventions.

    New functions should look for the self.channels object which stores the raw channel names.

    Output should be a new list of channel names called self.clean_channel_map.
    """

    def __init__(self):
        pass

    def pipeline(self,clean_method='HUP'):
        """
        Clean a vector of channel labels via the main pipeline.

        Args:
            clean_method (str, optional): _description_. Defaults to 'HUP'.
        """

        # Apply cleaning logic
        self.channel_logic(clean_method)

        # Add the cleaned labels to metadata
        self.metadata[self.file_cntr]['channels'] = self.clean_channel_map

    def direct_inputs(self,channels,clean_method="HUP"):
        """
        Clean a vector of channel labels via user provided input.

        Args:
            clean_method (str, optional): _description_. Defaults to 'HUP'.
        """

        self.channels = channels
        self.channel_logic(clean_method)
        return self.clean_channel_map

    def channel_logic(self,clean_method):

        # Logic gates for different cleaning methods
        if clean_method == 'HUP':
            self.HUP_clean()

    ###################################
    #### User Provided Logic Below ####
    ###################################

    def HUP_clean(self):
        """
        Return the channel names according to HUP standards.
        Adapted from Akash Pattnaik code.
        Updated to handle labels not typically generated at HUP (All leters, no numbers.)
        """

        self.clean_channel_map = []
        for ichannel in self.channels:
            regex_match = re.match(r"(\D+)(\d+)", ichannel)
            if regex_match != None:
                lead        = regex_match.group(1).replace("EEG", "").strip()
                contact     = int(regex_match.group(2))
                new_name    = f"{lead}{contact:02d}"
            else:
                new_name = ichannel.replace("EEG","").replace("-REF","").strip()
            self.clean_channel_map.append(new_name.upper())