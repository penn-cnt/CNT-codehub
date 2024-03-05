# General libraries
import re
import numpy as np
import pandas as PD

# Import the internal classes
from components.core.internal import *
from components.curation.internal import *
from components.features.internal import *
from components.metadata.internal import *
from components.validation.internal import *
from components.workflows.internal import *

# Import the public classes
from components.core.public import *
from components.curation.public import *
from components.features.public import *
from components.metadata.public import *
from components.validation.public import *
from components.workflows.public import *

class channel_clean:
    """
    Class devoted to cleaning different channel naming conventions.

    New functions should look for the self.channels object which stores the raw channel names.

    Output should be a new list of channel names called self.clean_channel_map.
    """

    def __init__(self):
        pass

    def pipeline(self):
        """
        Clean a vector of channel labels via the main pipeline.

        Args:
            clean_method (str, optional): _description_. Defaults to 'HUP'.
        """

        # Apply cleaning logic
        self.channel_clean_logic(self.args.channel_clean)

        # Add the cleaned labels to metadata
        self.metadata[self.file_cntr]['channels'] = self.clean_channel_map

    def direct_inputs(self,channels,clean_method="HUP"):
        """
        Clean a vector of channel labels via user provided input.

        Args:
            clean_method (str, optional): _description_. Defaults to 'HUP'.
        """

        self.channels = channels
        self.channel_clean_logic(clean_method)
        return self.clean_channel_map

    ###################################
    #### User Provided Logic Below ####
    ###################################

    def channel_clean_logic(self,clean_method):
        """
        Update this function for the pipeline and direct handler to find new functions.

        Args:
            filetype (str): cleaning method to use
        """

        # Logic gates for different cleaning methods
        if clean_method.lower() == 'hup':
            self.clean_hup()

    def clean_hup(self):
        """
        Return the channel names according to HUP standards.
        Adapted from Akash Pattnaik code.
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