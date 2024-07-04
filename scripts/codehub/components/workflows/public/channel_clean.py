# General libraries
import re
import numpy as np
import pandas as PD

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
        elif clean_method.lower() == 'temple':
            self.clean_temple()
        elif clean_method.lower() == 'neurovista':
            self.clean_neurovista()

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
        self.clean_channel_map = np.array(self.clean_channel_map)

    def clean_temple(self):
        """
        Return the channel names for Temple data.
        """

        self.clean_channel_map = []
        for ichannel in self.channels:
            regex_match = re.match(r"(\D+)(\d+)(.)", ichannel)
            if regex_match != None:

                # Make the new channel name
                lead        = regex_match.group(1).replace("EEG", "").strip()
                contact     = int(regex_match.group(2))
                new_name    = f"{lead}{contact:02d}"

                # Check the special character list for temple
                if regex_match.group(3).lower() in ['p']:
                    new_name = f"{new_name}{regex_match.group(3)}"

            else:
                new_name = ichannel.replace("EEG","").replace("-REF","").strip()
            self.clean_channel_map.append(new_name.upper())
        self.clean_channel_map = np.array(self.clean_channel_map)       

    def clean_neurovista(self):
        """
        TODO: This is essentially just a 'pass'. Neurovista cleaning logic will be added in the future.  
        """

        #self.clean_channel_map = self.metadata[self.file_cntr]['channels']
        pass