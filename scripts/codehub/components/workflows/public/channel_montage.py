# General libraries
import numpy as np
import pandas as PD

# Component imports
from components.metadata.public.metadata_handler import *

class channel_montage:
    """
    Class devoted to calculating the montages of interest for the data.

    New functions should look for the relevant data in the self.dataframe object.

    Output should create a new object self.montage_channels with the labels for the output montage data vectors. You must also return the actual montage data.
    (This is due to an inheritance issue passing this data directly to the dataframe manager.)

    Naming for new functions should follow f"montage_{option name given to allowed_montage_args}". If not in this format, the code will work, but the UI
    wrapper may not be able to find this function.
    """

    def __init__(self):
        pass

    def pipeline(self,DF):
        """
        Method for working within the larger pipeline environment to get channel montages.

        Returns:
            array: Array of montage data. (Issue with inheritance requires a direct passback and not through instance.)
        """

        # Flag for updating sampling frequency if using the pipeline
        self.pipeline_flag = True

        # Save the current dataframe
        self.dataframe_to_montage = DF

        # Apply the montage logic
        montage_data = self.channel_montage_logic(self.args.montage)

        # Update the metadata to note the montage channels
        metadata_handler.set_montage_channels(self,self.montage_channels)

        # Update the frequencies if needed
        try:
            metadata_handler.set_sampling_frequency(self,self.new_fs)
        except NameError:
            pass

        return PD.DataFrame(montage_data,columns=self.montage_channels)

    def direct_inputs(self,DF,montage):
        """
        Method for getting channel montages directly outside of the pipeline environment.

        Args:
            DF (datafram): Dataframe to get montage for.
            montage (str): Montage to perform

        Returns:
            dataframe: New dataframe with montage data and channel names
        """
        
        # Flag for updating sampling frequency if using the pipeline
        self.pipeline_flag = False

        # Save the user provided dataframe
        self.dataframe_to_montage = DF

        # Apply montage logic
        montage_data = self.channel_montage_logic(montage)

        return PD.DataFrame(montage_data,columns=self.montage_channels)

    ###################################
    #### User Provided Logic Below ####
    ###################################

    def channel_montage_logic(self, montage):
        """
        Update this function for the pipeline and direct handler to find new functions.

        Args:
            montage (str): User provided string for type of montage to perform.

        Returns:
            array: array of montage data
        """

        # Logic for different montages
        if montage.lower() == 'none':
            return self.no_montage()
        if montage.lower() == "hup1020":
            return self.montage_hup1020()
        elif montage.lower() == "common_average":
            return self.montage_common_average()   
        elif montage.lower() == "neurovista":
            return self.montage_neurovista()   

    def no_montage(self):
        self.montage_channels = list(self.dataframe_to_montage.columns)
        self.new_fs           = self.metadata[self.file_cntr]['fs']
        return self.dataframe_to_montage.values

    def montage_common_average(self):
        """
        Calculate the common average montage.

        Channel - AVG(Channels).
        """

        # Get the averages across all channels per time slice
        averages = np.average(self.dataframe_to_montage.values,axis=0).reshape((-1,1))

        # Get the montage data
        montage_df = self.dataframe_to_montage.copy()
        for idx,icol in enumerate(self.dataframe_to_montage.columns):
            montage_df[icol] = self.dataframe_to_montage[icol]-averages[idx]
        montage_data = montage_df.values

        # Get the montage labels
        self.montage_channels = [f"{ichannel}-car" for ichannel in self.dataframe_to_montage.columns]

        # Make the montage dataframe
        return montage_data

    def montage_hup1020(self):
        """
        Calculate the HUP 1020 montage and create the channel labels. Passes its data directly to the dataframe class.

        Montage map:
        FP01 - F07
        F07  - T03
        T03  - T05
        T05  - O01
        FP02 - F08
        F08  - T04
        T04  - T06
        T06  - O02
        FP01 - F03
        F03  - C03
        C03  - P03
        P03  - O01
        FP02 - F04
        F04  - C04
        C04  - P04
        P04  - O02
        FZ   - CZ
        """

        # Channel structure for the montage
        bipolar_array = [['FP01','F07'],
                         ['F07','T03'],
                         ['T03','T05'],
                         ['T05','O01'],
                         ['FP02','F08'],
                         ['F08','T04'],
                         ['T04','T06'],
                         ['T06','O02'],
                         ['FP01','F03'],
                         ['F03','C03'],
                         ['C03','P03'],
                         ['P03','O01'],
                         ['FP02','F04'],
                         ['F04','C04'],
                         ['C04','P04'],
                         ['P04','O02'],
                         ['FZ','CZ']]
        
        # Get the frequency array to reference and create the new output object
        if self.pipeline_flag:
            fs          = self.metadata[self.file_cntr]['fs']
            self.new_fs = []

        # Get the new values to pass to the dataframe class
        montage_data = np.zeros((self.dataframe_to_montage.shape[0],len(bipolar_array))).astype('float64')
        for ii,ival in enumerate(bipolar_array):

            # Update the data array
            try:
                montage_data[:,ii] = self.dataframe_to_montage[ival[0]].values-self.dataframe_to_montage[ival[1]].values
            except KeyError:
                montage_data[:,ii] = np.nan

            # Update the frequency
            if self.pipeline_flag:
                try:
                    fs0 = fs[np.where(self.dataframe.columns==ival[0])[0][0]]
                    fs1 = fs[np.where(self.dataframe.columns==ival[1])[0][0]]
                    if fs0 == fs1:
                        self.new_fs.append(fs0)
                    else:
                        raise ValueError("Cannot montage channels with different frequencies.")
                except IndexError:
                    self.new_fs.append(fs0)
            
        # Get the new montage channel labels
        self.montage_channels = [f"{ichannel[0]}-{ichannel[1]}" for ichannel in bipolar_array]

        # Pass the data to the dataframe class function for montages
        return montage_data 
    
    def montage_neurovista(self):
        """
        TODO: add Neurovista montaging.
        """
        montage_data          = self.dataframe.values
        self.montage_channels = self.dataframe.columns
        self.new_fs           = self.metadata[self.file_cntr]['fs']
        return montage_data