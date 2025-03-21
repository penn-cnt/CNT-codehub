from sys import exit
from tqdm import tqdm

# Component imports
from components.curation.public.data_loader import *
from components.core.internal.output_manager import *
from components.workflows.public.channel_clean import *
from components.workflows.public.preprocessing import *
from components.core.internal.dataframe_manager import *
from components.workflows.public.channel_mapping import *
from components.workflows.public.channel_montage import *
from components.metadata.public.metadata_handler import *

class project_handlers:
    """
    Class devoted the specific pipeline used to load data according to project needs. This is meant to provide a clean reproducable pipeline.

    New functions should follow all the data processing steps up to preprocessing and feature extraction that are relevant to their data type and data set.
    """

    def __init__(self):
        pass

    def file_manager(self):
        """
        Loop over the input files and send them to the correct data handler.

        Args:
            infiles (str list): Path to each dataset
            start_times (float list): Start times in seconds to start sampling
            end_times (float list): End times in seconds to end sampling
        """

        # Intialize a variable that stores the previous filepath. This allows us to cache data and only read in as needed. (i.e. new path != old path)
        self.oldfile = None  

        # Loop over files to read and store each ones data
        nfile = len(self.infiles)
        desc  = "Initial load with id %s:" %(self.unique_id)
        for ii,ifile in tqdm(enumerate(self.infiles), desc=desc, total=nfile, bar_format=self.bar_frmt, position=self.worker_number, leave=False, disable=self.args.silent,dynamic_ncols=True):            
        
            # Save current file info
            self.infile   = ifile
            self.t_start  = self.start_times[ii]
            self.t_end    = self.end_times[ii]
            self.t_window = self.ref_windows[ii]

            # Initialize the metadata container
            self.file_cntr = ii

            # Apply project specific pipeline
            try:
                self.project_logic()
            except ValueError: #Exception as e:
                if self.args.debug:
                    print("Encountered preprocessing error {e}.")
                self.active_workers.value -= 1
            
            # Update file strings for cached read in
            self.oldfile = self.infile

    ##########################
    #### Template Project ####
    ##########################

    def template(self):

        # Import data into memory
        load_flag = data_loader.pipeline(self)      # Load flag is a boolean that lets us know if the current data loaded correctly

        # If data loaded, begin the processing portion
        if load_flag:
            # Clean the channel names
            channel_clean.pipeline(self)

            # Get the correct channels for this merger
            channel_mapping.pipeline(self)

            # Once we have the cleaned channel names, and the appropriate column slices, make a dataframe.
            # Dataframes are formed from the self.raw_data object and self.master_channel_list.
            # Data up to this point is kept as a raw array due to variable input formats and because dataframes
            # tend to take up more memory and have slower operations. 
            dataframe_manager.__init__(self)
            dataframe_manager.column_subsection(self,self.channel_map_out)  

            # We can use the dataframe to set criteria for continued analysis.
            # In this example, the data must have at least the sampling frequency worth of values
            if self.dataframe.shape[0] > int(max(self.metadata[self.file_cntr]['fs'])):
                
                # You can either montage first, then preprocess, or vice versa.
                # At present you cannot mix these steps. But later updates will allow
                # to provide the ability to define multiple preprocessing blocks that
                # can be ordered independently.
                self.premontage_df = preprocessing.__init__(self, self.dataframe, self.metadata[self.file_cntr]['fs'])

                # Montage the data
                self.montaged_dataframe = channel_montage.pipeline(self,self.premontage_df)

                # Store the data to the output handler so we can pass everything to the feature extractor
                # Returning to a list of arrays so it can be passed to different modeling back-ends like PyTorch.
                output_manager.update_output_list(self,df.values)

    ###################################
    #### User Provided Logic Below ####
    ###################################

    def project_logic(self):
        """
        Update this function for the pipeline to find new pipelines.
        """

        # Case statement the workflow
        if self.args.project.lower() == 'scalp_basic':
            project_handlers.scalp_basic(self)
        elif self.args.project.lower() == 'imaging_basic':
            project_handlers.imaging_basic(self)
        elif self.args.project.lower() == 'neurovista_basic':
            project_handlers.neurovista_basic(self)

    def scalp_basic(self):
        """
        Basic pipeline to load EDF data for a scalp project.

        1) Load data.
        2) Clean channel data.
        3) Map channel names.
        4) Create dataframes.
        5) Preprocessing steps.
        6) Make montage.
        """

        # Import data into memory
        load_flag = data_loader.pipeline(self) 

        if load_flag:
            # Clean the channel names
            channel_clean.pipeline(self)

            # Get the correct channels for this merger
            channel_mapping.pipeline(self)

            # Create the dataframe for the object with the cleaned labels
            dataframe_manager.__init__(self)
            dataframe_manager.column_subsection(self,self.channel_map_out)

            # Perform next steps only if we have a viable dataset
            if self.dataframe.shape[0] > int(self.args.nfreq_window)*int(max(self.metadata[self.file_cntr]['fs'])):

                # Make the cleaned mne channel map
                mne_channels      = mne.channels.make_standard_montage("standard_1020").ch_names
                self.mne_channels = channel_clean.direct_inputs(self,mne_channels)

                # Preprocess the data
                self.premontage_df = preprocessing.__init__(self, self.dataframe, self.metadata[self.file_cntr]['fs'])

                # Put the data into a specific montage
                self.montaged_dataframe = channel_montage.pipeline(self,self.premontage_df)
                metadata_handler.set_montage_channels(self,self.montaged_dataframe.columns)

                # Update the output list
                output_manager.update_output_list(self,self.montaged_dataframe.values)

                # Update the metadata handler with the unmontaged data. Some features require specific montages to function
                metadata_handler.add_metadata(self,self.file_cntr,'unmontaged_data',self.premontage_df) 

    def imaging_basic(self):
        """
        Basic pipeline to processing imaging data.
        
        1) Save kittens
        2) Face goblins
        3) Do some shenanigans.
        4) Fight gods.
        """

        pass

    def neurovista_basic(self):
            """
            Basic pipeline to process Neurovista data.
            """
            # Import data into memory
            load_flag = data_loader.pipeline(self)      # Load flag is a boolean that lets us know if the current data loaded correctly

            # If data loaded, begin the processing portion
            if load_flag:
                # Clean the channel names
                channel_clean.pipeline(self)

                # Get the correct channels for this merger
                channel_mapping.pipeline(self)

                # Once we have the cleaned channel names, and the appropriate column slices, make a dataframe.
                # Dataframes are formed from the self.raw_data object and self.master_channel_list.
                # Data up to this point is kept as a raw array due to variable input formats and because dataframes
                # tend to take up more memory and have slower operations. 
                dataframe_manager.__init__(self)
                dataframe_manager.column_subsection(self,self.channel_map_out)  

                # We can use the dataframe to set criteria for continued analysis.
                # In this example, the data must have at least the sampling frequency worth of values
                min_fs = int(max(self.metadata[self.file_cntr]['fs']))
                if self.dataframe.shape[0] > min_fs:
                    # You can either montage first, then preprocess, or vice versa.
                    # At present you cannot mix these steps. But later updates will allow
                    # to provide the ability to define multiple preprocessing blocks that
                    # can be ordered independently.
                    #df = preprocessing.__init__(self, self.dataframe, self.metadata[self.file_cntr]['fs'])

                    # Montage the data
                    self.montaged_dataframe = channel_montage.pipeline(self,self.dataframe)

                    # Store the data to the output handler so we can pass everything to the feature extractor
                    # Returning to a list of arrays so it can be passed to different modeling back-ends like PyTorch.
                    output_manager.update_output_list(self,self.montaged_dataframe.values)
                else:
                    if not self.args.silent:
                        print(f"Dataframe of shape {self.dataframe.shape} does not contain at least {min_fs} samples.")