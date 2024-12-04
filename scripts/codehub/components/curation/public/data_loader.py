# General libraries
import random
import string
import pickle
import numpy as np
import pandas as PD
from sys import exit
from functools import partial
from mne.io import read_raw_edf
from pyedflib.highlevel import read_edf_header

# Component imports
from components.metadata.public.metadata_handler import *

class data_loader_test:
    """
    Class devoted to testing whether data can be loaded and used for analysis.

    New functions should return (True,) if it succeeds, or (False,e) if it raises an exception.

    test_logic handles the logic gates for different datatypes and should match the allowed_arguments.yaml options.
    """

    def __init__(self):
        pass

    def test_logic(self,ifile,ftype):
        if ftype.lower() == 'edf':
            return self.edf_test(ifile)
        elif ftype.lower() == 'pickle':
            return self.pickle_test(ifile)

    def edf_test(self,infile):
        try:
            read_edf_header(infile)
            return (True,)
        except Exception as e:
            return (False,e)
        
    def pickle_test(self,infile):
        try:
            idict = pickle.load(open(infile,'rb'))
            if 'data' not in idict.keys() or 'samp_freq' not in idict.keys():
                raise KeyError("Data or Sampling frequency not found in pickle file.")
            return (True,)
        except Exception as e:
            return (False,e)

class data_loader:
    """
    Class devoted to loading in raw data into the shared class instance.

    New functions should make use of the specific raw data handler for their dataset.
    """

    def __init__(self):
        pass

    def pipeline(self):
        """
        Method for working within the larger pipeline environment to load data.

        Args:
            filetype (str): filetype to read in (i.e. edf/mef/etc.)

        Returns:
            bool: Flag if data loaded correctly
        """
        
        # Get the host and username from the arguments
        self.ssh_host     = self.args.ssh_host
        self.ssh_username = self.args.ssh_username

        # Logic gate for filetyping, returns if load succeeded
        flag = self.data_loader_logic(self.args.datatype)

        if flag:
            # Create the metadata handler
            metadata_handler.highlevel_info(self)

            # Save the channel names
            self.channels = [ichannel.upper() for ichannel in self.channels]
            metadata_handler.set_channels(self,self.channels)

            # Calculate the sample frequencies to save the information and make time cuts
            sample_frequency = np.array([self.sfreq for ichannel in self.channel_metadata])
            metadata_handler.set_sampling_frequency(self,sample_frequency)

            # Get the rawdata
            self.raw_dataslice(sample_frequency,majoraxis=self.args.orientation)

            # Set the clip duration referenced to the whole file
            metadata_handler.set_ref_window(self)

            return True
        else:
            return False

    def direct_inputs(self,infile,filetype,ssh_host=None,ssh_username=None,majoraxis='column'):
        """
        Method for loading data directly outside of the pipeline environment.

        Args:
            infile (str): Path to the file to read in.
            filetype (str): filetype to read in (i.e. edf/mef/etc.)

        Returns:
            bool: Flag if data loaded correctly
        """

        # Define some instance variables needed to work within this pipeline
        self.infile       = infile
        self.oldfile      = '' 
        self.ssh_host     = ssh_host
        self.ssh_username = ssh_username

        # Check for valid major axis
        if majoraxis.lower() not in ['row','column']:
            raise ValueError(f"Invalid majoraxis {majoraxis}. Please select 'column' or 'row'.")

        # Try to load data
        flag = self.data_loader_logic(filetype)

        if flag:
            if majoraxis == 'row':
                self.indata = self.indata.T

            sample_frequency = np.array([self.sfreq for ichannel in self.channel_metadata])
            return PD.DataFrame(self.indata,columns=self.channels),sample_frequency[0]
        else:
            print("Unable to read in %s." %(self.infile))
            return None,None

    def raw_dataslice(self,sample_frequency,majoraxis='column'):
        """
        Logic for cutting the data up by time slices. Doing so at the beginning reduces memory load.

        Args:
            sample_frequency (int): Sampling frequency of the data
            majoraxis (str, optional): Orientation of the time vectors. Defaults to 'column'.
        """

        # Get only the time slices of interest
        self.raw_data = []
        for ii,isamp in enumerate(sample_frequency):
            
            # Calculate the index of the start
            samp_start = int(isamp*self.t_start)

            # Calculate the index of the end
            if self.t_end == -1:
                if majoraxis.lower() == 'column':
                    samp_end = int(len(self.indata[:,ii]))
                elif majoraxis.lower() == 'row':
                    samp_end = int(len(self.indata[ii]))
            else:
                samp_end = int(isamp*self.t_end)

            if majoraxis.lower() == 'column':
                self.raw_data.append(self.indata[samp_start:samp_end,ii])
            elif majoraxis.lower() == 'row':
                self.raw_data.append(self.indata[ii][samp_start:samp_end])

        # Get the underlying data shapes
        self.ncol = len(self.raw_data)
        self.nrow = max([ival.size for ival in self.raw_data])

        # Get the duration so we can more readily identify different length clips for post-hoc criteria (i.e. Find full file length rows to compare to a single clip)
        if majoraxis.lower() == 'column':
            self.duration = (samp_end-samp_start)/self.indata.shape[0]
        elif majoraxis.lower() == 'row':
            self.duration = (samp_end-samp_start)/self.indata.shape[1]

    ###################################
    #### User Provided Logic Below ####
    ###################################

    def data_loader_logic(self, filetype):
        """
        Update this function for the pipeline and direct handler to find new functions.

        Args:
            filetype (str): filetype to read in (i.e. edf/mef/etc.)

        Returns:
            bool: Flag if data loaded correctly
        """

        # Handle mix typing
        if filetype.lower() == 'mix':
            filetype = self.infile.split('.')[-1]

        if filetype.lower() == 'edf':
            flag = self.load_edf()
        elif filetype.lower() == 'pickle':
            flag = self.load_pickle()
        return flag

    def load_edf(self):
        """
        Load EDF data directly into the pipeline.
        """
 
        # Load current edf data into memory
        if self.infile != self.oldfile:
            try:
                # Read in the data via mne backend
                raw           = read_raw_edf(self.infile,verbose=False)
                self.indata   = raw.get_data().T
                self.channels = raw.ch_names
                self.sfreq    = raw.info.get('sfreq')

                # Keep a static copy of the channels so we can just reference this when using the same input data
                self.channel_metadata = self.channels.copy()
                return True
            except OSError:
                return False
        else:
            # Duplicate the channels from the last data load, since we are working with the same datafile
            self.channels = [ival for ival in self.channel_metadata]
            return True

    def load_pickle(self):
        """
        Load pickle data directly into the pipeline.

        The pickle data should be formatted as a dictionary. It looks for the following keys:
        data: A pandas dictionary with shape [samples,channels] and columns labeled by the raw channel label
        samp_freq: A float value of the sampling frequency
        """

        if self.infile != self.oldfile:
            try:
                # Read in the data via mne backend
                raw           = pickle.load(open(self.infile,'rb'))
                self.indata   = raw['data'].values
                self.channels = raw['data'].columns
                self.sfreq    = raw['samp_freq']

                # Keep a static copy of the channels so we can just reference this when using the same input data
                self.channel_metadata = self.channels.copy()
                return True
            except OSError:
                return False
        else:
            # Duplicate the channels from the last data load, since we are working with the same datafile
            self.channels = [ival for ival in self.channel_metadata]
            return True