import argparse
import numpy as np
import pandas as PD
from sys import exit
from mne.io import read_raw_edf
from pyedflib.highlevel import read_edf,read_edf_header
from components.workflows.public.channel_clean import channel_clean

class model_level:

    def __init__(self):
        pass

    def compare_to_standard(self,standard_file,direct_input=None,tol=1e-8):

        # Open the standard file
        data_standard = read_raw_edf(standard_file).get_data()

        # Read in the file to compare with if needed
        if direct_input == None:
            data_new = self.mne_data.copy()
        else:
            data_new = read_raw_edf(direct_input).get_data()

        # Find the difference to the standard
        diffs=data_standard-data_new
        if (diffs>tol).any():
            print("Fail")
            exit(1)

class machine_level(model_level):

    def __init__(self,args):

        # Save user inputs to instance variable
        self.args = args

        # Define the required keywords in the header
        self.required_dataset_headers = ['duration']
        self.required_channel_headers = ['label', 'dimension', 'sample_rate', 'sample_frequency', 'physical_max', 'physical_min', 'digital_max', 'digital_min']

    def run_tests(self):
        
        self.test_header()
        self.test_channels()
        self.test_sampfreq()
        self.load_data_mne()
        self.load_data_pyedf()
        self.compare_libraries()
        self.check_nan()
        self.check_running_stats(self.args.sampfreq+1)

    def failure(self,istr):
        print(istr)
        if not args.silent:
            exit(1)

    def test_header(self):
        
        # Read in the header
        self.header = read_edf_header(args.infile)
        
        print("HEADER:\n===============")
        print(self.header)
        print("\n===============")

        # Ensure casing of the keywords
        header_keys = list(self.header.keys())
        for ikey in header_keys:
            self.header[ikey.lower()] = self.header.pop(ikey)
        self.header_keys = list(self.header.keys())

        # Check the dataset level required header info
        for ikey in self.required_dataset_headers:
            if ikey.lower() not in self.header_keys:
                self.failure(f"Header missing the {ikey} information.")
            if self.header[ikey] == None or self.header[ikey] == '':
                self.failure(f"Header missing the {ikey} information.")

        # Check that the channel headers are all present and contain data
        channel_header_mask       = []
        channel_header_entry_mask = []
        for ival in self.header['signalheaders']:
            ikeys = list(ival.keys())
            channel_header_mask.append(all(tmp in ikeys for tmp in self.required_channel_headers))
            channel_header_entry_mask.extend([ival[tmp]==None for tmp in self.required_channel_headers])
        
        # Raise exceptions if poorly defined header is found
        if any(channel_header_mask) == False:
            self.failure("Header contains missing information.")
        if any(channel_header_entry_mask) == True:
            self.failure("Header contains missing information.")

    def test_sampfreq(self):

        # Obtain raw channel names
        samp_freqs = np.array([int(ival['sample_rate']) for ival in self.header['signalheaders']])

        # Check against the expected frequency
        freq_mask = (samp_freqs!=self.args.sampfreq)
        if (freq_mask).any():
            self.failure(f"Unexpted sampling frequency found in {self.channels[freq_mask]}")

    def test_channels(self):

        # Obtain the reference channels
        self.ref_channels = PD.read_csv(args.channel_file,names=['channels']).values

        # Obtain raw channel names
        raw_channels = self.header['channels']

        # Clean up the channel names
        CC            = channel_clean()
        self.channels = CC.direct_inputs(raw_channels)

        # Make sure we have at least the user required channels in the data
        channel_check = []
        for ichannel in self.ref_channels:
            if ichannel in self.channels:
                channel_check.append(True)
            else:
                channel_check.append(False)

        # Make sure all channels are present
        if not all(channel_check):
            self.failure("Could not find all the expected channels")
        
        # Check number of channels
        if self.channels.size != self.ref_channels.size:
            self.failure("Did not receive expected number of channels. This can arise due to poorly inputted channels.")

    def load_data_mne(self):
        self.mne_data = read_raw_edf(self.args.infile).get_data()[:,self.args.start_samp:self.args.end_samp]

    def load_data_pyedf(self):
        self.pyedf_data, self.pyedf_chan_info,_ = read_edf(self.args.infile)
        self.pyedf_data *= 1e-6
        self.pyedf_data  = self.pyedf_data[:,self.args.start_samp:self.args.end_samp]

    def compare_libraries(self,tol=1e-8):

        diffs=self.mne_data-self.pyedf_data
        if (diffs>tol).any():
            print("Tolerance issue.")
            exit(1)

        # Drop the pyedf data to reduce memory usage now that we dont need it
        self.pyedf_data = None

    def check_nan(self):

        if np.isnan(self.mne_data).any():
            self.failure("NaNs found in the data.")
        
    def check_running_stats(self,window_size):

        variance_array = np.zeros(self.mne_data.shape)
        for idx in range(self.channels.size):
            data = self.mne_data[idx]

            # Check parity of window size
            if window_size < 3:
                window_size = 3
            elif window_size%2 == 0:
                window_size -= 1

            # Make a padded data entry
            pad_size = int(window_size/2)
            pad_data = np.pad(data,(pad_size,pad_size), mode='constant', constant_values=np.nan)

            # Calculate the median and variance
            strided_data = np.lib.stride_tricks.sliding_window_view(pad_data, (window_size,))
            stride_inds  = ~np.isnan(strided_data)
            mean         = np.mean(strided_data, axis=1, where=stride_inds)
            variance     = np.mean((strided_data - mean[:, np.newaxis]) ** 2, axis=1, where=stride_inds)

            # Store to the channel wide measurement
            variance_array[idx] = variance

        # Get the channel wide variance sum. Zero means all channels had zero variance for the window size
        mask = variance_array.sum(axis=0)==0
        if (mask).any():
            self.failure(f"All channels have zero variance around second {self.args.sampfreq*np.arange(mask.size)[mask]} seconds.")

if __name__ == '__main__':

    # Argument parsing
    parser = argparse.ArgumentParser(description="Simplified data merging tool.")
    parser.add_argument("--infile", type=str, help='Input EDF Filepath')
    parser.add_argument("--sampfreq", type=int, default=256, help='Expected sampling frequency')
    parser.add_argument("--channel_file", type=str, default='configs/hup_standard.csv', help='CSV file containing the expected channels')
    parser.add_argument("--silent", action='store_true', default=False, help="Silence exceptions.")
    parser.add_argument("--start_samp", default=0, help="Start sample to read data in from. Useful if spot checking a large file. (Warning. Still requires initial load of full data into memory.)")
    parser.add_argument("--end_samp", default=-1, help="End sample to read data in from. Useful if spot checking a large file. (Warning. Still requires initial load of full data into memory.)")
    args = parser.parse_args()

    # Run machine level tests
    ML = machine_level(args)
    ML.run_tests()