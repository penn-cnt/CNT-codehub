import re
import mne
import numpy as np
import pandas as PD
from mne.io.constants import FIFF

# Local Imports
from components.internal.observer_handler import *

def return_backend(user_request='MNE'):
    if user_request == 'MNE':
        return MNE_handler()

class backend_observer(Observer):

    def listen_data(self):
        idata,itype = self.backend.workflow(self.args,self.data,self.channels,self.fs)
        self.data_list.append(idata)
        self.type_list.append(itype)

class MNE_handler:

    def __init__(self):
        pass

    def workflow(self,args,data,channels,fs):

        # Save the inputs to class instance
        self.args     = args
        self.indata   = data
        self.channels = channels
        self.fs       = fs

        # Prepare the data according to the backend
        try:
            passflag = self.get_channel_type()
            if passflag:
                self.make_info()
                self.make_raw()
            else:
                self.irow = None
                self.bids_datatype = None
        except Exception as e:
            if self.args.debug:
                print(f"Load error {e}")
        
        # Return raw to the list of raws being tracked by the Subject class
        return self.iraw,self.bids_datatype

    def make_raw(self):
        idata     = np.nan_to_num(self.indata.T, )
        self.iraw = mne.io.RawArray(idata, self.data_info, verbose=False)
        self.iraw.set_channel_types(self.channel_types.type)
    
    def make_info(self):
        self.data_info = mne.create_info(ch_names=list(self.channels), sfreq=self.fs, verbose=False)
        for idx,ichannel in enumerate(self.channels):
            if self.channel_types.loc[ichannel]['type'] in ['seeg','eeg']:
                self.data_info['chs'][idx]['unit'] = FIFF.FIFF_UNIT_V

    def get_channel_type(self, threshold=15):

        # Define the expression that gets lead info
        regex = re.compile(r"(\D+)(\d+)")

        # Get the outputs of each channel
        try:
            channel_expressions = [regex.match(ichannel) for ichannel in self.channels]

            # Make the channel types
            self.channel_types = []
            for (i, iexpression), channel in zip(enumerate(channel_expressions), self.channels):
                if iexpression == None:
                    if channel.lower() in ['fz','cz']:
                        self.channel_types.append('eeg')
                    else:
                        self.channel_types.append('misc')
                else:
                    lead = iexpression.group(1)
                    contact = int(iexpression.group(2))
                    if lead.lower() in ["ecg", "ekg"]:
                        self.channel_types.append('ecg')
                    elif lead.lower() in ['c', 'cz', 'cz', 'f', 'fp', 'fp', 'fz', 'fz', 'o', 'p', 'pz', 'pz', 't']:
                        self.channel_types.append('eeg')
                    elif "NVC" in iexpression.group(0):  # NeuroVista data 
                        self.channel_types.append('eeg')
                        self.channels[i] = f"{channel[-2:]}"
                    elif lead.lower() in ['a']:
                        self.channel_types.append('misc')
                    else:
                        self.channel_types.append(1)

            # Do some final clean ups based on number of leads
            lead_sum = 0
            for ival in self.channel_types:
                if isinstance(ival,int):lead_sum+=1
            if self.args.ch_type == None:
                if lead_sum > threshold:
                    remaining_leads = 'ecog'
                else:
                    remaining_leads = 'seeg'
            else:
                remaining_leads = self.args.ch_type
            for idx,ival in enumerate(self.channel_types):
                if isinstance(ival,int):self.channel_types[idx] = remaining_leads
            self.channel_types = np.array(self.channel_types)
        except:
            if self.args.ch_type != None:
                self.channel_types = np.array([self.args.ch_type for ichannel in self.channels])
            else:
                return False

        # Make the dictionary for mne
        self.channel_types = PD.DataFrame(self.channel_types.reshape((-1,1)),index=self.channels,columns=["type"])
        
        # Get the best guess datatype to send to bids writer
        raw_datatype = self.channel_types['type'].mode().values[0]
        
        # perform some common mappings to the bids keywords
        if raw_datatype == 'ecog':
            datatype = 'ieeg'
        elif raw_datatype == 'seeg':
            datatype = 'ieeg'
        else:
            datatype = raw_datatype

        # Store the data type to use for write out
        self.bids_datatype = datatype
        return True
