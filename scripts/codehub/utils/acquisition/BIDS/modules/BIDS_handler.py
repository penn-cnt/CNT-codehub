import os
import re
import mne
import glob
import pickle
import getpass
import mne_bids
import numpy as np
import pandas as PD
from os import path
from sys import exit
from tqdm import tqdm
from time import sleep
from datetime import date
from mne_bids import BIDSPath, write_raw_bids
from pyedflib.highlevel import read_edf_header

class BIDS_handler:

    def __init__(self):
        self.raws      = []
        self.data_info = {'iEEG_id':self.current_file}
        self.get_subject_number()
        self.get_session_number()

    def reset_variables(self):
            # Delete all variables in the object's namespace
            for var_name in list(self.__dict__.keys()):
                delattr(self, var_name)

    def get_subject_number(self):
        """
        Assigns a subject number to a dataset. 
        """

        # Load the mapping if available, otherwise dummy dataframe
        if not path.exists(self.subject_path):
            subject_uid_df = PD.DataFrame(np.empty((1,3)),columns=['iEEG file','uid','subject_number'])
        else:
            with self.semaphore:
                subject_uid_df = PD.read_csv(self.subject_path)

        # Check if we already have this subject
        uids = subject_uid_df['uid'].values
        if self.uid not in uids:
            self.subject_num = self.proposed_sub
        else:
            self.subject_num = int(subject_uid_df['subject_number'].values[np.where(uids==self.uid)[0][0]])

    def get_session_number(self):

        # Get the session number by file if possible, otherwise intuit by number of folders
        pattern = r'Day(\d+)'
        match = re.search(pattern, self.current_file)
        if self.proposed_ses != -1:
            self.session_number = self.proposed_ses
        elif match:
            self.session_number = int(match.group(1))
        else:
            # Get the folder strings
            folders = glob.glob("%ssub-%04d/*" %(self.args.bidsroot,self.subject_num))
            folders = [ifolder.split('/')[-1] for ifolder in folders]

            # Search for the session numbers
            regex = re.compile(r'\d+$')
            if len(folders) > 0:
                self.session_number = max([int(re.search(regex, ival).group()) for ival in folders])+1
            else:
                self.session_number = 1

    def get_channel_type(self, threshold=15):

        # Define the expression that gets lead info
        regex = re.compile(r"(\D+)(\d+)")

        # Get the outputs of each channel
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
                else:
                    self.channel_types.append(1)

        # Do some final clean ups based on number of leads
        lead_sum = 0
        for ival in self.channel_types:
            if isinstance(ival,int):lead_sum+=1
        if lead_sum > threshold:
            remaining_leads = 'ecog'
        else:
            remaining_leads = 'seeg'
        for idx,ival in enumerate(self.channel_types):
            if isinstance(ival,int):self.channel_types[idx] = remaining_leads
        self.channel_types = np.array(self.channel_types)

        # Make the dictionary for mne
        self.channel_types = PD.DataFrame(self.channel_types.reshape((-1,1)),index=self.channels,columns=["type"])

    def channel_cleanup(self):
        
        # Make a data copy for manipulation
        data = self.data.copy()

        # Repair the Nan values
        data[np.isnan(data)] = 0

        # Loop over channels to find ones to drop
        mask = []
        for idx in range(data.shape[1]):
            if (data[:,idx]==0).all():
                mask.append(False)
            else:
                mask.append(True)

        # Update the data and channel list
        self.data     = data[:,mask]
        self.channels = list(np.array(self.channels)[mask])

    def make_info(self):
        self.data_info = mne.create_info(ch_names=list(self.channels), sfreq=self.fs, verbose=False)

    def add_raw(self):
        # Put the data into an MNE raw array
        iraw = mne.io.RawArray(self.data.T, self.data_info, verbose=False)
        self.raws.append(iraw)

    def event_mapper(self):

        keys = np.unique(self.annotation_flats)
        vals = np.arange(keys.size)
        self.event_mapping = dict(zip(keys,vals))

    def annotation_save(self,idx,raw):
        """
        Save annotation layer data. This requires us to loop over runs and sessions, which can be inferred or provided. 

        Args:
            idx (_type_): _description_
            raw (_type_): _description_
        """

        # Make the events file and save the results
        events  = []
        alldesc = []
        for iannot in self.annotations[idx].keys():
            desc  = self.annotations[idx][iannot]
            index = (1e-6*iannot)*self.fs
            events.append([index,0,self.event_mapping[desc]])
            alldesc.append(desc)
        events = np.array(events)

        # Make the bids path
        session_str    = "%s%03d" %(self.args.session,self.session_number)
        self.bids_path = mne_bids.BIDSPath(root=self.args.bidsroot, datatype='eeg', session=session_str, subject='%05d' %(self.subject_num), run=idx+1, task='task')

        # Save the bids data
        write_raw_bids(bids_path=self.bids_path, raw=raw, events_data=events,event_id=self.event_mapping, allow_preload=True, format='EDF',verbose=False)

        # Save the targets with the edf path paired up to filetype
        target_path = str(self.bids_path.copy()).rstrip('.edf')+'_targets.pickle'
        target_dict = {'uid':self.uid,'target':self.target,'annotation':'||'.join(alldesc),
                        'ieeg_file':self.current_file,'ieeg_start_sec':1e-6*self.clip_start_times[idx],'ieeg_duration_sec':1e-6*self.clip_durations[idx]}
        pickle.dump(target_dict,open(target_path,"wb"))

        # Make sure that the data is EDF compliant
        try:
            read_edf_header(str(self.bids_path))

            # Update lookup table
            self.create_lookup(idx)
        except OSError:
            #os.system(f"rm {str(self.bids_path)}")   # Remove once we have a system for sorting inputs
            #self.pickle_save(idx,raw,events,self.event_mapping)
            pass
        self.pickle_save(idx,raw,events,self.event_mapping)

    def direct_save(self,idx,raw):
        """
        Save a single data chunk from iEEG.org.

        Args:
            idx (_type_): _description_
            raw (_type_): _description_
        """

        # Save the edf in bids format
        if self.proposed_run == -1:
            run_number = int(self.file_idx)+1
        else:
            run_number = int(self.proposed_run)
        session_str    = "%s%03d" %(self.args.session,self.session_number)
        self.bids_path = mne_bids.BIDSPath(root=self.args.bidsroot, datatype='eeg', session=session_str, subject='%05d' %(self.subject_num), run=run_number, task='task')

        # Ensure we have an output directory to write to
        rootdir = '/'.join(str(self.bids_path).split('/')[:-1])
        if not path.exists(rootdir):
            os.system(f"mkdir -p {rootdir}")

        # Save the targets with the edf path paired up to filetype
        target_path = str(self.bids_path.copy()).rstrip('.edf')+'_targets.pickle'
        target_dict = {'uid':self.uid,'target':self.target}
        pickle.dump(target_dict,open(target_path,"wb"))

        # Create the lookup table
        self.create_lookup(idx)

    def save_bids(self):

        # Loop over all the raw data, add annotations, save
        for idx, raw in tqdm(enumerate(self.raws),desc="Saving Clip Data", total=len(self.raws), leave=False, disable=self.args.multithread):

            if raw == 'SKIP':
                pass
            else:
                
                # Set the channel types
                try:
                    raw.set_channel_types(self.channel_types.type)

                    # Check for annotations
                    if self.args.annotations:
                        if len(self.annotations[idx].keys()):
                            self.annotation_save(idx,raw)
                    else:
                        self.direct_save(idx,raw)
                except Exception as e:
                    self.pickle_save(idx,raw,None,None)

    def pickle_save(self,idx,raw,events,event_mapping):
        """
        If the data fails to save correctly for any reason, we just save a pickle object of the MNE object
        """

        try:
            # Make the bids path
            session_str    = "%s%03d" %(self.args.session,self.session_number)
            self.bids_path = mne_bids.BIDSPath(root=self.args.bidsroot, datatype='eeg', session=session_str, subject='%05d' %(self.subject_num), run=idx+1, task='task')
            self.bids_dir  = '/'.join(str(self.bids_path.copy()).split('/')[:-1])

            # Make sure the directory exists
            if not os.path.exists(self.bids_dir):
                os.system(f"mkdir -p {self.bids_dir}")

            # Make the output object
            channels = raw.ch_names
            mne_obj  = {'data':PD.DataFrame(raw.get_data().T,columns=channels),'events':events,'event_mapping':event_mapping,'samp_freq':self.fs}

            # If the data fails to write in anyway, save the raw as a pickle so we can fix later without redownloading it
            #error_path = str(self.bids_path.copy()).rstrip('.edf')+'_eeg.pickle'
            error_path = str(self.bids_path.copy()).rstrip('.edf')+'.pickle'
            pickle.dump(mne_obj,open(error_path,"wb"))
            self.create_lookup(idx)
        except Exception as e:
            print("Unable to save data. Skipping.")
            pass

    def create_lookup(self,idx):

        # Prepare some metadata for download
        source  = np.array(['ieeg.org','edf'])
        inds    = [self.args.ieeg,self.args.edf]
        source  = source[inds][0]
        user    = getpass.getuser()
        gendate = date.today().strftime("%d-%m-%y")
        times   = f"{self.args.start}_{self.args.duration}"

        # Save the subject file info with source metadata
        columns = ['orig_filename','source','creator','gendate','uid','subject_number','session_number','run_number','start_sec','duration_sec']
        iDF     = PD.DataFrame([[self.current_file,source,user,gendate,self.uid,self.subject_num,self.session_number,idx+1,1e-6*self.clip_start_times[idx],1e-6*self.clip_durations[idx]]],columns=columns)

        if not path.exists(self.subject_path):
            subject_DF = iDF.copy()
        else:
            with self.semaphore:
                subject_DF = PD.read_csv(self.subject_path)
            subject_DF = PD.concat((subject_DF,iDF))
        subject_DF['subject_number'] = subject_DF['subject_number'].astype(str).str.zfill(4)
        subject_DF['session_number'] = subject_DF['session_number'].astype(str).str.zfill(4)
        subject_DF                   = subject_DF.drop_duplicates()

        # Check if new data is being added to the subject path, wait until it is closed for reading
        with self.semaphore:
            subject_DF.to_csv(self.subject_path,index=False)

