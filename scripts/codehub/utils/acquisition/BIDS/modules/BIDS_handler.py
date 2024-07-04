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

    def make_info(self):
        self.data_info = mne.create_info(ch_names=list(self.channels), sfreq=self.fs, verbose=False)

    def add_raw(self):
        self.raws.append(mne.io.RawArray(1e-6*self.data.T, self.data_info, verbose=False))

    def event_mapper(self):

        keys = np.unique(self.annotation_flats)
        vals = np.arange(keys.size)
        self.event_mapping = dict(zip(keys,vals))

    def annotation_save(self,idx,raw):

        # Make the events file and save the results
        try:
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

            # Overwrite the edf file only with set physical maxima/minima
            pmax = int(self.data.max())
            pmin = -pmax
            mne.export.export_raw(str(self.bids_path),raw,physical_range=(pmin,pmax),overwrite=True,verbose=False)

            # Save the targets with the edf path paired up to filetype
            target_path = str(self.bids_path.copy()).rstrip('.edf')+'_targets.pickle'
            target_dict = {'uid':self.uid,'target':self.target,'annotation':'||'.join(alldesc)}
            pickle.dump(target_dict,open(target_path,"wb"))

            # Update lookup table
            self.create_lookup(idx)

        except Exception as e:

            if self.args.debug:
                print(f"Annotation save error {e}")

            # If the data fails to write in anyway, save the raw as a pickle so we can fix later without redownloading it
            error_path = str(self.bids_path.copy()).rstrip('.edf')+'.pickle'
            pickle.dump((raw,events,self.event_mapping),open(error_path,"wb"))
            self.create_lookup(idx)

    def direct_save(self,idx,raw):

        # Save the edf in bids format
        if self.proposed_run == -1:
            run_number = int(self.file_idx)+1
        else:
            run_number = int(self.proposed_run)
        session_str    = "%s%03d" %(self.args.session,self.session_number)
        self.bids_path = mne_bids.BIDSPath(root=self.args.bidsroot, datatype='eeg', session=session_str, subject='%05d' %(self.subject_num), run=run_number, task='task')

        # Update the data to remove NaNs
        data = raw.get_data()
        data[np.isnan(data)] = 0
        raw._data = data

        # Ensure we have an output directory to write to
        rootdir = '/'.join(str(self.bids_path).split('/')[:-1])
        if not path.exists(rootdir):
            os.system(f"mkdir -p {rootdir}")

        # Write the bids file
        pmax = int(data.max())
        pmin = -pmax
        mne.export.export_raw(str(self.bids_path)+'.edf',raw,physical_range=(pmin,pmax),overwrite=True,verbose=False,fmt='edf')
        
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
                raw.set_channel_types(self.channel_types.type)

                # Check for annotations
                try:
                    if len(self.annotations[idx].keys()):
                        self.annotation_save(idx,raw)
                except AttributeError:
                    self.direct_save(idx,raw)

    def create_lookup(self,idx):

        # Prepare some metadata for download
        source  = np.array(['ieeg.org','edf'])
        inds    = [self.args.ieeg,self.args.edf]
        source  = source[inds][0]
        user    = getpass.getuser()
        gendate = date.today().strftime("%d-%m-%y")
        times   = f"{self.args.start}_{self.args.duration}"

        # Save the subject file info with source metadata
        columns = ['orig_filename','source','creator','gendate','uid','subject_number','session_number','run_number','start','duration']
        iDF     = PD.DataFrame([[self.current_file,source,user,gendate,self.uid,self.subject_num,self.session_number,idx+1,self.clip_start_times[idx],self.clip_durations[idx]]],columns=columns)

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

