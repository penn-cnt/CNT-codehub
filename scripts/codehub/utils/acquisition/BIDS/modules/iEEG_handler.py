import numpy as np
import pandas as PD
from sys import exit
from tqdm import tqdm
from time import sleep
from os import path,system
from ieeg.auth import Session

# Multicore support
import multiprocessing

# Local imports
from modules.BIDS_handler import BIDS_handler

# Allows us to catch ieeg api errors
import ieeg.ieeg_api as IIA
from requests.exceptions import ReadTimeout as RTIMEOUT
from pandas.errors import EmptyDataError

# API timeout class
import signal
class TimeoutException(Exception):
    pass

class Timeout:
    """
    Manage timeouts to the iEEG.org API call. It can go stale, and sit for long periods of time otherwise.
    """

    def __init__(self, seconds=1, multiflag=False, error_message='Function call timed out'):
        self.seconds       = seconds
        self.error_message = error_message
        self.multiflag     = multiflag

    def handle_timeout(self, signum, frame):
        raise TimeoutException(self.error_message)

    def __enter__(self):
        if not self.multiflag:
            signal.signal(signal.SIGALRM, self.handle_timeout)
            signal.alarm(self.seconds)
        else:
            pass

    def __exit__(self, exc_type, exc_value, traceback):
        if not self.multiflag:
            signal.alarm(0)
        else:
            pass

class iEEG_download(BIDS_handler):
    """
    Manages the actual downloads from iEEG.org. It then calls the BIDS handler to save the data.
    If using the annotation method, it will loop over each clip layer in this class.

    Args:
        BIDS_handler (_type_): _description_
    """

    def __init__(self, args, semaphore):
        
        # Store variables based on input params
        self.args           = args
        self.subject_path   = args.bidsroot+args.subject_file
        self.semaphore      = semaphore

        # Hard coded variables based on ieeg api
        self.n_retry        = 3
        self.global_timeout = 60
        self.clip_layer     = 'EEG clip times'
        self.natus_layer    = 'Imported Natus ENT annotations'

        # Get list of files to skip that already exist locally
        if path.exists(self.subject_path):
            # Read in the subject information
            with self.semaphore:
                self.subject_cache      = PD.read_csv(self.subject_path)
            self.processed_files    = self.subject_cache['orig_filename'].values
            self.processed_start    = self.subject_cache['start'].values
        else:
            self.processed_files    = []
            self.processed_start    = []

    def reset_variables(self):
        """
        Sometimes MNE bids needs to be reset. Use with caution.
        """            

        # Delete all variables in the object's namespace
        for var_name in list(self.__dict__.keys()):
            delattr(self, var_name)

    def get_annotations(self):
        """
        Query the annotation layer from iEEG.org and get the formatted start times and durations.
        The most important outputs are self.clip_start_times and self.clip_durations.
        """

        # Get the clip times
        self.session_method_handler(0,1e6,annotation_flag=True)

        if self.success_flag:
            # Remove start clip time if it is just the machine starting up
            if self.clips[0].type.lower() == 'clip end' and self.clips[0].end_time_offset_usec == 2000:
                self.clips = self.clips[1:]

            # Manage edge cases
            if self.clips[0].type.lower() == 'clip end':
                self.clips = list(np.concatenate(([0],self.clips), axis=0))
            if self.clips[-1].type.lower() == 'clip start':
                self.clips = list(np.concatenate((self.clips,[self.end_time-self.start_time]), axis=0))

            clip_vals = []
            for iclip in self.clips:
                try:
                    clip_vals.append(iclip.start_time_offset_usec)
                except AttributeError:
                    clip_vals.append(iclip)

            # Turn the clip times into start and end arrays
            self.clip_start_times = np.array([iclip for iclip in clip_vals[::2]])
            self.clip_end_times   = np.array([iclip for iclip in clip_vals[1::2]])
            self.clip_durations   = self.clip_end_times-self.clip_start_times

            # Match the annotations to the clips
            self.annotations      = {ival:{} for ival in range(self.clip_start_times.size)}
            self.annotation_flats = []
            for annot in self.raw_annotations:
                time = annot.start_time_offset_usec
                desc = annot.description
                for idx, istart in enumerate(self.clip_start_times):
                    if (time >= istart) and (time <= self.clip_end_times[idx]):
                        event_time_shift = (time-istart)
                        self.annotations[idx][event_time_shift] = desc
                        self.annotation_flats.append(desc)

    def download_by_cli(self, uid, file, target, start, duration, proposed_sub, proposed_ses, proposed_run, file_idx):
        """
        Download a single chunk of data from iEEG.org

        Args:
            uid (_type_): _description_
            file (_type_): _description_
            target (_type_): _description_
            start (_type_): _description_
            duration (_type_): _description_
            proposed_sub (_type_): _description_
            proposed_ses (_type_): _description_
            proposed_run (_type_): _description_
            file_idx (_type_): _description_
        """

        # Store the ieeg filename
        self.uid          = uid
        self.current_file = file
        self.target       = target
        self.success_flag = False
        self.proposed_sub = proposed_sub
        self.proposed_ses = proposed_ses
        self.proposed_run = proposed_run
        self.file_idx     = file_idx

        # Loop over clips
        BIDS_handler.__init__(self)
        self.session_method_handler(start,duration)
        if self.success_flag == True:
            BIDS_handler.channel_cleanup(self)
            BIDS_handler.get_channel_type(self)
            BIDS_handler.make_info(self)
            BIDS_handler.add_raw(self)

        # Save the bids files if we have any data
        try:
            if len(self.raws) > 0:
                BIDS_handler.save_bids(self)
        except AttributeError as e:
            if self.args.debug:
                print(f"Save error {e}")
            pass

    def download_by_annotation(self, uid, file, target, proposed_sub, proposed_ses, proposed_run):
        """
        Download each annotation layer within a given file on iEEG.org.

        Args:
            uid (_type_): _description_
            file (_type_): _description_
            target (_type_): _description_
            proposed_sub (_type_): _description_
            proposed_ses (_type_): _description_
            proposed_run (_type_): _description_
        """

        # Store the ieeg filename
        self.uid          = uid
        self.current_file = file
        self.target       = target
        self.success_flag = False
        self.proposed_sub = proposed_sub
        self.proposed_ses = proposed_ses
        self.proposed_run = proposed_run

        # Get the annotation times
        self.get_annotations()

        # Loop over clips
        if self.success_flag == True:
            BIDS_handler.__init__(self)
            for idx,istart in tqdm(enumerate(self.clip_start_times), desc="Downloading Clip Data", total=len(self.clip_start_times), leave=False, disable=self.args.multithread):

                if istart == 0 and self.args.skipzero:
                    self.raws.append("SKIP")
                else:
                    # Check if this combo has already been processed before
                    pinds = (self.processed_files==self.current_file)&(self.processed_start==istart)
                    if pinds.any():
                        self.raws.append("SKIP")
                        if self.args.multithread:
                            print(f"Skipping {self.current_file} annotation at {istart}.")
                    else:
                        if self.args.multithread:
                            print(f"Downloading {self.current_file} annotation at {istart}")
                        self.session_method_handler(istart, self.clip_durations[idx])
                        if self.success_flag == True:
                            BIDS_handler.channel_cleanup(self)
                            BIDS_handler.get_channel_type(self)
                            BIDS_handler.make_info(self)
                            BIDS_handler.add_raw(self)

        # Save the bids files if we have any data
        try:
            if len(self.raws) > 0:

                print(self.raws)
                exit()

                BIDS_handler.event_mapper(self)
                BIDS_handler.save_bids(self)
        except AttributeError as e:
            if self.args.debug:
                print(f"Download error {e}")

    def session_method_handler(self,start,duration,annotation_flag=False):
        """
        Wrapper to call ieeg. Due to ieeg errors, we want to make sure we can try to call it a few times before giving up.

        Args:
            start (float): Start time (referenced to data start) in microseconds to request data from
            duration (float): Duration in microseconds of data to request
            annotation_flag (bool, optional): Flag whether we just want annotation data or not. Defaults to False.
        """

        n_attempts = 0
        while True:
            with Timeout(self.global_timeout,self.args.multithread):
                try:
                    self.session_method(start,duration,annotation_flag)
                    self.success_flag = True
                    break
                except (IIA.IeegConnectionError,IIA.IeegServiceError,TimeoutException,RTIMEOUT,TypeError) as e:
                    if n_attempts<self.n_retry:
                        sleep(5)
                        n_attempts += 1
                    else:
                        if self.args.debug:
                            print(f"Connection Error: {e}")
                        self.success_flag = False
                        fp = open(self.args.bidsroot+self.args.failure_file,"a")
                        fp.write(f"{self.uid},{self.current_file},{start},{duration},{self.target},'{e}'\n")
                        fp.close()
                        break

    def session_method(self,start,duration,annotation_flag):
        """
        Call ieeg.org for data and return data or annotations.

        Args:
            start (float): Start time (referenced to data start) in microseconds to request data from
            duration (float): Duration in microseconds of data to request
            annotation_flag (bool, optional): Flag whether we just want annotation data or not. Defaults to False.

        Raises:
            IndexError: If there are multiple sampling frequencies, bids does not readily support this. Alerts user and stops.
        """

        with Session(self.args.username,self.args.password) as session:
            
            # Open dataset session
            dataset = session.open_dataset(self.current_file)
            
            # Logic gate for annotation call (faster, no time data needed) or get actual data
            if not annotation_flag:

                # Get the channel names and integer representations for data call
                self.channels = dataset.ch_labels
                channel_cntr  = list(range(len(self.channels)))

                # If duration is greater than 10 min, break up the call. Make array of start,duration with max 10 min each chunk
                time_cutoff = int(10*60*1e6)
                end_time    = start+duration
                ival        = start
                chunks      = []
                while ival < end_time:
                    if ival+time_cutoff >= end_time:
                        chunks.append([ival,end_time-ival])
                    else:
                        chunks.append([ival,time_cutoff])
                    ival += time_cutoff

                # Call data and concatenate calls if greater than 10 min
                self.data   = []
                for ival in chunks:
                    self.data.append(dataset.get_data(ival[0],ival[1],channel_cntr))
                if len(self.data) > 1:
                    self.data = np.concatenate(self.data)
                else:
                    self.data = self.data[0]

                # Get the samping frequencies
                self.fs = [dataset.get_time_series_details(ichannel).sample_rate for ichannel in self.channels]

                # Data quality checks before saving
                if np.unique(self.fs).size == 1:
                    self.fs = self.fs[0]
                else:
                    raise IndexError("Too many unique values for sampling frequency.")
            else:
                self.clips           = dataset.get_annotations(self.clip_layer)
                self.raw_annotations = dataset.get_annotations(self.natus_layer)
                self.start_time      = dataset.start_time
                self.end_time        = dataset.end_time
            session.close()

class ieeg_handler:

    def __init__(self,args,input_data):
        self.args         = args
        self.input_data   = input_data
        self.input_files  = input_data['orig_filename'].values
        self.start_times  = input_data['start'].values
        self.durations    = input_data['duration'].values
        self.proposed_sub = input_data['subject_number'].values
        self.proposed_ses = input_data['session_number'].values
        self.proposed_run = input_data['run_number'].values

    def single_pull(self):

        # Add a sempahore to allow orderly file access (to mimic multiprocesing for ease of argument definition)
        semaphore = multiprocessing.Semaphore(1)

        # Grab all the file indices to pass to the data pulling manager
        file_indices = np.array(range(self.input_files.size))
        self.pull_data(file_indices,semaphore)

    def multicore_pull(self):

        # Add a sempahore to allow orderly file access
        semaphore = multiprocessing.Semaphore(1)

        # Make the BIDS root data structure with a single core to avoid top-level directory issues
        self.pull_data(np.array([0]),semaphore)

        # Calculate the size of each subset based on the number of processes
        file_indices = np.array(range(self.input_files.size-1))+1
        subset_size  = (file_indices.size) // self.args.ncpu
        if subset_size > 0:
            list_subsets = [file_indices[i:i + subset_size] for i in range(0, self.args.ncpu*subset_size, subset_size)]

            # Handle leftovers
            remainder = file_indices[self.args.ncpu*subset_size:]
            for idx,ival in enumerate(remainder):
                list_subsets[idx] = np.concatenate((list_subsets[idx],np.array([ival])))
        else:
            list_subsets = [file_indices[:]]

        processes = []
        for data_chunk in list_subsets:
            process = multiprocessing.Process(target=self.pull_data, args=(np.array([data_chunk]).flatten(),semaphore))
            processes.append(process)
            process.start()
        
        # Wait for all processes to complete
        for process in processes:
            process.join()

    def pull_data(self,file_indices,semaphore):

        # Loop over files
        IEEG = iEEG_download(self.args,semaphore)
        for file_idx in file_indices:

            # Get the current file
            ifile  = self.input_files[file_idx]
            iid    = self.input_data['uid'].values[file_idx]
            target = self.input_data['target'].values[file_idx]

            if self.args.annotations:
                IEEG.download_by_annotation(iid,ifile,target,self.proposed_sub[file_idx],self.proposed_ses[file_idx],self.proposed_run[file_idx])
                IEEG = iEEG_download(self.args,semaphore)
            else:
                print(f"Working on {ifile}.")
                IEEG.download_by_cli(iid,ifile,target,self.start_times[file_idx],self.durations[file_idx],self.proposed_sub[file_idx],self.proposed_ses[file_idx],self.proposed_run[file_idx],file_idx)




