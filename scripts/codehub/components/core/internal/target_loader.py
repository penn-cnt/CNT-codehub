import re
import glob
import pickle
import numpy as np

class target_loader:

    def __init__(self):
        pass

    def find_matching_strings(self,reference_string, input_strings, trailing_characters_pattern):
        pattern = re.compile(f"{re.escape(reference_string)}{trailing_characters_pattern}")
        matching_strings = [string for string in input_strings if re.match(pattern, string)]
        return matching_strings

    def bids_finder(self,current_edf,target_substring):

        # Reformat the expected string structures based on bids logic
        base_bids_string = '.'.join(current_edf.split('.')[:-1])

        # Find all files that match the base bids string
        target_candidates = glob.glob(base_bids_string+"*")

        # Define the target rule
        trailing_characters_pattern = r".*"+target_substring+r".*"

        # Get the matched string
        target_files = self.find_matching_strings(base_bids_string, target_candidates, trailing_characters_pattern)
        
        # Return target is a one to one match. Otherwise return None
        if len(target_files) == 1:
            self.target_file = target_files[0]
        else:
            self.target_file = None

    def load_targets(self,current_edf,datatype,target_substring):

        # Find the target file based on datatype and substrings
        if datatype == 'bids':
            self.bids_finder(current_edf,target_substring)

        # Logic gates for type of target files
        if self.target_file != None:
            
            # Load the data
            raw_targets = pickle.load(open(self.target_file,"rb"))
            
            # Apply logic to known target types
            self.target_logic(raw_targets,current_edf)

    #################################################################
    ###### Logic options to clean up different target strings. ######
    #################################################################
                
    def target_logic(self,raw_targets,current_edf):
        if 'TUEG_dt_t0' in raw_targets.keys():
            self.TUEG_dt(raw_targets,current_edf)
        elif 'tueg_string' in raw_targets.keys():
            self.TUEG_string(raw_targets,current_edf)
        else:
            self.all_others(raw_targets,current_edf)

    def TUEG_dt(self,target_dict,current_edf):

        # Get the indices of the target df we need to update
        inds = (self.feature_df.file.values==current_edf)

        # Make the output array
        if 'TUEG' not in self.feature_df.columns:
            newvals = np.array(['' for idx in range(self.feature_df.shape[0])])
        else:
            newvals = self.feature_df['TUEG'].values
        
        # Get the start time and end time for each row
        t_start_array = self.feature_df['t_start'].values[inds]
        t_end_array   = self.feature_df['t_end'].values[inds]

        # Loop over the times and then check against temple tags
        for ii in range(t_start_array.size):

            # Make a time array we can compare against the temple time window. Overlap checking is easier than logic gates
            data_time_array = np.around(np.arange(float(t_start_array[ii]),float(t_end_array[ii]),0.1),1)

            # Loop over the possible temple tags
            TUEG_t0_array  = target_dict['TUEG_dt_t0'].split('_')
            TUEG_t1_array  = target_dict['TUEG_dt_t1'].split('_')
            TUEG_tag_array = target_dict['TUEG_dt_tag'].split('_')
            for jj,jtag in enumerate(TUEG_tag_array):

                # Make a TUEG time array
                tueg_time_array = np.around(np.arange(float(TUEG_t0_array[jj]),float(TUEG_t1_array[jj]),0.1),1)

                # Check for overlap
                if np.intersect1d(data_time_array,tueg_time_array).size > 0:
                    newvals[np.arange(self.feature_df.shape[0])[inds][ii]] = jtag

        self.feature_df['TUEG'] = newvals

    def TUEG_string(self,raw_targets,current_edf):

        # Make the output array
        if 'TUEG' not in self.feature_df.columns:
            newvals = np.array(['' for idx in range(self.feature_df.shape[0])])
        else:
            newvals = self.feature_df['TUEG'].values

        # Assign target values to the correct data files
        inds                  = (self.feature_df.file.values==current_edf)
        newvals[inds]         = raw_targets['tueg_string']
        self.feature_df['TUEG'] = newvals

    def all_others(self,raw_targets,current_edf):

        # Create the target columns as needed
        if type(raw_targets) == dict:
            for tkey in raw_targets.keys():
                if tkey not in self.feature_df.columns:
                    self.feature_df[tkey] = None

        # Assign target values to the correct data files
        inds    = (self.feature_df.file.values==current_edf)
        for tkey in raw_targets.keys():
            newvals               = self.feature_df[tkey].values
            newvals[inds]         = raw_targets[tkey]
            self.feature_df[tkey] = newvals