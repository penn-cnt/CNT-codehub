import re
import glob
import pickle

class target_loader:

    def __init__(self):
        pass

    def load_targets(self,current_edf,datatype,target_substring,file_format,target_label):

        # Find the target file based on datatype and substrings
        if datatype == 'bids':
            self.bids_finder(current_edf,target_substring)

        # Logic gates for type of target files
        if self.target_file != None:
            
            # Load the data
            raw_target = pickle.load(open(self.target_file,"rb"))

            # Based on file type, get the targert variable
            if file_format == 'dict':
                return raw_target[target_label]            

    def find_matching_strings(self,reference_string, input_strings, trailing_characters_pattern):
        pattern = re.compile(f"{re.escape(reference_string)}{trailing_characters_pattern}")
        matching_strings = [string for string in input_strings if re.match(pattern, string)]
        return matching_strings

    def bids_finder(self,current_edf,target_substring):

        # Reformat the expected string structures based on bids logic
        base_bids_string = '_'.join(current_edf.split('_')[:-1])

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

