import io
import getpass
import argparse
import paramiko
import pandas as PD
from prettytable import PrettyTable

class ssh_connection:

    def __init__(self,args,password):
        self.host     = args.host
        self.username = args.username
        self.password = password
        self.path     = args.path

    def read_data_collection(self):

        # Set up SSH connection to the remote system
        ssh = paramiko.SSHClient()

        # Check host file
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

        # Connect to the remote system
        ssh.connect(hostname=self.host, username=self.username, password=self.password)
            
        # Open an SFTP session using paramiko
        sftp = ssh.open_sftp()
        
        # Open the remote file as a file-like object
        remote_file = sftp.open(self.path, 'r')

        # Get the file contents
        file_contents = PD.read_csv(io.BytesIO(remote_file.read()))

        # Close the file and the SFTP session
        remote_file.close()
        sftp.close()
        
        # Close the SSH connection
        ssh.close()

        # Save the sorted results to be accessed by the viewer
        return file_contents.sort_values(by=['subject_number','session_number'])

class cache_search:

    def __init__(self,cache):

        # Save the original cache dataframe
        self.cache = cache

        # Create variables for initial display
        self.view_manager = True
        self.dataslice    = self.cache.copy()
        
        # Create loop for substring searching
        while self.view_manager:
            self.display()
            user_input = ""
            while user_input not in ['y','n','r']:
                user_input=input("Search for entry (y/n) or reset table (R/r)? ").lower()
            if user_input == 'n':
                self.view_manager = False
            elif user_input == 'r':
                self.dataslice = self.cache.copy()
            else:

                # Get the user input
                subcol = input("Please enter column to search within (default='orig_filename'): ").lower()
                substr = input("Please enter substring to search for: ").lower()
                
                # Clean up blank entries
                if subcol == '': subcol='orig_filename'

                # Make a dataslice based on user entries
                if substr != '':
                    self.create_dataslice(subcol,substr)

    def create_dataslice(self,subcol,substr):
        """
        Creates a temporary slice of the dataframe for viewing purposes only.

        Args:
            substr (str): Substring of the protocol names to display.
        """
        new_index = []
        for idx,ival in enumerate(list(self.dataslice[subcol].values)):
            if substr in ival.lower():
                new_index.append(idx)
        self.dataslice = self.dataslice.iloc[new_index]

    def display(self):
        
        # Initialize a pretty table for easy reading
        table = PrettyTable()
        table.field_names = self.dataslice.columns.tolist()

        # Step through the data and populate the pretty table with the current dataslice
        for idx, row in enumerate(self.dataslice.itertuples()):
            table.add_row(list(row)[1:])

        # Display pretty table
        print(table)

if __name__ == '__main__':
    
    # Command line options needed to obtain data.
    parser = argparse.ArgumentParser(description="Check for data in the CNT cache.")
    parser.add_argument("--username", required=True, help="Username on the host system.")
    parser.add_argument("--host", default="borel.seas.upenn.edu", help="Host system for the data cache.")
    parser.add_argument("--path", default='/mnt/leif/littlab/cache/Human_Data/BIDS/subject_map.csv', help="Path to the database file with cache info.")
    args = parser.parse_args()

    # Get the host password
    print("Please enter password for remote system:")
    password = getpass.getpass()

    # Get the data collection
    SC    = ssh_connection(args,password)
    cache = SC.read_data_collection()

    # Display the data collection
    CS = cache_search(cache[:150])