import os
import io
import paramiko
from functools import partial

class data_stream:

    def __init__(self):
        pass

    def ssh_copy(self,filepath,temp_path,host,username):
        """
        Open a file on a remote system and return the object in memory using ssh and sftp.

        Parameters
        ----------
        filepath : STR
            Absolute path to the file on the remote system that needs to be opened.
        host : STR
            Name of the host system.
        username : STR
            Username for the remote system.
        password : STR
            Password for the remote system.
        partial_fnc : functools.partial object
            A partial instantiation of the data load. Enter all keywords except for the file location and this will
            call the file across an ssh protocol and inherit the remaining keywords.

        Returns
        -------
        file_contents : object
            In memory copy of the remote data.

        """

        # Set up SSH connection to the remote system
        ssh = paramiko.SSHClient()

        # Check host file
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

        # Connect to the remote system
        ssh.connect(hostname=host, username=username)
            
        # Open an SFTP session using paramiko
        sftp = ssh.open_sftp()
        
        # Copy the file to a temporary file object
        sftp.get(filepath,temp_path)
        
        # Close sftp
        sftp.close()
        
        # Close the SSH connection
        ssh.close()
