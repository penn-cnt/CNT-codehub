import os

# Local imports
from modules.addons.data_loader import *
from modules.addons.channel_clean import *
from modules.addons.channel_mapping import *
from modules.addons.channel_montage import *

class data_handler:

    def __init__(self,infile):
        self.infile = infile

    def data_prep(self):

        # Create pointers to the relevant classes
        DL    = data_loader()
        CHCLN = channel_clean()
        CHMAP = channel_mapping()
        CHMON = channel_montage()

        # Get the raw data
        DF,self.fs = DL.direct_inputs(self.infile,'edf')

        # Get the cleaned channel names
        clean_channels = CHCLN.direct_inputs(DF.columns)

        # Get the needed channels for this project
        channel_map = CHMAP.direct_inputs(clean_channels,"HUP1020")

        # Clean up the dataframe with the new labels and the right channels
        channel_dict = dict(zip(DF.columns,clean_channels))
        DF.rename(columns=channel_dict,inplace=True)
        DF = DF[channel_map]

        # Get the montage
        self.DF = CHMON.direct_inputs(DF,"HUP1020")

        return self.DF

if __name__ == '__main__':

    # Path to example data
    script_path  = os.path.abspath(__file__)
    example_dir  = '/'.join(script_path.split('/')[:-2])
    example_path = f"{example_dir}/example_data/sample_000.edf"

    # Get the cleaned dataset
    DH = data_handler(example_path)
    DF = DH.data_prep()
    print(DF)