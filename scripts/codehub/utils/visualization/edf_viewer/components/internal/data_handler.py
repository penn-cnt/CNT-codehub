# Local imports
from components.curation.public.data_loader import *
from components.workflows.public.channel_clean import *
from components.workflows.public.channel_mapping import *
from components.workflows.public.channel_montage import *

class data_handler:

    def __init__(self,args,infile):
        self.args   = args
        self.infile = infile 

    def workflow(self):
        self.load_data()
        self.clean_channels()
        self.map_channels()
        self.montage_channels()
        self.get_time_info()
        return self.DF, self.fs, self.t_max, self.duration, self.t0

    def load_data(self):

        # Initialize class
        DL = data_loader()

        # Get the raw data and pointers
        if not self.args.pickle_load:
            self.DF,self.fs = DL.direct_inputs(self.infile,'edf')
        else:
            raw_input       = pickle.load(open(self.infile,"rb"))
            if type(raw_input) == np.ndarray or type(raw_input) == PD.core.frame.DataFrame:
                self.DF = raw_input
                if self.args.fs == None:
                    raise Exception("Must provide sampling frequency if passing a pickle file with only an array or dataframe.")
                else:
                    self.fs = self.args.fs
            else:
                self.DF = raw_input[0]
                self.fs = raw_input[1]

    def clean_channels(self):

        # Initialize class
        CHCLN = channel_clean()

        # Get the cleaned channel names
        clean_channels = CHCLN.direct_inputs(self.DF.columns,clean_method=self.args.chcln)
        channel_dict   = dict(zip(self.DF.columns,clean_channels))
        self.DF.rename(columns=channel_dict,inplace=True)

    def map_channels(self):

        # Initialize class
        CHMAP = channel_mapping()

        # Get the channel mapping
        if self.args.chmap != None:
            channel_map = CHMAP.direct_inputs(self.DF.columns,self.args.chmap)
            self.DF     = self.DF[channel_map]

    def montage_channels(self):

        # Initialize class
        CHMON = channel_montage()

        # Get the montage
        if self.args.montage != None:
            self.DF = CHMON.direct_inputs(self.DF,self.args.montage)

    def get_time_info(self):

        # Get the duration
        self.t_max = self.DF.shape[0]/self.fs
        if self.args.dur_frac:
            self.duration = self.args.dur*self.t_max
        else:
            self.duration = self.args.dur

        # Get the start time
        if self.args.t0_frac and self.args.t0 != None:
            self.t0 = self.args.t0*self.t_max
        else:
            if self.args.t0 != None:
                self.t0 = self.args.t0
            else:
                self.t0 = np.random.rand()*(self.t_max-self.args.dur)