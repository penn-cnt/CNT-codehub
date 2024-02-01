import glob
import numpy as np
import pandas as PD
import pylab as PLT
import seaborn as sns
from sys import argv,exit

# Local imports
from modules.addons.data_loader import *
from modules.addons.channel_clean import *
from modules.addons.channel_mapping import *

class pipeline_comparison:

    def __init__(self,filename,filepaths):
        self.filename  = filename
        self.filepaths = filepaths

    def process_data(self):

        # Iterate over the filepaths
        self.data_list = {}
        raw_output     = []
        for ifile in self.filepaths:
            grid_num       = int(ifile.split('grid/grid/')[1].split('/')[0])
            DF,channels,fs = self.data_prep(ifile)
            DF['grid_num'] = grid_num
            raw_output.append(DF)
        self.channels = channels

        # Merge the outputs into one dataframe
        output = PD.concat(raw_output)

        # Save the data channelwise
        for ichannel in self.channels:
            dataslice = output[['grid_num',ichannel]]
            dataslice = dataslice.loc[dataslice.grid_num < 10]
            dataslice.insert(0,'time',np.array(list(dataslice.index))/fs)
            self.data_list[ichannel] = dataslice
            print(f"{ichannel}, {dataslice.groupby(['time'])[ichannel].nunique()}")


    def data_prep(self,infile):

        # Create pointers to the relevant classes
        DL    = data_loader()
        CHCLN = channel_clean()
        CHMAP = channel_mapping()

        # Get the raw data and pointers
        DF,fs = pickle.load(open(infile,"rb"))
        fs    = fs[0]

        # Get the cleaned channel names
        clean_channels = CHCLN.direct_inputs(DF.columns)
        channel_dict   = dict(zip(DF.columns,clean_channels))
        DF.rename(columns=channel_dict,inplace=True)

        # Get the channel mapping
        channel_map = CHMAP.direct_inputs(DF.columns,"HUP1020")
        DF          = DF[channel_map]
        return DF,channel_map,fs

    def z_scores(self,dataset,grid_num,channel):
        
        # Get the reference value
        ref_value = dataset.loc[dataset.grid_num==grid_num][channel]

        # For each grid number, update the value to have z-value
        ugrids = dataset.grid_num.unique()
        for igrid in ugrids:
            ds                                             = dataset[(dataset.grid_num==igrid)]
            ds[channel]                                    = (ds[channel]-ref_value)#/np.std(ds[channel].values)
            dataset.loc[(dataset.grid_num==igrid),channel] = ds[channel]
        return dataset
    
    def plot_data(self):

        sns.set_theme(style="dark")

        # Plot each year's time series in its own facet
        for ichannel in self.channels:
            
            # Grab the relevant data for this channel
            dataset = self.data_list[ichannel].sort_values(by=['grid_num','time'])
            dataset.insert(0,"ref_z",np.zeros(dataset.shape[0]))
            
            # Create the plot
            g = sns.relplot(data=dataset,x="time", y=ichannel, col="grid_num", hue="grid_num",kind="line", palette="crest", linewidth=1.5, zorder=5, col_wrap=6, height=2, aspect=1.5, legend=False)

            # Iterate over each subplot to customize further
            for grid_num, ax in g.axes_dict.items():

                # Add the title as an annotation within the plot
                ax.text(.8, .85, grid_num, transform=ax.transAxes, fontweight="bold")

                # Plot every year's time series in the background
                ref_df          = dataset.copy()
                #self.ref_values = dataset.loc[dataset.grid_num==grid_num][ichannel]
                #ref_df          = self.z_scores(dataset,grid_num,ichannel)
                sns.lineplot(data=ref_df, x="time", y=ichannel, units="grid_num", estimator=None, color=".7", linewidth=1, ax=ax)

            # Tweak the supporting aspects of the plot
            g.set_titles("")
            g.set_axis_labels("", f"{ichannel} (uV)")
            g.tight_layout()
            PLT.show()
            exit()

if __name__ == '__main__':

    # Get the top level directory to search for data under
    #globstr = argv[1]
    globstr = '/Users/bjprager/Documents/GitHub/scalp_deep-learning/user_data/derivative/preprocessing_grid/grid/*/*/preprocessing_snapshot/*/*/*pickle'
    
    # Find all matching files
    filepaths = np.array(glob.glob(globstr))
    files     = np.array([ifile.split('/')[-1] for ifile in filepaths])
    ufiles    = np.unique(files)
    
    # Loop over unique datasets different grid results
    for ifile in ufiles:

        # Find the subsection of filepaths
        ipaths = filepaths[(files==ifile)]

        # Run the pipeline comparison
        print(f"Processing {ifile}.")
        PC = pipeline_comparison(ifile,ipaths)
        PC.process_data()
        PC.plot_data()
        exit()
