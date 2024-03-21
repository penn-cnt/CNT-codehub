import glob
import numpy as np
import pandas as PD
import pylab as PLT
import seaborn as sns
from sys import argv,exit
from matplotlib.gridspec import GridSpec

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
            #dataslice = dataslice.loc[dataslice.grid_num < 30]
            dataslice.insert(0,'time',np.array(list(dataslice.index))/fs)
            self.data_list[ichannel] = dataslice

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

    def z_scores(self,df,grid_num,channel,type='grid'):
        
        # Get the reference value
        if type == 'grid':
            ref_value = df.loc[df.grid_num==grid_num][channel].values
        elif type == 'median':
            ref_value = df.groupby(['time'])[channel].median().values

        # For each grid number, update the value to have z-value
        ugrids = df.grid_num.unique()
        for igrid in ugrids:
            ds                                             = df[(df.grid_num==igrid)].copy()
            stdevs                                         = np.std(ds[channel].values)
            offset                                         = (ds[channel].values-ref_value)
            z_score                                        = offset/stdevs
            df.loc[(df.grid_num==igrid),channel] = z_score
            
        return df
    
    def plot_data_by_grid(self):

        sns.set_theme(style="dark")

        # Plot each channel
        for ichannel in self.channels:
            
            # Grab the relevant data for this channel
            dataset = self.data_list[ichannel].sort_values(by=['grid_num','time'])
            dataset.insert(0,"ref_z",np.zeros(dataset.shape[0]))
            
            # Create the plot
            ncol   = 4
            nrow   = 4
            npages = np.ceil(dataset.grid_num.nunique()/(ncol*nrow)).astype('int')
            for ipage in range(npages):

                # Grab the current grid view for plotting in chunks
                grid_min      = (ipage)*(nrow*ncol)
                grid_max      = (ipage+1)*(nrow*ncol)
                dataset_slice = dataset.loc[(dataset.grid_num>=grid_min)&(dataset.grid_num<grid_max)] 

                # Plot the centered values for the slice
                plts   = sns.relplot(data=dataset_slice,x="time", y='ref_z', col="grid_num", hue="grid_num",kind="line", palette="crest", linewidth=1.5, zorder=5, col_wrap=4, height=2, aspect=1.5, legend=False)

                # Iterate over each subplot to customize further
                for grid_num, ax in plts.axes_dict.items():

                    # Add the title as an annotation within the plot
                    ax.text(.8, .85, grid_num, transform=ax.transAxes, fontweight="bold")

                    # Plot every year's time series in the background
                    self.ref_values = dataset.loc[dataset.grid_num==grid_num][ichannel]
                    ref_df          = self.z_scores(dataset.copy(),grid_num,ichannel)
                    try:
                        sns.lineplot(data=ref_df, x="time", y=ichannel, units="grid_num", estimator=None, color=".7", linewidth=1, ax=ax)
                    except:
                        pass

                # Define and output filename
                plotdir = '../../user_data/derivative/preprocessing_grid/grid/plots/'
                fpath   = f"{plotdir}{self.filename.split('.pickle')[0]}_{ichannel}_{ipage:02}.png"

                # Tweak the supporting aspects of the plot
                plts.set_titles("")
                plts.set_axis_labels("", f"{ichannel} (z-score)")
                plts.tight_layout()
                PLT.savefig(fpath)
                PLT.close("all")
    
    def plot_data_by_median(self):

        # Set the plot theme
        sns.set_theme(style="dark")

        # Loop over channel data
        for ichannel in self.channels:
            
            # Grab the relevant data for this channel
            dataset            = self.data_list[ichannel].sort_values(by=['grid_num','time'])
            ref_df             = dataset.loc[dataset.grid_num==0].copy()
            ref_df[ichannel]   = 0
            ref_df['grid_num'] = 'median'
            
            # get the reference values for grid wise plots
            grid_array = dataset.grid_num.unique()

            # Make the plotting environment
            fig = PLT.figure(dpi=100,figsize=(8.,6.))
            gs  = GridSpec(1, 2, width_ratios=[3, 1], wspace=0.05)
            ax1 = fig.add_subplot(gs[0])
            ax2 = fig.add_subplot(gs[1],sharey=ax1)
            
            # Plot the timeseries median reference lines
            plts = sns.lineplot(data=ref_df,x="time", y=ichannel, linewidth=1.5, ax=ax1, hue="grid_num", palette="crest", legend=False, zorder=5)

            # Plot the z-scores
            y_values = np.array([])
            for igrid in grid_array:
                grid_df  = self.z_scores(dataset.copy(),igrid,ichannel,type='median')
                y_values = np.concatenate((y_values,grid_df[ichannel].values)) 
                sns.lineplot(data=grid_df, x="time", y=ichannel, estimator=None, color=".7", linewidth=1, ax=ax1)
            ax1.set_ylabel(f"{ichannel} (z-value)",fontsize=16)
            ax1.set_xlabel(f"t (seconds)",fontsize=16)

            # Plot the distribution
            counts,bin_edges = np.histogram(y_values,bins=50)
            x_values         = 0.5*(bin_edges[:-1]+bin_edges[1:])
            pdf              = counts/counts.sum()
            ax2.hist(bin_edges[:-1], bin_edges, weights=pdf,orientation='horizontal')
            ax2.yaxis.tick_right()
            ax2.set_xticks([0.25,0.5,0.75])
            ax2.set_xlim([0,1])
            ax2.set_xlabel('P(z-values)',fontsize=16)
            
            # Add grids
            ax1.grid()
            ax2.grid()

            # Set the title
            PLT.suptitle(f"{self.filename}",fontsize=12)


            # Define and output filename
            plotdir = '../../user_data/derivative/preprocessing_grid/grid/plots/'
            fpath   = f"{plotdir}{self.filename.split('.pickle')[0]}_{ichannel}_zscore.png"
            PLT.savefig(fpath)
            PLT.close("all")


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
        PC.plot_data_by_median()
