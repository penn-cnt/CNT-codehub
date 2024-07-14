import numpy as np
import pandas as PD
from sys import argv,exit

class marsh_rejection:
    """
    Applies a marsh rejection mask to a dataframe. 
    Looks for dt=-1 from the pipeline manager to reference against the full file.
    """

    def __init__(self,DF):

        # Save the dataframe into self
        self.DF = DF

        # Find the channel labels
        ref_labels        = ['file', 't_start', 't_end', 't_window', 'method', 'tag', 'uid', 'target', 'annotation']
        self.channels     = np.setdiff1d(self.DF.columns, ref_labels)
        self.ref_cols     = np.setdiff1d(self.DF.columns, self.channels)
        self.merge_labels = np.concatenate((['file', 'method', 'tag'],self.channels))

    def reshape_dataframe(self):

        # Separate DF based on dt values
        DF_windows = self.DF[self.DF['t_window'] != -1]
        DF_file    = self.DF[self.DF['t_window'] == -1]

        # Merge the DataFrames on 'filename', 'tag_1', and 'tag_2'
        merged_DF = PD.merge(DF_windows,DF_file[self.merge_labels], on=['file', 'method', 'tag'],suffixes=('', '_ref'))

        return merged_DF
    
    def drop_file(self):

        self.DF = self.DF[self.DF['t_window'] != -1]

    def get_mean_stats(self):

        # Make a dataslice just for rms and just for ll
        DF_rms = self.DF.loc[self.DF.method=='rms']
        DF_ll = self.DF.loc[self.DF.method=='line_length']

        # Get the group level values
        DF_rms_mean  = DF_rms.groupby(['file'])[self.channels].mean()
        DF_rms_stdev = DF_rms.groupby(['file'])[self.channels].std()
        DF_ll_mean   = DF_ll.groupby(['file'])[self.channels].mean()
        DF_ll_stdev  = DF_ll.groupby(['file'])[self.channels].std()

        # Make an output column
        self.DF.loc[:,['marsh_rejection']] = True

        # Apply the filter
        DF_rms.set_index(['file'],inplace=True)
        DF_ll.set_index(['file'],inplace=True)
        DF_rms = DF_rms.sort_values(by=['t_start','t_end','t_window'])
        DF_ll  = DF_ll.sort_values(by=['t_start','t_end','t_window'])
        for ifile in DF_rms_mean.index:
            
            # Get the reference values
            ref_rms_mean  = DF_rms_mean.loc[ifile]
            ref_rms_stdev = DF_rms_stdev.loc[ifile]
            ref_ll_mean   = DF_ll_mean.loc[ifile]
            ref_ll_stdev  = DF_ll_stdev.loc[ifile]
            
            # Get the rms mask
            DF_rms_slice     = DF_rms.loc[ifile]
            channel_rms_mask = DF_rms_slice[self.channels]>(ref_rms_mean+2*ref_rms_stdev).values
            segment_rms_mask = []
            for irow in channel_rms_mask.values:
                segment_rms_mask.append(~irow.any())

            # Get the rms mask
            DF_ll_slice     = DF_ll.loc[ifile]
            channel_ll_mask = DF_ll_slice[self.channels]>(ref_ll_mean+2*ref_ll_stdev).values
            segment_ll_mask = []
            for irow in channel_ll_mask.values:
                segment_ll_mask.append(~irow.any())

            # Get the final marsh mask
            mask_arr = np.array(segment_rms_mask)*np.array(segment_ll_mask)

            # Reshape indices of subslice to we can iterate over time segments in the full file
            for ii in range(DF_rms_slice.shape[0]):
                
                # Get the references to find the right rows in the bigger dataframe
                irow     = DF_rms_slice.iloc[ii]
                t_start  = irow.t_start
                t_end    = irow.t_end
                t_window = irow.t_window
                mask     = mask_arr[ii]

                # Set the mask value
                self.DF.loc[(self.DF.file==ifile)&(self.DF.t_start==t_start)&(self.DF.t_end==t_end)&(self.DF.t_window==t_window),['marsh_rejection']] = mask                

    def return_df(self):
        return self.DF

if __name__ == '__main__':

    # Read in the data
    DF = PD.read_pickle(argv[1])
    DF.drop(['FZ-CZ'],axis=1,inplace=True)

    # Get the reformatted dataframe
    MR    = marsh_rejection(DF)
    MR.drop_file()
    MR.get_mean_stats()
    newDF = MR.return_df()
    newDF.to_pickle(argv[2])