# General libraries
import numpy as np
import pandas as PD

class dataframe_manager:
    """
    Class devoted to all things dataframe related. These dataframes will be used to populate the preprocessing, feature, tensor arrays etc.
    """

    def __init__(self):
        """
        Create the dataframe object with raw data.

        Allows for data samples without the same length.

        Row-major due to pyedflib limitations and need to work with 1-d slices for unequal lengths
        """

        ncol      = len(self.master_channel_list)
        values    = np.empty((self.nrow,ncol))
        values[:] = np.nan
        cols      = np.array(self.master_channel_list)
        for idx,icol in enumerate(self.clean_channel_map):
            ivals = self.raw_data[idx]
            try:
                column_idx                     = np.argwhere(cols==icol)[0][0]
                values[:ivals.size,column_idx] = ivals
            except IndexError:
                pass
        self.dataframe = PD.DataFrame(values, columns=self.master_channel_list)

    def column_subsection(self,keep_columns):
        """
        Return a dataframe with only the columns requested.
        
        Args:
            keep_columns (list of channel labels): List of columns to keep.
        """

        # Get the columns to drop
        drop_cols  = np.setdiff1d(self.dataframe.columns,keep_columns)
        self.dataframe = self.dataframe.drop(drop_cols, axis=1)