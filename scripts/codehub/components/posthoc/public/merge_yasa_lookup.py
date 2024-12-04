import numpy as np
import pandas as PD
from sys import argv
from tqdm import tqdm

if __name__ == '__main__':

    # Read in the raw data
    YASA_DF    = PD.read_csv(argv[1])
    FEATURE_DF = PD.read_csv(argv[2])

    # Clean up the labels to just be sleep or wake
    new_map      = {'N1':'S','N2':'S','N3':'S','R':'S','W':'W'}
    consensus_cols = [icol for icol in YASA_DF if 'yasa' in icol]
    for icol in consensus_cols:
        YASA_DF[icol] = YASA_DF[icol].apply(lambda x:new_map[x])

    # Get the consensus prediction
    preds                     = YASA_DF[consensus_cols].mode(axis=1).values
    YASA_DF['yasa_consensus'] = preds.flatten()

    # Drop the original columns
    YASA_DF = YASA_DF.drop(consensus_cols,axis=1)

    # Create the yasa lookup arrays
    yasa_files     = YASA_DF.file.values
    yasa_tstart    = YASA_DF.t_start.values
    yasa_tend      = YASA_DF.t_end.values
    unique_files   = np.unique(yasa_files)

    # Create the feature dataframe lookup arrays
    feature_files  = FEATURE_DF.file.values
    feature_tstart = FEATURE_DF.t_start.values

    # Populate the YASA feature column with unknowns that we can replace by index with the correct value
    YASA_FEATURE = np.array(['U' for ii in range(FEATURE_DF.shape[0])])
    YASA_LOOKUP  = YASA_DF['yasa_consensus'].values
    
    # Step through the unique files
    for ifile in tqdm(unique_files,total=unique_files.size):
                
        # Get the file indices
        yasa_file_inds    = (yasa_files==ifile)
        feature_file_inds = (feature_files==ifile)
        
        # The yasa lookup was made for more than just the PNES project. So we can cull for files in the feature df
        if feature_file_inds.sum() > 0:

            # Step through the time values
            unique_tstart = np.unique(yasa_tstart[yasa_file_inds])
            for istart in unique_tstart:

                # Get the time indices
                yasa_time_inds    = (yasa_tstart==istart)
                feature_time_inds = (feature_tstart>=istart)&(feature_tstart<(istart+30))

                # Get the current prediction, if available
                YASA_slice = YASA_LOOKUP[yasa_file_inds&yasa_time_inds]

                # Step through the possible outcomes for the yasa slice size
                combined_inds = feature_file_inds&feature_time_inds
                if combined_inds.sum() > 0:
                    if YASA_slice.size == 1:
                        YASA_FEATURE[combined_inds] = YASA_slice[0]
                    elif YASA_slice.size > 1:
                        raise Exception("Too many YASA values map to this feature. Check YASA generation.")
                    else:
                        pass

    # Add the prediction to the feature dataframe and save
    FEATURE_DF['yasa_prediction'] = YASA_FEATURE
    FEATURE_DF.to_csv(argv[3],index=False)
