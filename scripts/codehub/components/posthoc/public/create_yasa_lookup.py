import numpy as np
import pandas as PD
from sys import argv

if __name__ == '__main__':

    # Read in the dataframe
    raw_DF = PD.read_pickle(argv[1])

    # Only allow clips that had five minutes of data
    inds   = (raw_DF['t_end'].values-raw_DF['t_start'].values)>=300
    raw_DF = raw_DF.iloc[inds]

    # Make the lookup column list and get channel names
    lookup_cols = ['file', 't_start', 't_end', 't_window', 'method', 'tag']
    channels    = np.setdiff1d(raw_DF.columns,lookup_cols)

    # get the column headers for predictions
    cols = np.array(raw_DF.tag.values[0].split(','))

    # Get the YASA prediction. Which should be the same for all channels as we use a consensus across channels
    predictions = []
    for ival in raw_DF[channels[0]].values:
        formatted_pred = ival.replace('|',',')
        formatted_pred = np.array(formatted_pred.split(',')).reshape((-1,cols.size))
        predictions.append(formatted_pred)
    
    # Get the start time and filename for each row
    files   = raw_DF.file.values
    t_start = raw_DF.t_start.values
    
    # Make the final lookup tables
    outfile  = []
    outstart = []
    outend   = []
    outstage = []
    for idx,ifile in enumerate(files):
        istart = t_start[idx]
        ipred  = predictions[idx]
        for jdx,sleep_stage in enumerate(ipred):
            outfile.append(ifile)
            outstart.append(istart+(jdx*30))
            outend.append(istart+((jdx+1)*30))
            outstage.append(sleep_stage)
    outstage = np.array(outstage)

    # Make the lookup dataframe
    outDF = PD.DataFrame(outfile,columns=['file'])
    outDF['t_start']          = outstart
    outDF['t_end']            = outend
    for idx,icol in enumerate(cols):
        outDF[f"yasa_{icol}"] = outstage[:,idx]
    
    # Sort the results
    outDF = outDF.sort_values(by=['file','t_start'])

    # Save the results
    outDF.to_csv(argv[2],index=False)
