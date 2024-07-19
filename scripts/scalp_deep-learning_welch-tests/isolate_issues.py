import pickle
import numpy as np
import pandas as PD
from sys import argv

if __name__ == '__main__':

    duration = 30

    # Read in the data and apply marsh
    DF = PD.read_pickle(argv[1])
    DF = DF.loc[DF.t_window==duration]
    DF = DF.loc[DF.marsh_rejection]
    DF.drop(['uid','target','annotation','marsh_rejection'],axis=1,inplace=True)
    DF.dropna(inplace=True)

    # Get the channel labels
    id_cols  = ['file', 't_start', 't_end', 't_window', 'method', 'tag']
    channels = np.setdiff1d(DF.columns,id_cols)

    # Only keep the spectral_energy_welch data for now
    DF = DF.loc[DF.method=='spectral_energy_welch']

    # Make a reference dataframe to get some different views
    ref_df            = PD.DataFrame(columns=channels)
    ref_df.loc['min'] = np.argmin(DF[channels].values,axis=0)
    ref_df.loc['max'] = np.argmax(DF[channels].values,axis=0)
    medians           = np.median(DF[channels].values,axis=0)
    inds              = [np.argmin(np.fabs(DF[channels[itr]].values-medians[itr])) for itr in range(channels.size)]
    ref_df.loc['med'] = inds

    # Get the dataview
    inds   = np.unique(ref_df.values)
    mask   = [[channels[ref_df.iloc[0].values==ival],channels[ref_df.iloc[1].values==ival],channels[ref_df.iloc[2].values==ival]] for ival in inds]
    stype  = []
    schans = [] 
    for irow in mask:
        for idx,ival in enumerate(irow):
            if len(ival) > 0:
                if idx == 0:
                    stype.append('min')
                elif idx == 1:
                    stype.append('max')
                elif idx == 2:
                    stype.append('med')
                schans.append(','.join(ival))
    
    # Final clean up        
    DF               = DF.iloc[inds]
    DF['stat']       = stype
    DF['stat_chans'] = schans
    DF               = DF.sort_values(by=['stat','file','t_start','tag']).reset_index(drop=True)

    # Read in the metadata
    meta = pickle.load(open(argv[2],'rb'))

    # Get the metadata info
    keys    = []
    files   = []
    t_start = []
    durs    = [] 
    for ikey,idict in meta.items():
        keys.append(ikey)
        files.append(idict['file'])
        t_start.append(idict['t_start'])
        durs.append(idict['t_end']-idict['t_start'])
    keys    = np.array(keys)
    files   = np.array(files)
    t_start = np.array(t_start)
    durs    = np.array(durs) 

    # Add reference key numbers to the dataframe
    refkeys = []
    for idx,irow in DF.iterrows():
        finds = (files==irow.file)
        tinds = (t_start==irow.t_start)
        dinds = (durs==duration)
        refkeys.append(keys[finds&tinds&dinds][0])
    DF['keys'] = refkeys
    
    # Add the snapshot filename to the dataframe
    DF['snapshot_file'] = [f"{meta[ikey]['file'].split('/')[-1].split('.edf')[0]}_{meta[ikey]['t_start']:.1f}_{meta[ikey]['t_end']:.1f}_preprocess.pickle" for ikey in DF['keys'].values]   
