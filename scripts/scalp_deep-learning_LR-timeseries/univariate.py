import os
import pickle
import argparse
import numpy as np
import pandas as PD
import pylab as PLT
from sys import exit
import seaborn as sns
from scipy.stats import ttest_ind,ttest_rel
import matplotlib.gridspec as gridspec
from sklearn.feature_selection import SelectKBest, f_classif

def get_merged_data(args):

    # Create the merged dataframe
    fname = "merged_data.pickle"
    if not os.path.exists(fname):

        # Read in the raw data
        DF_sleep = PD.read_pickle(args.model_sleep)
        DF_wake  = PD.read_pickle(args.model_wake)
        
        # Read in the maps
        map_sleep = pickle.load(open(args.map_sleep,"rb"))
        map_wake  = pickle.load(open(args.map_wake,"rb"))

        # Read in the meta
        meta_sleep = pickle.load(open(args.meta_sleep,"rb"))
        meta_wake  = pickle.load(open(args.meta_wake,"rb"))

        # Add in the times to get the same time segments across bands later
        DF_sleep['t_start'] = meta_sleep['t_start']
        DF_wake['t_start']  = meta_wake['t_start']

        # Get the channel labels
        blacklist       = ['file','t_start','tag','uid','target','annotation']
        channels        = np.setdiff1d(DF_sleep.columns,blacklist)
        channel_map     = {}
        channel_map_rev = {} 
        for idx,ichannel in enumerate(channels):
            channel_map[ichannel]  = idx+1
            channel_map_rev[idx+1] = ichannel

        # Find the intersecting uids
        uid_intersect = np.intersect1d(DF_sleep.uid.unique(),DF_wake.uid.unique())

        # Get the current data slices for intersecting user ids
        DF_sleep_slice = DF_sleep.loc[DF_sleep.uid.isin(uid_intersect)]
        DF_wake_slice  = DF_wake.loc[DF_wake.uid.isin(uid_intersect)]
        
        # Add a sleep state tag
        DF_sleep_slice['sleep_state'] = 'sleep'
        DF_wake_slice['sleep_state']  = 'wake'

        # Get the merged dataframe
        DF_merged = PD.concat((DF_sleep_slice,DF_wake_slice))
        DF_merged = DF_merged.replace(to_replace='nan',value=np.nan)
        DF_merged.dropna(inplace=True)

        # Attempt downcasting
        for icol in channels:
            try:
                DF_merged[icol] = PD.to_numeric(DF_merged[icol],downcast='float')
            except ValueError:
                pass

        # Save the results
        pickle.dump((DF_merged,channels),open(fname,"wb"))
    else:
        DF_merged,channels = pickle.load(open(fname,"rb"))
    
    return DF_merged,np.array(channels)

def get_bandpowers(DF_merged,channels):

    # get alpha delta data
    fname = "bandpower_dataframe.csv"
    if not os.path.exists(fname):

        # Get the indices for each unique segment
        medians = DF_merged.groupby(['uid','tag','sleep_state'])[channels].median()
        medians = medians.drop(['FZ-CZ'],axis=1)

        # Calculate the alpha
        alpha_welch_sleep = []
        alpha_welch_wake  = []
        delta_welch_sleep = []
        delta_welch_wake  = []
        alpha_fooof_sleep = []
        alpha_fooof_wake  = []
        delta_fooof_sleep = []
        delta_fooof_wake  = []
        for iuid in uid_intersect:
            delta_welch = medians.loc[iuid,0].median(axis=1)
            alpha_welch = medians.loc[iuid,2].median(axis=1)
            delta_fooof = medians.loc[iuid,10].median(axis=1)
            alpha_fooof = medians.loc[iuid,12].median(axis=1)
            alpha_welch_sleep.append(alpha_welch.sleep)
            alpha_welch_wake.append(alpha_welch.wake)
            delta_welch_sleep.append(delta_welch.sleep)
            delta_welch_wake.append(delta_welch.wake)
            alpha_fooof_sleep.append(alpha_fooof.sleep)
            alpha_fooof_wake.append(alpha_fooof.wake)
            delta_fooof_sleep.append(delta_fooof.sleep)
            delta_fooof_wake.append(delta_fooof.wake)
        DF_bandpower = PD.DataFrame(alpha_welch_sleep,columns=['sleep_welch_8-13Hz'])
        DF_bandpower['wake_welch_8-13Hz']   = alpha_welch_wake
        DF_bandpower['sleep_welch_0.5-4Hz'] = delta_welch_sleep
        DF_bandpower['wake_welch_0.5-4Hz']  = delta_welch_wake
        DF_bandpower['sleep_fooof_8-13Hz']  = alpha_fooof_sleep
        DF_bandpower['wake_fooof_8-13Hz']   = alpha_fooof_wake
        DF_bandpower['sleep_fooof_2-4Hz']   = delta_fooof_sleep
        DF_bandpower['wake_fooof_2-4Hz']    = delta_fooof_wake

        # Save the results
        DF_bandpower.to_csv(fname,index=False)

    else:
        DF_bandpower = PD.read_csv(fname)

    return DF_bandpower

def get_alpha_delta(DF_merged,channels,outdir,powertype='fooof'):

    # Check which power type we are working with
    if powertype == 'fooof':
        alpha_ind = 12
        delta_ind = 10
        fname     = f"{outdir}alpha_delta_fooof.csv"
    elif powertype == 'welch':
        alpha_ind = 2
        delta_ind = 0
        fname     = f"{outdir}alpha_delta_welch.csv"

    # Check for existing data, otherwise, generate
    if not os.path.exists(fname):
        # Iterate over the user ids
        DF_alpha_delta = PD.DataFrame()
        uids           = DF_merged.uid.unique()

        for iuid in uids:

            # Get the relevant data slices
            DF_alpha_sleep = DF_merged.loc[(DF_merged.uid==iuid)&(DF_merged.tag==alpha_ind)&(DF_merged.sleep_state=='sleep')]
            DF_alpha_wake  = DF_merged.loc[(DF_merged.uid==iuid)&(DF_merged.tag==alpha_ind)&(DF_merged.sleep_state=='wake')]
            DF_delta_sleep = DF_merged.loc[(DF_merged.uid==iuid)&(DF_merged.tag==delta_ind)&(DF_merged.sleep_state=='sleep')]
            DF_delta_wake  = DF_merged.loc[(DF_merged.uid==iuid)&(DF_merged.tag==delta_ind)&(DF_merged.sleep_state=='wake')]

            # Get the alpha delta values
            alpha_delta_sleep = DF_alpha_sleep[channels].median()/DF_delta_sleep[channels].median()
            alpha_delta_wake  = DF_alpha_wake[channels].median()/DF_delta_wake[channels].median()

            # Transpose and save as a dataframe
            alpha_delta_sleep = PD.DataFrame(alpha_delta_sleep).transpose()
            alpha_delta_wake = PD.DataFrame(alpha_delta_wake).transpose()

            # Add the stratifying variables
            alpha_delta_sleep['uid']         = iuid
            alpha_delta_wake['uid']          = iuid
            alpha_delta_sleep['sleep_state'] = 'sleep'
            alpha_delta_wake['sleep_state']  = 'wake'

            # Append to output
            DF_alpha_delta = PD.concat((DF_alpha_delta,alpha_delta_sleep))
            DF_alpha_delta = PD.concat((DF_alpha_delta,alpha_delta_wake))

        # Save the results
        DF_alpha_delta.to_csv(fname,index=False)    
    else:
        DF_alpha_delta = PD.read_csv(fname)

    return DF_alpha_delta

def plot_alpha_delta(DF_merged,channels):

    # Loop over welch and fooof for plotting
    powers = ['fooof','welch']
    for ipower in powers:
        DF_alpha_delta      = get_alpha_delta(DF_merged,channels,powertype=ipower)
        channels            = np.setdiff1d(channels,['FZ-CZ'])
        try:
            DF_alpha_delta.drop(['FZ-CZ'],axis=1,inplace=True)
        except:
            pass
        
        # Get the t-test values
        ttest_results = {}
        inds          = DF_alpha_delta.groupby(['uid']).indices
        for iuid in inds.keys():

            # Prepare output for this patient id
            ttest_results[iuid] = {}
            
            # Get the relevant dataslices
            iDF      = DF_alpha_delta.iloc[inds[iuid]]
            DF_sleep = iDF.loc[iDF.sleep_state=='sleep']
            DF_wake  = iDF.loc[iDF.sleep_state=='wake']

            # Loop over the channels to get the p-value and sign of difference
            for ichan in channels:
                vals_sleep                 = DF_sleep[ichan].values
                vals_wake                  = DF_wake[ichan].values
                ttest_results[iuid][ichan] = ttest_ind(vals_wake,vals_sleep)

        # Make a more useful melted dataframe for plotting in seaborn
        DF_melt = PD.melt(DF_alpha_delta, id_vars=['uid','sleep_state'], value_vars=channels, var_name='channels', value_name='alpha_delta')
        indices = DF_melt.groupby(['uid','channels']).indices
        keys    = list(indices.keys())
        newarr  = np.zeros(DF_melt.shape[0])
        for ikey in keys:
            inds         = indices[ikey]
            t_output     = ttest_results[ikey[0]][ikey[1]]
            newarr[inds] = t_output[1]            #np.sign(t_output[0])*t_output[1]
        DF_melt['p-val'] = newarr

        # Loop over uids and plot
        sns.set_style("darkgrid", {"grid.color": ".6", "grid.linestyle": ":"})
        DF_melt.set_index('uid',inplace=True)
        for iuid in DF_melt.index.unique():

            # Prepare data slices for plots
            iDF = DF_melt.loc[iuid]
            jDF = iDF[['channels','p-val']].drop_duplicates()
            jDF.replace([np.inf, -np.inf], np.nan, inplace=True)
            jDF.dropna(inplace=True)

            # Make the plots
            try:
                fig = PLT.figure(dpi=100.,figsize=(8.,6.))
                gs  = gridspec.GridSpec(1, 2, width_ratios=[3, 1])
                ax1 = fig.add_subplot(gs[0])
                ax2 = fig.add_subplot(gs[1])
                sns.boxplot(data=iDF,y='channels',x='alpha_delta',hue='sleep_state',ax=ax1)
                sns.scatterplot(data=jDF,y='channels',x='p-val',ax=ax2)
                ax2.vlines(0.05,ax2.get_ylim()[0],ax2.get_ylim()[1],color='r',ls='--')
                
                # Clean up the axes
                ax2.yaxis.set_label_position("right")
                ax2.set_yticks([])
                ax2.set_ylabel('')
                ax2.set_xscale('log')
                ax2.set_xlim([-10,0])
                ax1.set_title(f"User ID {iuid:03d} with {ipower}",fontsize=13)
                ax2.set_title(f"2-samp Ttest")
                PLT.grid(True)
                fig.tight_layout()

                # Save the results and clean up
                PLT.savefig(f"{default_out}plots/alpha_delta/{iuid:04d}_{ipower}.png")
            except:
                pass
            PLT.close("all")

def get_scores(X,Y,nK):
    selector = SelectKBest(f_classif, k=nK)
    selector.fit(X, Y)
    scores = -np.log10(selector.pvalues_)
    inds   = np.isfinite(scores)
    scores /= scores[inds].max()
    return scores,selector.pvalues_

def get_features(DF_merged,channels):

    # Make sure we have paired matches
    DF_merged.reset_index(inplace=True,drop=True)
    DF_sleep        = DF_merged.loc[DF_merged.sleep_state=='sleep'].drop(['sleep_state'], axis=1)
    DF_wake         = DF_merged.loc[DF_merged.sleep_state=='wake'].drop(['sleep_state'], axis=1)
    """
    DF_sleep        = DF_sleep.set_index(['file', 'tag', 'uid','t_start'])
    DF_wake         = DF_wake.set_index(['file', 'tag', 'uid','t_start'])
    if len(DF_wake.index) > len(DF_sleep.index):
        index_intersect = DF_wake.index.intersection(DF_sleep.index)
    else:
        index_intersect = DF_sleep.index.intersection(DF_wake.index)
    DF_sleep        = DF_sleep.loc[index_intersect].reset_index()
    DF_wake         = DF_wake.loc[index_intersect].reset_index()
    """

    # Make some pivot tables to make a feature selector
    #DF_sleep['med'] = DF_sleep[channels].median(axis=1)
    #DF_wake['med']  = DF_wake[channels].median(axis=1)
    DF_sleep['med'] = np.quantile(DF_sleep[channels].values,q=0.75,method='median_unbiased',axis=1)
    DF_wake['med']  = np.quantile(DF_wake[channels].values,q=0.75,method='median_unbiased',axis=1)
    DF_sleep_med    = DF_sleep.drop(channels,axis=1)
    DF_wake_med     = DF_wake.drop(channels,axis=1)
    DF_wake_med     = DF_wake_med.drop_duplicates(subset=['file','uid','tag','target','t_start'])
    DF_sleep_med    = DF_sleep_med.drop_duplicates(subset=['file','uid','tag','target','t_start'])
    pivot_wake      = DF_wake_med.pivot(index=['file', 'uid', 'target', 't_start'], columns='tag', values='med')
    pivot_sleep     = DF_sleep_med.pivot(index=['file', 'uid', 'target', 't_start'], columns='tag', values='med')
    pivot_wake.columns  = [f'med_{col}' for col in pivot_wake.columns]
    pivot_sleep.columns = [f'med_{col}' for col in pivot_sleep.columns]
    pivot_wake  = pivot_wake.reset_index()
    pivot_sleep = pivot_sleep.reset_index()

    # Clean up the target vectors
    pivot_sleep = pivot_sleep.loc[pivot_sleep.target.isin([0,1])]
    pivot_wake  = pivot_wake.loc[pivot_wake.target.isin([0,3])]
    pivot_wake.loc[pivot_wake.target==0,['target']] = 1
    pivot_wake.loc[pivot_wake.target==3,['target']] = 0
    pivot_wake.dropna(inplace=True)
    pivot_sleep.dropna(inplace=True)

    # Get the input vectors and the targets
    X_cols  = [f"med_{idx}" for idx in range(14)]
    X_wake  = pivot_wake[X_cols].values
    X_sleep = pivot_sleep[X_cols].values
    Y_wake  =  pivot_wake['target'].values
    Y_sleep =  pivot_sleep['target'].values

    # get the scores and report back
    nK                            = 6
    scores_wake,raw_scores_wake   = get_scores(X_wake,Y_wake,nK)
    scores_sleep,raw_scores_sleep = get_scores(X_sleep,Y_sleep,nK)
    uwake,uwake_cnt               = np.unique(Y_wake,return_counts=True)
    usleep,usleep_cnt             = np.unique(Y_sleep,return_counts=True)
    print(f"For wake:")
    print(f"    PNES: {uwake_cnt[0]/uwake_cnt.sum():.2%}")
    print(f"    EPIL: {uwake_cnt[1]/uwake_cnt.sum():.2%}") 
    print(f"    Scores:",raw_scores_wake)
    print(f"For sleep:")
    print(f"    PNES: {usleep_cnt[0]/usleep_cnt.sum():.2%}")
    print(f"    EPIL: {usleep_cnt[1]/usleep_cnt.sum():.2%}") 
    print(f"    Scores:",raw_scores_sleep)

def paired_t_test(DF_merged,channels,outdir,powertype='fooof'):

    # Get the alpha/delta dataframe
    DF_ad = get_alpha_delta(DF_merged,channels,outdir,powertype=powertype)

    # Make the pairs for the paired t test
    AD_pair       = DF_ad.groupby(['uid','sleep_state']).median().median(axis=1).reset_index()
    AD_pair       = AD_pair.rename(columns={0: "AD"})
    AD_pair_pivot = AD_pair.pivot(index=['uid'],columns='sleep_state',values="AD")

    t_val,p_val = ttest_rel(AD_pair_pivot['sleep'].values,AD_pair_pivot['wake'].values)

    # Make the plot plot
    fig = PLT.figure(dpi=100,figsize=(8.,6.))
    ax1 = fig.add_subplot(111)
    sns.lineplot(data=AD_pair,x='sleep_state',y='AD',hue='uid')
    ax1.get_legend().remove()
    ax1.set_title(f"{powertype} Alpha/Delta Pairs. \nT statistic (sleep-wake): {t_val}. p-value:{p_val} ")
    PLT.savefig(f"{outdir}PLOTS/{powertype}_allchan_ad_pairs.png")
    PLT.close("all")

    # Loop over channels
    for ichannel in channels:
        # Make the pairs for the paired t test
        AD_pair       = DF_ad.groupby(['uid','sleep_state'])[ichannel].median().reset_index()
        AD_pair       = AD_pair.rename(columns={ichannel: "AD"})
        AD_pair_pivot = AD_pair.pivot(index=['uid'],columns='sleep_state',values="AD")

        t_val,p_val = ttest_rel(AD_pair_pivot['sleep'].values,AD_pair_pivot['wake'].values)

        # Make the plot plot
        fig = PLT.figure(dpi=100,figsize=(8.,6.))
        ax1 = fig.add_subplot(111)
        sns.lineplot(data=AD_pair,x='sleep_state',y='AD',hue='uid')
        ax1.get_legend().remove()
        ax1.set_title(f"{powertype} Alpha/Delta Pairs ({ichannel}). \nT statistic (sleep-wake): {t_val}. p-value:{p_val} ")
        PLT.savefig(f"{outdir}PLOTS/{powertype}_{ichannel}_ad_pairs.png")
        PLT.close("all")


if __name__ == '__main__':

    default_sleep = '/Users/bjprager/Documents/GitHub/scalp_deep-learning/user_data/derivative/epilepsy/outputs/DATA/052124/sleep/'
    default_wake  = '/Users/bjprager/Documents/GitHub/scalp_deep-learning/user_data/derivative/epilepsy/outputs/DATA/052124/wake/'
    default_out   = '/Users/bjprager/Documents/GitHub/scalp_deep-learning/user_data/derivative/epilepsy/outputs/MODEL/052124/univariate/'
    parser = argparse.ArgumentParser(description="iEEG to bids conversion tool.")
    parser.add_argument("--model_sleep", type=str, default=f"{default_sleep}merged_model.pickle", help="Sleep Merged Model file path.")
    parser.add_argument("--model_wake", type=str, default=f"{default_wake}merged_model.pickle", help="Wake Merged Model file path.")
    parser.add_argument("--map_sleep", type=str, default=f"{default_sleep}merged_map.pickle", help="Sleep Merged Map file path.")
    parser.add_argument("--map_wake", type=str, default=f"{default_wake}merged_map.pickle", help="Wake Merged Map file path.")
    parser.add_argument("--meta_sleep", type=str, default=f"{default_sleep}merged_meta.pickle", help="Sleep Merged Meta file path.")
    parser.add_argument("--meta_wake", type=str, default=f"{default_wake}merged_meta.pickle", help="Wake Merged Meta file path.")
    args = parser.parse_args()

    # Get the alpha delta bandpowers
    DF_merged, channels = get_merged_data(args)

    # Get the paired t-test results
    paired_t_test(DF_merged,channels,default_out)
    paired_t_test(DF_merged,channels,default_out,"welch")