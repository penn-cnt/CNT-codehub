import os
import pickle
import argparse
import pandas as PD
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Project imports
from models.lr import *
from cleaning.data_clean import *

def lr_handler(output_path,vectors,holdout,ptitle,lrtype,ncpu):

    # Run tests as needed
    if not os.path.exists(output_path):
        # Run a logistic regression model
        ntests     = 10
        lr_results = {'acc_train':{},'acc_test':{},'auc':{},'auc_holdout':{},'clf':{},'sratios':{}}
        stypes     = ['minmax','standard']
        for itype in stypes:
            
            # Let user know the current scaler type
            print(f"Testing the {itype} scaler.")

            # Add in the scaler types
            for ikey in list(lr_results.keys()):
                lr_results[ikey][itype] = []

            for idx in tqdm(range(ntests), total=ntests):

                # Run the logistic regression model
                LRH = LR_handler(vectors,holdout,ncpu)
                LRH.data_scale(stype=itype)
                LRH.data_split()
                acc_train,acc_test,auc,hold_auc,clf = LRH.LR_process(lrtype)

                # Save the results
                lr_results['acc_train'][itype].append(acc_train)
                lr_results['acc_test'][itype].append(acc_test)
                lr_results['auc'][itype].append(auc)
                lr_results['auc_holdout'][itype].append(hold_auc)
                lr_results['clf'][itype].append(clf)

                # Make sure we randomize the training data
                vectors.sample(frac=1)

        pickle.dump(lr_results,open(output_path,"wb"))
    else:
        lr_results = pickle.load(open(output_path,'rb'))
        ntests     = len(lr_results['auc'][list(lr_results['auc'].keys())[0]])

    # Make the outcome dataframe
    LRDF         = PD.DataFrame(columns=['solver','auc','flag'])
    LRDF2        = PD.DataFrame(columns=['solver','auc','flag'])
    svals        = np.repeat(list(lr_results['auc'].keys()),ntests)
    aucs         = np.array(list(lr_results['auc'].values())).flatten()
    aucs_holdout = np.array(list(lr_results['auc_holdout'].values())).flatten()
    accs         = np.array(list(lr_results['acc_test'].values())).flatten()
    LRDF['solver']   = svals
    LRDF['auc']      = aucs
    LRDF['dataset']  = 'testdata'
    LRDF['acc']      = accs
    LRDF2['solver']  = svals
    LRDF2['auc']     = aucs_holdout
    LRDF2['dataset'] = 'holdouts'
    LRDF_COMB = PD.concat((LRDF,LRDF2))

    # Plot the results
    fig = PLT.figure(dpi=100.,figsize=(5.,5.))
    ax1 = fig.add_subplot(111)
    sns.boxplot(data=LRDF_COMB, x="solver", y="auc", fill=False, hue="dataset",ax=ax1)
    ax1.set_title(f"{ptitle}")
    PLT.grid(True)
    PLT.show()

    return lr_results['clf']['standard']

def temporal_split(DF,test_fraction):

    # Get the indices
    index_dict = DF.groupby(['file','t_start',]).indices

    # Make the temporarl keyed dictionary
    temporal_dict = {}

    # Run over the keys to get the temporal splits
    index_keys = list(index_dict.keys())
    for ikey in index_keys:
        file  = ikey[0]
        itime = ikey[1]
        if file not in temporal_dict.keys():
            temporal_dict[file] = {}
        temporal_dict[file][itime] = index_dict[ikey]

    # Split each file up by the split fraction
    train_inds    = []
    test_inds     = []
    temporal_keys = temporal_dict.keys()
    for ikey in temporal_keys:
        time_keys = list(temporal_dict[ikey].keys())
        split_ind = int((1-test_fraction)*len(time_keys))
        for jkey in time_keys[:split_ind]:
            train_inds.append(temporal_dict[ikey][jkey][0])
        for jkey in time_keys[split_ind:]:
            test_inds.append(temporal_dict[ikey][jkey][0])

    return DF.iloc[train_inds].reset_index(drop=True),DF.iloc[test_inds].reset_index(drop=True)

def subject_split(DF,test_fraction,mapping_file):

    # Add in the mapping
    mapping = pickle.load(open(mapping_file,"rb"))
    DF['uid'] = DF.file.apply(lambda x: mapping[x])
    
    # Get a unique pairing for target and uid to ensure we also keep data balance
    uDF = DF[['uid','target']].drop_duplicates()
    X   = uDF['uid'].values
    Y   = uDF['target'].values
    uid_train,uid_test,targets_train,targets_test = train_test_split(X,Y,stratify=Y)

    # get the indices for different uids
    uid_inds = DF.groupby(['uid']).indices

    # Get the uids according to split
    train_inds = []
    test_inds  = []
    for iuid in uid_train:
        for iind in uid_inds[iuid]:
            train_inds.append(iind)
    for iuid in uid_test:
        for iind in uid_inds[iuid]:
            test_inds.append(iind)

    # Drop the uid and return the dataframe
    DF = DF.drop(columns=['uid'],axis=1)
    return DF.iloc[train_inds].reset_index(drop=True),DF.iloc[test_inds].reset_index(drop=True)

def prepare_TUEG(DF_TUEG,MAP_TUEG):

    # Clean the data
    CC = CLEAN_CLASS(DF_TUEG)
    DF_TUEG = CC.clean_data()

    # Prepare the data
    DP = DATA_PREP(DF_TUEG)
    DP.get_channels()
    DP.update_TUEG_targets(MAP_TUEG)
    DF_TUEG,CHAN_TUEG = DP.return_data()

    # Create the vectors
    MI                  = make_inputs(DF_TUEG)
    rawvectors,channels = MI.vector_creation()

    # Some cleanup
    colorder   = ['file','t_start']+list(channels)+['target']
    rawvectors = rawvectors[colorder]

    return rawvectors.drop_duplicates(subset=['file','t_start']),channels

def prepare_HUP(DF_HUP,MAP_HUP):

    # Drop unneeded columns
    DF_HUP = DF_HUP.drop(columns=['annotation','uid'],axis=1)

    # Clean the data
    CC = CLEAN_CLASS(DF_HUP)
    DF_HUP = CC.clean_data()

    # Prepare the data
    DP = DATA_PREP(DF_HUP)
    DP.get_channels()
    DF_HUP,CHAN_HUP = DP.return_data()

    # Make the binary classification
    DF_HUP = DF_HUP.loc[DF_HUP.target.isin([0,3])]
    DF_HUP['target'].replace(3,1,inplace=True)

    # Create the vectors
    MI                  = make_inputs(DF_HUP)
    rawvectors,channels = MI.vector_creation()

    # Some cleanup
    colorder   = ['file','t_start']+list(channels)+['target']
    rawvectors = rawvectors[colorder]

    return rawvectors.drop_duplicates(subset=['file','t_start']),channels

def add_slowing_prob(DF,clf_list):

    incols       = DF.columns
    X_cols       = incols[incols!='target']
    X            = DF[X_cols].values
    scaler       = StandardScaler()
    X_scaled     = scaler.fit_transform(X)
    slow_probs   = [iclf.predict_proba(X_scaled) for iclf in clf_list]
    posterior    = np.prod(slow_probs,axis=0)
    DF['slow_0'] = posterior[:,0] 
    DF['slow_1'] = posterior[:,1]
    return DF

def apply_pca(enc,DF,CHANNELS):

        rawvectors_trans = pca_enc.transform(DF[CHANNELS].values)
        channels         = [f"{ii:03d}" for ii in range(rawvectors_trans.shape[1])]
        new_df           = PD.DataFrame(rawvectors_trans,columns=channels)
        new_df['target'] = DF['target'].values
        return new_df,channels

if __name__ == '__main__':

    # Command line options needed to obtain data.
    #default_tueg = '/Users/bjprager/Documents/GitHub/scalp_deep-learning/user_data/derivative/slowing/041724/DATA/'
    #default_hup  = '/Users/bjprager/Documents/GitHub/scalp_deep-learning/user_data/derivative/epilepsy/outputs/DATA/'
    #default_out  = '/Users/bjprager/Documents/GitHub/scalp_deep-learning/user_data/derivative/epilepsy/outputs/MODEL/SUBJECT/'
    default_tueg = '/mnt/leif/littlab/users/bjprager/GitHub/scalp_deep-learning/user_data/derivative/epilepsy/DATA/TUEG/'
    default_hup  = '/mnt/leif/littlab/users/bjprager/GitHub/scalp_deep-learning/user_data/derivative/epilepsy/DATA/HUP/'
    default_out  = '/mnt/leif/littlab/users/bjprager/GitHub/scalp_deep-learning/user_data/derivative/epilepsy/MODEL/'
    parser = argparse.ArgumentParser(description="iEEG to bids conversion tool.")
    parser.add_argument("--model_TUEG", type=str, default=f"{default_tueg}merged_model.pickle", help="TUEG Merged Model file path.")
    parser.add_argument("--map_TUEG", type=str, default=f"{default_tueg}merged_map.pickle", help="TUEG Merged map file path.")
    parser.add_argument("--model_HUP", type=str, default=f"{default_hup}merged_model.pickle", help="TUEG Merged Model file path.")
    parser.add_argument("--map_HUP", type=str, default=f"{default_hup}merged_map.pickle", help="TUEG Merged map file path.")
    parser.add_argument("--output_TUEG", type=str, default=f"{default_out}dataset_TUEG.pickle", help="TUEG Merged map file path.")
    parser.add_argument("--output_HUP", type=str, default=f"{default_out}dataset_HUP.pickle", help="TUEG Merged map file path.")
    parser.add_argument("--slowing_output", type=str, default=f"{default_out}slowing.model", help="Slowing LR output.")
    parser.add_argument("--tueg_subject_map", type=str, default="/Users/bjprager/Documents/GitHub/scalp_deep-learning/user_data/derivative/slowing/041724/DATA/map_file_sub_tueg.pickle")
    parser.add_argument("--HUP_subject_map", type=str, default="/Users/bjprager/Documents/GitHub/scalp_deep-learning/user_data/derivative/epilepsy/outputs/DATA/map_file_sub.pickle")
    parser.add_argument("--epilepsy_output_noslow", type=str, default=f"{default_out}epilepsy_noslow.model", help="Epilepsy LR output.")
    parser.add_argument("--epilepsy_output_slow", type=str, default=f"{default_out}epilepsy_slow.model", help="Epilepsy LR output.")
    parser.add_argument("--ncpu", type=int, default=8, help="Number of cpus to use for cross validation.")
    parser.add_argument("--pca", action='store_true', default=False, help="Apply PCA first.")
    parser.add_argument("--npca", type=int, default=50, help="Number of PCA components")
    
    selection_group = parser.add_mutually_exclusive_group()
    selection_group.add_argument("--temporal", action='store_true', default=False, help="Temporal split.")
    selection_group.add_argument("--subject", action='store_true', default=False, help="Subject split.")

    fitter_group = parser.add_mutually_exclusive_group()
    fitter_group.add_argument("--lr_simple", action='store_true', default=False, help="Basic LR.")
    fitter_group.add_argument("--lr_cv", action='store_true', default=False, help="LR with cross validation.")
    args = parser.parse_args()

    if not os.path.exists(args.output_TUEG):
        # Read in the TUEG data
        DF_TUEG  = PD.read_pickle(args.model_TUEG)
        MAP_TUEG = pickle.load(open(args.map_TUEG,"rb"))

        # Prepare temple
        RAWVECTORS_TUEG,CHANNELS_TUEG = prepare_TUEG(DF_TUEG,MAP_TUEG)
        if args.temporal:
            TUEG_TRAIN, TUEG_TEST = temporal_split(RAWVECTORS_TUEG,0.2)
        else:
            TUEG_TRAIN, TUEG_TEST = subject_split(RAWVECTORS_TUEG,0.2,args.tueg_subject_map)

        # Save the output
        pickle.dump((TUEG_TRAIN,TUEG_TEST),open(args.output_TUEG,"wb"))
    else:
        TUEG_TRAIN, TUEG_TEST = pickle.load(open(args.output_TUEG,"rb"))

    if not os.path.exists(args.output_HUP):
        # Read in the HUP data
        DF_HUP  = PD.read_pickle(args.model_HUP)
        MAP_HUP = pickle.load(open(args.map_HUP,"rb"))

        # Prepare HUP
        RAWVECTORS_HUP,CHANNELS_HUP = prepare_HUP(DF_HUP,MAP_HUP)
        if args.temporal:
            HUP_TRAIN, HUP_TEST = temporal_split(RAWVECTORS_HUP,0.2)
        else:
            HUP_TRAIN, HUP_TEST = subject_split(RAWVECTORS_HUP,0.2,args.HUP_subject_map)

        # Save the output
        pickle.dump((HUP_TRAIN,HUP_TEST),open(args.output_HUP,"wb"))
    else:
        HUP_TRAIN, HUP_TEST = pickle.load(open(args.output_HUP,"rb"))

    # Make the LR flag string
    if args.lr_cv:
        lrtype = 'cv'
    else:
        lrtype = 'simple'

    # Apply PCA if needed
    if args.pca:
        
        # Make and fit the transformer
        pca_enc = SparsePCA(n_components=args.npca,n_jobs=args.ncpu)
        pca_enc.fit(TUEG_TRAIN[CHANNELS_TUEG].values)

        # Apply PCA to our data
        TUEG_TRAIN,CHANNELS_TUEG = apply_pca(pca_enc,TUEG_TRAIN,CHANNELS_TUEG)
        TUEG_TEST,CHANNELS_TUEG  = apply_pca(pca_enc,TUEG_TEST,CHANNELS_TUEG)
        HUP_TRAIN,CHANNELS_HUP   = apply_pca(pca_enc,HUP_TRAIN,CHANNELS_HUP)
        HUP_TEST,CHANNELS_HUP    = apply_pca(pca_enc,HUP_TEST,CHANNELS_HUP)

    # Make the logistic regression fit for TUEG slowing
    clf_slow       = lr_handler(args.slowing_output,TUEG_TRAIN,TUEG_TEST,'Slowing Prediction for time segments',lrtype,args.ncpu)
    clf_epi_noslow = lr_handler(args.epilepsy_output_noslow,HUP_TRAIN,HUP_TEST,'Epilepsy Prediction w/o slowing',lrtype,args.ncpu)

    # Get the predictions for slowing for HUP
    HUP_TRAIN_scaled = add_slowing_prob(HUP_TRAIN,clf_slow)
    HUP_TEST_scaled  = add_slowing_prob(HUP_TEST,clf_slow)
    
    # Get the predictions with slowing added
    clf_epi_slow = lr_handler(args.epilepsy_output_slow,HUP_TRAIN_scaled,HUP_TEST_scaled,'Epilepsy Prediction w/ slowing',lrtype,args.ncpu)
