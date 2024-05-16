import os
import nltk
import pickle
import argparse
import numpy as np
import pandas as PD
from sys import exit
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer

nltk.download('stopwords')

def return_tokens(istr):

    stop_words      = set(stopwords.words('english'))
    tokenizer       = RegexpTokenizer(r'\w+')
    tokens          = tokenizer.tokenize(istr.lower())
    filtered_tokens = [token for token in tokens if token not in stop_words and len(token) > 1]
    return filtered_tokens

if __name__ == '__main__':

    # Argument parsing
    parser = argparse.ArgumentParser(description="iEEG to bids conversion tool.")
    parser.add_argument("--rootdir", type=str, required=True, help="Root directory to search within for target data.")
    parser.add_argument("--outfile", type=str, required=True, help="Path to save results to")
    args = parser.parse_args()

    # Get all the target dictionaries
    target_files = []
    for dirpath, dirs, files in os.walk(args.rootdir):  
        for filename in files:
            fname = os.path.join(dirpath,filename) 
            if fname.endswith('targets.pickle'): 
                target_files.append(fname)
    
    # Loop through each file and read in the target information. Then tokenize and store to final search dictionary
    lookup_dict = {}
    for ifile in target_files:
        target_dict = pickle.load(open(ifile,"rb"))
        annot_str   = target_dict['annotation']
        target_str  = target_dict['target']
        all_tokens  = return_tokens(annot_str)
        all_tokens  = all_tokens + return_tokens(target_str)
        
        # Loop over the tokens to add to the lookup
        for itoken in all_tokens:
            if itoken not in lookup_dict.keys():
                lookup_dict[itoken]          = {}
                lookup_dict[itoken]['count'] = 0
                lookup_dict[itoken]['files'] = []
            lookup_dict[itoken]['count'] += 1
            lookup_dict[itoken]['files'].append(ifile.replace('_targets.pickle','.edf'))

    # Make a pretty lookup table for the user
    index         = np.array(list(lookup_dict.keys())).reshape((-1,1))
    counts        = np.array([lookup_dict[ikey]['count'] for ikey in lookup_dict.keys()]).reshape((-1,1))
    DF            = PD.DataFrame(counts,columns=['count'])
    DF['keyword'] = index
    DF            = DF.sort_values(by=['count'],ascending=False)
    DF.set_index('keyword',drop=True,inplace=True)
    
    # Print the results
    print(DF)

    # Ask the user for which keyword to save the filepaths for
    tokens = input("Enter the keyword (or comma separated keywords) you want the file list for? (Q/q quit). ")
    if tokens.lower() == 'q':exit()
    tokenlist = tokens.split(',')
    
    # Save the files to the output file
    outfiles = []
    for itoken in tokenlist:
        if itoken in DF.index:
            ifiles = lookup_dict[itoken]['files'] 
            outfiles.extend(ifiles)
        else:
            print(f"Could not find your token `{itoken}` in the token list from observed files.")

    # Make a dataframe, drop duplicates (which arise from list of tokens), and save
    DF = PD.DataFrame(outfiles,columns=['filepath'])
    DF.drop_duplicates(inplace=True)
    DF.to_csv(args.outfile,index=False)
