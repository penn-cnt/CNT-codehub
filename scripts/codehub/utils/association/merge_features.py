import yaml
import glob
import pickle
import argparse
import pandas as PD
from sys import argv,exit


if __name__ == '__main__':

    # Argument parsing
    parser = argparse.ArgumentParser(description="Simplified data merging tool.")
    parser.add_argument("--indir", type=str, help='Input directory')
    parser.add_argument("--col_config", type=str, help="Optional path to a yaml with drop_col and obj_col definitions.")
    parser.add_argument("--outfile_model", default="merged_model.pickle", type=str, help='Output filename for model data')
    parser.add_argument("--outfile_meta", default="merged_meta.pickle", type=str, help='Output filename for metadata')
    parser.add_argument("--outfile_map", default="merged_map.pickle", type=str, help='Output filename for any mapped data column dictionaries')
    args = parser.parse_args()

    # get the files to merge
    files = glob.glob(f"{args.indir}*feature*.pickle")
    files = [ifile for ifile in files if ifile != args.outfile_model and ifile != args.outfile_meta]

    if len(files) > 0:
        
        # Object columns
        if args.col_config == None:
            drop_cols = ['file', 't_end', 'method']
            obj_cols  = ['t_start', 'dt', 'uid']
        else:
            col_info = yaml.safe_load(open(args.col_config,'r'))
            for key, inner_dict in data.items():
                globals()[key] = inner_dict

        # Loop over the files and save the outputs
        meta_obj  = []
        model_obj = []
        for ifile in files:

            # Read in data and clean up
            print(f"Working on {ifile}.")
            iDF = PD.read_pickle(ifile)
            iDF = iDF.drop(drop_cols,axis=1)

            # Get the model columns
            model_cols = [icol for icol in iDF.columns if icol not in obj_cols]

            # Store results in serialized object
            meta_obj.append(iDF[obj_cols])
            model_obj.append(iDF[model_cols])
        
        # Meta generation
        print("Making the meta file")
        iDF = PD.concat(meta_obj)
        pickle.dump(iDF,open(f"{args.indir}{args.outfile_meta}","wb"))

        # Model generation
        print("Making the model file")
        iDF                                = PD.concat(model_obj)
        iDF['tag'], tag_mapping_dict       = PD.factorize(iDF['tag'])
        iDF['target'], target_mapping_dict = PD.factorize(iDF['target'])
        output_dict                        = {'tag':tag_mapping_dict,'target':target_mapping_dict}
        pickle.dump(iDF,open(f"{args.indir}{args.outfile_model}","wb"))
        pickle.dump(output_dict,open(f"{args.indir}{args.outfile_map}","wb"))
    else:
        print("No files found.")