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
            map_cols  = ['tag','target','annotation']
        else:
            col_info = yaml.safe_load(open(args.col_config,'r'))
            for key, inner_dict in col_info.items():
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

        # Make the cleaned up model view
        output_dict = {}
        iDF         = PD.concat(model_obj)
        for imap in map_cols:
            iDF[imap], output_dict[imap] = PD.factorize(iDF[imap])    
        if 'file' in iDF.columns:
            iDF['file'], file_mapping_dict = PD.factorize(iDF['file'])

        # Final downcasting attempt
        for icol in iDF:
            itype     = iDF[icol].dtype
            iDF[icol] = PD.to_numeric(iDF[icol],downcast='integer')
            if iDF[icol].dtype == itype:
                iDF[icol] = PD.to_numeric(iDF[icol],downcast='float')

        # Make the mapping dictionary
        output_dict = {'tag':tag_mapping_dict,'target':target_mapping_dict,'annotation':annotation_mapping_dict}
        if 'file' in iDF.columns:
            output_dict['file'] = file_mapping_dict

        print("Making the model file")
        pickle.dump(iDF,open(f"{args.indir}{args.outfile_model}","wb"))
        pickle.dump(output_dict,open(f"{args.indir}{args.outfile_map}","wb"))
    else:
        print("No files found.")