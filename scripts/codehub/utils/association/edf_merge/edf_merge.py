import argparse
import numpy as np
import pandas as PD
from tqdm import tqdm
from mne.io import read_raw_edf
from mne import concatenate_raws
from mne.export import export_raw
from sys import exit

if __name__ == '__main__':

    # Command line options needed to obtain data.
    parser = argparse.ArgumentParser(description="Merge EDF files together given a manifest document.")

    data_group = parser.add_argument_group('Data configuration options')
    data_group.add_argument("--outdir", type=str, required=True, default=None, help="Output directory to store merged files.")
    data_group.add_argument("--manifest", type=str, required=True, help="Filepath to the manifest document.")
    data_group.add_argument("--blocksize", type=int, default=-1, help="Number of files to combine together. -1 means merge all.")
    args = parser.parse_args()

    # Filepath cleanup
    if args.outdir[-1] != '/':args.outdir+='/'

    # Load the manifest
    manifest_DF = PD.read_csv(args.manifest)

    # Get the filepaths as an array
    filepaths = manifest_DF.filepath.values
    
    # Get the fileblocks
    if args.blocksize != -1:
        
        # Get the total number of blocks with remainder
        number_blocks = filepaths.size/args.blocksize

        # Get the number of blocks with the exact right sizing
        nitr       = int(np.floor(number_blocks))
        fileblocks = []
        for itr in range(nitr):
            fileblocks.append(filepaths[itr*args.blocksize:(itr+1)*args.blocksize])

        # Add the remainder if needed
        if number_blocks > nitr:
            fileblocks.append(filepaths[args.blocksize*nitr:])
    else:
        fileblocks = [filepaths]

    # Begin the merging
    for idx,iblock in enumerate(fileblocks):
        outraw = read_raw_edf(iblock[0])
        for ifile in tqdm(iblock[1:], total=len(iblock)-1, desc=f"Merging Block {idx:02d}"):
            newraw = read_raw_edf(ifile)
            outraw = concatenate_raws([outraw,newraw],on_mismatch='ignore')
        try:
            export_raw(f"{args.outdir}merged_{idx:03d}.edf",outraw,fmt='edf')
        except ValueError:
            export_raw(f"{args.outdir}merged_{idx:03d}.edf",outraw,fmt='edf',physical_range=(0,1))