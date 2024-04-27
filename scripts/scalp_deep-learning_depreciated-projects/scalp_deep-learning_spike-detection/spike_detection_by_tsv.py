import glob
import numpy as np
import pandas as PD
import pathlib as pl
from sys import argv

if __name__ == '__main__':

    # Read in the appropriate files
    pathing    = pl.Path(argv[1])
    posixpaths = pathing.glob("**/*events.tsv")
    files      = [str(ifile.absolute()) for ifile in posixpaths]

    # Spike word bank
    spike_words = ['Spike','IED','Epileptiform discharge','Sharp','EEC','UEO','Onset','SZ','Seizure','sp/w',',x,',
                   'SW','XLSpike','LPD','PLED','LIRDA','TIRDA','xlspike','evolve','evolution','ictal','BIRD','szr']

    # Loop over the tsv files to get proposed spike/no-spike lists
    no_spike_files = []
    spike_files    = []
    for ifile in files:
        
        # Flag for which list to populate
        spike_flag = False

        # Get the current tsv into memory, then iterate over tags (should be 1, but iterating for corner cases)
        DF = PD.DataFrame(ifile, delimiter='\t')
        for ival in DF['tag_type'].values:
            
            # Loop over spike words
            for iword in spike_words:
                if iword.lower() in ival.lower():
                    spike_flag = True
        
        # Populate lists
        if spike_flag:
            spike_files.append(ifile)
        else:
            no_spike_files.append(ifile)

    # Write out the results
    fp = open(f"{argv[2]}spike_files.txt",'w')
    for ival in spike_files: fp.write(f"{ival}\n")
    fp.close()
    fp = open(f"{argv[2]}spike_free_files.txt",'w')
    for ival in no_spike_files: fp.write(f"{ival}\n")
    fp.close()