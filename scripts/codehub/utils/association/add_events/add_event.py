import os
import re
import argparse
import pandas as PD


if __name__ == '__main__':

    # Command line options needed to obtain data.
    parser = argparse.ArgumentParser(description="Merge EDF files together given a manifest document.")

    data_group = parser.add_argument_group('Data configuration options')
    data_group.add_argument("--edfpath", type=str, required=True, default=None, help="Output directory to store merged files.")
    data_group.add_argument("--sampfreq", type=int,  help="Sampling frequency.")
    data_group.add_argument("--time", type=float, help="Time to add annotation for.")
    data_group.add_argument("--annot", type=str, help="Annotation to add.")
    args = parser.parse_args()

    # Determine the annotation filepath
    pattern  = r"(.+)_\w+\.edf$"
    match    = re.match(pattern, args.edfpath)
    basename = match.group(1)
    
    # Make the events path
    event_path = basename+'_events.tsv'

    # Try to find an existing annotation file
    if os.path.exists(event_path):
        event_DF = PD.read_csv(event_path,delimiter='\t')
        makeflag = False
    else:
        event_DF = PD.DataFrame(columns=['onset','duration','trial_type','value','sample'])
        makeflag = True

    # Add annotation as needed
    if args.annot != None:
        iDF      = PD.DataFrame([[args.time, 0.0, args.annot, 0, args.time*args.sampfreq]],columns=event_DF.columns)
        event_DF = PD.concat((event_DF,iDF))
        makeflag = True

    # Write out the results
    if makeflag:
        event_DF.to_csv(event_path,sep='\t')