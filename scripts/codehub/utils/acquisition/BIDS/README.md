# BIDS Data Generation

## EEG Data to BIDS

### iEEG.org

#### Sample Call
```
python EEG_BIDS.py --ieeg --username BJPrager --password ****** --bidsroot ../../user_data/BIDS/ --session preimplant --inputs_file samples/targets.csv --annotations --multithread --ncpu 2
```

Where
- ieeg : Selects ieeg.org as the source data
- username : iEEG.org Username
- password : iEEG.ord Password
- bidsroot : Output BIDS top level directory to save to
- session : Session label for the dataset (i.e. preimplant, postsurg, etc.)
- inputs_file : File with iEEG.org filenames, unique patient identifiers, and any targets you wish to associate with each dataset
- annotations : Download by annotations
- multithread : Download and prepare BIDS data using multiple cpus
- ncpu : Number of cpus to use if multithread is selected

For more information and other options, please consult:
```
python EEG_BIDS.py --help
```
