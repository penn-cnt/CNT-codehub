# EDF Viewer

Toolkit for visualizing all of the channel data for an EDF file. This uses the pipeline backend to make sure the data loading and data preparation is handled correctly. It also is optimized for quick data loads, visualizing targets, and marking sleep/spike states.

## Installation Guide

To use this toolkit, we highly recommend you create a python environment. This protects your base python environment from running into conflicts or versioning issues. We describe how to install the CNT environment below.

1. Clone this repository to your local workstation.
2. Install Anaconda
    - Please visit the [Anaconda Downloads](https://www.anaconda.com/download) page to download the appropriate Anaconda installer for your operating system. 
3. You will need to use your new conda installation to install the python environment. The build file can be found [here](https://github.com/penn-cnt/CNT-codehub/blob/main/core_libraries/python/cnt_codehub/envs/cnt_codehub.yml) 
    - Mac/Linux Users:
        - Within a terminal application, you can run the following command
        - > conda env create -f \<path-to-build-yml\>
    - Windows Users:
        - You will need to open the Anaconda Powershell program. Once you do, run the following command
        - > conda env create -f \<path-to-build-yml\>
4. To run the environment, within your terminal or Anaconda Powershell, run:
    - > conda activate cnt_codehub 
        - The environment is set to default to the name `cnt_codehub`. (You can change this by modifying the `name` entry in your local copy of the yaml file. If you change this, you would run the above command on the new name
5. Finally, all you need to do is add your new code repository to your anaconda path. Run the following command in the terminal or powershell
    -  conda develop \<path-to-repository-root\>/scripts/codehub
  
## Sample Commands
Please consult the `--help` flag for more detailed information on different inputs to the viewer. We provide a few common uses below.

(**Note:** We also provide the available keyboard shortcuts for a number of useful viewing situations at the top of the plot.)

### Random start time via seed (default behavior) with for all wildcard data matches
```
python utils/visualization/edf_viewer/edf_viewer.py --wildcard "../../../scalp_deep-learning/user_data/BIDS/BIDS/sub-0008/ses-preimplant01/eeg/sub-0008_ses-preimplant01_task-task_run-*_eeg.edf" --username bjprager --flagging
```

### Random start time via seed (default behavior) with for all wildcard data matches with a common average montage (default=bipolar)
```
python utils/visualization/edf_viewer/edf_viewer.py --wildcard "../../../scalp_deep-learning/user_data/BIDS/BIDS/sub-0008/ses-preimplant01/eeg/sub-0008_ses-preimplant01_task-task_run-*_eeg.edf" --username bjprager --flagging --montage common_average
```

### Random start time via seed (default behavior) with for all wildcard data matches without any montage
```
python utils/visualization/edf_viewer/edf_viewer.py --wildcard "../../../scalp_deep-learning/user_data/BIDS/BIDS/sub-0008/ses-preimplant01/eeg/sub-0008_ses-preimplant01_task-task_run-*_eeg.edf" --username bjprager --flagging --montage None
```

### Random start time via seed (default behavior) with flagging for all wildcard data matches
```
python utils/visualization/edf_viewer/edf_viewer.py --wildcard "../../../scalp_deep-learning/user_data/BIDS/BIDS/sub-0008/ses-preimplant01/eeg/sub-0008_ses-preimplant01_task-task_run-*_eeg.edf" --username bjprager --flagging
```
flagging enables an interactive mode where the user can denote if certain events occur within the observed time window and save the results to a csv file.

By default the code outputs to `./edf_viewer_flags.csv` but can be changed using the `--outfile` option at runtime.

### Set start time of t=0 with flagging for wildcard datamatches
```
python utils/visualization/edf_viewer/edf_viewer.py --wildcard "../../../scalp_deep-learning/user_data/BIDS/BIDS/sub-0008/ses-preimplant01/eeg/sub-0008_ses-preimplant01_task-task_run-*_eeg.edf" --username bjprager --flagging --t0 0
```

### Set start time of t=0 and duration 15 with flagging for wildcard datamatches
```
python utils/visualization/edf_viewer/edf_viewer.py --wildcard "../../../scalp_deep-learning/user_data/BIDS/BIDS/sub-0008/ses-preimplant01/eeg/sub-0008_ses-preimplant01_task-task_run-*_eeg.edf" --username bjprager --flagging --t0 0 --dur 15
```

### Load a single edf file into the viewer
```
python utils/visualization/edf_viewer/edf_viewer.py --cli ../../user_data/EDF/sub-00149_ses-preimplant002_task-task_run-06_eeg.edf 
```

