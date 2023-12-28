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
  
## Sample Command
Random start time via seed (default behavior)
```
python utils/visualization/edf_viewer/edf_viewer.py --wildcard "../../../scalp_deep-learning/user_data/BIDS/BIDS/sub-0008/ses-preimplant01/eeg/sub-0008_ses-preimplant01_task-task_run-*_eeg.edf" --username bjprager --flagging
```

Set start time (t=0 in this example)
```
python utils/visualization/edf_viewer/edf_viewer.py --wildcard "../../../scalp_deep-learning/user_data/BIDS/BIDS/sub-0008/ses-preimplant01/eeg/sub-0008_ses-preimplant01_task-task_run-*_eeg.edf" --username bjprager --flagging --t0 0
```

Loading data through an ssh tunnel
```
python utils/visualization/edf_viewer/edf_viewer.py --file "files.tmp" --username bjprager --flagging --ssh_host borel.seas.upenn.edu --username bjprager
```

where files.tmp is a single column file of filepaths to edf data to view. In this case, paths on the remote system.
