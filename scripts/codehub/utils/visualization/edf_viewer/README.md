# EDF Viewer

Toolkit for visualizing all of the channel data for an EDF file. This uses the pipeline backend to make sure the data loading and data preparation is handled correctly. It also is optimized for quick data loads, visualizing targets, and marking sleep/spike states.

## Installation Guide

To use this toolkit, we highly recommend you create a python environment. This protects your base python environment from running into conflicts or versioning issues. We describe how to install the CNT environment below.

### Conda
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

### Venv and pip

1. To create a virtual environment, you need to create a location for the environment to install to. For this example, we will specify `/demonstration/environment/cnt_codehub` as our environment location. Using the python version of your choice, in this example we will select 3.10, run the following command:

    > python3.10 -m venv /demonstration/environment/cnt_codehub
2. To enter the envrionment, simply run:

    > source /demonstration/environment/cnt_codehub/bin/activate
3. Once in the environment, a requirements.txt file with all the needed packages to run this suite of code can be found at the following location:

    > [CNT Codehub YAML](core_libraries/python/cnt_codehub/envs/requirements.txt)

    This file can be installed using the following call to pip from the envs subdirectory:

    > pip install -r requirements.txt

    which will install everything to your current virual environment. 
4. Add the codehub to your virtual environment path. For a virtual environment, an easy way to add `<path-to-git-head>/scripts/codehub/` to your path would be to add a text file with a .pth extention (any filename is fine) to the site-packages subfolder in your virtual environment folder. Within the text file you can just copy and paste the absolute path as the only contents.

    Typically, the path your your site-packages can be found at: `<path-to-environment-folder>/lib/python<version-number>/site-packages`.

## Sample Commands
Please consult the `--help` flag for more detailed information on different inputs to the viewer. We provide a few common uses below.

(**Note:** We also provide the available keyboard shortcuts for a number of useful viewing situations at the top of the plot.)

### Random start time via seed (default behavior) with for all wildcard data matches
```
python utils/visualization/edf_viewer/edf_viewer.py --wildcard "../../../scalp_deep-learning/user_data/BIDS/BIDS/sub-0008/ses-preimplant01/eeg/sub-0008_ses-preimplant01_task-task_run-*_eeg.edf"
```

### Random start time via seed (default behavior) with for all wildcard data matches with a common average montage (default=bipolar)
```
python utils/visualization/edf_viewer/edf_viewer.py --wildcard "../../../scalp_deep-learning/user_data/BIDS/BIDS/sub-0008/ses-preimplant01/eeg/sub-0008_ses-preimplant01_task-task_run-*_eeg.edf" --montage common_average
```

### Random start time via seed (default behavior) with for all wildcard data matches without any montage
```
python utils/visualization/edf_viewer/edf_viewer.py --wildcard "../../../scalp_deep-learning/user_data/BIDS/BIDS/sub-0008/ses-preimplant01/eeg/sub-0008_ses-preimplant01_task-task_run-*_eeg.edf" --montage None
```

### Set start time of t=0 for wildcard datamatches
```
python utils/visualization/edf_viewer/edf_viewer.py --wildcard "../../../scalp_deep-learning/user_data/BIDS/BIDS/sub-0008/ses-preimplant01/eeg/sub-0008_ses-preimplant01_task-task_run-*_eeg.edf" --t0 0
```

### Set start time of t=0 and duration 15 for wildcard datamatches
```
python utils/visualization/edf_viewer/edf_viewer.py --wildcard "../../../scalp_deep-learning/user_data/BIDS/BIDS/sub-0008/ses-preimplant01/eeg/sub-0008_ses-preimplant01_task-task_run-*_eeg.edf" --username bjprager --flagging --t0 0 --dur 15
```

### Load a single edf file into the viewer
```
python utils/visualization/edf_viewer/edf_viewer.py --cli ../../user_data/EDF/sub-00149_ses-preimplant002_task-task_run-06_eeg.edf 
```

