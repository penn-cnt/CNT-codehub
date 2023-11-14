CNT Scalp Concatenation Software
================
![version](https://img.shields.io/badge/version-0.2.1-blue)
![pip](https://img.shields.io/pypi/v/pip.svg)
![https://img.shields.io/pypi/pyversions/](https://img.shields.io/pypi/pyversions/4)

This code is designed to help with the processing of epilepsy datasets commonly used within the Center for Neuroengineering & Therapeutics (CNT) at the University of Pennsylvania. 

This code is meant to be researcher driven, allowing new code libraries to be added to modules that represent common research tasks (i.e. Channel Cleaning, Montaging, Preprocessing, etc.). The code can be accessed both as independent libraries that can be called on for a range of tasks, or as part of a large framework meant to ingest, clean, and prepare data for analysis or deep-learning tasks.

For more information on how to use our code, please see the examples folder for specific use-cases and common practices.

# Prerequisites
In order to use this repository, you must have access to Python 3+. 

# Installation

The python environment required to run this code can be found in the following location. [Concatenation YAML](/core_libraries/python/scalp/envs/CNT_ENVIRON_SCALP_CONCAT.yml)

This file can be installed using the following call to conda:

> conda create --name <env> --file CNT_ENVIRON_SCALP_CONCAT.yml

where <env> is the name of the environment you wish to save this work under.

More information about creating conda environments can be found [here](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html).

# Documentation

Due to the required flexibility of this code, multiple runtime options are available. We have aimed to reduce the need for extensive preparation of sidecar configuration files. Any sidecar files that are needed can be generated at runtime via a cli user-interace that queries the user for processing steps. If the sidecar files are already provided, this step is skipped. An example instantiation of this code is as follows:

> %run -i main.py --input GLOB --preprocess_file configs/preprocessing.yaml --feature_file configs/features.yaml --t_window 60 --n_input 2

**main.py** is the main body of this code. The additional flags are:
 * -i is a special runtime flag for interactive environments. This is required if running multiprocessing in most interactive environments
 * --input GLOB : Query the user for a wildcard path to files to read in via the GLOB library.
 * --preprocess_file : Path to the sidecar yaml configuration file that defines preprocessing steps. An example can be found [here](scripts/SCALP_CONCAT-EEG/configs/preprocessing.yaml)
 * --feature_file : Path to the sidecar yaml configuration file that defines feature extraction steps. An example can be found [here](scripts/SCALP_CONCAT-EEG/configs/features.yaml)
 * --t_window : Break the data in each file into windows of the provided size in seconds.
 * --n_input : Limit how many files to read in. This is useful for testing code or data and not wanting to read in all the data found along the provided path or in the pathing file.

**NOTE:** The provided features.yaml file shows a special use case of how to create looped data. This is useful if trying to perform one analysis step many times with slightly different windows. An example yaml code block is shown as follows:
```
spectral_energy_welch:
    step_nums:
        - 1
        - 2:
            - 99
            - 1
    low_freq:
        - -np.inf
        - 0:
            - 99
            - 1
    hi_freq:
        - np.inf
        - 1:
            - 100
            - 1
    win_size:
        - 2
        - 2:
            - 99
    win_stride:
        - 1
        - 1:
            - 99
```

If you wish to duplicate a step many times with slight variations, you can use the double indented blocks. If there are two entries under each entry, then it will iterate from the key out to the first nested value  by the amount of the second. (i.e. Step number 2, with a 99 and 1 below would give you range(2,99,1). If you only provide one value, then you simple tile the key by the number shown. (i.e. The win_size value of 2 with a 99 underneath means all 99 steps will have a win_size of 2.)

For more information, here is the full help documentation that can be found at runtime.

```
%run main.py --help
usage: main.py [-h] [--input {CSV,MANUAL,GLOB}] [--n_input N_INPUT] [--dtype {EDF}] [--t_start T_START] [--t_end T_END] [--t_window T_WINDOW] [--multithread] [--ncpu NCPU] [--channel_list {HUP1020,RAW}]
               [--montage {HUP1020,COMMON_AVERAGE}] [--viability {VIABLE_DATA,VIABLE_COLUMNS}] [--interp] [--n_interp N_INTERP] [--no_preprocess_flag] [--preprocess_file PREPROCESS_FILE] [--no_feature_flag]
               [--feature_file FEATURE_FILE] [--outdir OUTDIR]

Simplified data merging tool.

optional arguments:
  -h, --help            show this help message and exit

Data Merging Options:
  --input {CSV,MANUAL,GLOB}
                        Choose an option:
                        CSV            : Use a comma separated file of files to read in. (default)
                        MANUAL         : Manually enter filepaths.
                        GLOB           : Use Python glob to select all files that follow a user inputted pattern.
  --n_input N_INPUT     Limit number of files read in. Useful for testing.
  --dtype {EDF}         Choose an option:
                        EDF            : EDF file formats. (default)
  --t_start T_START     Time in seconds to start data collection.
  --t_end T_END         Time in seconds to end data collection. (-1 represents the end of the file.)
  --t_window T_WINDOW   List of window sizes, effectively setting multiple t_start and t_end for a single file.
  --multithread         Multithread flag.
  --ncpu NCPU           Number of CPUs to use if multithread.

Channel label Options:
  --channel_list {HUP1020,RAW}
                        Choose an option:
                        HUP1020        : Channels associated with a 10-20 montage performed at HUP.
                        RAW            : Use all possible channels. Warning, channels may not match across different datasets.

Montage Options:
  --montage {HUP1020,COMMON_AVERAGE}
                        Choose an option:
                        HUP1020        : Use a 10-20 montage.
                        COMMON_AVERAGE : Use a common average montage.

Data viability Options:
  --viability {VIABLE_DATA,VIABLE_COLUMNS}
                        Choose an option:
                        VIABLE_DATA    : Drop datasets that contain a NaN column. (default)
                        VIABLE_COLUMNS : Use the minimum cross section of columns across all datasets that contain no NaNs.
  --interp              Interpolate over NaN values of sequence length equal to n_interp.
  --n_interp N_INTERP   Number of contiguous NaN values that can be interpolated over should the interp option be used.

Preprocessing Options:
  --no_preprocess_flag  Do not run preprocessing on data.
  --preprocess_file PREPROCESS_FILE
                        Path to preprocessing YAML file. If not provided, code will walk user through generation of a pipeline.

Feature Extraction Options:
  --no_feature_flag     Do not run feature extraction on data.
  --feature_file FEATURE_FILE
                        Path to preprocessing YAML file. If not provided, code will walk user through generation of a pipeline.

Output Options:
  --outdir OUTDIR       Output directory.
```

# Major Features Remaining
- Associating target variables with the each subject

# License
Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.

# Contact Us
Any questions should be directed to the data science team. Contact information is provided below:

[Brian Prager](mailto:bjprager@seas.upenn.edu)

