# Data Audit Tool

Due to the increasing data volume, regular data audits are required to remove duplicate data and identify non-essential data to send to deep storage. This tool is meant to help staff interact with audit data in an easier format.

## Installation

1. Download and install anaconda.
2. Install the needed python environment using the provided environment file.
    > conda env create --file ./envs/cnt_audit.yml 
3. Activate the conda environment
    > conda activate cnt_audit
4. Run the code
    > python multiaudit.py --search_root `<directory-to_audit>` --outdir `<directory-to-store-audit-files>` --username `<username>` --system `<systemname>` --merge --ncpu XYZ

where 
- `<directory-to_audit>` is the directory you wish to perform the audit on.
- `<directory-to-store-audit-files>` is the output directory to store the temporary and final results.
- `<username>` your username on the system
- `<systemname>` is the name of the system you are on (i.e. leif/bsc/etc.) 
- XYZ is the number of cores to use for the audit.

*NOTE* Please make sure to use a value for NCPU that makes sense for your system. (We recommend using `top` and `lscpu` to gather information about usage of and the total number of cpus.)

## Potential Issues

This code is meant to be a wrapper to bash/powershell scripts to get some basic file information. We have tested it against a few different systems, and a variety of file paths, but its possible you will encounter an error. A few notes if this happens:

- If a single core fails, the other cores will continue working. So the audit will continue to get as much information as it can.
- If a core crashes, the code will hang upon completion. This means you will likely need to manually kill it. We apologize for the inconvenience, but this way we prevent run conditions on writing the final output.
- You can restart the code and it will pick up where it left off. So if your code crashes/ends for any reason, your work wont be lost.
- If you encounter a partically problematic folder, let Brian P. know. It will be helpful to figure out what other use cases we need to prepare for, and I can probably help create a solution to finish your audit.

## Basic Workflow

At a high level, this code is just a wrapper to various shell commands that find data, calculates the checksum, and saves basic information about the file.

To do this effectively, it will recursively search down to each folder and get information on all the **files** in a folder.

This method lets the code *finish* a task in a reasonable timeframe. By doing a single folder at a time, it can save the temporary results to your output directory and update the audit history saying that folder is done.

Once all the audits are done, it will merge the results and remove all the temporary data.

If the code dies halfway through, or you need to restart, it should be able to find where it left off. It will inform you that it is skipping folders it has already completed.

You can use the same output location for different input audit folders. Results will append to previous audit data.

## Sample Output

A sample output audit file can be found [here](interface/modules/samples/sample.audit).

## Configuration Files

The configuration file set by default has been tested to work on most CNT systems. If you find that you are getting errors, you can try using the `--cmd_path` command and point to one of the other files in the config folder to see if it resolves. If you still have trouble, please reach out to Brian P. for more help.

## More Commands

For more information about any given input argument, you can always run
> python multiaudit.py --help
