# Search Data Cache

This code is meant to acquire the latest data cache report and help the user search for data in the cache. This tool allows users to find out if data already exists in the CNT data cache before downloading data needlessly.

## Installation

You will need the following packages to run the search software. You can run the following commands to install the packages.

```
pip install argparse
pip install paramiko
pip install pandas
pip install prettytable
```

## Example usage

> python data_search.py --username bjprager

The code will by default search in the borel data cache for existing datasets. You will need to provide a username to acquire the new data cache record over ssh protocol, and will be prompted for a password at run time.

For more information, you can run
> python data_search.py --help

## Expected Behavior

This code will create a table of all the data available in our data cache. You can then query the table to find data that matches your criteria. You will be requested for:

    - subcol : The column name to search within. Defaults to orig_filename. (i.e. iEEG.org filenames)
    - substr : The substring to search for

You will get back a table with all the entries that match your query. You can then use the subject number and session numbers to find the correct dataset within the data cache.

### Sample output

```
python data_search.py --username bjprager
Please enter password for remote system:
Password: 
+-----------------+----------+----------+----------+------+----------------+----------------+--------+
|  orig_filename  |  source  | creator  | gendate  | uid  | subject_number | session_number | times  |
+-----------------+----------+----------+----------+------+----------------+----------------+--------+
| EMU0562_Day01_1 | ieeg.org | bjprager | 01-04-24 | 0.0  |       1        |       1        | annots |
| EMU1169_Day01_1 | ieeg.org | bjprager | 01-04-24 | 0.0  |       1        |       1        | annots |
| EMU0562_Day02_1 | ieeg.org | bjprager | 01-04-24 | 0.0  |       1        |       2        | annots |
| EMU1169_Day02_1 | ieeg.org | bjprager | 01-04-24 | 0.0  |       1        |       2        | annots |
| EMU0562_Day03_1 | ieeg.org | bjprager | 01-04-24 | 0.0  |       1        |       3        | annots |
| EMU1169_Day03_1 | ieeg.org | bjprager | 01-04-24 | 0.0  |       1        |       3        | annots |
| EMU0562_Day04_1 | ieeg.org | bjprager | 01-04-24 | 0.0  |       1        |       4        | annots |
| EMU1169_Day04_1 | ieeg.org | bjprager | 01-04-24 | 0.0  |       1        |       4        | annots |
| EMU0562_Day05_1 | ieeg.org | bjprager | 01-04-24 | 0.0  |       1        |       5        | annots |
| EMU1169_Day05_1 | ieeg.org | bjprager | 01-04-24 | 0.0  |       1        |       5        | annots |
| EMU0562_Day06_1 | ieeg.org | bjprager | 01-04-24 | 0.0  |       1        |       6        | annots |
| EMU0562_Day07_1 | ieeg.org | bjprager | 01-04-24 | 0.0  |       1        |       7        | annots |
| EMU1048_Day01_1 | ieeg.org | bjprager | 01-04-24 | 1.0  |       2        |       1        | annots |
| EMU1048_Day02_1 | ieeg.org | bjprager | 01-04-24 | 1.0  |       2        |       2        | annots |
| EMU1048_Day03_1 | ieeg.org | bjprager | 01-04-24 | 1.0  |       2        |       3        | annots |
| EMU1048_Day03_2 | ieeg.org | bjprager | 01-04-24 | 1.0  |       2        |       3        | annots |
| EMU1048_Day04_1 | ieeg.org | bjprager | 01-04-24 | 1.0  |       2        |       4        | annots |
| EMU1252_Day01_1 | ieeg.org | bjprager | 01-04-24 | 2.0  |       3        |       1        | annots |
| EMU1494_Day01_1 | ieeg.org | bjprager | 02-04-24 | 3.0  |       7        |       1        | annots |
| EMU1494_Day01_1 | ieeg.org | bjprager | 02-04-24 | 3.0  |       7        |       1        | annots |
| EMU1872_Day01_1 | ieeg.org | bjprager | 02-04-24 | 3.0  |       7        |       1        | annots |
| EMU1872_Day01_1 | ieeg.org | bjprager | 02-04-24 | 3.0  |       7        |       1        | annots |
| EMU2014_Day01_1 | ieeg.org | bjprager | 02-04-24 | 3.0  |       7        |       1        | annots |
| EMU2014_Day01_1 | ieeg.org | bjprager | 02-04-24 | 3.0  |       7        |       1        | annots |
| EMU1494_Day02_1 | ieeg.org | bjprager | 02-04-24 | 3.0  |       7        |       2        | annots |
| EMU1494_Day02_1 | ieeg.org | bjprager | 02-04-24 | 3.0  |       7        |       2        | annots |
| EMU2014_Day02_1 | ieeg.org | bjprager | 02-04-24 | 3.0  |       7        |       2        | annots |
| EMU2014_Day02_1 | ieeg.org | bjprager | 02-04-24 | 3.0  |       7        |       2        | annots |
+-----------------+----------+----------+----------+------+----------------+----------------+--------+
Search for entry (y/n) or reset table (R/r)? y
Please enter column to search within (default='orig_filename'): 
Please enter substring to search for: EMU2014
+-----------------+----------+----------+----------+-----+----------------+----------------+--------+
|  orig_filename  |  source  | creator  | gendate  | uid | subject_number | session_number | times  |
+-----------------+----------+----------+----------+-----+----------------+----------------+--------+
| EMU2014_Day01_1 | ieeg.org | bjprager | 02-04-24 | 3.0 |       7        |       1        | annots |
| EMU2014_Day01_1 | ieeg.org | bjprager | 02-04-24 | 3.0 |       7        |       1        | annots |
| EMU2014_Day02_1 | ieeg.org | bjprager | 02-04-24 | 3.0 |       7        |       2        | annots |
| EMU2014_Day02_1 | ieeg.org | bjprager | 02-04-24 | 3.0 |       7        |       2        | annots |
+-----------------+----------+----------+----------+-----+----------------+----------------+--------+
Search for entry (y/n) or reset table (R/r)? 
```

## Notes

Nomenclature notes:
    - annots : Portions of the iEEG.org dataset within clip times (and containing all clip annotations.)
