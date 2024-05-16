# Find Target data

This code is designed to create lists of files that contain annotations or target keywords of interest.

At a high level, the code looks for the **_targets.pickle** file associated with each edf file in our bids repository. It then creates a list of tokens from the annotations and target fields. It reports how many times it saw that word across the entire search directory, and then you can enter in a list of tokens you want a file list for.

## Expected Output

```
python utils/association/find_targets/find_targets.py --rootdir ~/Documents/GitHub/scalp_deep-learning/user_data/BIDS --outfile ../../user_data/scratch/wake_files.txt
[nltk_data] Downloading package stopwords to
[nltk_data]     /Users/bjprager/nltk_data...
[nltk_data]   Package stopwords is already up-to-date!
           count
keyword         
uncertain     81
clip          36
note          36
pnes          25
spike         12
wake          10
ec             9
tag            7
sleep          7
firda          7
rem            7
slow           6
ed             6
temporal       4
emu            4
asleep         4
frontal        3
drowsy         3
n2             3
event          2
left           2
pdr            2
hz             2
head           1
twitch         1
n3             1
hr             1
video          1
change         1
Enter the keyword (or comma separated keywords) you want the file list for? (Q/q quit). wake
```

The output file `wake_files.txt` looks like:

```
filepath
/Users/bjprager/Documents/GitHub/scalp_deep-learning/user_data/BIDS/sub-0268/ses-preimplant002/eeg/sub-0268_ses-preimplant002_task-task_run-07_eeg.edf
/Users/bjprager/Documents/GitHub/scalp_deep-learning/user_data/BIDS/sub-0268/ses-preimplant002/eeg/sub-0268_ses-preimplant002_task-task_run-04_eeg.edf
/Users/bjprager/Documents/GitHub/scalp_deep-learning/user_data/BIDS/sub-0268/ses-preimplant003/eeg/sub-0268_ses-preimplant003_task-task_run-04_eeg.edf
/Users/bjprager/Documents/GitHub/scalp_deep-learning/user_data/BIDS/sub-0268/ses-preimplant003/eeg/sub-0268_ses-preimplant003_task-task_run-02_eeg.edf
/Users/bjprager/Documents/GitHub/scalp_deep-learning/user_data/BIDS/sub-0268/ses-preimplant003/eeg/sub-0268_ses-preimplant003_task-task_run-03_eeg.edf
/Users/bjprager/Documents/GitHub/scalp_deep-learning/user_data/BIDS/sub-0268/ses-preimplant003/eeg/sub-0268_ses-preimplant003_task-task_run-05_eeg.edf
/Users/bjprager/Documents/GitHub/scalp_deep-learning/user_data/BIDS/sub-0268/ses-preimplant001/eeg/sub-0268_ses-preimplant001_task-task_run-02_eeg.edf
/Users/bjprager/Documents/GitHub/scalp_deep-learning/user_data/BIDS/sub-00001/ses-preimplant002/eeg/sub-00001_ses-preimplant002_task-task_run-09_eeg.edf
/Users/bjprager/Documents/GitHub/scalp_deep-learning/user_data/BIDS/sub-00001/ses-preimplant005/eeg/sub-00001_ses-preimplant005_task-task_run-02_eeg.edf
/Users/bjprager/Documents/GitHub/scalp_deep-learning/user_data/BIDS/sub-00001/ses-preimplant004/eeg/sub-00001_ses-preimplant004_task-task_run-06_eeg.edf
```

If I wanted an output with both `wake` and `pdr` in the annotations/targets field, I can input the following

```
python utils/association/find_targets/find_targets.py --rootdir ~/Documents/GitHub/scalp_deep-learning/user_data/BIDS --outfile ../../user_data/scratch/wake_files.txt
[nltk_data] Downloading package stopwords to
[nltk_data]     /Users/bjprager/nltk_data...
[nltk_data]   Package stopwords is already up-to-date!
           count
keyword         
uncertain     81
clip          36
note          36
pnes          25
spike         12
wake          10
ec             9
tag            7
sleep          7
firda          7
rem            7
slow           6
ed             6
temporal       4
emu            4
asleep         4
frontal        3
drowsy         3
n2             3
event          2
left           2
pdr            2
hz             2
head           1
twitch         1
n3             1
hr             1
video          1
change         1
Enter the keyword (or comma separated keywords) you want the file list for? (Q/q quit). wake,pdr
```