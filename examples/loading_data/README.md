# Loading data into memory

Loading data directly into memory can be accomplished via directly calling the data loader class.

For a complete list of allowed data formats, please reference the allowed_datatypes keyword [here](../scripts/codehub/allowed_arguments.yaml).

## Sample data load

A simple example on how to load data follows:

```
from components.curation.public.data_loader import *

class data_handler:

    def __init__(self,infile):
        self.infile = infile

    def get_data(self,datatype='edf'):

        # Create pointers to the relevant classes
        DL = data_loader()

        # Get the raw data
        DF,self.fs = DL.direct_inputs(self.infile,datatype)

        return DF,self.fs

if __name__ == '__main__':

    # Path to example data
    script_path  = os.path.abspath(__file__)
    example_dir  = '/'.join(script_path.split('/')[:-2])
    example_path = f"{example_dir}/example_data/sample_000.edf"

    # Get the cleaned dataset
    DH    = data_handler(example_path)
    DF,fs = DH.get_data()
    print(DF)
```

For a complete example on how to clean and montage the data as well, please see [here](../using_core_libraries_directly/README.md).
