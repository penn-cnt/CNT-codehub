# Directly accessing the core libraries

There are two methods for using the core libraries kept within this framework. The first is via the pipeline manager, and is designed primarily for use with the main EPIPY package for processing large volumes of data. The second is via direct invocation of the underlying code. This allows for scripting, interactive calls, and building new pipeline logic with the underlying code.

To directly invoke a library, you simply call the `direct_inputs` method for each class. You can call the document string for more info on how each class' direct method works, as different tasks may require different inputs.

To find out what arguments are allowed for each class, you can either run the pipeline manager (found [here](../scripts/codehub/)) as follows `pipeline_manager.py --help` to get all the documentation, or you can directly view the allowed arguments and their help strings [here](../scripts/codehub/allowed_arguments.yaml).

## Example Direct Inputs Method

An example for the montage class is as follows:
```
    def direct_inputs(self,DF,montage):
        """
        Method for getting channel montages directly outside of the pipeline environment.

        Args:
            DF (datafram): Dataframe to get montage for.
            montage (str): Montage to perform

        Returns:
            dataframe: New dataframe with montage data and channel names
        """
        
        # Save the user provided dataframe
        self.dataframe_to_montage = DF

        # Apply montage logic
        montage_data = self.channel_montage_logic(montage)

        return PD.DataFrame(montage_data,columns=self.montage_channels)
```

For a complete example of how to read in, clean, and montage a dataframe please see [here](./example_direct_invocation.py).

## Sample Instantiation

An example of montaging data can be accomplished as follows (assuming you have the scripts/codehub/ in my PythonPath):

```
from modules.addons.channel_montage import channel_montage

# Dataframe generation
DF = **Your dataframe here**

# Montage choice
montage = "hup1020"

CM     = channel_montage()
new_df = CM.direct_inputs(DF,montage)
```

And you would have a new dataframe that is montaged and labeled accordingly.
