# Sample User Directory

This is a placeholder folder to help you orient yourself to the repository for the first time. You can either rename this to your eventual project folder, or just delete it.

We provide some information on how to make a project folder/rename this folder below.

## Naming conventions

This is an example user folder. Personal project work takes place in folders saved at the same level as the central `codehub' folder.

We require that the naming of the folder follow the following design pattern:

> {device name}\_{project name}\_{optional subdirectory}

We require this naming structure in order to join different projects within the CNT data ecosystem. Multiple users can work within a single repository, either within the same directory or within their own optional subdirectories.

### Example

If I am working on a scalp multi-layer perceptron (MLP) project to predict sleep stages, spikes, and pnes predictions. I might go about making folders like follows:

- `device_name`: I am working on scalp data, so I will go with `scalp`
- `project_name`: I am using MLP for a a few different tasks, so lets just summarize the project as `MLP`.

Now I could stop there and just make my folder: `scalp_MLP` and place all of my work within. Or, if I wanted to be careful about the environment for each sub-goal, or maybe I was collaborating and each person was doing their own sub-goal, I could make the following folders:
- scalp_MLP_sleep
- scalp_MLP_sleep-stags
- scalp_MLP_pnes-predictor

## Updating the codehub libraries

Any changes to scripts within the [modules](../codehub/modules) subdiretory can be submitted to the main lab repository as its own branch, at which point a pull request will be reviewed before changes are accepted or rejected.
