import yaml
import json
import argparse
import numpy as np
from os import path,system
from itertools import permutations

class yaml_loader:

    def __init__(self,config,grid,outdir,cmd):
        self.yaml_config = config
        self.grid        = grid
        self.outdir      = outdir
        self.cmd_file    = cmd
        if self.outdir[-1] != '/':
            self.outdir += '/'

    def step_combos(self):

        # Get the unique permutations
        step_perms = np.array(list(set(permutations(self.step_keys))))
        
        # Get the current step to configure grid to
        for ref_step in list(self.grid.keys()):
            ref_rules  = self.grid[ref_step]

            # Get the current rule applying to this step
            for current_rule in list(ref_rules.keys()):
                current_params = ref_rules[current_rule]

                # Generate a mask and step through permutations
                mask = []
                for iperm in step_perms:

                    # Step though the possible rules and apply logic
                    if current_rule == 'gt':

                        # Find the current permutations index for the current step
                        grid_step_index = np.argwhere(np.array(iperm)==ref_step)[0][0]

                        # Get the index for each of the rule params
                        grid_params_index = [np.argwhere(np.array(iperm)==val)[0][0] for val in current_params]

                        # Make the mask
                        flag = (grid_step_index>grid_params_index).all()
                        mask.append(flag)
                    elif current_rule == 'set':
                        # Find the current permutations index for the current step
                        grid_step_index = current_params[0]-1

                        # Get the index for each of the rule params
                        grid_params_index = np.argwhere(np.array(iperm)==ref_step)[0][0]

                        # Make the mask
                        flag = (grid_step_index==grid_params_index)
                        mask.append(flag)
                    elif current_rule == 'next':
                        # Find the current permutations index for the current step
                        grid_step_index = np.argwhere(np.array(iperm)==ref_step)[0][0]

                        # Get the index for each of the rule params
                        grid_params_index = [np.argwhere(np.array(iperm)==val)[0][0] for val in current_params][0]

                        # Make the mask
                        flag = ((grid_step_index+1)==grid_params_index)
                        mask.append(flag)

                # For testing, print stats 
                before = step_perms.shape[0]
                step_perms = step_perms[mask]
                after = step_perms.shape[0]
                print(f"Current rule took the grid from {before} permutations down to {after}.")
        self.step_perms = step_perms
        raw_accept_flag = input("Proceed with current grid size (y/Y)? ")
        if raw_accept_flag.lower() == 'y':
            self.accept_flag = True
        else:
            self.accept_flag = False

    def make_grid_configs(self):

        # Loop and make a copy of the step dict so we can swap
        if self.accept_flag:
            for idx,iperm in enumerate(self.step_perms):

                # Make the step dict
                new_step  = (np.arange(iperm.size)+1)
                map_dict  = dict(zip(iperm.ravel(),new_step.ravel()))
                grid_step = {int(map_dict.get(old_key, old_key)): value for old_key, value in self.yaml_step.items()}

                # Save the grid data
                base_dir     = f"{self.outdir}{idx:03}"
                config_dir   = f"{base_dir}/config"
                output_dir   = f"{base_dir}/output"
                current_file = f"{config_dir}/preprocessing.json"
                if not path.exists(config_dir):
                    system(f"mkdir -p {config_dir}")
                                
                with open(current_file, 'w') as json_file:
                    json.dump(grid_step, json_file, indent=4)

                # Update the command line call as needed
                if self.cmd_file != None:

                    if not path.exists(output_dir):
                        system(f"mkdir -p {output_dir}")

                    # Update the preprocessing path
                    preprocess_str = "--preprocess_file"
                    cmd_arr        = cmd.split(preprocess_str)
                    cmd_arr2       = cmd_arr[1].split()[1:]
                    cmd_tail       = ' '.join(cmd_arr2) 
                    new_cmd        = f"{cmd_arr[0]} {preprocess_str} {current_file} {cmd_tail}"

                    # Update the output path
                    output_str = "--outdir"
                    cmd_arr    = cmd.split(output_str)
                    cmd_arr2   = cmd_arr[1].split()[1:]
                    cmd_tail   = ' '.join(cmd_arr2) 
                    new_cmd    = f"{cmd_arr[0]} {output_str} {output_dir}/ {cmd_tail}"
                    cmd_file       = f"{base_dir}/cmd.txt"

                    # Write the new command out
                    fp = open(cmd_file,"w")
                    fp.write(new_cmd)
                    fp.close()

    def convert_to_step(self):
        """
        Modify the human readable yaml to a more machine friendly step sorted dictionary.
        """
        
        # Convert human readable config to easier format for coded
        self.yaml_step = {}
        for ikey in list(self.yaml_config.keys()):
            steps = self.yaml_config[ikey]['step_nums']
            for idx,istep in enumerate(steps):

                # Get the argument list for the current command
                args = self.yaml_config[ikey].copy()
                args.pop('step_nums')

                # Clean up the current argument list to only show current step
                for jkey in list(args.keys()):
                    args[jkey] = args[jkey][idx]

                # Make the step formatted command list
                self.yaml_step[istep] = {}
                self.yaml_step[istep]['method'] = ikey
                self.yaml_step[istep]['args']   = args
        self.step_keys = list(self.yaml_step.keys())

if __name__ == '__main__':

    # Command line options needed to obtain data.
    parser = argparse.ArgumentParser(description="iEEG to bids conversion tool.")
    parser.add_argument("--yaml_file", type=str, required=True, help="Yaml file to duplicate into grid format.")
    parser.add_argument("--grid_file", type=str, required=True, help="Yaml grid file with rules for duplication.")
    parser.add_argument("--outdir", type=str, required=True, help="Path to output directory to make grid structure in.")
    parser.add_argument("--cmd", type=str, help="File containing the command line call to the pipeline. Will create updated commands.")
    args = parser.parse_args()
    
    # Read in config
    config = yaml.safe_load(open(args.yaml_file))
    grid   = yaml.safe_load(open(args.grid_file))['step']
    
    # Read in the cmd file if provided
    if args.cmd != None:
        fp  = open(args.cmd,"r")
        cmd = fp.read()
        fp.close()

    # Send data to class for testing
    YL = yaml_loader(config,grid,args.outdir,cmd)

    # Make the step dict
    YL.convert_to_step()

    # Get the combos
    YL.step_combos()

    # Save the new grid
    YL.make_grid_configs()