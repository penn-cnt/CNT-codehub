import os
import ast
import pyperclip
import numpy as np
import dearpygui.dearpygui as dpg

# Local imports to get documentation
from modules.addons.channel_clean import *
from modules.addons.channel_mapping import *
from modules.addons.channel_montage import *
from modules.addons.project_handler import *

class callback_handler:

    def __init__(self):
        pass

    ############################
    ###### Helper Functions ####
    ############################
    def height_fnc(self):
        """
        Find a suitable height for the yaml multiline text object.
        Could use a better method for figuring out a decent height.
        """

        height     = dpg.get_viewport_client_height()
        open_space = 1-self.yaml_frac
        modifier   = np.amin([open_space,np.log10(height/self.height)])
        if height>=self.height:
            scale = (self.yaml_frac+modifier)
        else:
            scale = (self.yaml_frac-modifier) 
        return height*scale
    
    def yaml_example_url(self):
        """"
        Copy the examples url to the clipboard of the user.
        """
        pyperclip.copy(self.url)

    ################################
    ###### Resizing Functions ######
    ################################

    def update_submit_widget(self, sender, app_data):
        """
        Fix the height of the submission text widget if rescaled.
        """
        widget_height = self.height_fnc()
        dpg.configure_item(self.submit_widget_text, height=widget_height)

    def update_yaml_input_preprocess_widget(self, sender, app_data):
        """
        Fix the height of the preprocessing yaml text widget if rescaled.
        """
        widget_height = self.height_fnc()
        dpg.configure_item(self.yaml_input_preprocess_widget, height=widget_height)

    def update_yaml_input_features_widget(self, sender, app_data):
        """
        Fix the height of the feature extraction yaml text widget if rescaled.
        """
        widget_height = self.height_fnc()
        dpg.configure_item(self.yaml_input_features_widget, height=widget_height)


    ###############################
    #### File/Folder Selection ####
    ###############################
    
    def init_folder_selection(self,obj,sender,app_data):
        """
        Intialize the folder selection here. Need to send the object to populate with a path, and using a direct show_item doesn't allow this.
        """

        # Make a file and folder dialoge item
        self.current_path_obj = obj
        try:
            dpg.add_file_dialog(directory_selector=True, show=False, callback=self.path_selection_callback, tag="folder_dialog_id", height=400)
        except SystemError:
            pass
        dpg.show_item("folder_dialog_id")

    def init_file_selection(self,obj,sender,app_data):
        """
        Intialize the file selection here. Need to send the object to populate with a path, and using a direct show_item doesn't allow this.
        """

        # Make a file and folder dialoge item
        self.current_path_obj = obj
        try:
            dpg.add_file_dialog(directory_selector=False, show=False, callback=self.path_selection_callback, tag="file_dialog_id", height=400)
            dpg.add_file_extension(".*",parent="file_dialog_id")
        except SystemError:
            pass
        dpg.show_item("file_dialog_id")

    def path_selection_callback(self, sender, app_data):
        """
        Select a file/folder and update the folder path field.
        """
        selected_path = list(app_data['selections'].values())[0]
        dpg.set_value(self.current_path_obj,selected_path)

    #################################
    ###### Save/Load Functions ######
    #################################

    def load_preprocess_yaml(self):
        """
        Load the preprocessing yaml selected by the user and populate the yaml textbox with formatted results.
        """

        yaml_path = dpg.get_value(self.preprocess_yaml_path_widget_text)
        if yaml_path != '' and yaml_path != None:
            if os.path.exists(yaml_path):
                fp       = open(yaml_path,'r')
                contents = fp.readlines()
                fp.close()
                yaml_text = ''.join(contents)
                dpg.set_value(self.yaml_input_preprocess_widget,yaml_text) 

    def load_feature_yaml(self):
        """
        Load the feature extraction yaml selected by the user and populate the yaml textbox with formatted results.
        """

        yaml_path = dpg.get_value(self.features_yaml_path_widget_text)
        if yaml_path != '' and yaml_path != None:
            if os.path.exists(yaml_path):
                fp       = open(yaml_path,'r')
                contents = fp.readlines()
                fp.close()
                yaml_text = ''.join(contents)
                dpg.set_value(self.yaml_input_features_widget,yaml_text) 

    def save_preprocess_yaml(self):
        """
        Save the preprocessing yaml that was created.
        """

        # Define variables
        writeflag       = False
        outpath         = dpg.get_value(self.preprocess_output_yaml_widget_text)
        preprocess_yaml = dpg.get_value(self.yaml_input_preprocess_widget)
        
        # Handle different possible outcomes from widgets and pathing
        if outpath == '' or outpath == None:
            dpg.set_value(self.yaml_input_preprocess_widget,"Please specify an output directory/path.")
        else:
            if preprocess_yaml == '' or preprocess_yaml == None:
                dpg.set_value(self.yaml_input_preprocess_widget,"Cannot save empty YAML entry.")
            else:
                if os.path.exists(outpath):
                    if os.path.isdir(outpath):
                        outpath   = outpath+"preprocess.yaml"
                        writeflag = True
                    else:
                         dpg.set_value(self.yaml_input_preprocess_widget,"Output filepath already exists.")
                else:
                    writeflag = True

        # If the path is available, and we have data, write output
        if writeflag:
            fp = open(outpath,'w')
            fp.write(preprocess_yaml)
            fp.close()
            dpg.set_value(self.yaml_input_preprocess_widget,f"Output saved to {outpath}.")

        # Update the input pathing widget so the CLI gets the new filepath
        dpg.set_value(self.preprocess_yaml_path_widget_text,outpath)

    def save_features_yaml(self):
        """
        Save the featuresing yaml that was created.
        """

        # Define variables
        writeflag       = False
        outpath         = dpg.get_value(self.features_output_yaml_widget_text)
        features_yaml = dpg.get_value(self.yaml_input_features_widget)
        
        # Handle different possible outcomes from widgets and pathing
        if outpath == '' or outpath == None:
            dpg.set_value(self.yaml_input_features_widget,"Please specify an output directory/path.")
        else:
            if features_yaml == '' or features_yaml == None:
                dpg.set_value(self.yaml_input_features_widget,"Cannot save empty YAML entry.")
            else:
                if os.path.exists(outpath):
                    if os.path.isdir(outpath):
                        outpath   = outpath+"features.yaml"
                        writeflag = True
                    else:
                         dpg.set_value(self.yaml_input_features_widget,"Output filepath already exists.")
                else:
                    writeflag = True

        # If the path is available, and we have data, write output
        if writeflag:
            fp = open(outpath,'w')
            fp.write(features_yaml)
            fp.close()
            dpg.set_value(self.yaml_input_features_widget,f"Output saved to {outpath}.")

        # Update the input pathing widget so the CLI gets the new filepath
        dpg.set_value(self.features_yaml_path_widget_text,outpath)

    ############################
    ###### Misc Functions ######
    ############################    

    def display_example_preprocess(self):
        """
        Display the example preprocessing yaml example.
        """
        dpg.configure_item(self.yaml_input_preprocess_widget, default_value=self.preprocess_example)

    def clear_preprocess(self):
        """
        Clear the preprocessing yaml text widget.
        """
        dpg.configure_item(self.yaml_input_preprocess_widget, default_value='')

    def display_example_features(self):
        """
        Display the example feature extraction yaml example.
        """
        dpg.configure_item(self.yaml_input_features_widget, default_value=self.features_example)

    def clear_features(self):
        """
        Clear the feature extraction yaml text widget.
        """
        dpg.configure_item(self.yaml_input_features_widget, default_value='')

    def combo_callback(self, sender, app_data):
        """
        Get the selected value from a drop down widget.
        """
        selected_item = dpg.get_value(sender)

    def radio_button_callback(self, sender, app_data):
        """
        Get the selected value from a radio button widget.
        """
        selected_item = dpg.get_value(sender)

    def update_help(self, help_obj, sender, app_data):
        """
        Update the help text widget.
        """
        button_alias = dpg.get_item_alias(sender)
        new_text     = f"{self.help[button_alias]}"
        new_text     = '\n'.join([new_text[i:i+self.nchar_help] for i in range(0, len(new_text), self.nchar_help)])
        dpg.set_value(help_obj, new_text)

    def update_combo_help(self, help_obj, sender, app_data):
        """
        For drop down menus, where we need more specific help options, case statement through known pipeline arguments.
        """
        button_alias = dpg.get_item_alias(sender)
        if button_alias == 'project':
            combo_value = dpg.get_value(self.project_widget)
            try:
                new_text = getattr(project_handlers, combo_value.lower()).__doc__
            except AttributeError:
                new_text = self.options['allowed_project_args'][combo_value]
        elif button_alias == 'input':
            combo_value = dpg.get_value(self.input_widget)
            new_text    = self.options['allowed_input_args'][combo_value]
        elif button_alias == 'channel_clean':
            combo_value = dpg.get_value(self.channel_clean_widget)
            try:
                new_text = getattr(channel_clean, f"clean_{combo_value.lower()}").__doc__
            except AttributeError:
                new_text = self.options['allowed_clean_args'][combo_value]
        elif button_alias == 'channel_list':
            combo_value = dpg.get_value(self.channel_list_widget)
            try:
                new_text = getattr(channel_mapping, f"mapping_{combo_value.lower()}").__doc__
            except AttributeError:
                new_text = self.options['allowed_channel_args'][combo_value]
        elif button_alias == 'montage':
            combo_value = dpg.get_value(self.montage_widget)
            try:
                new_text = getattr(channel_montage, f"montage_{combo_value.lower()}").__doc__
            except AttributeError:
                new_text = self.options['allowed_montage_args'][combo_value]
        elif button_alias == 'viability':
            combo_value = dpg.get_value(self.viability_widget)
            new_text    = self.options['allowed_viability_args'][combo_value]
        elif button_alias.split('_')[0] == 'imaging':
            combo_value = button_alias.split('_')[1]
            new_text    = self.options['allowed_imaging_programs'][combo_value]['help']

        # Remove tabs
        new_text = new_text.replace("    ","")
        
        # Set the text
        dpg.set_value(help_obj, new_text)

    def create_submit_cmd(self):
        """
        Make the submission command.
        """

        # Base command to the code
        base_cmd = 'python pipeline_manager.py'
        excl     = ['',None]
        
        while True:
            # Input handler
            input_type = dpg.get_value(self.input_widget)
            base_cmd   = f"{base_cmd} --input {input_type}"
            input_path = dpg.get_value(self.input_path_widget_text)
            if input_path not in excl:
                base_cmd   = f"{base_cmd} --input_str {input_path}"
            
            # Output Handler
            output_path = dpg.get_value(self.output_widget_text)
            if output_path in excl:
                base_cmd = "Please provide an output path."
                break
            else:
                base_cmd = f"{base_cmd} --outdir {output_path}"

            # Target join handler
            target = ast.literal_eval(dpg.get_value(self.target_widget))
            if target:
                base_cmd = f"{base_cmd} --targets"

            # Multiprocessing handler
            multithread = ast.literal_eval(dpg.get_value(self.multithread_widget))
            ncpu        = dpg.get_value(self.ncpu_widget)
            if multithread:
                base_cmd = f"{base_cmd} --multithread --ncpu {ncpu}"

            # File jumping options
            n_input  = dpg.get_value(self.n_input_widget)
            n_offset = dpg.get_value(self.n_offset_widget)
            if n_input not in excl and n_input > 0 :
                base_cmd = f"{base_cmd} --n_input {n_input}"
            if n_offset not in excl:
                base_cmd = f"{base_cmd} --n_offset {n_offset}"

            # Timing handler
            t_start   = dpg.get_value(self.t_start_widget)
            t_end     = dpg.get_value(self.t_end_widget)
            t_window  = dpg.get_value(self.t_window_widget)
            t_overlap = np.round(dpg.get_value(self.t_overlap_widget),2)
            if t_start not in excl and t_start > 0:
                base_cmd = f"{base_cmd} --t_start {t_start}"
            if t_end not in excl and t_end != -1:
                base_cmd = f"{base_cmd} --t_start {t_end}"
            if t_window not in excl:
                base_cmd = f"{base_cmd} --t_window {t_window}"
            if t_overlap not in excl and t_overlap > 0:
                base_cmd = f"{base_cmd} --t_overlap {t_overlap}"

            # Channel options
            channel_clean   = dpg.get_value(self.channel_clean_widget)
            channel_list    = dpg.get_value(self.channel_list_widget)
            channel_montage = dpg.get_value(self.montage_widget)
            base_cmd        = f"{base_cmd} --channel_clean {channel_clean} --channel_list {channel_list} --montage {channel_montage}"

            # Data Viability options
            viability = dpg.get_value(self.viability_widget) 
            interp    = ast.literal_eval(dpg.get_value(self.interp_widget))
            n_interp  = dpg.get_value(self.n_interp_widget)
            base_cmd  = f"{base_cmd} --viability {viability}"
            if interp:
                base_cmd  = f"{base_cmd} --interp --n_interp {n_interp}"

            # Project options
            project = dpg.get_value(self.project_widget)
            base_cmd = f"{base_cmd} --project {project}"

            # Preprocessing config options
            skip_preprocess = ast.literal_eval(dpg.get_value(self.skip_preprocess_widget))
            if not skip_preprocess:
                use_preprocess = ast.literal_eval(dpg.get_value(self.use_preprocess_yaml_widget))
                if use_preprocess:
                    preprocess_fpath = dpg.get_value(self.preprocess_yaml_path_widget_text)
                    if preprocess_fpath == None or preprocess_fpath == '':
                        base_cmd = "You selected to use an existing preprocessing YAML file, but selected no file."
                        base_cmd = f"{base_cmd} Please select a file or uncheck this option for runtime generation."
                        break
                    else:
                        base_cmd = f"{base_cmd} --preprocess_file {preprocess_fpath}"
            else:
                base_cmd = f"{base_cmd} --no_preprocess_flag"

            # Feature config options
            skip_feature = ast.literal_eval(dpg.get_value(self.skip_feature_widget))
            if not skip_feature:
                use_features = ast.literal_eval(dpg.get_value(self.use_features_yaml_widget))
                if use_features:
                    features_fpath = dpg.get_value(self.features_yaml_path_widget_text)
                    if features_fpath == None or features_fpath == '':
                        base_cmd = "You selected to use an existing feature extraction YAML file, but selected no file."
                        base_cmd = f"{base_cmd} Please select a file or uncheck this option for runtime generation."
                        break
                    else:
                        base_cmd = f"{base_cmd} --feature_file {preprocess_fpath}"
            else:
                base_cmd = f"{base_cmd} --no_feature_flag"

            # Verbose
            silent = ast.literal_eval(dpg.get_value(self.verbose_widget))
            if silent:
                base_cmd = f"{base_cmd} --silent"

            break

        # Make a display version and update
        self.submit_str      = base_cmd
        self.submit_str_disp = '\n'.join([self.submit_str[i:i+self.nchar_child] for i in range(0, len(self.submit_str), self.nchar_child)])
        dpg.set_value(self.submit_widget_text, self.submit_str_disp)
