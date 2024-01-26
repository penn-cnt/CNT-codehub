import os
import dearpygui.dearpygui as dpg
from configs.makeconfigs import *

import modules.addons.features as FEATS

class epi_features_handler:

    def __init__(self):
        pass

    def showFeatures(self, main_window_width = 1280):

        # Child Window Geometry
        child_window_width = int(0.65*main_window_width)
        help_window_width  = int(0.32*main_window_width)

        # Get the module info
        MC         = make_config(FEATS,None)
        method_str = MC.print_methods(silent=True)
        method_str = method_str.replace("    ","")

        # Get the example yaml
        script_path             = os.path.abspath(__file__)
        script_dir              = '/'.join(script_path.split('/')[:-1])
        fp                      = open(f"{script_dir}/defaults/default_features_yaml.txt")
        example_str_arr         = fp.readlines()
        self.features_example   = ''.join(example_str_arr)
        fp.close()

        with dpg.group(horizontal=True):
            with dpg.child_window(width=child_window_width):

                ################################
                ###### Skip Feature Block ######
                ################################
                # Skip Button
                with dpg.group(horizontal=True):
                    dpg.add_text(f"{'Skip Feature Extraction?':40}")
                    self.skip_feature_widget = dpg.add_radio_button(items=[True,False], callback=self.radio_button_callback, horizontal=True, default_value=False)

                ######################### 
                ###### Input Block ######
                #########################
                dpg.add_spacer(height=10)
                dpg.add_separator()
                # Input Options
                with dpg.group(horizontal=True):
                    dpg.add_text(f"{'Load Features YAML File?':40}")
                    self.use_features_yaml_widget  = dpg.add_radio_button(items=[True,False], callback=self.radio_button_callback, horizontal=True, default_value=True)

                # Input pathing
                with dpg.group(horizontal=True):
                    dpg.add_text(f"{'Input YAML Path':40}")
                    self.features_yaml_path_widget_text = dpg.add_input_text(width=int(0.35*child_window_width))
                    self.features_yaml_path_widget      = dpg.add_button(label="Select File", callback=lambda sender, app_data:self.init_file_selection(self.features_yaml_path_widget_text, sender, app_data))

                ########################## 
                ###### Output Block ######
                ##########################
                dpg.add_spacer(height=10)
                dpg.add_separator()
                # Output directory selection
                with dpg.group(horizontal=True):
                    dpg.add_text(f"{'Output YAML Filename':40}")
                    self.features_output_yaml_widget_text = dpg.add_input_text(width=int(0.35*child_window_width))
                    self.features_output_yaml_widget      = dpg.add_button(label="Select Folder", callback=lambda sender, app_data:self.init_folder_selection(self.features_output_yaml_widget_text, sender, app_data))

                ############################### 
                ###### YAML Editor Block ######
                ###############################
                dpg.add_spacer(height=10)
                dpg.add_separator()
                default = "If blank, or no input YAML found, you will be prompted for inputs at runtime." 
                height  = 0.85*self.height_fnc()
                width   = int(0.98*child_window_width)
                dpg.add_text(f"Please enter YAML configuration settings below.")
                self.yaml_input_features_widget = dpg.add_input_text(default_value=default,height=height,width=width,multiline=True)
                with dpg.group(horizontal=True):
                    dpg.add_button(label="Resize Textbox", callback=self.update_yaml_input_features_widget)
                    dpg.add_button(label="Load YAML", callback=self.load_feature_yaml)
                    dpg.add_button(label="Save YAML", callback=self.save_features_yaml)
                    dpg.add_button(label="Show Example", callback=self.display_example_features)
                    dpg.add_button(label="Clear Text", callback=self.clear_features)
                
                with dpg.group(horizontal=True):
                    dpg.add_text(f"More examples:{self.url}")
                    dpg.add_button(label="Copy URL",callback=self.yaml_example_url)


            # Text widget
            with dpg.child_window(width=help_window_width):
                with dpg.group():
                    dpg.add_text("Preprocessing Options:")
                    self.features_help = dpg.add_text(method_str, wrap=0.95*help_window_width)