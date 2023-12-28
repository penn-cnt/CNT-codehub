import os
import dearpygui.dearpygui as dpg
from configs.makeconfigs import *

import modules.addons.preprocessing as PP

class epi_preprocess_handler:

    def __init__(self):
        pass

    def showPreprocess(self, main_window_width = 1280):

        # Child Window Geometry
        child_window_width = int(0.66*main_window_width)
        help_window_width  = int(0.33*main_window_width)

        # Get the module info
        MC         = make_config(PP,None)
        method_str = MC.print_methods(silent=True)
        method_str = method_str.replace("    ","")

        # Get the example yaml
        script_path             = os.path.abspath(__file__)
        script_dir              = '/'.join(script_path.split('/')[:-1])
        fp                      = open(f"{script_dir}/defaults/default_preprocess_yaml.txt")
        example_str_arr         = fp.readlines()
        self.preprocess_example = ''.join(example_str_arr)
        fp.close()

        with dpg.group(horizontal=True):
            with dpg.child_window(width=child_window_width):
                ######################### 
                ###### Input Block ######
                #########################

                # Input Options
                with dpg.group(horizontal=True):
                    dpg.add_text(f"{'Load Preprocessing YAML File?':40}")
                    self.preprocess_yaml_widget  = dpg.add_radio_button(items=[True,False], callback=self.radio_button_callback, horizontal=True, default_value=True)

                # Input pathing
                with dpg.group(horizontal=True):
                    dpg.add_text(f"{'Input YAML Path':40}")
                    self.preprocess_yaml_path_widget_text = dpg.add_input_text(width=int(0.35*child_window_width))
                    self.preprocess_yaml_path_widget      = dpg.add_button(label="Select File", callback=lambda: dpg.configure_item(file_dialog, show=True),width=int(0.14*child_window_width))

                ########################## 
                ###### Output Block ######
                ##########################
                dpg.add_spacer(height=10)
                dpg.add_separator()
                # Output directory selection
                with dpg.group(horizontal=True):
                    dpg.add_text(f"{'Output YAML Filename':40}")
                    self.output_yaml_widget_text = dpg.add_input_text(width=int(0.35*child_window_width))
                    self.output_yaml_widget = dpg.add_button(label="Select Output Folder", callback=lambda: dpg.configure_item(file_dialog, show=True),width=int(0.14*child_window_width))

                ############################### 
                ###### YAML Editor Block ######
                ###############################
                dpg.add_spacer(height=10)
                dpg.add_separator()
                default = "If blank, or no input YAML found, you will be prompted for inputs at runtime." 
                height  = 0.925*self.height_fnc()
                width   = int(0.98*child_window_width)
                dpg.add_text(f"Please enter YAML configuration settings below.")
                self.yaml_input_preprocess_widget = dpg.add_input_text(default_value=default,height=height,width=width,multiline=True)
                with dpg.group(horizontal=True):
                    dpg.add_button(label="Resize Textbox", callback=self.update_yaml_input_preprocess_widget)
                    dpg.add_button(label="Save YAML", callback=self.update_yaml_input_preprocess_widget)
                    dpg.add_button(label="Show Example", callback=self.display_example_preprocess)
                    dpg.add_button(label="Clear Text", callback=self.clear_preprocess)
                
                with dpg.group(horizontal=True):
                    dpg.add_text(f"More examples:{self.url}")
                    dpg.add_button(label="Copy URL",callback=self.yaml_example_url)


            # Text widget
            with dpg.child_window(width=help_window_width):
                with dpg.group():
                    dpg.add_text("Preprocessing Options:")
                    self.preprocess_help = dpg.add_text(method_str, wrap=0.95*help_window_width)