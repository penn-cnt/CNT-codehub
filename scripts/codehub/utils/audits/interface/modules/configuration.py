import os
import yaml
import dearpygui.dearpygui as dpg

class configuration_handler:

    def __init__(self):
        pass

    def update_leifborel(self,sender,app_data):
        self.show_all_data(dpg.get_value(self.leifborel_input_path_widget_text),self.leifborel_text_id,'leifborel')

    def update_bsc(self,sender,app_data):
        self.show_all_data(dpg.get_value(self.bsc_input_path_widget_text),self.bsc_text_id,'bsc')

    def update_habitat(self,sender,app_data):
        self.show_all_data(dpg.get_value(self.habitat_input_path_widget_text),self.habitat_text_id,'habitat')

    def showConfiguration(self, main_window_width = 1280):

        # Load the default paths
        auditpaths      = yaml.safe_load(open(f"{self.script_dir}/../config/auditpaths.yaml","r"))
        default_leif    = auditpaths['leifborel']
        default_bsc     = auditpaths['bsc']
        default_habitat = auditpaths['habitat']

        str_width = 30
        ######################### 
        ###### Input Block ######
        #########################

        # Input pathing
        with dpg.group(horizontal=True):
            dpg.add_text(f"{'Leif/Borel Audit Data':{str_width}}")
            self.leifborel_input_path_widget_text = dpg.add_input_text(default_value=default_leif)
            self.leifborel_input_path_widget      = dpg.add_button(label="Select File", width=int(0.1*main_window_width), callback=lambda sender, app_data:self.init_file_selection(self.leifborel_input_path_widget_text, sender, app_data))
            self.leifborel_update_path            = dpg.add_button(label="Update", width=int(0.1*main_window_width), callback=self.update_leifborel)
        with dpg.group(horizontal=True):
            dpg.add_text(f"{'BSC Audit Data':{str_width}}")
            self.bsc_input_path_widget_text = dpg.add_input_text(default_value=default_bsc)
            self.bsc_input_path_widget      = dpg.add_button(label="Select File", width=int(0.1*main_window_width), callback=lambda sender, app_data:self.init_file_selection(self.bsc_input_path_widget_text, sender, app_data))
            self.bsc_update_path            = dpg.add_button(label="Update", width=int(0.1*main_window_width), callback=self.update_bsc)
        with dpg.group(horizontal=True):
            dpg.add_text(f"{'Habitat Audit Data':{str_width}}")
            self.habitat_input_path_widget_text = dpg.add_input_text(default_value=default_habitat)
            self.habitat_input_path_widget      = dpg.add_button(label="Select File", width=int(0.1*main_window_width), callback=lambda sender, app_data:self.init_file_selection(self.habitat_input_path_widget_text, sender, app_data))
            self.habitat_update_path            = dpg.add_button(label="Update", width=int(0.1*main_window_width), callback=self.update_habitat)

        self.show_all_data(dpg.get_value(self.leifborel_input_path_widget_text),self.leifborel_text_id,'leifborel')
        self.show_all_data(dpg.get_value(self.bsc_input_path_widget_text),self.bsc_text_id,'bsc')
        self.show_all_data(dpg.get_value(self.habitat_input_path_widget_text),self.habitat_text_id,'habitat')