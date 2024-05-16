import pandas as PD
import dearpygui.dearpygui as dpg

class habitat_handler:

    def __init__(self):
        pass

    def showhabitat(self):

        # Search String
        str_width = 15
        with dpg.group(horizontal=True):
            dpg.add_text(f"{'Search for: ':{str_width}}")
            self.habitat_search_text_widget = dpg.add_input_text(width=int(0.65*self.width))
            self.habitat_search_widget      = dpg.add_button(label="Show Results", width=int(0.10*self.width), callback=lambda sender, app_data:self.search_fnc(
                                                                self.habitat_search_text_widget, self.habitat_text_id, self.habitat_search_type_widget, 'habitat', sender, app_data))
            self.habitat_reset_widget       = dpg.add_button(label="Reset", width=int(0.10*self.width), callback=lambda sender, app_data:self.reset_fnc(self.habitat_text_id, 'habitat', sender, app_data))
        with dpg.group(horizontal=True):
            dpg.add_text(f"{'Search by: ':{str_width}}")
            self.habitat_search_type_widget = dpg.add_radio_button(items=['path','md5'], horizontal=True, default_value='path')
        with dpg.group(horizontal=True):
            dpg.add_text(f"{'Fuzzy Match? ':{str_width}}")
            self.habitat_fuzzysearch_widget = dpg.add_radio_button(items=[True,False], horizontal=True, default_value=False)
            dpg.add_text(f"(Only applies to *filenames*. Do not use for md5 or absolute path.)")
        with dpg.group(horizontal=True):
            dpg.add_text(f"{'Apply to: ':{str_width}}")
            self.apply_leifborel_widget = dpg.add_button(label="Leif/Borel", width=int(0.10*self.width), callback=lambda sender, app_data:self.apply_leifborel(self.habitat_search_text_widget))
            self.apply_bsc_widget       = dpg.add_button(label="BSC", width=int(0.10*self.width), callback=lambda sender, app_data:self.apply_bsc(self.habitat_search_text_widget))
            self.apply_habitat_widget   = dpg.add_button(label="Habitat", width=int(0.10*self.width), callback=lambda sender, app_data:self.apply_habitat(self.habitat_search_text_widget))
        with dpg.group(horizontal=True):
            dpg.add_text(f"{'Shrink path: ':{str_width}}")
            self.folder_shrink_input_widget = dpg.add_input_int(default_value=0,step_fast=2,min_value=1,width=int(0.10*self.width),callback=lambda sender, app_data:self.shrink_path(self.habitat_text_id,'habitat',sender,app_data))
        dpg.add_spacer(height=10)
        dpg.add_separator()
        with dpg.group(horizontal=True):
            dpg.add_text(f"{'Sort by: ':{str_width}}")
            self.sort_path  = dpg.add_button(label="path", width=int(0.10*self.width), callback=lambda sender, app_data:self.sort_data(self.habitat_text_id,'habitat','path',sender,app_data))
            self.sort_size  = dpg.add_button(label="size", width=int(0.10*self.width), callback=lambda sender, app_data:self.sort_data(self.habitat_text_id,'habitat','size-(MB)',sender,app_data))
            self.sort_date  = dpg.add_button(label="last-modified-date", width=int(0.10*self.width), callback=lambda sender, app_data:self.sort_data(self.habitat_text_id,'habitat','last-modified-date',sender,app_data))
            self.sort_owner = dpg.add_button(label="owner", width=int(0.10*self.width), callback=lambda sender, app_data:self.sort_data(self.habitat_text_id,'habitat','owner',sender,app_data))
        dpg.add_spacer(height=10)

        # Add a multiline text input widget
        height = 1.0*self.height_fnc()
        self.habitat_text_id = dpg.add_input_text(multiline=True, readonly=True, width=0.95*self.width,height=height)
        dpg.add_button(label="Resize Box",callback=self.update_habitat_widget)
        
