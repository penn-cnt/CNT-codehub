import os
import ast
import numpy as np
import pandas as PD
from thefuzz import fuzz
import dearpygui.dearpygui as dpg
from prettytable import PrettyTable, ALL

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

    def update_leif_widget(self, sender, app_data):
        """
        Fix the height of the submission text widget if rescaled.
        """
        widget_height = self.height_fnc()
        dpg.configure_item(self.leifborel_text_id, height=widget_height)

    def update_bsc_widget(self, sender, app_data):
        """
        Fix the height of the submission text widget if rescaled.
        """
        widget_height = self.height_fnc()
        dpg.configure_item(self.bsc_text_id, height=widget_height)

    def update_habitat_widget(self, sender, app_data):
        """
        Fix the height of the submission text widget if rescaled.
        """
        widget_height = self.height_fnc()
        dpg.configure_item(self.habitat_text_id, height=widget_height)

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
        # Get the selected path
        selected_path = list(app_data['selections'].values())[0]

        # Handle dpg bug about folder selection appearing twice in a string
        selected_path_arr = selected_path.split('/')
        if selected_path_arr[-2] == selected_path_arr[-1]:
            selected_path = '/'.join(selected_path_arr[:-1])

        dpg.set_value(self.current_path_obj,selected_path)

    #############################
    ###### Search Function ######
    #############################

    def apply_leifborel(self, new_text_widget):
        search_str = dpg.get_value(new_text_widget)
        dpg.set_value(self.leifborel_search_text_widget,search_str)
        self.search_fnc(self.leifborel_search_text_widget, self.leifborel_text_id, self.leifborel_search_type_widget, 'leifborel', None, None)

    def apply_bsc(self, new_text_widget):
        search_str = dpg.get_value(new_text_widget)
        dpg.set_value(self.bsc_search_text_widget,search_str)
        self.search_fnc(self.bsc_search_text_widget, self.bsc_text_id, self.bsc_search_type_widget, 'bsc', None, None)

    def apply_habitat(self, new_text_widget):
        search_str = dpg.get_value(new_text_widget)
        dpg.set_value(self.habitat_search_text_widget,search_str)
        self.search_fnc(self.habitat_search_text_widget, self.habitat_text_id, self.habitat_search_type_widget, 'habitat', None, None)

    def search_fnc(self, search_widget, text_widget, search_type_widget, systag, sender, app_data):
        
        # Get the search string
        search_str = dpg.get_value(search_widget)

        # Get the search type
        search_type = dpg.get_value(search_type_widget)

        # Check if fuzzy matching
        fuzz_flag = ast.literal_eval(dpg.get_value(self.leifborel_fuzzysearch_widget))

        # Find the index to display
        if search_type == 'path':
            if not fuzz_flag:
                self.display_index = self.DF[systag].index[self.DF[systag]['path'].apply(lambda x:search_str in x).values]
            else:
                ratios             = self.DF[systag]['path'].apply(lambda x:fuzz.token_sort_ratio(x.split('/')[-1],search_str)).values
                self.display_index = self.DF[systag].index[np.argsort(ratios)[::-1]]
        else:
            self.display_index = self.DF[systag].index[self.DF[systag]['md5'].apply(lambda x:search_str in x).values]
        
        # Make the pretty table
        self.table[str(text_widget)] = self.make_pretty_table(self.DF[systag].loc[self.display_index])

        # Clear the current pretty table
        dpg.configure_item(text_widget, default_value='')

        # Display the new slice
        dpg.set_value(text_widget, self.table[str(text_widget)])

    def reset_fnc(self, text_widget, systag, sender, app_data):

        # Set the display index to be all of the data
        self.display_index = list(self.DF[systag].index)

        # Make the pretty table
        self.table[str(text_widget)] = self.make_pretty_table(self.DF[systag].loc[self.display_index])

        # Clear the current pretty table
        dpg.configure_item(text_widget, default_value='')

        # Display the new slice
        dpg.set_value(text_widget, self.table[str(text_widget)])

    ############################
    ###### Misc Functions ######
    ############################    

    def make_pretty_table(self,DF):

        # Initialize a pretty table for easy reading
        table = PrettyTable(hrules=ALL)
        table.field_names = self.cols
        for irow in DF.index[:250]:
            iDF           = DF.loc[irow]
            ipath         = iDF['smallpath']
            ipath         = '\n'.join([ipath[i:i+self.path_width-1] for i in range(0, len(ipath), self.path_width-1)])
            formatted_row = [f"{ipath}",f"{iDF['md5']:33}",f"{iDF['size-(MB)']:.2f}",f"{iDF['last-modified-date']:12}",f"{iDF['owner']:12}"]
            table.add_row(formatted_row)
        table.align['path'] = 'l'
        return table
    
    def sort_data(self, text_widget, systag, axis, sender, app_data):
        
        # Get the sort flag
        sort_flag = self.sort_order[axis]

        # Update the sort order for next time
        self.sort_order[axis] = not self.sort_order[axis]

        # Make a temporary dataslice for sorting
        iDF = self.DF[systag].copy()
        iDF = iDF.loc[self.display_index]

        # Sort the dataframe by the requested axis
        #self.DF[systag] = self.DF[systag].sort_values(by=[axis],ascending=sort_flag)
        iDF = iDF.sort_values(by=[axis],ascending=sort_flag)

        # Set the display index to be all of the data
        #self.display_index = list(self.DF[systag].index)

        # Make the pretty table
        #self.table[str(text_widget)] = self.make_pretty_table(self.DF[systag].loc[self.display_index])
        self.table[str(text_widget)] = self.make_pretty_table(iDF)

        # Clear the current pretty table
        dpg.configure_item(text_widget, default_value='')

        # Display the new slice
        dpg.set_value(text_widget, self.table[str(text_widget)])

    def shrink_path(self, text_widget, systag, sender, app_data):
        
        # Make a new shrink path
        self.nfolder_shrink = dpg.get_value(sender)
        self.DF[systag]['smallpath'] = self.DF[systag]['path'].apply(lambda x:'/'.join(x.split('/')[self.nfolder_shrink:]))        
        
        # Make the pretty table
        self.table[str(text_widget)] = self.make_pretty_table(self.DF[systag].loc[self.display_index])

        # Clear the current pretty table
        dpg.configure_item(text_widget, default_value='')

        # Display the new slice
        dpg.set_value(text_widget, self.table[str(text_widget)])

    def show_all_data(self,fpath,widget,systag):

        # Load data
        self.DF[systag] = PD.read_csv(fpath,nrows=10000)

        # Get the columns for pretty table
        self.cols = list(self.DF[systag].columns)

        # Make the shrinked path
        self.DF[systag]['smallpath'] = self.DF[systag]['path'].apply(lambda x:'/'.join(x.split('/')[self.nfolder_shrink:]))

        # Set the display index to be all of the data
        self.display_index = list(self.DF[systag].index)

        # Make the pretty table
        self.table[str(widget)] = self.make_pretty_table(self.DF[systag].loc[self.display_index])

        # Set the text values
        dpg.set_value(widget, self.table[str(widget)])
