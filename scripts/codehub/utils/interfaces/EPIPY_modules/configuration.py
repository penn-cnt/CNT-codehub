import dearpygui.dearpygui as dpg

class configuration_handler:

    def __init__(self):
        pass

    def showConfiguration(self, main_window_width = 1280):

        # Child Window Geometry
        child_window_width = int(0.65*main_window_width)
        help_window_width  = int(0.32*main_window_width)
        
        # Get the approximate number of characters allowed per-line. One time call to self to be visible across all widgets.
        max_pixel_width = 8
        self.nchar_help = int(help_window_width/max_pixel_width)

        with dpg.file_dialog(directory_selector=True, show=False, callback=self.folder_selection_callback, height=400) as file_dialog:
            dpg.add_input_text(label="Output Folder", width=400, callback=self.folder_selection_callback)
            dpg.add_button(label="Select Folder", callback=lambda: dpg.configure_item(file_dialog, show=True))

        with dpg.group(horizontal=True):
            with dpg.child_window(width=child_window_width):

                ######################### 
                ###### Input Block ######
                #########################

                # Input Options
                input_list = list(self.options['allowed_input_args'].keys())
                with dpg.group(horizontal=True):
                    arg_var = 'input'
                    dpg.add_text(f"{'Input Type':40}")
                    self.input_widget = dpg.add_combo(items=input_list, callback=self.combo_callback, default_value=self.defaults[arg_var],width=int(0.5*child_window_width))
                    dpg.add_button(label="Help", callback=lambda sender, app_data: self.update_combo_help(self.configuration_help,sender,app_data), tag=arg_var)

                # Input pathing
                with dpg.group(horizontal=True):
                    dpg.add_text(f"{'Input Path':40}")
                    self.input_path_widget_text = dpg.add_input_text(width=int(0.35*child_window_width))
                    self.input_path_widget      = dpg.add_button(label="Select File", callback=lambda: dpg.configure_item(file_dialog, show=True),width=int(0.14*child_window_width))
                    dpg.add_button(label="Help", callback=lambda sender, app_data: self.update_help(self.configuration_help, sender, app_data), tag="input_path")

                ########################## 
                ###### Output Block ######
                ##########################
                dpg.add_spacer(height=10)
                dpg.add_separator()
                # Output directory selection
                with dpg.group(horizontal=True):
                    dpg.add_text(f"{'Output Directory':40}")
                    output_widget_text = dpg.add_input_text(width=int(0.35*child_window_width))
                    self.output_widget = dpg.add_button(label="Select Folder", callback=lambda: dpg.configure_item(file_dialog, show=True),width=int(0.14*child_window_width))
                    dpg.add_button(label="Help", callback=lambda sender, app_data: self.update_help(self.configuration_help, sender, app_data), tag="outdir")

                ########################## 
                ###### Config Block ######
                ##########################

                # Multithread Options
                dpg.add_spacer(height=10)
                dpg.add_separator()
                with dpg.group(horizontal=True):
                    arg_var = 'multithread'
                    default = self.defaults[arg_var]
                    dpg.add_text(f"{'Multithreaded':40}")
                    self.multithread_widget  = dpg.add_radio_button(items=[True,False], callback=self.radio_button_callback, horizontal=True, default_value=default)
                    dpg.add_button(label="Help", callback=lambda sender, app_data: self.update_help(self.configuration_help, sender, app_data), tag=arg_var)

                # Number of CPUs
                with dpg.group(horizontal=True):
                    arg_var = 'ncpu'
                    default = self.defaults[arg_var]
                    dpg.add_text(f"{'# CPUs':40}")
                    self.ncpu_widget = dpg.add_input_int(default_value=default,step_fast=4,min_value=1,width=int(0.5*child_window_width))
                    dpg.add_button(label="Help", callback=lambda sender, app_data: self.update_help(self.configuration_help, sender, app_data), tag=arg_var)

                # Limit number of input files to this amount
                dpg.add_spacer(height=10)
                dpg.add_separator()
                with dpg.group(horizontal=True):
                    arg_var = 'n_input'
                    default = self.defaults[arg_var]
                    dpg.add_text(f"{'# of input files':40}")
                    self.n_input_widget = dpg.add_input_int(default_value=default,step_fast=25,min_value=1,width=int(0.5*child_window_width))
                    dpg.add_button(label="Help", callback=lambda sender, app_data: self.update_help(self.configuration_help, sender, app_data), tag=arg_var)
                
                # Skip ahead to this file number
                with dpg.group(horizontal=True):
                    arg_var = 'n_offset'
                    default = self.defaults[arg_var]
                    dpg.add_text(f"{'# of skipped inputs':40}")
                    self.n_offset_widget = dpg.add_input_int(default_value=default,step_fast=25,min_value=0,width=int(0.5*child_window_width))
                    dpg.add_button(label="Help", callback=lambda sender, app_data: self.update_help(self.configuration_help, sender, app_data), tag=arg_var)

                ########################## 
                ###### Timing Block ######
                ##########################
                dpg.add_spacer(height=10)
                dpg.add_separator()
                # Start time
                with dpg.group(horizontal=True):
                    arg_var = 't_start'
                    default = self.defaults[arg_var]
                    dpg.add_text(f"{'File start times (s)':40}")
                    self.t_start_widget = dpg.add_input_int(default_value=default,step_fast=25,width=int(0.5*child_window_width))
                    dpg.add_button(label="Help", callback=lambda sender, app_data: self.update_help(self.configuration_help, sender, app_data), tag=arg_var)

                # End Time
                with dpg.group(horizontal=True):
                    arg_var = 't_end'
                    default = self.defaults[arg_var]
                    dpg.add_text(f"{'File end times (s)':40}")
                    self.t_end_widget = dpg.add_input_int(default_value=default,step_fast=25,width=int(0.5*child_window_width))
                    dpg.add_button(label="Help", callback=lambda sender, app_data: self.update_help(self.configuration_help, sender, app_data), tag=arg_var)

                # Time Window
                with dpg.group(horizontal=True):
                    arg_var = 't_window'
                    default = self.defaults[arg_var]
                    if default == None: default = ''
                    dpg.add_text(f"{'Time windows (comma list,secs,""=all)':40}")
                    self.t_window_widget = dpg.add_input_text(default_value=default,width=int(0.5*child_window_width))
                    dpg.add_button(label="Help", callback=lambda sender, app_data: self.update_help(self.configuration_help, sender, app_data), tag=arg_var)

                # Time Overlap
                with dpg.group(horizontal=True):
                    arg_var = 't_overlap'
                    default = self.defaults[arg_var]
                    dpg.add_text(f"{'Time window overlap (fractional)':40}")
                    self.t_overlap_widget = dpg.add_input_float(default_value=default,step_fast=5,max_value=1.0,min_value=0.0,width=int(0.5*child_window_width))
                    dpg.add_button(label="Help", callback=lambda sender, app_data: self.update_help(self.configuration_help, sender, app_data), tag=arg_var)

                ########################### 
                ###### Verbose Block ######
                ###########################
                dpg.add_spacer(height=10)
                dpg.add_separator()
                with dpg.group(horizontal=True):
                    arg_var = 'silent'
                    default = self.defaults[arg_var]
                    dpg.add_text(f"{'Silent':40}")
                    self.multithread_widget  = dpg.add_radio_button(items=[True,False], callback=self.radio_button_callback, horizontal=True, default_value=default)
                    dpg.add_button(label="Help", callback=lambda sender, app_data: self.update_help(self.configuration_help, sender, app_data), tag=arg_var)

                ########################### 
                ###### Project Block ######
                ###########################
                dpg.add_spacer(height=10)
                dpg.add_separator()
                project_list = list(self.options['allowed_project_args'].keys())
                with dpg.group(horizontal=True):
                    arg_var = 'project'
                    dpg.add_text(f"{'Project Workflow':40}")
                    self.project_widget = dpg.add_combo(items=project_list, callback=self.combo_callback, default_value=self.defaults[arg_var],width=int(0.5*child_window_width))
                    dpg.add_button(label="Help", callback=lambda sender, app_data: self.update_combo_help(self.configuration_help,sender,app_data), tag=arg_var)

                # Submit a job
                dpg.add_spacer(height=175)
                dpg.add_separator()
                with dpg.group(horizontal=True):
                    dpg.add_button(label="Submit Job", callback=self.submit_fnc)

            # Text widget
            with dpg.child_window(width=help_window_width):
                with dpg.group():
                    dpg.add_text("Help:")
                    self.configuration_help = dpg.add_text("", wrap=0.95*help_window_width)