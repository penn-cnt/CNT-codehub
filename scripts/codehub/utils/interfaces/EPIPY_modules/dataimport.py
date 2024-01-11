import dearpygui.dearpygui as dpg

class dataimport_handler:

    def __init__(self):
        pass

    def showDataImport(self, main_window_width = 1280):

        # Child Window Geometry
        child_window_width = int(0.65*main_window_width)
        help_window_width  = int(0.32*main_window_width)

        with dpg.group(horizontal=True):
            with dpg.child_window(width=child_window_width):

                ########################### 
                ###### Channel Block ######
                ###########################
                # Channel cleaning info
                channel_clean_list = list(self.options['allowed_clean_args'].keys())
                with dpg.group(horizontal=True):
                    arg_var = 'channel_clean'
                    dpg.add_text(f"{'Channel Cleaning Method':40}")
                    self.channel_clean_widget = dpg.add_combo(items=channel_clean_list, callback=self.combo_callback, default_value=self.defaults[arg_var],width=int(0.5*child_window_width))
                    dpg.add_button(label="Help", callback=lambda sender, app_data: self.update_combo_help(self.channel_help,sender,app_data), tag=arg_var)

                # Channel mapping info
                channel_list = list(self.options['allowed_channel_args'].keys())
                with dpg.group(horizontal=True):
                    arg_var = 'channel_list'
                    dpg.add_text(f"{'Channel Mapping':40}")
                    self.channel_list_widget = dpg.add_combo(items=channel_list, callback=self.combo_callback, default_value=self.defaults[arg_var],width=int(0.5*child_window_width))
                    dpg.add_button(label="Help", callback=lambda sender, app_data: self.update_combo_help(self.channel_help,sender,app_data), tag=arg_var)

                # Channel Montaging info
                montage_list = list(self.options['allowed_montage_args'].keys())
                with dpg.group(horizontal=True):
                    arg_var = 'montage'
                    dpg.add_text(f"{'Montage Method':40}")
                    self.montage_widget = dpg.add_combo(items=montage_list, callback=self.combo_callback, default_value=self.defaults[arg_var],width=int(0.5*child_window_width))
                    dpg.add_button(label="Help", callback=lambda sender, app_data: self.update_combo_help(self.channel_help,sender,app_data), tag=arg_var)

                ############################# 
                ###### Viability Block ######
                #############################
                dpg.add_spacer(height=10)
                dpg.add_separator()
                # Viability info
                viability_list = list(self.options['allowed_viability_args'].keys())
                with dpg.group(horizontal=True):
                    arg_var = 'viability'
                    dpg.add_text(f"{'Viability Method':40}")
                    self.viability_widget = dpg.add_combo(items=viability_list, callback=self.combo_callback, default_value=self.defaults[arg_var],width=int(0.5*child_window_width))
                    dpg.add_button(label="Help", callback=lambda sender, app_data: self.update_combo_help(self.channel_help,sender,app_data), tag=arg_var)

                # NaN interpolation flag
                with dpg.group(horizontal=True):
                    arg_var = 'interp'
                    default = self.defaults[arg_var]
                    dpg.add_text(f"{'Interpolation of NaNs':40}")
                    self.interp_widget  = dpg.add_radio_button(items=[True,False], callback=self.radio_button_callback, horizontal=True, default_value=default)
                    dpg.add_button(label="Help", callback=lambda sender, app_data: self.update_help(self.channel_help, sender, app_data), tag=arg_var)

                # If interpolating of NaNs, how many to interp over
                with dpg.group(horizontal=True):
                    arg_var = 'n_interp'
                    default = self.defaults[arg_var]
                    dpg.add_text(f"{'Max # of NaNs to interpolate over':40}")
                    self.n_interp_widget = dpg.add_input_int(default_value=default,step_fast=4,min_value=1,width=int(0.5*child_window_width))
                    dpg.add_button(label="Help", callback=lambda sender, app_data: self.update_help(self.channel_help, sender, app_data), tag=arg_var)

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
                    dpg.add_button(label="Help", callback=lambda sender, app_data: self.update_combo_help(self.channel_help,sender,app_data), tag=arg_var)


            # Text widget
            with dpg.child_window(width=help_window_width):
                with dpg.group():
                    dpg.add_text("Help:")
                    self.channel_help = dpg.add_text("", wrap=0.95*help_window_width)