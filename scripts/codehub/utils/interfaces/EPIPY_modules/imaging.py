import dearpygui.dearpygui as dpg

class imaging_handler:

    def __init__(self):
        pass

    def showImaging(self, main_window_width = 1280):

        # Child Window Geometry
        child_window_width = int(0.65*main_window_width)
        help_window_width  = int(0.32*main_window_width)

        # Since there are variable image widgets, store via dict
        self.imaging_widgets = {}

        with dpg.group(horizontal=True):
            with dpg.child_window(width=child_window_width):

                # Obtain Imaging Programs to display
                keys = list(self.options['allowed_imaging_programs'].keys())

                for ikey in keys:
                    with dpg.group(horizontal=True):
                        default = self.options['allowed_imaging_programs'][ikey]['path']
                        dpg.add_text(f"{ikey:40}")
                        self.imaging_widgets[ikey] = dpg.add_input_text(width=int(0.35*child_window_width),default_value=default)
                        dpg.add_button(label="Select File", width=int(0.14*child_window_width), callback=lambda sender, app_data:self.init_file_selection(self.imaging_widgets[ikey], sender, app_data))
                        dpg.add_button(label="Help", callback=lambda sender, app_data: self.update_combo_help(self.image_help,sender,app_data), tag=f"imaging_{ikey}")

            # Text widget
            with dpg.child_window(width=help_window_width):
                with dpg.group():
                    dpg.add_text("Help:")
                    self.image_help = dpg.add_text("", wrap=0.95*help_window_width)