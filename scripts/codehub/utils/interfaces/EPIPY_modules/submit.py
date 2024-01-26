import pyperclip
import dearpygui.dearpygui as dpg

class submit_handler:

    def __init__(self):
        pass

    def showSubmit(self, main_window_width = 1280):

        # Child Window Geometry
        child_window_width = int(0.65*main_window_width)
        help_window_width  = int(0.32*main_window_width)

        with dpg.group(horizontal=True):
            with dpg.child_window(width=child_window_width):
                height = 0.75*self.height_fnc()
                width  = int(0.98*child_window_width)
                dpg.add_text(f"Command Line Call:")
                self.submit_widget_text = dpg.add_input_text(multiline=True,width=width,height=height)
                with dpg.group(horizontal=True):
                    dpg.add_button(label="Show Command",callback=self.create_submit_cmd)
                    dpg.add_button(label="Copy Command",callback=lambda sender,app_data:pyperclip.copy(dpg.get_value(self.submit_widget_text).replace("\n","")))
                    dpg.add_button(label="Resize Box",callback=self.update_submit_widget)

                dpg.add_button(label="Submit Job")

