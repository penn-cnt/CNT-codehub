import os
import dearpygui.dearpygui as dpg

# Interface imports
from modules.theme import applyTheme
from modules.configuration import configuration_handler
from modules.bsc import bsc_handler
from modules.habitat import habitat_handler
from modules.leifborel import leifborel_handler
from modules.callbacks import callback_handler

class Interface(callback_handler,configuration_handler,leifborel_handler):

    def __init__(self):
        """
        Initialize the interface class. Convert passed arguments to instance variables

        Args:
            args (dict): Command line arguments for the lab pipeline.
            metadata (tuple): Metadata about the cli arguments (defaults,help strings, etc.)
        """

        # Program wide variable initialization
        self.DF             = {}
        self.table          = {}
        self.nfolder_shrink = 0
        self.old_fpath      = ''

        # Set some hardcoded values for the GUI dimensions
        self.yaml_frac = 0.6
        self.height    = 900
        self.width     = 1600

        # Table sizing variables
        self.path_width = 120

        # Store sort orders
        self.sort_order                       = {}
        self.sort_order['path']               = False
        self.sort_order['size-(MB)']          = False
        self.sort_order['last-modified-date'] = False
        self.sort_order['owner']              = False

        # Get some relative path info for various fields to use
        self.script_dir  = '/'.join(os.path.abspath(__file__).split('/')[:-1])

        # Make the GUI
        self.show()
        
    def show(self):

        # Create the window object dpg will populate and define some starting parametwers
        dpg.create_context()
        dpg.create_viewport(title='Tool for Visualizing CNT Data Audits', width=self.width, height=self.height, min_height=600, min_width=900)

        # Set the theme for the window
        with dpg.window(tag="Main"):
            applyTheme()
            self.showTabBar()
            pass
        
        # Prepare the window for interactive use
        dpg.setup_dearpygui()

        # Render the window
        dpg.show_viewport()

        # Set the window as the primary viewport (meaning it will remain behind other windows, be the one called by default, etc.)
        dpg.set_primary_window("Main", True)

        # Start the interactive loop for dpg
        dpg.start_dearpygui()

        # On exit of the main window, clear dpg resources and exit interactive use
        dpg.destroy_context()
        pass

    def showTabBar(self):
        with dpg.tab_bar():
            self.showTabs()
        pass

    def showTabs(self):
        """
        Define what tabs the user will see in the main window.
        """

        # Honestly, not sure what it does exactly. Makes the tabs look nicer, but not sure how. This is boilerplate code,.
        dpg.add_texture_registry(show=False, tag='textureRegistry')
        
        # Define the different tabs. Associate with classes that handle the different pages of the gui to show.
        with dpg.tab(label='Leif/Borel'):
            leifborel_handler.showLeifBorel(self)
            pass
        with dpg.tab(label='BSC'):
            bsc_handler.showbsc(self)
            pass
        with dpg.tab(label='HABITAT'):
            habitat_handler.showhabitat(self)
            pass
        with dpg.tab(label='Configurations',tag="configtab"):
            configuration_handler.showConfiguration(self)
            pass
        pass

class App:
    """
    Application handling class for DPG.
    Boilerplate code. Unsure why most examples call DPG via an intermediate class.
    """

    def __init__(self):
        self.interface = Interface()
        pass


if __name__ == '__main__':

    # Run the dearpygui app
    app = App()