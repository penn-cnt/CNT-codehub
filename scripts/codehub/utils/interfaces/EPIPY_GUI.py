import os
import dearpygui.dearpygui as dpg

# Local imports
import epipy as PM

# Interface imports
from EPIPY_modules.theme import applyTheme
from EPIPY_modules.submit import submit_handler
from EPIPY_modules.imaging import imaging_handler
from EPIPY_modules.callbacks import callback_handler
from EPIPY_modules.dataimport import dataimport_handler
from EPIPY_modules.epi_features import epi_features_handler
from EPIPY_modules.configuration import configuration_handler
from EPIPY_modules.epi_preprocess import epi_preprocess_handler

class Interface(callback_handler,configuration_handler,dataimport_handler,epi_preprocess_handler,epi_features_handler):

    def __init__(self,args,metadata):
        """
        Initialize the interface class. Convert passed arguments to instance variables

        Args:
            args (dict): Command line arguments for the lab pipeline.
            metadata (tuple): Metadata about the cli arguments (defaults,help strings, etc.)
        """
        # Save passed arguments to class
        self.args     = args
        self.help     = metadata[0]
        self.types    = metadata[1]
        self.defaults = metadata[2]
        self.options  = metadata[3]

        # Set some hardcoded values for the GUI dimensions
        self.yaml_frac = 0.55
        self.height    = 720
        self.width     = 1280 

        # Save the URL to how to make configuration files for the user to copy if needed.
        self.url       = "https://github.com/penn-cnt/CNT-codehub/tree/main/examples/making_configuration_files"

        # Make the GUI
        self.show()
        
    def show(self):

        # Create the window object dpg will populate and define some starting parametwers
        dpg.create_context()
        dpg.create_viewport(title='EPIPY: Epilepsy Processing and Interpretation using PYthon', width=self.width, height=self.height, min_height=600, min_width=900)

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
        with dpg.tab(label='Configurations',tag="configtab"):
            configuration_handler.showConfiguration(self)
            pass
        with dpg.tab(label='Data Preparation'):
            dataimport_handler.showDataImport(self)
            pass
        with dpg.tab(label='Imaging Options'):
            imaging_handler.showImaging(self)
            pass
        with dpg.tab(label='Preprocessing Options'):
            epi_preprocess_handler.showPreprocess(self)
            pass
        with dpg.tab(label='Feature Extraction'):
            epi_features_handler.showFeatures(self)
            pass
        with dpg.tab(label="Submit Job"):
            submit_handler.showSubmit(self)
            pass
        pass

class App:
    """
    Application handling class for DPG.
    Boilerplate code. Unsure why most examples call DPG via an intermediate class.
    """

    def __init__(self,args,metadata):
        self.interface = Interface(args,metadata)
        pass


if __name__ == '__main__':
    
    # Get the pathing to the pipeline manager. Allows us to find the argument file
    pipeline_path = os.path.dirname(os.path.abspath(PM.__file__))+'/'

    # Get the arguments
    args, metadata = PM.argument_handler(argument_dir=pipeline_path,require_flag=False)
    args           = vars(args)

    # Run the dearpygui app
    app = App(args,metadata)