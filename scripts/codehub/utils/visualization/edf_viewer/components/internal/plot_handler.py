# User interface imports
import getpass
import pyautogui
import numpy as np

# Matplotlib import and settings
import matplotlib.pyplot as PLT
from matplotlib.ticker import MultipleLocator

# Local Imports
from components.internal.data_handler import *
from components.internal.event_handler import *
from components.internal.observer_handler import *

class data_viewer(Subject,event_handler):

    def __init__(self, infile, args, tight_layout_dict):
        """
        Initialize the data viewer class.

        Args:
            infile (string): Filepath to an edf or pickle file.
            args (object): UI arguments
            tight_layout_dict (object): If `None` the code will calculate the best tight layout.
                                        Otherwise, a dictionary with layout arguments will be used for plotting.
                                        Saves time on loading multiple datasets.
        """
        
        # Save the input info
        self.infile            = infile
        self.fname             = infile.split('/')[-1]
        self.args              = args
        self.tight_layout_dict = tight_layout_dict

        # Get the approx screen dimensions and set some plot variables
        self.height  = args.winfrac*pyautogui.size().height/100
        self.width   = args.winfrac*pyautogui.size().width/100
        self.supsize = self.fontsize_scaler(16,14,self.width)
        self.supsize = np.min([self.supsize,16])

        # Prepare the data
        DH            = data_handler(args,infile)
        rawobj        = DH.workflow()
        self.DF       = rawobj[0]
        self.fs       = rawobj[1]
        self.t_max    = rawobj[2]
        self.duration = rawobj[3]
        self.t0       = rawobj[4]

    def workflow(self):
        """
        Workflow for plotting data and managing flow of information.
        """

        # Attach the observers
        self.attach_objects()

        # Make the initial plot info
        self.create_plot_info()

        # Draw the plot for the first time
        self.draw_base_plots()

        # Save the annotations, if any
        if self.plot_info['annots'].keys():
            self.save_annotations()

    def attach_objects(self):
        """
        Attach observers here so we can have each multiprocessor see the pointers correctly.

        event_observer: Manages what happens when a button or key is pressed. This also sends info to other scripts as needed.
        """

        # Create the observer objects
        self._event_observers = []

        # Attach observers
        self.add_event_observer(event_observer)

    def create_plot_info(self):

        # Store some valuable information about the plot to reference for events and modifications
        self.plot_info                = {}
        self.plot_info['axes']        = {}
        self.plot_info['ylim']        = {}
        self.plot_info['shade']       = {}
        self.plot_info['xlim_orig']   = [self.t0,self.t0+self.duration]
        self.plot_info['xvals']       = np.arange(self.DF.shape[0])/self.fs
        self.plot_info['zoom_cntr']   = 0
        self.plot_info['zoom_lines']  = [0,0]
        self.plot_info['zlim']        = [0,0]
        self.plot_info['annots']      = {}

    def save_annotations(self):
        """
        Save any user annotations to an output CSV.

        Looks for the self.plot_info['annots'] object and iterates over the keys.
        """

        # Loop over the annotations and make the output array object
        output = []
        for ikey in self.plot_info['annots'].keys():
            ival = self.plot_info['annots'][ikey]
            output.append([getpass.getuser(),ival[0],ikey,ival[1]])
        
        # Make the output dataframe
        outDF = PD.DataFrame(output,columns=['user','channel','time','annotation'])
        
        # Append as needed to existing records
        if os.path.exists(self.args.outfile):
            annot_DF = PD.read_csv(self.args.outfile)
            outDF    = PD.concat((annot_DF,outDF))
        
        # Write out the results
        outDF.to_csv(self.args.outfile,index=False)

    ############################
    #### Plotting functions ####
    ############################    

    def draw_base_plots(self):

        # Set the label shift. 72 points equals ~1 inch in pyplot
        width_frac = (0.025*self.width)
        npnt       = int(72*width_frac)
        
        # Create the plotting environment
        nrows           = len(self.DF.columns)
        self.fig        = PLT.figure(dpi=100,figsize=(self.width,self.height))
        gs              = self.fig.add_gridspec(nrows, 1, hspace=0)
        for idx,ichan in enumerate(self.DF.columns):
            # Define the axes
            if idx == 0:
                self.plot_info['axes'][ichan] = self.fig.add_subplot(gs[idx, 0])
                self.first_chan               = ichan
            else:
                self.plot_info['axes'][ichan] = self.fig.add_subplot(gs[idx, 0],sharex=self.plot_info['axes'][self.first_chan])

            # Get the data stats 
            idata,ymin,ymax = self.get_stats(ichan)

            # Plot the data
            self.plot_info['axes'][ichan].plot(self.plot_info['xvals'][::self.args.nstride],idata[::self.args.nstride],color='k')
            self.plot_info['axes'][ichan].set_ylim([ymin,ymax])
            self.plot_info['ylim'][ichan] = [ymin,ymax]

            # Add in shading for the original axes limits
            self.plot_info['shade'][ichan] = self.plot_info['axes'][ichan].axvspan(self.plot_info['xlim_orig'][0], self.plot_info['xlim_orig'][1], facecolor='orange',alpha=0.2)

            # Clean up the plot
            for label in self.plot_info['axes'][ichan].get_xticklabels():
                label.set_alpha(0)
            self.plot_info['axes'][ichan].set_yticklabels([])
            self.plot_info['axes'][ichan].set_ylabel(ichan,fontsize=12,rotation=0,labelpad=npnt)
            self.plot_info['axes'][ichan].xaxis.grid(True)
        
        # X-axis cleanup
        self.last_chan = ichan
        self.plot_info['axes'][ichan].set_xlim(self.plot_info['xlim_orig'])

        # Add an xlabel to the final object
        self.plot_info['axes'][self.last_chan].xaxis.set_major_locator(MultipleLocator(1))
        self.plot_info['axes'][self.last_chan].set_xlabel("Time (s)",fontsize=14)
        for label in self.plot_info['axes'][self.last_chan].get_xticklabels():
            label.set_alpha(1)

        # Set the axes title object
        self.generate_title_str()
        self.plot_info['axes'][self.first_chan].set_title(self.title_str,fontsize=10)
        
        # Set the figure title object
        self.generate_suptitle_str()
        PLT.suptitle(self.suptitle,fontsize=self.supsize)
        
        # Layout handling using previous plot layout or find it for the first time
        if self.tight_layout_dict == None:
            self.fig.tight_layout()
        else:
            self.fig.subplots_adjust(**self.tight_layout_dict)
        
        # Event associations
        self.fig.canvas.mpl_connect('button_press_event', self.notify_event_observers)
        self.fig.canvas.mpl_connect('key_press_event', self.notify_event_observers)

        # Show the results
        PLT.show()

        # Store and return tight layout params for faster subsequent plots
        if self.tight_layout_dict == None:
            self.tight_layout_dict = {par : getattr(self.fig.subplotpars, par) for par in ["left", "right", "bottom", "top", "wspace", "hspace"]}
        return self.tight_layout_dict

    def draw_annotations(self,xpos,annotation,ichannel):

        # Add the annotation
        ymin,ymax = self.plot_info['axes'][ichannel].get_ylim()
        ypos      = 0.5*(ymin+ymax)
        pltobj    = self.plot_info['axes'][ichannel].annotate(text=annotation, xy =(xpos,ypos),bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
        self.plot_info['annots'][xpos] = (ichannel,annotation,pltobj,[])

        # Draw the line for the user to see
        for ikey in self.plot_info['axes'].keys():
            self.plot_info['annots'][xpos][3].append(self.plot_info['axes'][ikey].axvline(xpos, color='g', linestyle='--'))

    def draw_zoom(self,xpos):

        zoom_lines = []
        for ikey in self.plot_info['axes'].keys():
            zoom_lines.append(self.plot_info['axes'][ikey].axvline(xpos, color='r', linestyle='--'))
        return zoom_lines

    def enlarged_plot(self,channel):
        
        # Get the data view
        idata,ymin,ymax = self.get_stats(channel)
        xvals           = np.arange(idata.size)/self.fs

        # Get the current limits of the main viewer
        xlims = self.ax_dict[self.first_chan].get_xlim()
        ylims = [ymin,ymax]

        # Plot the enlarged view
        fig     = PLT.figure(dpi=100,figsize=(self.width,self.height))
        self.ax_enl = fig.add_subplot(111)
        self.ax_enl.plot(xvals,idata,color='k')
        self.ax_enl.set_xlabel("Time (s)",fontsize=14)
        self.ax_enl.set_ylabel(channel,fontsize=14)
        self.ax_enl.set_xlim(xlims)
        self.ax_enl.set_ylim(ylims)
        PLT.title(self.fname,fontsize=14)
        fig.tight_layout()
        PLT.show()

    ##########################
    #### Helper functions ####
    ##########################

    def fontsize_scaler(self,font_ref,width_ref,width_val):
        return font_ref+2*np.floor((width_val-width_ref))

    def get_stats(self,ichan):

        idata  = self.DF[ichan].values
        median = np.median(idata)
        stdev  = np.std(idata)
        idata -= median
        ymin   = -5*stdev
        ymax   = 5*stdev
        return idata,ymin,ymax

    def yscaling(self,ikey,dy):

        # Get the limits of the current plot for rescaling and recreating
        xlim      = self.plot_info['axes'][ikey].get_xlim()
        ylim      = self.plot_info['axes'][ikey].get_ylim()

        # Get the approximate new scale
        scale     = ylim[1]-ylim[0]
        ymin      = ylim[0]+dy*scale
        ymax      = ylim[1]-dy*scale

        # Get the data and limits with a good vertical offset
        vals      = self.DF[ikey].values
        inds      = (vals>=ymin)&(vals<=ymax)
        vals      = vals[inds]
        offset    = np.median(vals)
        vals     -= offset
        ymin     -= offset
        ymax     -= offset

        # Generate new limits
        self.plot_info['axes'][ikey].set_ylim([ymin,ymax])

    def generate_title_str(self):
        upa        = u'\u2191'  # Up arrow
        downa      = u'\u2193'  # Down arrow
        lefta      = u'\u2190'  # Left arrow
        righta     = u'\u2192'  # Right arrow
        self.title_str  = r"z=Zoom between mouse clicks; 'r'=reset x-scale; 'x'=Show entire x-axis; '0'=reset y-scale; 't'=Toggle targets; 'q'=quit current plot; 'Q'=quit the program entirely"
        self.title_str += '\n'
        self.title_str += r"'%s'=Increase Gain; '%s'=Decrease Gain; '%s'=Shift Left; '%s'=Shift Right; '<'=Minor Shift Left; '>'=Minor Shift Right; 'e'=Zoom-in plot of axis the mouse is on;" %(upa, downa, lefta, righta)

    def generate_suptitle_str(self):

        # Base string
        self.suptitle = self.fname