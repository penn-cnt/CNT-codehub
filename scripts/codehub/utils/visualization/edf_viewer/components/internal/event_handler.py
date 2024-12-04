import numpy as np
import pylab as PLT
from sys import exit
from prompt_toolkit import prompt
from prompt_toolkit.completion import WordCompleter

# Local Imports
from components.internal.observer_handler import *
from components.internal.annotation_handler import *

class event_observer(Observer):

    def listen_event(self,event):
        
        # Event logic
        button_flag = hasattr(event,'button')
        if button_flag:
            event_handler.button_response(self,event)
        else:
            # Quit the plot
            if event.key == 'Q':
                event_handler.quit(self)
            # Enlarge subplot options
            elif event.key == 'e':
                event_handler.enlarge(self,event)
            # Annotation options
            elif event.key == 'a':
                event_handler.annotate(self,event)
            elif event.key == 'backspace':
                event_handler.delete_annotation(self,event)
            # Shift back in time
            elif event.key == 'left':
                event_handler.shift_time(self,-1*self.duration)
            # Shift forward in time
            elif event.key == 'right':
                event_handler.shift_time(self,self.duration)
            # Small shift back in time
            elif event.key == '<':
                event_handler.shift_time(self,-0.5*self.duration)
            # Small shift forward in time
            elif event.key == '>':
                event_handler.shift_time(self,0.5*self.duration)
            # Increase gain
            elif event.key == 'up':
                event_handler.change_gain(self,0.1)
            # Decrease gain
            elif event.key == 'down':
                event_handler.change_gain(self,-0.1)
            # Reset x range
            elif event.key == 'r':
                event_handler.change_time(self,self.plot_info['xlim_orig'])
            # Show entire x range
            elif event.key == 'x':
                event_handler.change_time(self,[0,self.t_max])
            # Reset the gain
            elif event.key == '0':
                event_handler.reset_gain(self)
            # Zoom in on the user requested portion
            elif event.key == 'z':
                event_handler.zoom_lines(self)
            
            #########################################
            #### Send results to other observers ####
            #########################################
            elif event.key == 'q':
                event_handler.quit_action(self)

class event_handler:

    def __init__(self):
        pass

    def button_response(self,event,line_container=None,line_color='r'):

        # Add zoom lines
        if event.button == 1:
            
            # Get the zoom line index
            zcnt     = self.plot_info['zoom_cntr']
            zcnt_mod = zcnt%2

            # Loop around the zoom lines as needed to we only have two at a time
            if zcnt >= 2:
                for iobj in self.plot_info['zoom_lines'][zcnt_mod]: iobj.remove()

            # Draw the zoom lines
            self.plot_info['zoom_lines'][zcnt_mod] = self.draw_zoom(event.xdata)

            # Save the positions
            self.plot_info['zlim'][zcnt_mod] = event.xdata

            # iterate the zoom count
            self.plot_info['zoom_cntr'] += 1

            # Draw the lines
            PLT.draw()


    #####################
    #### Key options ####
    #####################

    def quit(self):
        PLT.close("all")
        exit()

    def enlarge(self,event):

        for ikey in self.plot_info['axes'].keys():
            if event.inaxes == self.plot_info['axes'][ikey]:
                self.enlarged_plot(ikey)

    def annotate(self,event):

        # Read in the selection variables
        script_dir        = '/'.join(os.path.abspath(__file__).split('/')[:-3])
        self.annot_config = yaml.safe_load(open(f"{script_dir}/configs/annotation_config.yaml","r"))

        # Do some cleanup on the input options
        for ikey in self.annot_config.keys():self.annot_config[ikey] = dict(self.annot_config[ikey])
        for ikey in list(self.annot_config.keys()):
            self.annot_config[ikey.lower()] = self.annot_config.pop(ikey)

        # Print the options for the user.
        if len(self.plot_info['annots'].keys()) == 0:
            print("Available options:")
            for ikey in self.annot_config.keys():print(f"    - {ikey}")
            print("Inputs not in this list will be taken as your annotation.")

        # Query the user for information
        initial_options = list(self.annot_config.keys())
    
        # Setup auto-completion for the prompt
        options_completer = WordCompleter(initial_options)
        
        # Ask user to choose one of the options
        selected_option = prompt('Please choose an option (Q/q to quit): ', completer=options_completer)

        if selected_option.lower() != 'q':
            # Check for annotations in the config to give specific answers
            if selected_option.lower() in self.annot_config.keys():
                
                # get the current option list and display for user
                new_options       = list(self.annot_config[selected_option.lower()]['options'])
                print("Available options:")
                for ival in new_options:
                    print(f"    - {ival}")

                options_completer = WordCompleter(new_options)
                selected_value    = prompt('Please choose an option: ', completer=options_completer)
                annotation        = f"{selected_option}_{selected_value}"
            else:
                annotation = selected_option

            # Get the channel the user clicked on
            for ikey in self.plot_info['axes'].keys():
                if event.inaxes == self.plot_info['axes'][ikey]:
                    ichannel = ikey

            # Draw the annotation line at the event location
            self.draw_annotations(event.xdata,annotation,ichannel)
                
            # Redraw the plot to update the display
            PLT.draw()

    def delete_annotation(self,event):

        # Get the position of the click
        xpos = event.xdata
        for ikey in self.plot_info['axes'].keys():
            if event.inaxes == self.plot_info['axes'][ikey]:
                ichannel = ikey

        # Get the current x range so we can try to approximate the annotation
        xlim   = self.plot_info['axes'][self.first_chan].get_xlim()
        xrange = .025*(xlim[1]-xlim[0])

        # Look across the annotation labels to try and remove the selected entry
        lo = xpos-xrange
        hi = xpos+xrange
        for ikey in self.plot_info['annots']:
            if (ikey >= lo) & (ikey <=hi):
                annot_obj = self.plot_info['annots'][ikey]
                if ichannel == annot_obj[0]:
                    annot_obj[2].remove()
                    for iline in annot_obj[3]:iline.remove()
                    PLT.draw()
                    self.plot_info['annots'].pop(ikey)

    def shift_time(self,shiftvalue):
        current_xlim = self.plot_info['axes'][self.first_chan].get_xlim()
        current_xlim = [ival+shiftvalue for ival in current_xlim]
        self.change_time(current_xlim)

    def change_time(self,new_range):
        for ikey in self.plot_info['axes'].keys():
            self.plot_info['axes'][ikey].set_xlim(new_range)
        PLT.draw()
    
    def zoom_lines(self):
        if self.plot_info['zoom_cntr'] >= 2:
            new_range = np.sort(self.plot_info['zlim'])
            for iobj in self.plot_info['zoom_lines'][0]: iobj.remove()
            for iobj in self.plot_info['zoom_lines'][1]: iobj.remove()
            self.plot_info['zoom_cntr'] = 0

            for ikey in self.plot_info['axes'].keys():
                self.plot_info['axes'][ikey].set_xlim(new_range)
            PLT.draw()

    def change_gain(self,frac_change):
        for ikey in self.plot_info['axes'].keys():
            self.yscaling(ikey,frac_change)
        PLT.draw()

    def reset_gain(self):
        for ikey in self.plot_info['axes'].keys():
            self.plot_info['axes'][ikey].set_ylim(self.plot_info['ylim'][ikey])
        PLT.draw()

    def quit_action(self):
        print("Foo")