import os
import yaml
from sys import exit
import tkinter as tk
from tkinter import ttk

# Local imports
import pipeline_manager as PM

class pipegui:

    def __init__(self,args,allowed_dict,help):
        self.args         = args
        self.keys         = list(args.keys())
        self.allowed_dict = allowed_dict
        self.help         = help

    def update_text_widget(self,selected_item):

        # Get and clean up the help string
        newtext = self.help[selected_item]
        newtext = newtext.lstrip('R|')

        # Update the text object
        self.text_widget.delete("1.0", tk.END)  # Clear previous text
        self.text_widget.insert(tk.END, newtext)

    def main(self):

        # Create the main window
        root = tk.Tk()
        root.title("Pipeline Manager GUI")

        # Window sizing
        window_width  = 1600
        window_height = 900
        root.geometry(f"{window_width}x{window_height}")

        # Create a frame for the left side (containing widgets)
        left_frame = ttk.Frame(root, padding=10)
        left_frame.grid(row=0, column=0, sticky="nsew")

        # Create a canvas for the right side (containing text widget)
        canvas = tk.Canvas(root, borderwidth=0, background="white")
        canvas.grid(row=0, column=1, sticky="nsew")

        # Configure the grid layout
        root.grid_rowconfigure(0, weight=1)
        root.grid_columnconfigure(0, weight=1)
        root.grid_columnconfigure(1, weight=2)

        # Create a vertical separator line
        separator = ttk.Separator(root, orient="vertical")
        separator.grid(row=0, column=2, sticky="ns", padx=5)

        # Make the labels
        for idy,ikey in enumerate(self.keys):
            label=tk.Label(left_frame,text=f"{ikey}")
            label.grid(row=idy, column=0, padx=5, pady=3, sticky="ew")

        # Make the drop-down widgets
        dropdown_dict = {}
        for idy,ikey in enumerate(self.keys):
            try:
                values = allowed_dict[ikey]
            except KeyError:
                values = ['a','b']

            dropdown_dict[ikey] = tk.StringVar()
            dropdown = ttk.Combobox(left_frame, textvariable=dropdown_dict[ikey], values=values, width=40)
            dropdown.grid(row=idy, column=1, padx=5, pady=3, sticky="ew")

        # Make some help buttons
        for idy,ikey in enumerate(self.keys):
            button = tk.Button(left_frame, text="Show Help", command=lambda x=ikey: self.update_text_widget(x))
            button.grid(row=idy, column=2, padx=5, pady=3, sticky="ew")

        # Text widget on the right side
        self.text_widget = tk.Text(canvas, wrap="word")
        self.text_widget.pack(padx=10, pady=10)

        # Run the GUI
        root.mainloop()

if __name__ == '__main__':

    # Get the pathing to the pipeline manager. Allows us to find the argument file
    pipeline_path = os.path.dirname(os.path.abspath(PM.__file__))+'/'

    # Get the arguments
    args, metadata = PM.argument_handler(argument_dir=pipeline_path,require_flag=False)
    help           = metadata[0]
    args           = vars(args)

    # Read in the allowed arguments
    allowed_dict = {}
    raw_args     = yaml.safe_load(open(f"{pipeline_path}allowed_arguments.yaml","r"))
    for ikey in raw_args.keys():
        exec(f"allowed_dict['{ikey}']={raw_args[ikey]}", globals())

    # Create the GUI
    PG = pipegui(args,allowed_dict,help)
    PG.main()