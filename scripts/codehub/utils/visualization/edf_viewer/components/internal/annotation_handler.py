import os
import yaml
import tkinter as tk
from tkinter import messagebox

class annotation_handler:

    def __init__(self,root,width=450,height=300):

        # Create the tkinter object
        self.root = root
        self.root.title("Annotation Widget")
        self.root.geometry(f"{width}x{height}")

        # Read in the selection variables
        script_dir        = '/'.join(os.path.abspath(__file__).split('/')[:-3])
        self.annot_config = yaml.safe_load(open(f"{script_dir}/configs/annotation_config.yaml","r"))

        # Use the annotation config to make the inputs
        self.selection_vars = {ikey:tk.StringVar() for ikey in self.annot_config.keys()}

        # Make an object to break us out of waiting for user input
        self.submitted = False

    def workflow(self):

        # Create the row widgets from annotation config
        for current_key in self.annot_config.keys():
            self.create_widgets(current_key)

        # Make the user input widget
        self.create_user_input()

        # Add the submit and reset buttons
        self.create_submit_reset()

    def reset_action(self):
        for var in self.selection_vars.values():
            var.set('')
        self.user_input_entry.delete(0, tk.END)

    def submit_action(self):
        self.selections         = {label: var.get() for label, var in self.selection_vars.items()}
        self.selections['user'] = self.user_input_entry.get()
        self.root.destroy()
        self.submitted = True

    def create_widgets(self, current_key):

        row_frame = tk.Frame(self.root)
        row_frame.pack(anchor=tk.W, pady=5)

        label = tk.Label(row_frame, text=current_key, width=15)
        label.pack(side=tk.LEFT)

        # Create buttons
        if self.annot_config[current_key]['type'] == 'radio':
            for ivalue in self.annot_config[current_key]['options']:
                tk.Radiobutton(row_frame, text=ivalue, variable=self.selection_vars[current_key], value=ivalue,indicatoron=False).pack(side=tk.LEFT, padx=5)

    def create_user_input(self):

        # Row for user provided text input (on the same line)
        user_frame = tk.Frame(self.root)
        user_frame.pack(anchor=tk.W, pady=5)

        label_user = tk.Label(user_frame, text="User Provided", width=15)
        label_user.pack(side=tk.LEFT)

        self.user_input_entry = tk.Entry(user_frame, width=30)
        self.user_input_entry.pack(side=tk.LEFT, padx=5)

    def create_submit_reset(self):

        # Row for Submit and Reset buttons
        buttons_frame = tk.Frame(self.root)
        buttons_frame.pack(pady=20)

        submit_button = tk.Button(buttons_frame, text="Submit", command=self.submit_action)
        submit_button.pack(side=tk.LEFT, padx=10)

        reset_button = tk.Button(buttons_frame, text="Reset", command=self.reset_action)
        reset_button.pack(side=tk.LEFT, padx=10)

def annot_main(return_dict):

    #root = tk.Toplevel()
    #input_window = annotation_handler(root)
    #input_window.workflow()
    #root.mainloop()

    #return_dict = input_window.selections
    pass

# Main function just for testing
if __name__ == '__main__':
    results = annot_main()