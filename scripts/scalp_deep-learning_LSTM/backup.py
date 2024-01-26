import pickle
import argparse
import numpy as np
import pandas as PD
from sys import exit
from os import path,system
from itertools import product
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc   = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out, _ = self.lstm(x)
        out    = self.fc(out[:, -1, :])
        return out

class LSTM_prep:

    def __init__(self,vectors,input_path,timesteps):

        # Define some variables for the models
        self.vectors    = vectors
        self.input_path = input_path
        self.timesteps  = timesteps
        self.data       = []
        self.verbose    = True

    def run_LSTM(self):

        # Example usage
        input_size  = self.X_train.shape[-1]
        hidden_size = 128 
        num_layers  = 2
        num_classes = 2

        print(np.unique(self.y_train))
        exit()

        # Make target structure
        ytrain = self.y_train[:,0,:]
        ytest  = self.y_test[:,0,:]
        encoder = LabelBinarizer()
        encoder.fit(ytrain)
        ytrain = encoder.transform(ytrain)
        ytest  = encoder.transform(ytest)

        # Convert NumPy arrays to PyTorch tensors
        X_train_tensor = torch.tensor(self.X_train, dtype=torch.float32)
        X_test_tensor  = torch.tensor(self.X_test, dtype=torch.float32)
        y_train_tensor = torch.tensor(ytrain, dtype=torch.int8)
        y_test_tensor  = torch.tensor(ytest, dtype=torch.int8)

        print(X_train_tensor.shape)
        print(y_train_tensor.shape)
        
        # Create DataLoader for training and testing sets
        bsize         = 64
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        test_dataset  = TensorDataset(X_test_tensor, y_test_tensor)
        train_loader  = DataLoader(train_dataset, batch_size=bsize, shuffle=True)
        test_loader   = DataLoader(test_dataset, batch_size=bsize, shuffle=False)

        model     = LSTMModel(input_size, hidden_size, num_layers, num_classes)
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        # Train the model
        num_epochs = 5
        for epoch in range(num_epochs):
            model.train()
            for inputs, labels in train_loader:
                optimizer.zero_grad()
                outputs = model(inputs)
                loss    = criterion(outputs.squeeze(), labels)
                loss.backward()
                optimizer.step()

        # Evaluate the model on the test set
        model.eval()
        with torch.no_grad():
            y_pred = model(X_test_tensor).squeeze().numpy()
            y_pred_binary = (y_pred > 0.5).astype(np.float32)
            accuracy = np.mean(y_pred_binary == y_test)
        print(f"Test Accuracy: {accuracy:.4f}")

    def data_split(self):

        # Get the endo/exo cols
        X_cols = []
        for ival in self.columns:
            if ival!='target':
                X_cols.append(True)
            else:
                X_cols.append(False)
        Y_cols = np.invert(X_cols)

        # Split the data into train and test
        X_train, X_test, y_train, y_test = train_test_split(self.data[:,:,X_cols], self.data[:,:,Y_cols], test_size=0.33, random_state=42)
        print(np.unique(y_train))

        return X_train,X_test,y_train,y_test

    def make_inputs(self):
        # Prepare the data
        self.pad_times()
        self.drop_missing_times()
        self.make_arrays()

    def load_inputs(self):
        print("Loading inputs.")
        self.X_train,self.X_test,self.y_train,self.y_test = pickle.load(open(self.input_path,'rb'))

    def pad_times(self):

        # Get the times
        time_array = np.array([])
        keys       = list(self.vectors.keys())
        for ikey in keys:
            time_array = np.concatenate((time_array,self.vectors[ikey].index))
        self.time_array = np.sort(np.unique(time_array))
        
        # Zero pad the ending
        for ikey in keys:
            max_time     = self.vectors[ikey].index.max()
            append_times = self.time_array[(self.time_array>max_time)]
            if (append_times.size>0):
                for itime in append_times:
                    self.vectors[ikey].loc[itime] = 0 

    def drop_missing_times(self):

        self.good_vectors = {}
        keys              = list(self.vectors.keys())
        for ikey in keys:
            current_times = np.array(list(self.vectors[ikey].index))
            if current_times.size < self.time_array.size:
                pass
            else:
                self.good_vectors[ikey] = self.vectors[ikey]

    def make_arrays(self):

        # Make sure timesteps are valid
        if self.timesteps == 0:
            raise ValueError("Please provide an odd number of timesteps greater than 0.")
        if (self.timesteps%2==0):
            self.timesteps -= 1
            raise Warning(f"Even number of timesteps provided. Using timesteps-1={self.timesteps}")
        
        # Loop over the vector keys to make the right shapes
        keys = list(self.good_vectors.keys())
        for ikey in keys:

            # Get the current array
            iDF         = self.good_vectors[ikey]
            iDF['file'] = ikey
            arr         = iDF.values

            # From the number of timesteps, set the dimensions
            outer_dim = arr.shape[0]-self.timesteps+1
            offset    = int((self.timesteps-1)/2)

            # user information
            if self.verbose:
                print(f"Original data dimensions: {arr.shape}. New dimensions: {(outer_dim,self.timesteps,arr.shape[-1])}.")
            
            new_arr = np.zeros((outer_dim,self.timesteps,arr.shape[-1]))
            for idx in range(offset,arr.shape[0]-offset):
                new_arr[idx-offset] = arr[idx-offset:idx+offset+1]

            # Save the new results to the input list
            self.data.append(new_arr)
        self.data = np.concatenate(self.data)
        
        # Make the column vector
        self.columns = list(self.good_vectors[ikey].columns)

        # Split data
        self.X_train,self.X_test,self.y_train,self.y_test = self.data_split()

        # Save the results
        pickle.dump((self.X_train,self.X_test,self.y_train,self.y_test),open(self.input_path,"wb"))

class data_prep:

    def __init__(self,infile):
        self.DF    = PD.read_pickle(infile)
        self.t_min = 100
        self.t_max = 2000 
    
    def channel_prep(self):
        
        # Get the base channel names
        skip          = ['file','t_start','tag','uid','target','annotation']
        self.channels = np.setdiff1d(self.DF.columns,skip)

        # Create the LSTM feature channel names (channel+tag)
        tags                = self.DF['tag'].unique()
        self.feature_combos = list(product(self.channels,tags))
        self.feature_labels = [f"{ival[0]}_{ival[1]}" for ival in self.feature_combos]

    def make_input_vectors(self):

        # Do the initial target cut here to reduce memory volume
        inds    = (self.DF['target'].values <= 1)
        self.DF = self.DF.iloc[inds].reset_index(drop=True) 

        # Make the dataset container
        self.vectors = {}

        # Outer loop 
        file_arr = self.DF['file'].values 
        ufiles   = np.unique(file_arr)
        cols     = np.concatenate((['t_start'],self.channels)) 
        for ifile in ufiles:

            # Make the current dataset object
            dataset      = PD.DataFrame()
            dataset_list = []
            
            # Get the dataslice by index
            iDF = self.DF.iloc[(file_arr==ifile)]

            # Go through the slice to populate the new values
            tag_arr = iDF['tag'].values
            utags   = np.unique(tag_arr)
            for idx,itag in enumerate(utags):

                # Get the correct dataslice
                jDF = iDF.iloc[(tag_arr==itag)]

                # Channel slice
                channel_data            = jDF[self.channels].copy()
                channel_data.columns    = [f"{ichan}_{itag}" for ichan in channel_data.columns]
                channel_data['t_start'] = jDF['t_start'].values
                channel_data.set_index('t_start',inplace=True)
                
                # Store for joining
                dataset_list.append(channel_data)

            # Due to overhead, direct join is slow. Onetime concat then groupby produces outer join faster.
            dataset = PD.concat(dataset_list).groupby(level=0).max()
            
            # Add in the target. Target is the same for entire dataset, one time entry will suffice.
            dataset['target'] = jDF['target'].values[0]

            # Save output to the trainging vectors if within tolerance
            if dataset.index.max() >= self.t_min and dataset.index.max() <= self.t_max:
                self.vectors[ifile] = dataset
        return self.vectors

if __name__ == '__main__':

    # Argument parsing
    parser = argparse.ArgumentParser(description="Simplified data merging tool.")
    parser.add_argument("--rawfile", required=True, type=str, help='Raw Input data file')
    parser.add_argument("--vectorfile", required=True, type=str, help="Vector file. If it exits, use and skip restructuring. Otherwise, save to this location")
    parser.add_argument("--modelinput", required=True, type=str, help="Vector file. If it exits, use and skip restructuring/vectorizing. Otherwise, save to this location")
    args = parser.parse_args()

    # Data prep
    if not path.exists(args.modelinput):
        if not path.exists(args.vectorfile):
            # Make sure we have all the required inputs from the user
            if args.rawfile == None:
                raise FileNotFoundError("Please provide rawfile")
            
            # Make the training vectors
            DP = data_prep(args.rawfile)
            DP.channel_prep()
            vectors = DP.make_input_vectors()

            # Make sure we have a place to write data to
            outdir = '/'.join(args.vectorfile.split('/')[:-1])
            if not path.exists(outdir):
                system(f"mkdir -p {outdir}")

            # Save the result
            pickle.dump(vectors,open(args.vectorfile,'wb'))
        else:
            print('Reading vectors in.')
            vectors = pickle.load(open(args.vectorfile,'rb'))

        # Start the modeling
        LSTM_handler = LSTM_prep(vectors,args.modelinput,5)
        LSTM_handler.make_inputs()
    else:
        vectors      = pickle.load(open(args.vectorfile,'rb'))
        LSTM_handler = LSTM_prep(vectors,args.modelinput,5)
        LSTM_handler.load_inputs()

    LSTM_handler.run_LSTM()