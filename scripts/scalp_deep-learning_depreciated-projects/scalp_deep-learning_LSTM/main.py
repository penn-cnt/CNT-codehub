import pickle
import argparse
import numpy as np
import pandas as PD
from sys import exit
from os import path,system
from itertools import product
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder,MinMaxScaler

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import TensorDataset, DataLoader
from torch.optim.lr_scheduler import _LRScheduler

class CyclicLR(_LRScheduler):
    
    def __init__(self, optimizer, schedule, last_epoch=-1):
        assert callable(schedule)
        self.schedule = schedule
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        return [self.schedule(self.last_epoch, lr) for lr in self.base_lrs]

class LSTMClassifier(nn.Module):
    """Very simple implementation of LSTM-based time-series classifier."""
    
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim  = layer_dim
        self.rnn        = nn.LSTM(input_dim, hidden_dim, layer_dim, dropout=0.2 ,batch_first=True)
        self.drop       = nn.Dropout(p=0.2)
        self.fc         = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        h0, c0        = self.init_hidden(x)
        out, (hn, cn) = self.rnn(x, (h0, c0))
        #out           = self.drop(out)
        out           = self.fc(out[:, -1, :])
        return out
    
    def init_hidden(self, x):
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim)
        c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim)
        return h0, c0

class data_prep:

    def __init__(self,infile):
        self.DF    = PD.read_pickle(infile)
        self.t_min = 100
        self.t_max = 2000 

    def channel_prep(self):
        """
        Define some variables for channel referencing.
        """

        # Get the base channel names
        skip          = ['file','t_start','tag','uid','target','annotation']
        self.channels = np.setdiff1d(self.DF.columns,skip)

        # Create the LSTM feature channel names (channel+tag)
        tags                = self.DF['tag'].unique()
        self.feature_combos = list(product(self.channels,tags))
        self.feature_labels = [f"{ival[0]}_{ival[1]}" for ival in self.feature_combos]

    def target_criteria(self):
        """
        Make cuts on the dataframe based on targets
        """

        # Cut by target list
        targets = [0,1]
        self.DF = self.DF[self.DF['target'].isin(targets)].reset_index(drop=True)

    def min_max_times(self):

        # Get the list of file times greater than min
        groups     = self.DF.groupby(['file'])['t_start'].max()
        times      = groups.values 
        inds       = (times>=self.t_min)
        white_list = np.array(list(groups.index))[inds]

        # Update the dataframe with the whitelist of files
        self.DF = self.DF[self.DF['file'].isin(white_list)].reset_index(drop=True)

        # Clip at the max time
        self.DF = self.DF.loc[self.DF['t_start']<=self.t_max]
        
        # Clean up the order
        self.DF = self.DF.sort_values(by=['file','tag','t_start']).reset_index(drop=True)

    def drop_times(self):

        # Get a list of all times so we know the values to pad
        t_vals = self.DF['t_start'].unique()

        # Group by file and tag
        groupby   = self.DF.groupby(['file','tag'])['t_start']
        max_times = groupby.max()
        cnts      = groupby.nunique()

        # Loop through the indices to see if we need to drop
        bad_files = []
        for index in max_times.index:
            itime = max_times.loc[index]
            ntime = (t_vals<=itime).sum()
            if cnts.loc[index] != ntime:
                bad_files.append(index[0])

        # If any bad files were found, drop them
        if len(bad_files) > 0:
            files      = self.DF['file'].unique()
            white_list = np.setdiff1d(files,np.unique(bad_files))
            self.DF    = self.DF[self.DF['file'].isin(white_list)].reset_index(drop=True)

    def pad_times(self):

        # Get a list of all times so we know the values to pad
        t_vals = self.DF['t_start'].unique()

        # Group by file and tag
        grpby     = self.DF.groupby(['file','tag'])
        max_times = grpby['t_start'].max()
        groups    = grpby[['uid','target','annotation']].max()
        
        # Loop through the group indices to make the new value to pad with
        output = []
        for index in groups.index:

            # get the current max time and see if we need to pad
            itime = max_times.loc[index]
            if itime < self.t_max:

                # Make a blank dataframe to populate
                iDF = PD.DataFrame(columns=self.DF.columns)

                # get the values that need population
                file       = index[0]
                tag        = index[1] 
                uid        = groups.loc[index].uid
                target     = groups.loc[index].target
                annotation = groups.loc[index].annotation
                pad_times  = t_vals[(t_vals>itime)]

                # Add values to the new temporary dataframe
                iDF['file']        = np.tile(file,pad_times.size)
                iDF['t_start']     = pad_times
                iDF['tag']         = tag
                iDF['uid']         = uid
                iDF['target']      = target
                iDF['annotation']  = annotation
                iDF[self.channels] = 0

                # Concatenate to the main dataframe
                output.append(iDF)            

        # Make the new dataframe
        output.append(self.DF)
        self.DF = PD.concat(output)
        self.DF = self.DF.sort_values(by=['file','tag','t_start']).reset_index(drop=True).drop_duplicates(ignore_index=True)

    def scale_data(self):

        # Scale the channel data
        encoder = MinMaxScaler()
        self.DF[self.channels] = encoder.fit_transform(self.DF[self.channels])

    def create_inputs(self, drop_cols, group_cols=['file','tag']):

        # Get the targets out
        targets = self.DF['target']

        # Make the input vectors
        X_list = []
        Y_list = []
        for _, group in self.DF.groupby(group_cols):
            Y_list.append(group['target'].values[0])
            X_list.append(group.drop(columns=drop_cols).values[None])
        columns   = group.drop(columns=drop_cols).columns
        X_vectors = np.row_stack(X_list)
        Y_vectors = np.array(Y_list).reshape((-1,1))

        # Binarize the targets
        encoder   = OneHotEncoder()
        Y_vectors = encoder.fit_transform(Y_vectors).toarray()

        # Create the train/test splits
        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(X_vectors, Y_vectors, test_size=0.3)

        # Make the tensors
        X_train, X_test = [torch.tensor(arr, dtype=torch.float32) for arr in (self.X_train, self.X_test)]
        Y_train, Y_test = [torch.tensor(arr, dtype=torch.float16) for arr in (self.Y_train, self.Y_test)]
        self.train_ds   = TensorDataset(X_train, Y_train)
        self.test_ds    = TensorDataset(X_test, Y_test)

    def create_loaders(self, bs=512, jobs=1):
        self.train_dl = DataLoader(self.train_ds, bs, shuffle=True, num_workers=jobs)
        self.test_dl  = DataLoader(self.test_ds, bs, shuffle=False, num_workers=jobs)

    def cosine(self, t_max, eta_min=0):
    
        def scheduler(epoch, base_lr):
            t = epoch % t_max
            return eta_min + (base_lr - eta_min)*(1 + np.cos(np.pi*t/t_max))/2
        
        return scheduler

    def start_training(self):

        input_dim            = self.X_train.shape[-1]
        hidden_dim           = 2*input_dim
        layer_dim            = 4
        output_dim           = 2
        lr                   = 0.0005
        n_epochs             = 25
        iterations_per_epoch = len(self.train_dl)
        best_acc             = 0
        patience, trials     = 100, 0

        model     = LSTMClassifier(input_dim, hidden_dim, layer_dim, output_dim)
        criterion = nn.CrossEntropyLoss()
        opt       = torch.optim.RMSprop(model.parameters(), lr=lr)
        sched     = CyclicLR(opt, self.cosine(t_max=iterations_per_epoch * 2, eta_min=lr/100))

        print('Start model training')
        for epoch in range(1, n_epochs + 1):
            
            for i, (x_batch, y_batch) in enumerate(self.train_dl):
                model.train()
                sched.step()
                opt.zero_grad()
                out = model(x_batch)
                loss = criterion(out, y_batch)
                loss.backward()
                opt.step()
            
            # Calculate the test accuracy
            model.eval()
            correct, total = 0, 0
            for x_val, y_val in self.test_dl:
                out      = model(x_val)
                preds    = F.log_softmax(out, dim=1).argmax(dim=1)
                total   += y_val.size(0)
                correct += (preds == y_val.argmax(dim=1)).sum().item()
            acc = correct / total

            # Update the user on the current progress
            if epoch % 5 == 0:
                print(f'Epoch: {epoch:3d}. Loss: {loss.item():.4f}. Acc.: {acc:2.2%}')

            # Save the best results so far
            if acc > best_acc:
                trials   = 0
                best_acc = acc
                torch.save(model.state_dict(), 'best.pth')
                print(f'Epoch {epoch} best model saved with accuracy: {best_acc:2.2%}')
            else:
                trials += 1
                if trials >= patience:
                    print(f'Early stopping on epoch {epoch}')
                    break
        
if __name__ == '__main__':

    # Argument parsing
    parser = argparse.ArgumentParser(description="Simplified data merging tool.")
    parser.add_argument("--rawfile", required=True, type=str, help='Raw Input data file')
    args = parser.parse_args()

    # Make the training vectors
    DP = data_prep(args.rawfile)
    DP.channel_prep()
    DP.target_criteria()
    DP.min_max_times()
    DP.drop_times()
    DP.pad_times()
    DP.scale_data()
    DP.create_inputs(drop_cols=['t_start','uid','target','annotation'])
    DP.create_loaders(bs=128)
    DP.start_training()