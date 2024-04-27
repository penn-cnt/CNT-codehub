# Set the random seed
import random as rnd
rnd.seed(42)

import os
import pickle
import argparse
import numpy as np
import pandas as PD
import pylab as PLT
from sys import argv
from tqdm import tqdm
import seaborn as sns
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler,StandardScaler,RobustScaler

# Torch loaders
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

class mlp_network(nn.Module):

    def __init__(self, input_size, hidden_size1, hidden_size2, output_size, dropout_rate1, dropout_rate2):
        super(mlp_network, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(p=dropout_rate1)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(p=dropout_rate2)
        self.fc3 = nn.Linear(hidden_size2, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        x = self.sigmoid(x)
        return x
    
class mlp:

    def __init__(self,vectors,holdout):
        self.incols  = vectors.columns
        self.vectors = vectors
        self.X_cols  = self.incols[self.incols!='target']
        self.Y_cols  = self.incols[self.incols=='target']
        self.X       = vectors[self.X_cols].values
        self.Y_flat  = vectors[self.Y_cols].values.flatten()
        self.hold_X  = holdout[self.X_cols].values
        self.hold_Y  = holdout[self.Y_cols].values

        # Encode the Y vectors
        self.Y                                = np.zeros((self.Y_flat.shape[0],2))
        self.Y[np.where(self.Y_flat==0)[0],0] = 1
        self.Y[np.where(self.Y_flat==1)[0],1] = 1
    
    def data_scale(self,stype='minmax'):

        if stype.lower() not in ['minmax','standard','robust']:
            print("Could not find {stype} scaler. Defaulting to Standard")
            stype = 'standard'

        if stype.lower() == 'minmax':
            scaler = MinMaxScaler()
        elif stype.lower() == 'standard':
            scaler = StandardScaler()
        elif stype.lower() == 'robust':
            scaler = RobustScaler()

        scaler.fit(self.X)
        self.X_scaled = scaler.transform(self.X)

        # Get the holdout fit
        self.hold_X_scaled = scaler.transform(self.hold_X)

    def data_split(self):

        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(self.X_scaled, self.Y, stratify=self.Y, test_size=0.3)

    def run_network(self):

        # Set the input size, hidden layer sizes, and output size
        input_size    = self.X_train.shape[1]
        hidden_size1  = int(0.6*self.X_train.shape[1])
        hidden_size2  = int(0.3*self.X_train.shape[1])
        dropout_rate1 = 0.6
        dropout_rate2 = 0.3
        output_size   = 2

        # Convert NumPy arrays to PyTorch tensors
        self.X_train_tensor = torch.tensor(self.X_train, dtype=torch.float32)
        self.y_train_tensor = torch.tensor(self.Y_train, dtype=torch.float32)
        self.X_test_tensor = torch.tensor(self.X_test, dtype=torch.float32)
        self.y_test_tensor = torch.tensor(self.Y_test, dtype=torch.float32)

        # Make the dataset objects
        self.train_dataset = TensorDataset(self.X_train_tensor, self.y_train_tensor)
        self.test_dataset  = TensorDataset(self.X_test_tensor, self.y_test_tensor)
        self.train_loader  = DataLoader(self.train_dataset, batch_size=32, shuffle=True)
        self.test_loader   = DataLoader(self.test_dataset, batch_size=32, shuffle=False)

        # Create the model, loss function, and optimizer
        model     = mlp_network(input_size, hidden_size1, hidden_size2, output_size, dropout_rate1, dropout_rate2)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        # Tracking values
        loss_vals = []

        # Train the model
        num_epochs = 25
        for epoch in range(num_epochs):
            model.train()
            for inputs, labels in self.train_loader:
                optimizer.zero_grad()
                outputs = model(inputs)
                loss    = criterion(outputs.squeeze(), labels)
                loss.backward()
                optimizer.step()

            print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")
            loss_vals.append(loss.item())

        # Evaluate the model on the test set
        model.eval()
        with torch.no_grad():
            y_pred        = model(self.X_test_tensor).squeeze().numpy()
            y_pred_binary = (y_pred > 0.5).astype(np.float32)
            accuracy = np.mean(y_pred_binary == self.Y_test)
        print(accuracy)
        pickle.dump((self.Y_test,y_pred,y_pred_binary),open("yvals.pickle","wb"))
        pickle.dump(loss,open("loss.pickle","wb"))