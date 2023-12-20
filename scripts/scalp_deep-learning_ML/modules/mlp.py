import numpy as np

# Sklearn loaders
from sklearn.model_selection import GridSearchCV

# Torch loaders
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

class mlp_network(nn.Module):

    def __init__(self, input_size, hidden_size1, hidden_size2, output_size, dropout_rate1, dropout_rate2):
        super(MLPWithDropout, self).__init__()
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
    
class mlp_handler:

    def __init__(self, X_train, X_test, Y_train, Y_test):
        
        # Convert NumPy arrays to PyTorch tensors
        self.X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        self.y_train_tensor = torch.tensor(Y_train, dtype=torch.float32)
        self.X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
        self.y_test_tensor = torch.tensor(Y_test, dtype=torch.float32)

        # Make the dataset objects
        self.train_dataset = TensorDataset(self.X_train_tensor, self.y_train_tensor)
        self.test_dataset  = TensorDataset(self.X_test_tensor, self.y_test_tensor)
        self.train_loader  = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
        self.test_loader   = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)