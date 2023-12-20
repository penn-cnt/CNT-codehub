import pickle
import argparse
import numpy as np
import pandas as PD
from os import path
from sys import exit
from itertools import product
import matplotlib.pyplot as PLT

# Sklearn imports
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score,roc_auc_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import SGDClassifier,LogisticRegression

# Keras imports
#from keras.optimizers import Adam
#from keras.models import Sequential
#from keras.utils import to_categorical
#from keras.layers import Dense,Dropout

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Build the MLP model
class MLPWithDropout(nn.Module):
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

def histogram_channel(channel,chname,plotdir,nbin=200):
    """
    Plot a histogram of the raw channel inputs. Helps to see how it is distributed.

    Args:
        channel (_type_): _description_
    """

    imin   = np.floor(np.log10(channel.min()))
    imax   = np.ceil(np.log10(channel.max()))
    logbin = np.logspace(imin,imax,nbin)
    
    fig = PLT.figure(dpi=100,figsize=(7.,7.))
    ax  = fig.add_subplot(111)
    ax.hist(channel,bins=logbin,edgecolor='k')
    ax.set_yscale('log')
    ax.set_xscale('log')
    fig.tight_layout()
    PLT.savefig(f"{plotdir}{chname}.png")
    PLT.close("all")

def SGD(X_train_normalized, X_test_normalized, y_train, y_test):

    # Create an SGD model
    sgd_model = SGDClassifier(verbose=True,random_state=42)

    # Train the model on the normalized training data
    sgd_model.fit(X_train_normalized, y_train)

    # Get the predicted values
    y_test_pred = sgd_model.predict(X_test_normalized)

    # Test the model on the normalized testing data
    accuracy = sgd_model.score(X_test_normalized, y_test)
    print(f"Accuracy on the test set: {accuracy:.2f}")

    # Perform cross-validation
    cv_scores = cross_val_score(sgd_model, X_train_normalized, y_train, cv=5)
    print("Cross-validation scores:", cv_scores)
    print(f"Mean cross-validation accuracy: {cv_scores.mean():.2f}")

    # Get the f1-scores
    y_naive   = np.ones(y_test.size)
    f1        = f1_score(y_test,y_test_pred,average='micro')
    f1_naive  = f1_score(y_test,y_naive,average='micro')
    auc       = roc_auc_score(y_test,y_test_pred,average='micro')
    auc_naive = roc_auc_score(y_test,y_naive,average='micro')
    print(f"Calculated f1-score: {f1}")
    print(f"Naive f1-score: {f1_naive}")
    print(f"Calculated AUC: {auc}")
    print(f"Naive AUC: {auc_naive}")

def logistic(X_train_normalized, X_test_normalized, y_train, y_test):

    # Create the logistic regression model
    LR_model = LogisticRegression(random_state=42)

    # Train the model on the normalized training data
    LR_model.fit(X_train_normalized, y_train)

    # Test the model on the normalized testing data
    accuracy = LR_model.score(X_test_normalized, y_test)
    print(f"Accuracy on the test set: {accuracy:.2f}")

    # Perform cross-validation
    cv_scores = cross_val_score(LR_model, X_train_normalized, y_train, cv=5)
    print("Cross-validation scores:", cv_scores)
    print(f"Mean cross-validation accuracy: {cv_scores.mean():.2f}")

def mlp(args, X_train_normalized, X_test_normalized, y_train, y_test, n1, n2, d1, d2, bval, verbose=True):

    # Tracking values
    loss_vals = []

    # Convert NumPy arrays to PyTorch tensors
    X_train_tensor = torch.tensor(X_train_normalized, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test_normalized, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

    # Create DataLoader for training and testing sets
    bsize         = int(bval)
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset  = TensorDataset(X_test_tensor, y_test_tensor)
    train_loader  = DataLoader(train_dataset, batch_size=bsize, shuffle=True)
    test_loader   = DataLoader(test_dataset, batch_size=bsize, shuffle=False)

    # Set the input size, hidden layer sizes, and output size
    input_size    = X_train.shape[1]
    hidden_size1  = int(n1*X_train_normalized.shape[1])
    hidden_size2  = int(n2*X_train_normalized.shape[1])
    dropout_rate1 = d1
    dropout_rate2 = d2
    output_size   = 1

    # Create the model, loss function, and optimizer
    model     = MLPWithDropout(input_size, hidden_size1, hidden_size2, output_size, dropout_rate1, dropout_rate2)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train the model
    print(f"Training: {n1},{n2},{d1},{d2},{bval}")
    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), labels)
            loss.backward()
            optimizer.step()

        if verbose:
            print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")
        loss_vals.append(loss.item())

    # Evaluate the model on the test set
    model.eval()
    with torch.no_grad():
        y_pred = model(X_test_tensor).squeeze().numpy()
        y_pred_binary = (y_pred > 0.5).astype(np.float32)
        accuracy = np.mean(y_pred_binary == y_test)

    # Get the f1-scores
    y_test_pred = y_pred_binary
    y_naive   = np.ones(y_test.size)
    f1        = f1_score(y_test,y_test_pred,average='weighted')
    auc       = roc_auc_score(y_test,y_test_pred,average='weighted')
    
    if verbose:
        print(f"Test Accuracy: {accuracy:.4f}")
        print(f"Calculated f1-score: {f1}")
        print(f"Calculated AUC: {auc}")

    # Generate and return the output object
    return (accuracy,auc,loss_vals)

if __name__ == '__main__':

    # Argument parsing
    parser = argparse.ArgumentParser(description="Simplified data merging tool.")
    parser.add_argument("--infile", type=str, help='Input file')
    parser.add_argument("--plotdir", type=str, help='Output plot directory')
    parser.add_argument("--datadir", type=str, help='Output data directory')

    scaled_group = parser.add_mutually_exclusive_group()
    scaled_group.add_argument("--create_scaled", help="Save the scaled dataset to this path.")
    scaled_group.add_argument("--load_scaled", help="Load the scaled data from this path.")
    args = parser.parse_args()

    # Ensure that the directory strings have proper trailing character
    if args.plotdir[-1] != '/':
        args.plotdir += '/'
    if args.datadir[-1] != '/':
        args.datadir += '/'

    # Handle exclusion options
    if args.create_scaled == None and args.load_scaled == None:
        args.create_scaled = 'scaled_data.pickle'

    spath = f"{args.datadir}{args.create_scaled}"
    if args.create_scaled != None:
        # Read in the data
        rawdata = PD.read_pickle(args.infile)
        rawdata = rawdata.loc[rawdata.target.isin([0,1])]

        # Break out the exogenous and endogenous portions
        raw_Y = rawdata.target.values
        raw_X = rawdata.drop(['target','annotation'],axis=1)

        tmp      = raw_X['FZ-CZ'].values
        goodinds = (tmp>0)
        raw_X    = raw_X.iloc[goodinds]
        raw_Y    = raw_Y[goodinds]

        # Try a log transformation on the data
        for icol in raw_X.columns:
            if icol != 'tag':
                print(icol,raw_X[icol].values.min(),raw_X[icol].values.max())
                raw_X[icol] = np.log10(raw_X[icol].values)
                
        # Apply train test split
        X_train, X_test, y_train, y_test = train_test_split(raw_X, raw_Y, test_size=0.33, random_state=42)

        # Normalize the input vectors using StandardScaler
        print("Scaling data. ")
        scaler = StandardScaler()
        X_train_normalized = scaler.fit_transform(X_train)
        X_test_normalized = scaler.transform(X_test)

        # Save the output
        pickle.dump((X_train_normalized,X_test_normalized,y_train,y_test),open(spath,"wb"))
    else:
        X_train_normalized, X_test_normalized, y_train, y_test = pickle.load(open(spath,"rb"))

    # Run a simple grid search
    #n1        = np.arange(0.5,1.75,.25)
    #n2        = np.arange(0.25,1.0,.25)
    n1        = np.array([1.25])
    n2        = np.array([0.5])
    d1        = np.array([0.2])
    d2        = np.array([0.2])
    bvals     = np.array([256])
    combos    = list(product(n1,n2,d1,d2,bvals))
    
    # Create output object
    columns   = ['n1','n2','d1','d2','bval','accuracy','auc']
    columns  += [f"{ival:02}" for ival in range(10)]
    output_df = PD.DataFrame(columns=columns)

    # Do the grid search
    outpath = f"{args.datadir}gridsearch.csv"
    if path.exists(outpath):
        output_df = PD.read_csv(outpath)

    for ival in combos:
        n1        = ival[0]
        n2        = ival[1]
        d1        = ival[2]
        d2        = ival[3]
        bval      = ival[4]
        check_df  = output_df.loc[(output_df.n1==n1)&(output_df.n2==n2)&(output_df.d1==d1)&(output_df.d2==d2)&(output_df.bval==bval)]

        if check_df.shape[0] == 0:
            acc, auc, loss = mlp(args, X_train_normalized, X_test_normalized, y_train, y_test, ival[0], ival[1], ival[2], ival[3], ival[4])
            new_row        = [ival[0],ival[1],ival[2],ival[3],ival[4],acc,auc]
            for jdx,jval in enumerate(loss):
                new_row.append(jval)
            iDF = PD.DataFrame([new_row],columns=columns)
            output_df = PD.concat((output_df,iDF),ignore_index=True)
            output_df.to_csv(outpath,index=False)
        else:
            print("Skipping:",ival)