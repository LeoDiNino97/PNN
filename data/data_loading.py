import torch
import numpy as np
from utils import *
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_dataset(name, seed=42):

    metric = None
    if name.startswith("adni"):
        X = np.load(f"data/datasets/adni/X_{name}_age.npy") # Shape T x N
        y = np.load(f"data/datasets/adni/y_{name}_age.npy")

        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        X = torch.FloatTensor(X)
        y = torch.FloatTensor(y)

        Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.4, random_state=42)
        Xval, Xtest, yval, ytest = train_test_split(Xtest, ytest, test_size=0.2, random_state=42)
    
        Xtrain = Xtrain.unsqueeze(2)
        Xval = Xval.unsqueeze(2)
        Xtest = Xtest.unsqueeze(2)

        task = "regression"
        task_level = "graph"
        metric = "mae"
        F_out = 1

    elif name == "abide":
        X = np.load(f"data/datasets/abide/X_age.npy") # Shape T x N
        y = np.load(f"data/datasets/abide/y_age.npy")

        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        X = torch.FloatTensor(X)
        y = torch.FloatTensor(y)

        Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.4, random_state=42)
        Xval, Xtest, yval, ytest = train_test_split(Xtest, ytest, test_size=0.2, random_state=42)

        Xtrain = Xtrain.unsqueeze(2)
        Xval = Xval.unsqueeze(2)
        Xtest = Xtest.unsqueeze(2)

        task = "regression"
        task_level = "graph"
        metric = "mae"
        F_out = 1

    else:
        raise NotImplementedError(f"Dataset {name} not available")
    
    return Xtrain, ytrain, Xval, yval, Xtest, ytest, task_level, F_out, task, metric
