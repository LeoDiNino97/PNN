"""
Run experiments for PCA+MLP readout on real data
"""

import torch
from data.data_loading import load_dataset
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error
from utils.utils import perform_pca

iterations = 5
dset = 'adni2' # adni2 or abide
hidden_sizes = [(32)] 

pca_metrics = []

for it in range(iterations):

    Xtrain, ytrain, Xval, yval, Xtest, ytest, task_level, F_out, task, metric = \
        load_dataset(dset, seed=it)

    K = Xtrain.shape[1]
    X_train_np, pca = perform_pca(Xtrain.numpy(), K=K)
    X_val_np = perform_pca(Xval.numpy(), K=K, pca=pca)[0].reshape((Xval.shape[0],-1))
    X_test_np = perform_pca(Xtest.numpy(), K=K, pca=pca)[0].reshape((Xtest.shape[0],-1))
    X_train_np = X_train_np.reshape((Xtrain.shape[0],-1))
    
    mlp = MLPRegressor(hidden_layer_sizes=hidden_sizes, max_iter=200, learning_rate_init=0.01)  

    # Fit the model
    mlp.fit(X_train_np, ytrain)
    y_val_pred = mlp.predict(X_val_np)
    y_test_pred = mlp.predict(X_test_np)
    pca_metrics.append(mean_absolute_error(ytest, y_test_pred))

print(f"MAE {torch.tensor(pca_metrics).mean()} +- {torch.tensor(pca_metrics).std()}")

