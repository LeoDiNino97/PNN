"""
Train and evaluate a PNN on the synthetic dataset for
the specifiec hyperparameters. The results are 
returned by the function.
"""

import numpy as np
import torch 
from torch import nn, optim
from models.vnn import VNN as CovNN
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from utils.utils import (compute_covariance_and_precision, compute_metric, 
                         compute_loss, graphical_lasso_with_reg, soft_threshold)


def train_pnn(X, y, hidden_sizes, K, dropout, gamma_max, 
              lambda_0, eta, batch_norm, nEpochs, schedule_gamma,
            iterations=1, prec_type="sample", true_Prec=None):

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    true_Prec = torch.FloatTensor(true_Prec)
    all_test_acc, all_test_loss = [], []
    theta_mae, theta_sparsity = [], []
    
    # Parameters of the experiment
    it_H, it_Theta_tilde, it_Theta = 20, 20, 20
    task_level = "graph"
    F_out = 1
    task = "regression"
    metric = "mae"    
    bias = True
    lr = 0.01
    gamma_min = 0.1
    lambda_ = lambda_0
    hidden_sizes, hidden_mlp_sizes = hidden_sizes


    for it in tqdm(range(iterations)):

        # Split dataset 
        Xtrain, Xvaltest, ytrain, yvaltest = train_test_split(X, y, test_size=0.4, random_state=it)
        Xval, Xtest, yval, ytest = train_test_split(Xvaltest, yvaltest, test_size=0.5, random_state=it)
        
        Xtrain, Xval, Xtest = torch.FloatTensor(Xtrain).to(device), torch.FloatTensor(Xval).to(device), torch.FloatTensor(Xtest).to(device)
        if task == 'classification':
            ytrain, yval, ytest = torch.LongTensor(ytrain).to(device), torch.LongTensor(yval).to(device), torch.LongTensor(ytest).to(device)
        elif task == 'regression':
            ytrain, yval, ytest = torch.FloatTensor(ytrain).to(device), torch.FloatTensor(yval).to(device), torch.FloatTensor(ytest).to(device)

        if len(Xtrain.shape) == 2:
            Xtrain = Xtrain.unsqueeze(-1)
            Xval = Xval.unsqueeze(-1)
            Xtest = Xtest.unsqueeze(-1)
        
        _, N, F_in = Xtrain.shape

        # Initialize precision and covariance matrixes
        sampleC, Prec = compute_covariance_and_precision(Xtrain, prec_type, lambda_, eta)
        sampleC, Prec = sampleC.to(device), Prec.to(device)
        
        if prec_type in ["joint", "naive"]:
            Theta_tilde = nn.Parameter(torch.eye(N, device=device), requires_grad=True)
            optimizer_Theta_tilde = optim.SGD([Theta_tilde], lr=lr, weight_decay=0.0)
        else:
            Theta_tilde = Prec

        Theta = Prec

        # Initialize model, loss and optimizers
        VNN = CovNN(          
                    input_size=F_in, 
                    output_size=F_out,
                    hidden_sizes=hidden_sizes, 
                    hidden_mlp_sizes=hidden_mlp_sizes,
                    K=K, 
                    bias=bias, 
                    dropout=dropout,
                    task_level=task_level,
                    node_readout="mean",
                    use_batch_norms=batch_norm,
                    use_layer_norms=False,
                    task=task,
                ).to(device)
        

        if task == "regression":
            Loss = nn.MSELoss()
            MAE = nn.L1Loss()
        elif task == "classification":
            Loss = nn.NLLLoss()

        optimizer = optim.Adam(VNN.parameters(), lr=lr, weight_decay=0.001)

        batchSize = 1000000
        nTrainBatches = int(np.ceil(Xtrain.shape[0] / batchSize))
        nTestBatches = int(np.ceil(Xtest.shape[0] / batchSize))

        for epoch in range(nEpochs):

            if schedule_gamma:# scheduler for gamma
                gamma = gamma_min + (gamma_max - gamma_min) * min(1, epoch / (nEpochs / 10))
            else:
                gamma = gamma_max

            tot_train_loss, tot_train_acc = [], []
            train_perm_idx = torch.randperm(Xtrain.shape[0])

            for altern_opt_i in range(3): # alternatively optimize C, C_gl and H
            
                if altern_opt_i == 2: # optimize H
                    for _ in range(it_H):
                        for batch in range(nTrainBatches):
                            thisBatchIndices = train_perm_idx[batch * batchSize : (batch + 1) * batchSize]
                            xTrainBatch = Xtrain[thisBatchIndices]
                            yTrainBatch = ytrain[thisBatchIndices]
                            VNN.train()
                            VNN.zero_grad()

                            yHatTrain = VNN(xTrainBatch, Theta_tilde)
                            lossValueTrain = compute_loss(Loss, prec_type, yTrainBatch, yHatTrain, gamma, Theta_tilde, Theta, sampleC, lambda_)

                            if task == "classification":
                                accTrain = compute_metric(yHatTrain.squeeze(), yTrainBatch, metric)
                            else:
                                accTrain = MAE(yHatTrain, yTrainBatch)
                            
                            lossValueTrain.backward()
                            optimizer.step()
                                
                            tot_train_loss.append(lossValueTrain.detach())
                            tot_train_acc.append(accTrain)

                elif altern_opt_i == 1: # optimize Theta_tilde
                    if prec_type not in ["joint", "naive"]:
                        continue
                    
                    for _ in range(it_Theta_tilde):
                        for batch in range(nTrainBatches):
                            thisBatchIndices = train_perm_idx[batch * batchSize : (batch + 1) * batchSize]
                            xTrainBatch = Xtrain[thisBatchIndices]
                            yTrainBatch = ytrain[thisBatchIndices]
                            optimizer_Theta_tilde.zero_grad()

                            yHatTrain = VNN(xTrainBatch, Theta_tilde)
                            lossValueTrain = compute_loss(Loss, prec_type, yTrainBatch, yHatTrain, gamma, Theta_tilde, Theta, sampleC, lambda_)

                            if task == "classification":
                                accTrain = compute_metric(yHatTrain.squeeze(), yTrainBatch, metric)
                            else:
                                accTrain = MAE(yHatTrain, yTrainBatch)
                            
                            lossValueTrain.backward()
                            optimizer_Theta_tilde.step()
                            tot_train_loss.append(lossValueTrain.detach())
                            tot_train_acc.append(accTrain)
                
                elif altern_opt_i == 0: # optimize C_gl
                    if prec_type not in ["joint"]:
                        continue
                    Theta = graphical_lasso_with_reg(sampleC, Theta_tilde.detach(), gamma, cur_Theta=Theta, max_iter=it_Theta, tau=lambda_*eta, eta=eta)

        if prec_type == "naive": # remove small values to enforce sparsity
            with torch.no_grad():
                Theta_tilde = soft_threshold(Theta_tilde, tau=1e-3)

        if prec_type == "naive": # we only have Theta_tilde to check for sparsity and support reconstruction
            Theta = Theta_tilde.detach().cpu()

        if true_Prec is not None:
            theta_mae.append((Theta.cpu() - true_Prec).abs().mean())
            theta_sparsity.append((Theta.cpu() == 0).sum())

        test_perm_idx = torch.randperm(Xtest.shape[0])
        tot_test_acc, tot_test_loss = [], []
        with torch.no_grad():
            for batch in range(nTestBatches):
                thisBatchIndices = test_perm_idx[batch * batchSize : (batch + 1) * batchSize]
                xtestBatch = Xtest[thisBatchIndices]
                ytestBatch = ytest[thisBatchIndices]

                yHatTest = VNN(xtestBatch, Theta_tilde)
                lossValueTest = Loss(yHatTest.squeeze(), ytestBatch.squeeze())
                if task == "classification":
                    acc = compute_metric(yHatTest.squeeze(), ytestBatch, metric).item()
                else:
                    acc = MAE(yHatTest.squeeze(), ytestBatch.squeeze()).detach()
                tot_test_acc.append(acc)
                tot_test_loss.append(lossValueTest.item())

        all_test_loss.append(torch.tensor(tot_test_loss).mean())
        all_test_acc.append(torch.tensor(tot_test_acc).mean())

    return torch.tensor(all_test_loss), torch.tensor(all_test_acc), \
        torch.tensor(theta_mae), torch.FloatTensor(theta_sparsity)