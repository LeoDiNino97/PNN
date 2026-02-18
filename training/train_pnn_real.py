"""
Train and evaluate a PNN on the specified dataset for all
combinations of the parameter lists. Saves results in a 
csv file.
"""

import numpy as np
import torch 
from torch import nn, optim
from models.vnn import VNN as CovNN
from tqdm import tqdm
from itertools import product
from utils.utils import (compute_covariance_and_precision, compute_metric, 
                         compute_loss, graphical_lasso_with_reg)
from data.data_loading import load_dataset
import pandas as pd


def train_pnn(dset, iterations=1, prec_type="sample", load_path=None 
              gamma_list=[0.1], lambda_list=[0.1],
              valid=True, hidden_sizes_list=[([32]*2,[32,16])],
              K_list=[1], dropout_list=[0.0], eta_list=[0.1], batch_norm_list=[True],
              nEpochs_list=[10], schedule_gamma_list=[False]):

    df_res = pd.DataFrame(columns=['it', 'hidden_sizes_all', 'K', 'dropout', 'gamma', 'lambda_',
                                   'loss', 'acc', 'theta_tilde_sparsity', 'theta_sparsity', 'eta', 'batch_norm',
                                    'nEpochs', 'schedule_gamma'])

    all_thetas, all_theta_tilde = [], []

    ## Training values to track
    all_training_loss, all_training_mae = [], []
    theta_tilde_updates, theta_updates = [], []
    
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    params_iterator = product( 
        hidden_sizes_list, K_list, dropout_list, 
        gamma_list, lambda_list, eta_list, batch_norm_list, nEpochs_list,
        schedule_gamma_list
    )

    for hidden_sizes_all, K, dropout, gamma_max, lambda_0, eta, batch_norm, nEpochs, schedule_gamma in params_iterator:

        hidden_sizes, hidden_mlp_sizes = hidden_sizes_all
        all_test_acc, all_test_loss = [], []
        all_test_acc_other, all_test_loss_other = [], []
        theta_tilde_sparsity = []
        theta_sparsity = []

        for it in tqdm(range(iterations)):

            it_H, it_Theta_tilde, it_Theta = 20, 20, 20

            Xtrain, ytrain, Xval, yval, Xtest, ytest, task_level, F_out, task, metric = \
            load_dataset(dset, seed=it)
            F_in = Xtrain.shape[2]
            Xtrain, ytrain, Xval, yval, Xtest, ytest = Xtrain.to(device), ytrain.to(device), Xval.to(device), yval.to(device), Xtest.to(device), ytest.to(device)

            if len(Xtrain.shape) == 2:
                Xtrain = Xtrain.unsqueeze(-1)
                Xval = Xval.unsqueeze(-1)
                Xtest = Xtest.unsqueeze(-1)

            if valid:
                Xtest, ytest = Xval, yval

            bias = True
            lr = 0.01
            gamma_min = 0.1
            T, N, F_in = Xtrain.shape

            lambda_ = lambda_0 * float(np.sqrt(np.log(N) / T))
            if prec_type == 'SCGL':
                _, Prec = torch.from_numpy(np.load(load_path)).float().to(device)
            sampleC, Prec = compute_covariance_and_precision(Xtrain, prec_type, lambda_, eta)
            sampleC, Prec = sampleC.to(device), Prec.to(device)
            
            if prec_type in ["joint"]:
                Theta_tilde = nn.Parameter(torch.eye(N, device=device), requires_grad=True)
                optimizer_Theta_tilde = optim.SGD([Theta_tilde], lr=lr, weight_decay=0.0)
            else:
                Theta_tilde = Prec

            Theta = Prec

            N = Xtrain.shape[1]
            model = CovNN

            VNN = model(          
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


            optimizer = optim.Adam(VNN.parameters(), lr=lr, weight_decay=0.0001)
            batchSize = 1000000
            nTrainBatches = int(np.ceil(Xtrain.shape[0] / batchSize))
            nTestBatches = int(np.ceil(Xtest.shape[0] / batchSize))

            for epoch in range(nEpochs):
                if schedule_gamma:# scheduler for gamma
                    gamma = gamma_min + (gamma_max - gamma_min) * min(1, epoch / (nEpochs / 10))
                else:
                    gamma = gamma_max

                train_perm_idx = torch.randperm(Xtrain.shape[0])

                for altern_opt_i in range(3): # alternatively optimize C, C_gl and H
                
                    if altern_opt_i == 2: # optimize H
                        these_it_H = it_H
                        for _ in range(these_it_H):
                            for batch in range(nTrainBatches):
                                thisBatchIndices = train_perm_idx[batch * batchSize : (batch + 1) * batchSize]
                                xTrainBatch = Xtrain[thisBatchIndices]
                                yTrainBatch = ytrain[thisBatchIndices]
                                VNN.train()
                                VNN.zero_grad()

                                yHatTrain = VNN(xTrainBatch, Theta_tilde)
                                lossValueTrain = compute_loss(Loss, prec_type, yTrainBatch, yHatTrain, gamma, Theta_tilde, Theta)

                                if task == "classification":
                                    accTrain = compute_metric(yHatTrain.squeeze(), yTrainBatch, metric)
                                else:
                                    accTrain = MAE(yHatTrain, yTrainBatch)
                                
                                lossValueTrain.backward()
                                optimizer.step()

                                all_training_loss.append(lossValueTrain.detach())
                                all_training_mae.append(accTrain.detach())

                    elif altern_opt_i == 1: # optimize Theta_tilde
                        if prec_type not in ["joint"]:
                            continue
                        for _ in range(it_Theta_tilde):
                            for batch in range(nTrainBatches):
                                thisBatchIndices = train_perm_idx[batch * batchSize : (batch + 1) * batchSize]
                                xTrainBatch = Xtrain[thisBatchIndices]
                                yTrainBatch = ytrain[thisBatchIndices]
                                optimizer_Theta_tilde.zero_grad()

                                yHatTrain = VNN(xTrainBatch, Theta_tilde)
                                lossValueTrain = compute_loss(Loss, prec_type, yTrainBatch, yHatTrain, gamma, Theta_tilde, Theta)

                                if task == "classification":
                                    accTrain = compute_metric(yHatTrain.squeeze(), yTrainBatch, metric)
                                else:
                                    accTrain = MAE(yHatTrain, yTrainBatch)
                                
                                old_Theta_tilde = Theta_tilde.detach().clone()

                                lossValueTrain.backward()
                                optimizer_Theta_tilde.step()

                                theta_tilde_updates.append(torch.linalg.norm(old_Theta_tilde - Theta_tilde.detach(), 'fro'))
                    
                    elif altern_opt_i == 0: # optimize Theta
                        if prec_type not in ["joint"]:
                            continue
                        old_Theta = Theta.clone()
                        Theta = graphical_lasso_with_reg(sampleC, Theta_tilde.detach(), gamma, cur_Theta=Theta, max_iter=it_Theta, tau=lambda_*eta, eta=eta)
                        theta_updates.append(torch.linalg.norm(old_Theta - Theta.detach(), 'fro'))


            theta_tilde_sparsity.append((Theta_tilde.detach() == 0).sum())
            theta_sparsity.append((Theta == 0).sum())
            all_thetas.append(Theta)
            all_theta_tilde.append(Theta_tilde.detach().cpu())

            test_perm_idx = torch.randperm(Xtest.shape[0])
            tot_test_acc, tot_test_loss = [], []
            tot_test_acc_other, tot_test_loss_other = [], []
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
            all_test_loss_other.append(torch.tensor(tot_test_loss_other).mean())
            all_test_acc_other.append(torch.tensor(tot_test_acc_other).mean())


            df_new_row = pd.DataFrame(
                            data=np.array([[it, str(hidden_sizes_all), K, dropout, gamma, lambda_0, lossValueTest.item(), torch.tensor(tot_test_acc).mean().item(),
                                            theta_tilde_sparsity[-1].item(), theta_sparsity[-1].item(), eta, batch_norm,
                                            nEpochs, schedule_gamma]]),                                             
                            columns=['it', 'hidden_sizes_all', 'K', 'dropout', 'gamma', 'lambda_',
                                   'loss', 'acc', 'theta_tilde_sparsity', 'theta_sparsity', 'eta', 'batch_norm',
                                   'nEpochs', 'schedule_gamma'])
                            
            df_res = pd.concat([df_res,df_new_row], ignore_index=True)

    return df_res



def pick_best_params(df):
    cols = df.columns.tolist()
    cols = [c for c in cols if c not in ['hidden_sizes_all', 'K', 'dropout', 'gamma', 'lambda_','eta', 'batch_norm', 'nEpochs', 'schedule_gamma']]
    df[cols] = df[cols].astype(float)
    res = pd.pivot_table(df, index=['hidden_sizes_all', 'K', 'dropout', 'gamma', 'lambda_','eta', 'batch_norm', 'nEpochs', 'schedule_gamma'], aggfunc="mean")
    cols = ['acc']
    best_params = {}
    for col in cols:
        best_params[col] = res[res[col] == res[col].values.min()].index[0]
    
    return best_params


