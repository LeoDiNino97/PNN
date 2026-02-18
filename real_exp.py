"""
Run experiments on real datasets for PNN variants.
Load best parameter configuration for each dataset and
precision estimation type and run PNN on test set.
Results are saved in a CSV file and printed on terminal.
"""

from training.train_pnn_real import train_pnn
import json
import torch
dset = "abide" # abide or adni2
prec_type = "glasso" # sample, glasso, joint or cov (for VNN baseline)
print("CUDA available:", torch.cuda.is_available())
print("Number of GPUs:", torch.cuda.device_count())

if torch.cuda.is_available():
    print("GPU name:", torch.cuda.get_device_name(0))
    print("Current device:", torch.cuda.current_device())

with open(f"best_params/{dset}_{prec_type}.json", "r") as f:
    loaded_data = json.load(f)
    res = train_pnn(dset, iterations=20, prec_type=prec_type, 
                gamma_list=loaded_data["gamma_list"], lambda_list=loaded_data["lambda_list"],
                valid=False, hidden_sizes_list=loaded_data["hidden_sizes_list"],
                K_list=loaded_data["K_list"], dropout_list=loaded_data["dropout_list"], 
                eta_list=loaded_data["eta_list"], batch_norm_list=loaded_data["batch_norm_list"], nEpochs_list=loaded_data["nEpochs_list"],
                schedule_gamma_list=loaded_data["schedule_gamma_list"])

print(f"MSE {res.loss.to_numpy().astype(float).mean()} +- {res.loss.to_numpy().astype(float).std()} MAE {res.acc.to_numpy().astype(float).mean()} +- {res.acc.to_numpy().astype(float).std()} \
       Theta sparsity {res.theta_sparsity.to_numpy().astype(float).mean()} +- {res.theta_sparsity.to_numpy().astype(float).std()}")
