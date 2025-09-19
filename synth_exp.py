"""
Generate synthetic datasets and evaluate the PNN variants on it.
The script loads the best parameter configuration for each PNN
variant and prints results on terminal.
"""

from training.train_pnn_synth import train_pnn
import torch
from utils.utils import generate_task
import json

# Experiment parameters
alpha = .9 # Parameter for precision sparsity, see https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_sparse_spd_matrix.html
n_samples = 100
n_features = 20
SNR = 10
n_iterations = 5
prec_type = "sample" # sample, glasso, naive or joint

mse, mae, prec_mae, prec_sparsity = [], [], [], []

for seed in range(n_iterations):
    X, y, cov, prec = generate_task(n_samples=n_samples, n_features=n_features, SNR=SNR, alpha=alpha, seed=seed)
    with open(f"best_params/synth_{prec_type}.json", "r") as f:
        loaded_data = json.load(f)

        this_mse, this_mae, this_prec_mae, this_prec_sparsity = train_pnn( X, y, 
            loaded_data["hidden_sizes"], loaded_data["K"], loaded_data["dropout"], loaded_data["gamma"], 
            loaded_data["lambda"], loaded_data["eta"], loaded_data["batch_norm"], loaded_data["nEpochs"], 
            loaded_data["schedule_gamma"], iterations=1, prec_type=prec_type, true_Prec=prec)
    
    mse.append(this_mse)
    mae.append(this_mae)
    prec_mae.append(this_prec_mae)
    prec_sparsity.append(this_prec_sparsity)


print(f"MSE {torch.tensor(mse).mean()} +- {torch.tensor(mse).std()} MAE {torch.tensor(mae).mean()} +- {torch.tensor(mae).std()}")
print(f"Theta MAE {torch.tensor(prec_mae).mean()} +- {torch.tensor(prec_mae).std()} Theta #NZ {torch.tensor(prec_sparsity).mean()} +- {torch.tensor(prec_sparsity).std()}")
