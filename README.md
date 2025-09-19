# PNN

This repository contains the code to replicate the experiments in the paper **Precision Neural Networks: Joint Graph And Relational Learning** ([link](https://arxiv.org/abs/2509.14821)). 

## Description

Precision Neural Networks (PNNs) are graph convolutional neural networks operating on the graph defined by the precision matrix. Their weights are optimized jointly with the estimation of a sparse precision matrix, leading to task-aware dependency structures.


## Requirements
- Python 3.11.13
- `pip install -r requirements.txt`

## Datasets

The file `data_preprocess.ipynb` contains the code to preprocess the real datasets. Below the preliminary steps to download the datasets.

#### ABIDE: 
Following instructions at [http://preprocessed-connectomes-project.org/abide/](http://preprocessed-connectomes-project.org/abide/), download the file `Phenotypic_V1_0b_preprocessed1.csv` and place it at the path `data/datasets/abide/Phenotypic_V1_0b_preprocessed1.csv`

#### ADNI

From the website [https://ida.loni.usc.edu/](https://ida.loni.usc.edu/), download the files `ADSP_PHC_T1_FS_DATADIC` and `ADSP_PHC_T1_FS` and place them in the folder `data/datasets/adni`

## Experiments

To replicate experiments in the paper, run the following scripts. The dataset and the experiment parameters should be specified inside the scripts. The folder `best_params` contains the best hyperparameter configuration obtained for each model and experimental setting on each dataset, and it is automatically loaded when running the experiment.
- `synth_exp.py` to run the experiments on synthetic data
- `real_exp.py` to run the experiments for PNNs and VNN on real data
- `pca.py` to run the experiments for PCA on real data

