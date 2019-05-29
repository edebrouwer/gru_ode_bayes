#!/usr/bin/python
import itertools
import pandas as pd
import numpy as np
import torch
from mimic_gruode import train_gruode_mimic
import argparse
import os

parser = argparse.ArgumentParser(description="Running Cross validation on Neural ODE")
parser.add_argument('--fold_number', type=int, help="Model to use", default=0)
parser.add_argument('--regression', action="store_true", help = "Set for regression training (no labels)")


args = parser.parse_args()
fold = args.fold_number
if args.regression:
    type = "LogLik"
else:
    type = "AUC"


hyper_dict = np.load("../../hyper_dict.npy",allow_pickle = True).item()
if type=="LogLik":
    hyper_dict["lambda"] = [0]
epoch_max = 60


keys = hyper_dict.keys()
values = (hyper_dict[key] for key in keys)
combinations = [dict(zip(keys, combination)) for combination in itertools.product(*values)]

results_df = pd.DataFrame(columns = ["fold_num","dropout_rate","weight_decay","lambda","last_val_metric","best_val_metric","test_loglik","test_auc"])


train_idx = np.load(f"../../Datasets/MIMIC/fold_idx_{fold}/train_idx.npy")
val_idx = np.load(f"../../Datasets/MIMIC/fold_idx_{fold}/val_idx.npy")
test_idx = np.load(f"../../Datasets/MIMIC/fold_idx_{fold}/test_idx.npy")

for c in combinations:
    print(c)

    dropout_rate = c["dropout_rate"]
    weight_decay = c["weight_decay"]
    lambda_factor = c["lambda"]

    simulation_name = f"Xval_{type}_GRU_ODE_MIMIC_Binned60_NoImpute_dropout{dropout_rate}_weightdecay{weight_decay}_lambda{lambda_factor}_fold{fold}"

    device = torch.device("cuda")

    params_dict=dict()
    params_dict["hidden_size"] = 100
    params_dict["p_hidden"] = 25
    params_dict["prep_hidden"] = 10
    params_dict["logvar"] = True
    params_dict["mixing"] = 1e-4 #Weighting between KL loss and MSE loss.
    params_dict["delta_t"]=0.05
    params_dict["T"]=100

    params_dict["classification_hidden"] = 2
    params_dict["cov_hidden"] = 50
    params_dict["lr"]=0.001
    params_dict["full_gru_ode"] = True
    params_dict["no_cov"]=True

    params_dict["lambda"] = lambda_factor #Weighting between reconstruction and classification loss.
    params_dict["dropout_rate"] = dropout_rate
    params_dict["weight_decay"] = weight_decay
    params_dict["impute"] = False

    last_val_results, max_val_metric, test_loglik, test_auc, test_mse =train_gruode_mimic(simulation_name = simulation_name,
                                                            params_dict = params_dict,
                                                            device = device,
                                                            train_idx = train_idx,
                                                            val_idx = val_idx,
                                                            test_idx = test_idx,
                                                            epoch_max = epoch_max,
                                                            binned60 = False)
    if type =="AUC":
        val_result_index = "AUC_validation"
    else:
        val_result_index = "loglik_loss"

    results_df = pd.DataFrame({"fold_num": [fold], "dropout_rate": [dropout_rate], "weight_decay": [weight_decay],
                            "lambda": [lambda_factor], "last_val_metric": [last_val_results[val_result_index]],
                            "best_val_metric": [max_val_metric], "test_loglik": [test_loglik], "test_auc": [test_auc], "test_mse": [test_mse] })

    df_file_name = f"Results_{type}_EXTRAAA_GRU_ODE_{fold}.csv"

    if os.path.isfile(df_file_name):
        print("Saving the results...")
        df = pd.read_csv(df_file_name)
        df = df.append(results_df, ignore_index=True)
        df.to_csv(df_file_name,index=False)
    else:
        print("Creating the results dataframe...")
        df = results_df
        df.to_csv(df_file_name,index=False)

print(f"Fold computation finished for MIMIC GRU ODE : fold {fold}")
