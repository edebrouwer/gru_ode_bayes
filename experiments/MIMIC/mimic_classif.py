import numpy as np
import gru_ode
from gru_ode import train_gru_ode_bayes
#Script for training gru-ode-bayes on the MIMIC dataset.

simulation_name="MIMIC_in_mortality_classification"
device = torch.device("cuda")


train_idx = np.load("./Datasets/MIMIC/fold_idx_0/train_idx.npy")
val_idx = np.load("./Datasets/MIMIC/fold_idx_0/val_idx.npy")
test_idx = np.load("./Datasets/MIMIC/fold_idx_0/test_idx.npy")

#Model parameters.
params_dict=dict()
params_dict["hidden_size"] = 100
params_dict["p_hidden"] = 25
params_dict["prep_hidden"] = 10
params_dict["logvar"] = True
params_dict["mixing"] = 1e-4 #Weighting between KL loss and MSE loss.
params_dict["delta_t"]=0.1
params_dict["T"]=100
params_dict["lambda"] = 500 #Weighting between classification and MSE loss.

params_dict["classification_hidden"] = 2
params_dict["cov_hidden"] = 50
params_dict["weight_decay"] = 0.1
params_dict["dropout_rate"] = 0.3
params_dict["lr"]=0.001
params_dict["full_gru_ode"] = True
params_dict["no_cov"] = True

params_dict["train_idx"] = train_idx
params_dict["val_idx"] = val_idx
params_dict["test_idx"] = test_idx

val_options = {"T_val": 75, "max_val_samples": 3}

info,val_metric_prev, test_loglik, test_auc, test_mse = train_gru_ode_bayes(params_dict = params_dict, device = device, csv_input_file= "./Datasets/MIMIC/Processed_MIMIC.csv", csv_tags_file="./Datasets/MIMIC/MIMIC_tags.csv", csv_covs_file="./Datasets/MIMIC/MIMIC_covs.csv",val_options= val_options,simulation_name= simulation_name, epoch_max = 40 )
