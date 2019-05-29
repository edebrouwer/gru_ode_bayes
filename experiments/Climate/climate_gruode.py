import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import gru_ode_bayes
import gru_ode_bayes.data_utils as data_utils
import time
import tqdm
from sklearn.metrics import roc_auc_score
from gru_ode_bayes import Logger

def train_gruode(simulation_name,params_dict,device, train_idx, val_idx, test_idx, epoch_max=40):
    csv_file_path = params_dict["csv_file_path"]
    csv_file_cov = params_dict["csv_file_cov"]
    csv_file_tags = params_dict["csv_file_tags"]

    N = pd.read_csv(csv_file_path)["ID"].nunique()
    
    if params_dict["lambda"]==0:
        validation = True
        val_options = {"T_val": params_dict["T_val"], "max_val_samples": params_dict["max_val_samples"]}
    else:
        validation = False
        val_options = None

    if params_dict["lambda"]==0:
        logger = Logger(f'./Logs/{simulation_name}')
    else:
        logger = Logger(f'./Logs/{simulation_name}')


    data_train = data_utils.ODE_Dataset(csv_file=csv_file_path,label_file=csv_file_tags, cov_file= csv_file_cov, idx=train_idx)
    data_val   = data_utils.ODE_Dataset(csv_file=csv_file_path,label_file=csv_file_tags,
                                        cov_file= csv_file_cov, idx=val_idx, validation = validation,
                                        val_options = val_options)
    data_test   = data_utils.ODE_Dataset(csv_file=csv_file_path,label_file=csv_file_tags,
                                        cov_file= csv_file_cov, idx=test_idx, validation = validation,
                                        val_options = val_options)

    dl   = DataLoader(dataset=data_train, collate_fn=data_utils.custom_collate_fn, shuffle=True, batch_size=500,num_workers=2)
    dl_val = DataLoader(dataset=data_val, collate_fn=data_utils.custom_collate_fn, shuffle=True, batch_size=len(val_idx))
    dl_test = DataLoader(dataset=data_test, collate_fn=data_utils.custom_collate_fn, shuffle=True, batch_size=len(test_idx))


    params_dict["input_size"]=data_train.variable_num
    params_dict["cov_size"] = data_train.cov_dim

    np.save(f"./../trained_models/{simulation_name}_params.npy",params_dict)

    nnfwobj = gru_ode_bayes.NNFOwithBayesianJumps(input_size = params_dict["input_size"], hidden_size = params_dict["hidden_size"],
                                            p_hidden = params_dict["p_hidden"], prep_hidden = params_dict["prep_hidden"],
                                            logvar = params_dict["logvar"], mixing = params_dict["mixing"],
                                            classification_hidden=params_dict["classification_hidden"],
                                            cov_size = params_dict["cov_size"], cov_hidden = params_dict["cov_hidden"],
                                            dropout_rate = params_dict["dropout_rate"],full_gru_ode= params_dict["full_gru_ode"], impute = params_dict["impute"])
    nnfwobj.to(device)

    optimizer = torch.optim.Adam(nnfwobj.parameters(), lr=params_dict["lr"], weight_decay= params_dict["weight_decay"])
    class_criterion = torch.nn.BCEWithLogitsLoss(reduction='sum')
    print("Start Training")
    val_metric_prev = -1000
    for epoch in range(epoch_max):
        nnfwobj.train()
        total_train_loss = 0
        auc_total_train  = 0
        tot_loglik_loss = 0
        for i, b in enumerate(tqdm.tqdm(dl)):

            optimizer.zero_grad()
            times    = b["times"]
            time_ptr = b["time_ptr"]
            X        = b["X"].to(device)
            M        = b["M"].to(device)
            obs_idx  = b["obs_idx"]
            cov      = b["cov"].to(device)
            labels = b["y"].to(device)
            batch_size = labels.size(0)

            h0 = 0# torch.zeros(labels.shape[0], params_dict["hidden_size"]).to(device)
            hT, loss, class_pred, mse_loss  = nnfwobj(times, time_ptr, X, M, obs_idx, delta_t=params_dict["delta_t"], T=params_dict["T"], cov = cov)

            total_loss = (loss + params_dict["lambda"]*class_criterion(class_pred, labels))/batch_size
            total_train_loss += total_loss
            tot_loglik_loss +=mse_loss
            try:
                auc_total_train += roc_auc_score(labels.detach().cpu(),torch.sigmoid(class_pred).detach().cpu())
            except ValueError:
                if params_dict["verbose"]>=3:
                    print("Single CLASS ! AUC is erroneous")
                pass
            
            total_loss.backward()
            optimizer.step()
    
 
        
        info = { 'training_loss' : total_train_loss.detach().cpu().numpy()/(i+1), 'AUC_training' : auc_total_train/(i+1), "loglik_loss" :tot_loglik_loss.detach().cpu().numpy()}
        for tag, value in info.items():
            logger.scalar_summary(tag, value, epoch)
        print(f"NegLogLik Loss train : {tot_loglik_loss.detach().cpu().numpy()}")

        data_utils.adjust_learning_rate(optimizer,epoch,params_dict["lr"])

        with torch.no_grad():
            nnfwobj.eval()
            total_loss_val = 0
            auc_total_val = 0
            loss_val = 0
            mse_val  = 0
            corr_val = 0
            num_obs = 0
            for i, b in enumerate(dl_val):
                times    = b["times"]
                time_ptr = b["time_ptr"]
                X        = b["X"].to(device)
                M        = b["M"].to(device)
                obs_idx  = b["obs_idx"]
                cov      = b["cov"].to(device)
                labels   = b["y"].to(device)
                batch_size = labels.size(0)

                if b["X_val"] is not None:
                    X_val     = b["X_val"].to(device)
                    M_val     = b["M_val"].to(device)
                    times_val = b["times_val"]
                    times_idx = b["index_val"]

                h0 = 0 #torch.zeros(labels.shape[0], params_dict["hidden_size"]).to(device)
                hT, loss, class_pred, t_vec, p_vec, h_vec, _, _  = nnfwobj(times, time_ptr, X, M, obs_idx, delta_t=params_dict["delta_t"], T=params_dict["T"], cov=cov, return_path=True)
                total_loss = (loss + params_dict["lambda"]*class_criterion(class_pred, labels))/batch_size

                try:
                    auc_val=roc_auc_score(labels.cpu(),torch.sigmoid(class_pred).cpu())
                except ValueError:
                    auc_val = 0.5
                    if params_dict["verbose"]>=3:
                        print("Only one class : AUC is erroneous")
                    pass

                if params_dict["lambda"]==0:
                    t_vec = np.around(t_vec,str(params_dict["delta_t"])[::-1].find('.')).astype(np.float32) #Round floating points error in the time vector.
                    p_val = data_utils.extract_from_path(t_vec,p_vec,times_val,times_idx)
                    m, v = torch.chunk(p_val,2,dim=1)
                    last_loss = (data_utils.log_lik_gaussian(X_val,m,v)*M_val).sum()
                    mse_loss = (torch.pow(X_val-m,2)*M_val).sum()
                    corr_val_loss = data_utils.compute_corr(X_val, m, M_val)

                    loss_val += last_loss.cpu().numpy()
                    num_obs += M_val.sum().cpu().numpy()
                    mse_val += mse_loss.cpu().numpy()
                    corr_val += corr_val_loss.cpu().numpy()
                else:
                    num_obs=1

                total_loss_val += total_loss.cpu().detach().numpy()
                auc_total_val += auc_val

            loss_val /= num_obs
            mse_val /=  num_obs
            info = { 'validation_loss' : total_loss_val/(i+1), 'AUC_validation' : auc_total_val/(i+1),
                     'loglik_loss' : loss_val, 'validation_mse' : mse_val, 'correlation_mean' : np.nanmean(corr_val),
                    'correlation_max': np.nanmax(corr_val), 'correlation_min': np.nanmin(corr_val)}
            for tag, value in info.items():
                logger.scalar_summary(tag, value, epoch)

            if params_dict["lambda"]==0:
                val_metric = - loss_val
            else:
                val_metric = auc_total_val/(i+1)

            if val_metric > val_metric_prev:
                print(f"New highest validation metric reached ! : {val_metric}")
                print("Saving Model")
                torch.save(nnfwobj.state_dict(),f"./../trained_models/{simulation_name}_MAX.pt")
                val_metric_prev = val_metric
                test_loglik, test_auc, test_mse = test_evaluation(nnfwobj, params_dict, class_criterion, device, dl_test)
                print(f"Test loglik loss at epoch {epoch} : {test_loglik}")
                print(f"Test AUC loss at epoch {epoch} : {test_auc}")
                print(f"Test MSE loss at epoch{epoch} : {test_mse}")
            else:
                if epoch % 10:
                    torch.save(nnfwobj.state_dict(),f"./../trained_models/{simulation_name}.pt")

        print(f"Total validation loss at epoch {epoch}: {total_loss_val/(i+1)}")
        print(f"Validation AUC at epoch {epoch}: {auc_total_val/(i+1)}")
        print(f"Validation loss (loglik) at epoch {epoch}: {loss_val:.5f}. MSE : {mse_val:.5f}. Correlation : {np.nanmean(corr_val):.5f}. Num obs = {num_obs}")

    print(f"Finished training GRU-ODE for Climate. Saved in ./../trained_models/{simulation_name}")

    return(info, val_metric_prev, test_loglik, test_auc, test_mse)

def test_evaluation(model, params_dict, class_criterion, device, dl_test):
    with torch.no_grad():
        model.eval()
        total_loss_test = 0
        auc_total_test = 0
        loss_test = 0
        mse_test  = 0
        corr_test = 0
        num_obs = 0
        for i, b in enumerate(dl_test):
            times    = b["times"]
            time_ptr = b["time_ptr"]
            X        = b["X"].to(device)
            M        = b["M"].to(device)
            obs_idx  = b["obs_idx"]
            cov      = b["cov"].to(device)
            labels   = b["y"].to(device)
            batch_size = labels.size(0)

            if b["X_val"] is not None:
                X_val     = b["X_val"].to(device)
                M_val     = b["M_val"].to(device)
                times_val = b["times_val"]
                times_idx = b["index_val"]

            h0 = 0 #torch.zeros(labels.shape[0], params_dict["hidden_size"]).to(device)
            hT, loss, class_pred, t_vec, p_vec, h_vec, _ , _ = model(times, time_ptr, X, M, obs_idx, delta_t=params_dict["delta_t"], T=params_dict["T"], cov=cov, return_path=True)
            total_loss = (loss + params_dict["lambda"]*class_criterion(class_pred, labels))/batch_size

            try:
                auc_test=roc_auc_score(labels.cpu(),torch.sigmoid(class_pred).cpu())
            except ValueError:
                if params_dict["verbose"]>=3:
                    print("Only one class. AUC is wrong")
                auc_test = 0
                pass

            if params_dict["lambda"]==0:
                t_vec = np.around(t_vec,str(params_dict["delta_t"])[::-1].find('.')).astype(np.float32) #Round floating points error in the time vector.
                p_val = data_utils.extract_from_path(t_vec,p_vec,times_val,times_idx)
                m, v = torch.chunk(p_val,2,dim=1)
                last_loss = (data_utils.log_lik_gaussian(X_val,m,v)*M_val).sum()
                mse_loss = (torch.pow(X_val-m,2)*M_val).sum()
                corr_test_loss = data_utils.compute_corr(X_val, m, M_val)

                loss_test += last_loss.cpu().numpy()
                num_obs += M_val.sum().cpu().numpy()
                mse_test += mse_loss.cpu().numpy()
                corr_test += corr_test_loss.cpu().numpy()
            else:
                num_obs=1

            total_loss_test += total_loss.cpu().detach().numpy()
            auc_total_test += auc_test

        loss_test /= num_obs
        mse_test /=  num_obs
        auc_total_test /= (i+1)

        return(loss_test, auc_total_test, mse_test)

if __name__ =="__main__":

    simulation_name="small_climate"
    device = torch.device("cuda")


    train_idx = np.load("../../gru_ode_bayes/datasets/Climate/folds/small_chunk_fold_idx_0/train_idx.npy",allow_pickle=True)
    val_idx = np.load("../../gru_ode_bayes/datasets/Climate/folds/small_chunk_fold_idx_0/val_idx.npy",allow_pickle=True)
    test_idx = np.load("../../gru_ode_bayes/datasets/Climate/folds/small_chunk_fold_idx_0/test_idx.npy",allow_pickle=True)

    #Model parameters.
    params_dict=dict()

    params_dict["csv_file_path"] = "../../gru_ode_bayes/datasets/Climate/small_chunked_sporadic.csv" 
    params_dict["csv_file_tags"] = None
    params_dict["csv_file_cov"]  = None

    params_dict["hidden_size"] = 50
    params_dict["p_hidden"] = 25
    params_dict["prep_hidden"] = 10
    params_dict["logvar"] = True
    params_dict["mixing"] = 1e-4 #Weighting between KL loss and MSE loss.
    params_dict["delta_t"]=0.1
    params_dict["T"]=200
    params_dict["lambda"] = 0 #Weighting between classification and MSE loss.

    params_dict["classification_hidden"] = 2
    params_dict["cov_hidden"] = 50
    params_dict["weight_decay"] = 0.0001
    params_dict["dropout_rate"] = 0.2
    params_dict["lr"]=0.001
    params_dict["full_gru_ode"] = True
    params_dict["no_cov"] = True
    params_dict["impute"] = False
    params_dict["verbose"] = 0 #from 0 to 3 (highest)

    params_dict["T_val"] = 150
    params_dict["max_val_samples"] = 3



    info, val_metric_prev, test_loglik, test_auc, test_mse = train_gruode(simulation_name = simulation_name,
                        params_dict = params_dict,
                        device = device,
                        train_idx = train_idx,
                        val_idx = val_idx,
                        test_idx = test_idx,
                        epoch_max=100)
