import torch
import numpy as np
import gru_ode_bayes
from torch.utils.data import DataLoader
import gru_ode_bayes.data_utils as data_utils

from gru_ode_bayes.datasets.double_OU import double_OU

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import argparse
from matplotlib import cm

#import sys; sys.argv.extend(["--model_name", "paper_random_r", "--random_r"])
#import sys; sys.argv.extend(["--model_name", "BXLator", "--data_type", "BXLator","--format","pdf"])


def plot_trained_model(model_name = "paper_random_r", format_image = "pdf", random_r= True, max_lag = 0, jitter = 0, random_theta = False, data_type = "double_OU" ):

    style = "fill"



    summary_dict=np.load(f"./../trained_models/{model_name}_params.npy",allow_pickle=True).item()

    params_dict = summary_dict["model_params"]
    metadata    = summary_dict["metadata"]

    if type(params_dict) == np.ndarray:
        ## converting np array to dictionary:
        params_dict = params_dict.tolist()

    #Loading model
    model = gru_ode_bayes.NNFOwithBayesianJumps(input_size = params_dict["input_size"], hidden_size = params_dict["hidden_size"],
                                            p_hidden = params_dict["p_hidden"], prep_hidden = params_dict["prep_hidden"],
                                            logvar = params_dict["logvar"], mixing = params_dict["mixing"],
                                            full_gru_ode = params_dict["full_gru_ode"],impute = params_dict["impute"], solver = params_dict["solver"],store_hist = True)

    model.load_state_dict(torch.load(f"./../trained_models/{model_name}.pt"))
    model.eval()

    #Test data :
    N           = 10
    T           = metadata["T"]
    delta_t     = metadata["delta_t"]
    theta       = metadata.pop("theta",None)
    sigma       = metadata["sigma"]
    rho         = metadata["rho"]
    r_mu        = metadata.pop("r_mu",None)
    sample_rate = metadata["sample_rate"]
    sample_rate = 1
    dual_sample_rate = metadata["dual_sample_rate"]
    r_std       = metadata.pop("r_std",None)
    #print(f"R std :{r_std}")
    max_lag     = metadata.pop("max_lag",None)


    if data_type=="double_OU":
        T = 6
        df = double_OU.OU_sample(T = T, dt = delta_t,
                    N = N, sigma = 0.1,
                    theta = theta,
                    r_mu = r_mu, r_std = r_std,
                    rho = rho, sample_rate = sample_rate,
                    dual_sample_rate = dual_sample_rate, max_lag = max_lag, random_theta= random_theta,full=True,seed=432)


        ## for 10 time-points
        times_1 = [1.0, 2.0, 4.0, 5.0, 7.0,7.5]
        times_2 = [2.0, 3.0, 4.0, 6.0]
    else:
        df = gru_ode_bayes.datasets.BXLator.datagen.BXL_sample(T = metadata["T"], dt = metadata["delta_t"],N = N, sigma = metadata["sigma"], a = 0.3, b= 1.4, rho = metadata["rho"], sample_rate = 10 , dual_sample_rate = 1, full = True)

        ## for 10 time-points
        times_1 = [2.0,5.0, 12.0,15.0, 23.0, 32.0,35.0, 41.0, 43.0]
        times_2 = [1.0,7.0, 12.0, 15.0, 25.0, 32.0, 38.0, 45.0]


    times   = np.union1d(times_1,times_2)
    obs     = df.loc[df["Time"].isin(times)].copy()
    obs[["Mask_1","Mask_2"]]                   = 0
    obs.loc[df["Time"].isin(times_1),"Mask_1"] = 1
    obs.loc[df["Time"].isin(times_2),"Mask_2"] = 1


    data = data_utils.ODE_Dataset(panda_df=obs, jitter_time=jitter)
    dl   = DataLoader(dataset=data, collate_fn=data_utils.custom_collate_fn, shuffle=False, batch_size=1)

    with torch.no_grad():
        for sample, b in enumerate(dl):
            times    = b["times"]
            time_ptr = b["time_ptr"]
            X        = b["X"]
            M        = b["M"]
            obs_idx  = b["obs_idx"]
            cov      = b["cov"]

            y = b["y"]
            hT, loss, _, t_vec, p_vec, _, eval_times, eval_vals = model(times, time_ptr, X, M, obs_idx, delta_t=delta_t, T=T, cov = cov, return_path=True)

            if params_dict["solver"]=="dopri5":
                p_vec = eval_vals
                t_vec = eval_times.cpu().numpy()
            
            observations=X.detach().numpy()
            m, v = torch.chunk(p_vec[:,0,:],2,dim=1)

            if params_dict["logvar"]:
                up   = m + torch.exp(0.5*v) * 1.96
                down = m - torch.exp(0.5*v) * 1.96
            else:
                up   = m + torch.sqrt(v) * 1.96
                down = m - torch.sqrt(v) * 1.96

            plots_dict = dict()
            plots_dict["t_vec"] = t_vec
            plots_dict["up"] = up.numpy()
            plots_dict["down"] = down.numpy()
            plots_dict["m"] = m.numpy()
            plots_dict["observations"] = observations
            plots_dict["mask"] = M.cpu().numpy()

            fill_colors = [cm.Blues(0.25), cm.Greens(0.25)]

            line_colors = [cm.Blues(0.6), cm.Greens(0.6)]
            colors=["blue","green"]

            ## sde trajectory
            df_i = df.query(f"ID == {sample}")

            plt.figure(figsize=(6.4, 4.8))
            if style == "fill":
                for dim in range(2):
                    plt.fill_between(x  = t_vec,
                                     y1 = down[:,dim].numpy(),
                                     y2 = up[:,dim].numpy(),
                                     facecolor = fill_colors[dim],
                                     alpha=1.0, zorder=1)
                    plt.plot(t_vec, m[:,dim].numpy(), color=line_colors[dim], linewidth=2, zorder=2, label=f"Dimension {dim+1}")
                    observed_idx = np.where(plots_dict["mask"][:, dim]==1)[0]
                    plt.scatter(times[observed_idx], observations[observed_idx,dim], color=colors[dim], alpha=0.5, s=60)
                    plt.plot(df_i.Time, df_i[f"Value_{dim+1}"], ":", color=colors[dim], linewidth=1.5, alpha=0.8, label="_nolegend_")
            else:
                for dim in range(2):
                    plt.plot(t_vec, up[:,dim].numpy(),"--", color="red", linewidth=2)
                    plt.plot(t_vec, down[:,dim].numpy(),"--", color="red",linewidth=2)
                    plt.plot(t_vec, m[:,dim].numpy(), color=colors[dim], linewidth=2)
                    observed_idx = np.where(plots_dict["mask"][:, dim]==1)[0]
                    plt.scatter(times[observed_idx], observations[observed_idx,dim], color=colors[dim], alpha=0.5, s=60)
                    plt.plot(df_i.Time, df_i[f"Value_{dim+1}"], ":", color=colors[dim], linewidth=1.5, alpha=0.8)

            #plt.title("Test trajectory of a double OU process")
            plt.xlabel("Time")
            plt.grid()
            plt.legend(loc="lower right")
            plt.ylabel("Predicton (+/- 1.96 st. dev)")
            fname = f"{model_name}_sample{sample}_{style}.{format_image}"
            plt.tight_layout()
            plt.savefig(fname)
            plt.close()
            print(f"Saved sample into '{fname}'.")
            #dict_name = f"paper-plots/{model_name}_sample{sample}_dict.npy"
            #np.save(dict_name, plots_dict)
