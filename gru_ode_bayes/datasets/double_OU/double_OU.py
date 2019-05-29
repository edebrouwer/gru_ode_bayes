import numpy as np
import pandas as pd
import argparse

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt


def OU(T, dt, r_mu, r_std, theta=10, N_sims=1, sigma=0.1, rho=-0.99, random_theta=False):
    N_t  = int(T//dt)
    mu   = np.sqrt(12)*r_std*np.random.uniform(low=-0.5,high=0.5,size=(N_sims,2)) + r_mu #uniform distribution
    if random_theta:
        theta = np.random.uniform(low=0.1,high=1.5,size=(N_sims,1))
    sims = np.zeros((N_sims, N_t, 2))
    cov  = dt * np.array(
            [[sigma**2,       sigma**2 * rho],
             [sigma**2 * rho, sigma**2]])
    dW   = np.random.multivariate_normal([0, 0], cov, size=(N_sims, N_t))
    for i in range(1,N_t):
        sims[:, i] = sims[:,(i-1)] - theta * (sims[:,(i-1)] - mu)*dt + dW[:,i]
    return sims.astype(np.float32)


def OU_sample(T,dt,theta,N,sigma,r_mu,r_std,rho,sample_rate,dual_sample_rate,max_lag, random_theta, full=False,seed=432):
    '''
    Samples from N 2 dimensional OU process with opposite means.
    The sample rate should be expressed in samples per unit of time. (on average there will be sample_rate*T sample per series)
    The dual_sample rate gives the proportion of samples wich are jointly sampled (for both dimensions)
    We generate dummy covariates (all 0)
    '''
    np.random.seed(seed)
    y_vec = OU(T+max_lag, dt=dt, r_mu=r_mu, r_std=r_std, theta=theta, rho=rho, N_sims=N, random_theta=random_theta)
    N_t = int(T//dt)
    p_single=1-dual_sample_rate
    p_both=dual_sample_rate

    col=["ID","Time","Value_1","Value_2","Mask_1","Mask_2","Cov"]
    df = pd.DataFrame(columns=col)


    for i in range(N):
        variability_num_samples=0.2 #variability in number of samples for each trajectory.
        #Make sure that there is enough possibilities for sampling the number of observations.
        if variability_num_samples*2*sample_rate*T<1:
            num_samples=int(sample_rate*T)
        else:
            num_samples=np.random.randint(sample_rate*T*(1-variability_num_samples),sample_rate*T*(1+variability_num_samples)) #number of sample varies around the mean with 20% variability

        index_max_lag = int(max_lag//dt)
        lag = np.random.randint(low=0,high=index_max_lag+1)

        if full:
            sample_times = np.arange(N_t)
            sample_type = (np.ones(N_t)*2).astype(np.int)
            num_samples=N_t
        else:
            sample_times=np.random.choice(N_t,num_samples,replace=False)
            sample_type=np.random.choice(3,num_samples,replace=True,p=[p_single/2,p_single/2,p_both])
        samples=y_vec[i,sample_times+lag,:]

        #non observed samples are set to 0
        samples[sample_type==0,1] = 0
        samples[sample_type==1,0] = 0

        #Observed samples have mask 1, others have 0.
        mask=np.ones((num_samples,2))
        mask[sample_type==0,1]=0
        mask[sample_type==1,0]=0

        covs=np.zeros((num_samples,1))

        individual_data=pd.DataFrame(np.concatenate((i*np.ones((num_samples,1)),dt*np.expand_dims(sample_times,1),samples,mask,covs),1),columns=col)
        df=df.append(individual_data)
    df.reset_index(drop=True,inplace=True)
    return(df)

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Generating 2D OU datasets.")
    parser.add_argument('--rho', type=float, help="Correlation between the two variables.", default=0.99)
    parser.add_argument('--prefix', type=str, help="Prefix for generated data", default="double_OU")
    parser.add_argument('--random_r',action="store_true", help="Generates random targets (r) from uniform distribution with mean 1 (-1) and std = 0.5",default = True)
    parser.add_argument('--max_lag',type=float, help = "Shift trajectories with a random positive lag. Insert max lag allowed.",default=0)
    parser.add_argument('--random_theta',action="store_true", help="Generates random thetas for each sample")

    args = parser.parse_args()

    T       = 10
    delta_t = 0.05
    theta   = 1.0
    rho     = args.rho
    sigma   = 0.1
    r_mu    = [1.0, -1.0]
    sample_rate = 2
    dual_sample_rate = 0.2
    if args.random_r:
        r_std   = 1/np.sqrt(12)
    else:
        r_std   = 0
    random_theta = args.random_theta

    N  = 10000
    df = OU_sample(T = T, dt = delta_t,
                N = N, sigma = sigma,
                theta = theta,
                r_mu = r_mu, r_std = r_std,
                rho = rho, sample_rate = sample_rate,
                dual_sample_rate = dual_sample_rate, max_lag = args.max_lag,
                random_theta = random_theta)

    df.to_csv(f"{args.prefix}.csv",index=False)

    #Save metadata dictionary
    metadata_dict = {"T":T, "delta_t":delta_t, "theta":theta, "rho": args.rho,
                    "r_mu":r_mu, "sample_rate": sample_rate, "dual_sample_rate": dual_sample_rate,
                    "r_std":r_std,"N": N, "max_lag":args.max_lag, "sigma":sigma}
    np.save(f"{args.prefix}_metadata.npy",metadata_dict)

    #Plot some examples and store them.
    import os
    N_examples = 10
    examples_dir = f"{args.prefix}_paths_examples/"
    if not os.path.exists(examples_dir):
        os.makedirs(examples_dir)
    for ex in range(N_examples):
        idx = np.random.randint(low=0,high=df["ID"].nunique())
        plt.figure()
        for dim in range(2):
            random_sample = df.loc[df["ID"]==idx].sort_values(by="Time").values
            obs_mask = random_sample[:,4+dim]==1
            plt.scatter(random_sample[obs_mask,1],random_sample[obs_mask,2+dim])
            plt.title("Example of generated trajectory")
            plt.xlabel("Time")
        plt.savefig(f"{examples_dir}{args.prefix}_{ex}.pdf")
        plt.close()
