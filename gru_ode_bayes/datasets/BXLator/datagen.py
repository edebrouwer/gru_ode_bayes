import numpy as np
import pandas as pd
import argparse

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt


def Brusselator(T, dt, b=2, a = 1, N_sims=2, sigma=0.1, rho=-0.99, random_theta=False):
    N_t  = int(T//dt)
    sims = np.zeros((N_sims, N_t, 2))
    sims[:,0,:] = np.ones((N_sims,2))*np.array([1,b/a])+0.3*np.random.randn(N_sims,2)
    beta = np.ones((N_sims,2))
    beta[:,1] = 0
    tensor = np.zeros((N_sims,2,2))
    tensor[:,0,0] = -(b+1)
    tensor[:,1,0] = b
    cov  = dt * np.array(
            [[sigma**2,       sigma**2 * rho],
             [sigma**2 * rho, sigma**2]])
    dW   = np.random.multivariate_normal([0, 0], cov, size=(N_sims, N_t))
    for i in range(1,N_t):
        tensor[:,0,1] = a*sims[:,i-1,0]**2
        tensor[:,1,1] = -a*sims[:,i-1,0]**2
        sims[:, i, :] = sims[:,i-1,:] + beta*dt + np.einsum('ijk,ik->ij',tensor,sims[:,i-1,:])*dt + sigma*dW[:,i]
    return sims.astype(np.float32)

def BXL_sample(T,dt,N,a,b,sigma,rho,sample_rate,dual_sample_rate, full=False,seed=432):
    '''
    Samples from N 2 dimensional BXL process with opposite means.
    The sample rate should be expressed in samples per unit of time. (on average there will be sample_rate*T sample per series)
    The dual_sample rate gives the proportion of samples wich are jointly sampled (for both dimensions)
    We generate dummy covariates (all 0)
    '''
    np.random.seed(seed)
    y_vec = Brusselator(T, dt=dt, b= b, a = a,N_sims=N, sigma=sigma)
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
        elif (sample_rate*dt)==1:
            num_samples = int(sample_rate*T)
        else:
            num_samples=np.random.randint(sample_rate*T*(1-variability_num_samples),sample_rate*T*(1+variability_num_samples)) #number of sample varies around the mean with 20% variability

        #index_max_lag = int(max_lag//dt)
        #lag = np.random.randint(low=0,high=index_max_lag+1)
        lag = 0
        
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
    parser = argparse.ArgumentParser(description="Generating 2D Brusselator datasets.")
    parser.add_argument('--rho', type=float, help="Correlation between the two variables.", default=0.99)
    parser.add_argument('--prefix', type=str, help="Prefix for generated data", default="BXLator")

    args = parser.parse_args()

    T       = 50
    delta_t = 0.1
    a       = 0.3
    b       = 1.4
    sigma   = 0.1
    rho = args.rho
    sample_rate = 0.5
    dual_sample_rate = 0.8

    N  = 1000
    df = BXL_sample(T = T, dt = delta_t,
                N = N, sigma = sigma, a = a, b= b,
                rho = rho, sample_rate = sample_rate,
                dual_sample_rate = dual_sample_rate)

    df.to_csv(f"{args.prefix}.csv",index=False)

    #Save metadata dictionary
    metadata_dict = {"T":T, "delta_t":delta_t, "rho": args.rho,
                    "sample_rate": sample_rate, "dual_sample_rate": dual_sample_rate,
                    "sigma":sigma, "a" : a, "b": b}
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
        #plt.savefig(f"{examples_dir}{args.prefix}_{ex}.pdf")
        plt.savefig(f"full_example_{ex}.pdf")
        plt.close()
