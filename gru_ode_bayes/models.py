import torch
import math
import numpy as np
from torchdiffeq import odeint

from torch.nn.utils.rnn import pack_padded_sequence

# GRU-ODE: Neural Negative Feedback ODE with Bayesian jumps

class GRUODECell(torch.nn.Module):
    def __init__(self, input_size, hidden_size, bias=True):
        """
        For p(t) modelling input_size should be 2x the x size.
        """
        super().__init__()
        self.input_size  = input_size
        self.hidden_size = hidden_size
        self.bias        = bias

        self.lin_xz = torch.nn.Linear(input_size, hidden_size, bias=bias)
        self.lin_xn = torch.nn.Linear(input_size, hidden_size, bias=bias)

        self.lin_hz = torch.nn.Linear(hidden_size, hidden_size, bias=False)
        self.lin_hn = torch.nn.Linear(hidden_size, hidden_size, bias=False)


    def forward(self, x, h):
        """
        Returns a change due to one step of using GRU-ODE for all h.
        The step size is given by delta_t.

        Args:
            x        input values
            h        hidden state (current)
            delta_t  time step

        Returns:
            Updated h
        """
        z = torch.sigmoid(self.lin_xz(x) + self.lin_hz(h))
        n = torch.tanh(self.lin_xn(x) + self.lin_hn(z * h))

        dh = (1 - z) * (n - h)
        return dh

class GRUODECell_Autonomous(torch.nn.Module):
    def __init__(self, hidden_size, bias=True):
        """
        For p(t) modelling input_size should be 2x the x size.
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.bias        = bias

        #self.lin_xz = torch.nn.Linear(input_size, hidden_size, bias=bias)
        #self.lin_xn = torch.nn.Linear(input_size, hidden_size, bias=bias)

        self.lin_hz = torch.nn.Linear(hidden_size, hidden_size, bias=False)
        self.lin_hn = torch.nn.Linear(hidden_size, hidden_size, bias=False)


    def forward(self, t, h):
        """
        Returns a change due to one step of using GRU-ODE for all h.
        The step size is given by delta_t.

        Args:
            t        time
            h        hidden state (current)

        Returns:
            Updated h
        """
        x = torch.zeros_like(h)
        z = torch.sigmoid(x + self.lin_hz(h))
        n = torch.tanh(x + self.lin_hn(z * h))

        dh = (1 - z) * (n - h)
        return dh


class FullGRUODECell(torch.nn.Module):
    def __init__(self, input_size, hidden_size, bias=True):
        """
        For p(t) modelling input_size should be 2x the x size.
        """
        super().__init__()

        #self.lin_xh = torch.nn.Linear(input_size, hidden_size, bias=bias)
        #self.lin_xz = torch.nn.Linear(input_size, hidden_size, bias=bias)
        #self.lin_xr = torch.nn.Linear(input_size, hidden_size, bias=bias)

        self.lin_x = torch.nn.Linear(input_size, hidden_size * 3, bias=bias)

        self.lin_hh = torch.nn.Linear(hidden_size, hidden_size, bias=False)
        self.lin_hz = torch.nn.Linear(hidden_size, hidden_size, bias=False)
        self.lin_hr = torch.nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, x, h):
        """
        Executes one step with GRU-ODE for all h.
        The step size is given by delta_t.

        Args:
            x        input values
            h        hidden state (current)
            delta_t  time step

        Returns:
            Updated h
        """
        xr, xz, xh = torch.chunk(self.lin_x(x), 3, dim=1)
        r = torch.sigmoid(xr + self.lin_hr(h))
        z = torch.sigmoid(xz + self.lin_hz(h))
        u = torch.tanh(xh + self.lin_hh(r * h))

        dh = (1 - z) * (u - h)
        return dh

class FullGRUODECell_Autonomous(torch.nn.Module):
    
    def __init__(self, hidden_size, bias=True):
        """
        For p(t) modelling input_size should be 2x the x size.
        """
        super().__init__()

        #self.lin_xh = torch.nn.Linear(input_size, hidden_size, bias=bias)
        #self.lin_xz = torch.nn.Linear(input_size, hidden_size, bias=bias)
        #self.lin_xr = torch.nn.Linear(input_size, hidden_size, bias=bias)

        #self.lin_x = torch.nn.Linear(input_size, hidden_size * 3, bias=bias)

        self.lin_hh = torch.nn.Linear(hidden_size, hidden_size, bias=False)
        self.lin_hz = torch.nn.Linear(hidden_size, hidden_size, bias=False)
        self.lin_hr = torch.nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, t, h):
        """
        Executes one step with autonomous GRU-ODE for all h.
        The step size is given by delta_t.

        Args:
            t        time of evaluation
            h        hidden state (current)

        Returns:
            Updated h
        """
        #xr, xz, xh = torch.chunk(self.lin_x(x), 3, dim=1)
        x = torch.zeros_like(h)
        r = torch.sigmoid(x + self.lin_hr(h))
        z = torch.sigmoid(x + self.lin_hz(h))
        u = torch.tanh(x + self.lin_hh(r * h))

        dh = (1 - z) * (u - h)
        return dh

class GRUObservationCellLogvar(torch.nn.Module):
    """Implements discrete update based on the received observations."""

    def __init__(self, input_size, hidden_size, prep_hidden, bias=True):
        super().__init__()
        self.gru_d     = torch.nn.GRUCell(prep_hidden * input_size, hidden_size, bias=bias)
        self.gru_debug = torch.nn.GRUCell(prep_hidden * input_size, hidden_size, bias=bias)

        ## prep layer and its initialization
        std            = math.sqrt(2.0 / (4 + prep_hidden))
        self.w_prep    = torch.nn.Parameter(std * torch.randn(input_size, 4, prep_hidden))
        self.bias_prep = torch.nn.Parameter(0.1 + torch.zeros(input_size, prep_hidden))

        self.input_size  = input_size
        self.prep_hidden = prep_hidden

    def forward(self, h, p, X_obs, M_obs, i_obs):
        ## only updating rows that have observations
        p_obs        = p[i_obs]

        mean, logvar = torch.chunk(p_obs, 2, dim=1)
        sigma        = torch.exp(0.5 * logvar)
        error        = (X_obs - mean) / sigma

        ## log normal loss, over all observations
        log_lik_c    = np.log(np.sqrt(2*np.pi))
        losses       = 0.5 * ((torch.pow(error, 2) + logvar + 2*log_lik_c) * M_obs)
        if losses.sum()!=losses.sum():
            import ipdb; ipdb.set_trace()

        ## TODO: try removing X_obs (they are included in error)
        gru_input    = torch.stack([X_obs, mean, logvar, error], dim=2).unsqueeze(2)
        gru_input    = torch.matmul(gru_input, self.w_prep).squeeze(2) + self.bias_prep
        gru_input.relu_()
        ## gru_input is (sample x feature x prep_hidden)
        gru_input    = gru_input.permute(2, 0, 1)
        gru_input    = (gru_input * M_obs).permute(1, 2, 0).contiguous().view(-1, self.prep_hidden * self.input_size)

        temp = h.clone()
        temp[i_obs] = self.gru_d(gru_input, h[i_obs])
        h = temp

        return h, losses

class GRUObservationCell(torch.nn.Module):
    """Implements discrete update based on the received observations."""

    def __init__(self, input_size, hidden_size, prep_hidden, bias=True):
        super().__init__()
        self.gru_d     = torch.nn.GRUCell(prep_hidden * input_size, hidden_size, bias=bias)
        self.gru_debug = torch.nn.GRUCell(prep_hidden * input_size, hidden_size, bias=bias)

        ## prep layer and its initialization
        std            = math.sqrt(2.0 / (4 + prep_hidden))
        self.w_prep    = torch.nn.Parameter(std * torch.randn(input_size, 4, prep_hidden))
        self.bias_prep = torch.nn.Parameter(0.1 + torch.zeros(input_size, prep_hidden))

        self.input_size  = input_size
        self.prep_hidden = prep_hidden
        self.var_eps     = 1e-6

    def forward(self, h, p, X_obs, M_obs, i_obs):
        ## only updating rows that have observations
        p_obs     = p[i_obs]
        mean, var = torch.chunk(p_obs, 2, dim=1)
        ## making var non-negative and also non-zero (by adding a small value)
        var       = torch.abs(var) + self.var_eps
        error     = (X_obs - mean) / torch.sqrt(var)

        ## log normal loss, over all observations
        loss         = 0.5 * ((torch.pow(error, 2) + torch.log(var)) * M_obs).sum()


        ## TODO: try removing X_obs (they are included in error)
        gru_input    = torch.stack([X_obs, mean, var, error], dim=2).unsqueeze(2)
        gru_input    = torch.matmul(gru_input, self.w_prep).squeeze(2) + self.bias_prep
        gru_input.relu_()
        ## gru_input is (sample x feature x prep_hidden)
        gru_input    = gru_input.permute(2, 0, 1)
        gru_input    = (gru_input * M_obs).permute(1, 2, 0).contiguous().view(-1, self.prep_hidden * self.input_size)

        temp = h.clone()
        temp[i_obs] = self.gru_d(gru_input, h[i_obs])
        h = temp

        return h, loss


def init_weights(m):
    if type(m) == torch.nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0.05)

class NNFOwithBayesianJumps(torch.nn.Module):
    ## Neural Negative Feedback ODE with Bayesian jumps
    def __init__(self, input_size, hidden_size, p_hidden, prep_hidden, bias=True, cov_size=1, cov_hidden=1, classification_hidden=1, logvar=True, mixing=1, dropout_rate=0, full_gru_ode=False, solver="euler", impute = True, **options):
        """
        The smoother variable computes the classification loss as a weighted average of the projection of the latents at each observation.
        impute feeds the parameters of the distribution to GRU-ODE at each step.
        """

        super().__init__()

        self.impute = impute
        self.p_model = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, p_hidden, bias=bias),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=dropout_rate),
            torch.nn.Linear(p_hidden, 2 * input_size, bias=bias),
        )

        self.classification_model = torch.nn.Sequential(
            torch.nn.Linear(hidden_size,classification_hidden,bias=bias),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=dropout_rate),
            torch.nn.Linear(classification_hidden,1,bias=bias)
        )
        if full_gru_ode:
            if impute is False:
                self.gru_c   = FullGRUODECell_Autonomous(hidden_size, bias = bias)
            else:
                self.gru_c   = FullGRUODECell(2 * input_size, hidden_size, bias=bias)

        else:
            if impute is False:
                self.gru_c = GRUODECell_Autonomous(hidden_size, bias= bias)
            else:
                self.gru_c   = GRUODECell(2 * input_size, hidden_size, bias=bias)

        if logvar:
            self.gru_obs = GRUObservationCellLogvar(input_size, hidden_size, prep_hidden, bias=bias)
        else:
            self.gru_obs = GRUObservationCell(input_size, hidden_size, prep_hidden, bias=bias)

        self.covariates_map = torch.nn.Sequential(
            torch.nn.Linear(cov_size, cov_hidden, bias=bias),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=dropout_rate),
            torch.nn.Linear(cov_hidden, hidden_size, bias=bias),
            torch.nn.Tanh()
        )

        assert solver in ["euler", "midpoint", "dopri5"], "Solver must be either 'euler' or 'midpoint' or 'dopri5'."

        self.solver     = solver
        self.store_hist = options.pop("store_hist",False)
        self.input_size = input_size
        self.logvar     = logvar
        self.mixing     = mixing #mixing hyperparameter for loss_1 and loss_2 aggregation.

        self.apply(init_weights)

    def ode_step(self, h, p, delta_t, current_time):
        """Executes a single ODE step."""
        eval_times = torch.tensor([0],device = h.device, dtype = torch.float64)
        eval_ps = torch.tensor([0],device = h.device, dtype = torch.float32)
        if self.impute is False:
            p = torch.zeros_like(p)
            
        if self.solver == "euler":
            h = h + delta_t * self.gru_c(p, h)
            p = self.p_model(h)

        elif self.solver == "midpoint":
            k  = h + delta_t / 2 * self.gru_c(p, h)
            pk = self.p_model(k)

            h = h + delta_t * self.gru_c(pk, k)
            p = self.p_model(h)

        elif self.solver == "dopri5":
            assert self.impute==False #Dopri5 solver is only compatible with autonomous ODE.
            solution, eval_times, eval_vals = odeint(self.gru_c,h,torch.tensor([0,delta_t]),method=self.solver,options={"store_hist":self.store_hist})
            if self.store_hist:
                eval_ps = self.p_model(torch.stack([ev[0] for ev in eval_vals]))
            eval_times = torch.stack(eval_times) + current_time
            h = solution[1,:,:]
            p = self.p_model(h)
        
        current_time += delta_t
        return h,p,current_time, eval_times, eval_ps

        raise ValueError(f"Unknown solver '{self.solver}'.")

    def forward(self, times, time_ptr, X, M, obs_idx, delta_t, T, cov,
                return_path=False, smoother = False, class_criterion = None, labels=None):
        """
        Args:
            times      np vector of observation times
            time_ptr   start indices of data for a given time
            X          data tensor
            M          mask tensor (1.0 if observed, 0.0 if unobserved)
            obs_idx    observed patients of each datapoint (indexed within the current minibatch)
            delta_t    time step for Euler
            T          total time
            cov        static covariates for learning the first h0
            return_path   whether to return the path of h

        Returns:
            h          hidden state at final time (T)
            loss       loss of the Gaussian observations
        """

        h = self.covariates_map(cov)

        p            = self.p_model(h)
        current_time = 0.0
        counter      = 0

        loss_1 = 0 #Pre-jump loss
        loss_2 = 0 #Post-jump loss (KL between p_updated and the actual sample)

        if return_path:
            path_t = [0]
            path_p = [p]
            path_h = [h]

        if smoother:
            class_loss_vec = torch.zeros(cov.shape[0],device = h.device)
            num_evals_vec  = torch.zeros(cov.shape[0],device = h.device)
            class_criterion = class_criterion
            assert class_criterion is not None

        assert len(times) + 1 == len(time_ptr)
        assert (len(times) == 0) or (times[-1] <= T)

        eval_times_total = torch.tensor([],dtype = torch.float64, device = h.device)
        eval_vals_total  = torch.tensor([],dtype = torch.float32, device = h.device)

        for i, obs_time in enumerate(times):
            ## Propagation of the ODE until next observation
            while current_time < (obs_time-0.001*delta_t): #0.0001 delta_t used for numerical consistency.
                 
                if self.solver == "dopri5":
                    h, p, current_time, eval_times, eval_ps = self.ode_step(h, p, obs_time-current_time, current_time)
                else:
                    h, p, current_time, eval_times, eval_ps = self.ode_step(h, p, delta_t, current_time)
                eval_times_total = torch.cat((eval_times_total, eval_times))
                eval_vals_total  = torch.cat((eval_vals_total, eval_ps))

                #Storing the predictions.
                if return_path:
                    path_t.append(current_time)
                    path_p.append(p)
                    path_h.append(h)

            ## Reached an observation
            start = time_ptr[i]
            end   = time_ptr[i+1]

            X_obs = X[start:end]
            M_obs = M[start:end]
            i_obs = obs_idx[start:end]

            ## Using GRUObservationCell to update h. Also updating p and loss
            h, losses = self.gru_obs(h, p, X_obs, M_obs, i_obs)
           
            if smoother:
                class_loss_vec[i_obs] += class_criterion(self.classification_model(h[i_obs]),labels[i_obs]).squeeze(1)
                num_evals_vec[i_obs] +=1
            if losses.sum()!=losses.sum():
                import ipdb;ipdb.set_trace()
            loss_1    = loss_1+ losses.sum()
            p         = self.p_model(h)

            loss_2 = loss_2 + compute_KL_loss(p_obs = p[i_obs], X_obs = X_obs, M_obs = M_obs, logvar=self.logvar)

            if return_path:
                path_t.append(obs_time)
                path_p.append(p)
                path_h.append(h)

        ## after every observation has been processed, propagating until T
        while current_time < T:
            if self.solver == "dopri5":
                h, p, current_time,eval_times, eval_ps = self.ode_step(h, p, T-current_time, current_time)
            else:
                h, p, current_time,eval_times, eval_ps = self.ode_step(h, p, delta_t, current_time)
            eval_times_total = torch.cat((eval_times_total,eval_times))
            eval_vals_total  = torch.cat((eval_vals_total, eval_ps))
            #counter += 1
            #current_time = counter * delta_t
            
            #Storing the predictions
            if return_path:
                path_t.append(current_time)
                path_p.append(p)
                path_h.append(h)

        loss = loss_1 + self.mixing * loss_2

        if smoother:
            class_loss_vec += class_criterion(self.classification_model(h),labels).squeeze(1)
            class_loss_vec /= num_evals_vec
        
        class_pred = self.classification_model(h)
       
        if return_path:
            if smoother:
                return h, loss, class_pred, np.array(path_t), torch.stack(path_p), torch.stack(path_h), class_loss_vec
            else:
                return h, loss, class_pred, np.array(path_t), torch.stack(path_p), torch.stack(path_h), eval_times_total, eval_vals_total
        else:
            if smoother:
                return h, loss, class_pred, class_loss_vec
            else:
                return h, loss, class_pred, loss_1

def compute_KL_loss(p_obs, X_obs, M_obs, obs_noise_std=1e-2, logvar=True):
    obs_noise_std = torch.tensor(obs_noise_std)
    if logvar:
        mean, var = torch.chunk(p_obs, 2, dim=1)
        std = torch.exp(0.5*var)
    else:
        mean, var = torch.chunk(p_obs, 2, dim=1)
        ## making var non-negative and also non-zero (by adding a small value)
        std       = torch.pow(torch.abs(var) + 1e-5,0.5)

    return (gaussian_KL(mu_1 = mean, mu_2 = X_obs, sigma_1 = std, sigma_2 = obs_noise_std)*M_obs).sum()


def gaussian_KL(mu_1, mu_2, sigma_1, sigma_2):
    return(torch.log(sigma_2) - torch.log(sigma_1) + (torch.pow(sigma_1,2)+torch.pow((mu_1 - mu_2),2)) / (2*sigma_2**2) - 0.5)


class GRUODEBayesSeq(torch.nn.Module):
    def __init__(self, input_size, hidden_size, p_hidden, prep_hidden, bias=True, cov_size=1, cov_hidden=1, classification_hidden=1, logvar=True, mixing=1, dropout_rate=0, obs_noise_std=1e-2, full_gru_ode=False):
        super().__init__()
        self.obs_noise_std = obs_noise_std
        self.classification_model = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, classification_hidden, bias=bias),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=dropout_rate),
            torch.nn.Linear(classification_hidden, 1, bias=bias),
        )
        if full_gru_ode:
            self.gru_c   = FullGRUODECell(2 * input_size, hidden_size, bias=bias)
        else:
            self.gru_c   = GRUODECell(2 * input_size, hidden_size, bias=bias)

        self.gru_bayes = SeqGRUBayes(input_size=input_size, hidden_size=hidden_size, prep_hidden=prep_hidden, p_hidden=p_hidden, bias=bias)

        self.covariates_map = torch.nn.Sequential(
            torch.nn.Linear(cov_size, cov_hidden, bias=bias),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=dropout_rate),
            torch.nn.Linear(cov_hidden, hidden_size, bias=bias),
        )
        self.input_size = input_size
        self.mixing     = mixing #mixing hyperparameter for loss_1 and loss_2 aggregation.
        self.apply(init_weights)

    def forward(self, times, time_ptr, Xpadded, Fpadded, X, M, lengths,
                obs_idx, delta_t, T, cov, return_path=False):
        """
        Args:
            times      np vector of observation times
            time_ptr   start indices of data for a given time
            Xpadded    data tensor (padded)
            Fpadded    feature id of each data point (padded)
            X          observation tensor
            M          mask tensor
            obs_idx    observed patients of each datapoint (current minibatch)
            delta_t    time step for Euler
            T          total time
            cov        static covariates for learning the first h0
            return_path   whether to return the path of h

        Returns:
            h          hidden state at final time (T)
            loss       loss of the Gaussian observations
        """

        h       = self.covariates_map(cov)
        p       = self.gru_bayes.p_model(h)
        time    = 0.0
        counter = 0

        loss_1 = 0 # Pre-jump loss
        loss_2 = 0 # Post-jump loss

        if return_path:
            path_t = [0]
            path_p = [p]

        assert len(times) + 1 == len(time_ptr)
        assert (len(times) == 0) or (times[-1] <= T)

        for i, obs_time in enumerate(times):
            ## Propagation of the ODE until obs_time
            while time < obs_time:
                h = h + delta_t * self.gru_c(p, h)
                p = self.gru_bayes.p_model(h)

                ## using counter to avoid numerical errors
                counter += 1
                time = counter * delta_t
                ## Storing the predictions.
                if return_path:
                    path_t.append(time)
                    path_p.append(p)

            ## Reached obs_time
            start = time_ptr[i]
            end   = time_ptr[i+1]

            L_obs = lengths[start:end]
            X_obs = pack_padded_sequence(Xpadded[start:end], L_obs, batch_first=True)
            F_obs = pack_padded_sequence(Fpadded[start:end], L_obs, batch_first=True)
            i_obs = obs_idx[start:end]

            Xf_batch = X[start:end]
            Mf_batch = M[start:end]

            ## Using GRU-Bayes to update h. Also updating p and loss
            h, loss_i, loss_pre = self.gru_bayes(h, X_obs, F_obs, i_obs, X=Xf_batch, M=Mf_batch)
            loss_1    = loss_1 + loss_i + loss_pre.sum()
            p         = self.gru_bayes.p_model(h)

            loss_2 = loss_2 + compute_KL_loss(p_obs = p[i_obs], X_obs = Xf_batch, M_obs = Mf_batch, obs_noise_std=self.obs_noise_std)

            if return_path:
                path_t.append(obs_time)
                path_p.append(p)

        while time < T:
            h = h + delta_t * self.gru_c(p, h)
            p = self.gru_bayes.p_model(h)

            counter += 1
            time = counter * delta_t
            if return_path:
                path_t.append(time)
                path_p.append(p)

        loss = loss_1 + self.mixing * loss_2
        class_pred = self.classification_model(h)
        if return_path:
            return h, loss, class_pred, np.array(path_t), torch.stack(path_p)
        return h, loss, class_pred


class SeqGRUBayes(torch.nn.Module):
    """

    Inputs to forward:
        h      tensor of hiddens
        X_obs  PackedSequence of observation values
        F_obs  PackedSequence of feature ids
        i_obs  indices of h that have been observed

    Returns updated h.
    """
    def __init__(self, input_size, hidden_size, prep_hidden, p_hidden, bias=True):
        super().__init__()
        self.p_model = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, p_hidden, bias=bias),
            torch.nn.ReLU(),
            torch.nn.Linear(p_hidden, 2 * input_size, bias=bias),
        )
        self.gru = torch.nn.GRUCell(prep_hidden, hidden_size, bias=bias)

        ## prep layer and its initialization
        std            = math.sqrt(2.0 / (4 + prep_hidden))
        self.w_prep    = torch.nn.Parameter(std * torch.randn(input_size, 4, prep_hidden))
        self.bias_prep = torch.nn.Parameter(0.1 + torch.zeros(input_size, prep_hidden))

        self.input_size  = input_size
        self.prep_hidden = prep_hidden
        self.var_eps     = 1e-6

    def p_mean_logvar(self, h):
        p      = self.p_model(h)
        mean, logvar = torch.chunk(p, 2, dim=1)
        return mean, logvar

    def step_1feature(self, hidden, X_step, F_step):
        ## 1) Compute error on the observed features
        mean, logvar = self.p_mean_logvar(hidden)
        ## mean, logvar both are [ Batch  x  input_size ]
        hrange = torch.arange(hidden.shape[0])
        mean   = mean[   hrange, F_step ]
        logvar = logvar[ hrange, F_step ]

        sigma  = torch.exp(0.5 * logvar)
        error  = (X_step - mean) / sigma

        ## log normal loss, over all observations
        loss   = 0.5 * (torch.pow(error, 2) + logvar).sum()

        ## TODO: try removing X_obs (they are included in error)
        gru_input = torch.stack([X_step, mean, logvar, error], dim=1).unsqueeze(1)
        ## 2) Select the matrices from w_prep and bias_prep; multiply
        W         = self.w_prep[F_step, :, :]
        bias      = self.bias_prep[F_step]
        gru_input = torch.matmul(gru_input, W).squeeze(1) + bias
        gru_input.relu_()

        return self.gru(gru_input, hidden), loss

    def ode_step(self, h, p, delta_t):
        if self.solver == "euler":
            h = h + delta_t * self.gru_c(p, h)
            p = self.p_model(h)
            return h, p

        if self.solver == "midpoint":
            k  = h + delta_t / 2 * self.gru_c(p, h)
            pk = self.p_model(k)

            h2 = h + delta_t * self.gru_c(pk, k)
            p2 = self.p_model(h2)
            return h2, p2

        raise ValueError(f"Unknown solver '{self.solver}'.")

    def forward(self, h, X_obs, F_obs, i_obs, X, M):
        """
        See https://github.com/pytorch/pytorch/blob/a462edd0f6696a4cac4dd04c60d1ad3c9bc0b99c/torch/nn/_functions/rnn.py#L118-L154
        """
        ## selecting h to be updated
        hidden = h[i_obs]

        output          = []
        input_offset    = 0
        last_batch_size = X_obs.batch_sizes[0]
        hiddens         = []
        #flat_hidden     = not isinstance(hidden, tuple)


        ## computing loss before any updates
        mean, logvar = self.p_mean_logvar(hidden)
        sigma        = torch.exp(0.5 * logvar)
        error        = (X - mean) / sigma
        losses_pre   = 0.5 * ((torch.pow(error, 2) + logvar) * M)

        ## updating hidden
        loss = 0
        input_offset = 0
        for batch_size in X_obs.batch_sizes:
            X_step = X_obs.data[input_offset:input_offset + batch_size]
            F_step = F_obs.data[input_offset:input_offset + batch_size]
            input_offset += batch_size

            dec = last_batch_size - batch_size
            if dec > 0:
                hiddens.append(hidden[-dec:])
                hidden = hidden[:-dec]
            last_batch_size = batch_size

            hidden, loss_b = self.step_1feature(hidden, X_step, F_step)
            loss = loss + loss_b

        hiddens.append(hidden)
        hiddens.reverse()

        hidden = torch.cat(hiddens, dim=0)

        ## updating observed trajectories
        h2        = h.clone()
        h2[i_obs] = hidden

        return h2, loss, losses_pre

class Discretized_GRU(torch.nn.Module):
    ## Discretized GRU model (GRU-ODE-Bayes without ODE and without Bayes)
    def __init__(self, input_size, hidden_size, p_hidden, prep_hidden, bias=True, cov_size=1, cov_hidden=1, classification_hidden=1, logvar=True, mixing=1, dropout_rate=0, impute=True):
        """
        The smoother variable computes the classification loss as a weighted average of the projection of the latents at each observation.
        """
        
        super().__init__()
        self.impute = impute
        self.p_model = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, p_hidden, bias=bias),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=dropout_rate),
            torch.nn.Linear(p_hidden, 2 * input_size, bias=bias),
        )

        self.classification_model = torch.nn.Sequential(
            torch.nn.Linear(hidden_size,classification_hidden,bias=bias),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=dropout_rate),
            torch.nn.Linear(classification_hidden,1,bias=bias)
        )

        self.gru = torch.nn.GRUCell(2*input_size, hidden_size, bias = bias)

        if logvar:
            self.gru_obs = GRUObservationCellLogvar(input_size, hidden_size, prep_hidden, bias=bias)
        else:
            self.gru_obs = GRUObservationCell(input_size, hidden_size, prep_hidden, bias=bias)

        self.covariates_map = torch.nn.Sequential(
            torch.nn.Linear(cov_size, cov_hidden, bias=bias),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=dropout_rate),
            torch.nn.Linear(cov_hidden, hidden_size, bias=bias),
            torch.nn.Tanh()
        )


        self.input_size = input_size
        self.logvar     = logvar
        self.mixing     = mixing #mixing hyperparameter for loss_1 and loss_2 aggregation.

        self.apply(init_weights)

    def ode_step(self, h, p, delta_t):
 
        h = self.gru(p,h)

        raise ValueError(f"Unknown solver '{self.solver}'.")

    def forward(self, times, time_ptr, X, M, obs_idx, delta_t, T, cov,
                return_path=False, smoother = False, class_criterion = None, labels=None):
        """
        Args:
            times      np vector of observation times
            time_ptr   start indices of data for a given time
            X          data tensor
            M          mask tensor (1.0 if observed, 0.0 if unobserved)
            obs_idx    observed patients of each datapoint (indexed within the current minibatch)
            delta_t    time step for Euler
            T          total time
            cov        static covariates for learning the first h0
            return_path   whether to return the path of h

        Returns:
            h          hidden state at final time (T)
            loss       loss of the Gaussian observations
        """


        h = self.covariates_map(cov)

        p            = self.p_model(h)
        current_time = 0.0
        counter      = 0

        loss_1 = 0 #Pre-jump loss
        loss_2 = 0 #Post-jump loss (KL between p_updated and the actual sample)

        if return_path:
            path_t = [0]
            path_p = [p]
            path_h = [h]

        if smoother:
            class_loss_vec = torch.zeros(cov.shape[0],device = h.device)
            num_evals_vec  = torch.zeros(cov.shape[0],device = h.device)
            class_criterion = class_criterion
            assert class_criterion is not None

        assert len(times) + 1 == len(time_ptr)
        assert (len(times) == 0) or (times[-1] <= T)

        for i, obs_time in enumerate(times):
            ## Propagation of the ODE until next observation
            while current_time < (obs_time-0.001*delta_t): #0.0001 delta_t used for numerical consistency.
                
                if self.impute is False:
                    p = torch.zeros_like(p)
                h = self.gru(p, h)
                p = self.p_model(h)

                ## using counter to avoid numerical errors
                counter += 1
                current_time = counter * delta_t
                #Storing the predictions.
                if return_path:
                    path_t.append(current_time)
                    path_p.append(p)
                    path_h.append(h)

            ## Reached an observation
            start = time_ptr[i]
            end   = time_ptr[i+1]

            X_obs = X[start:end]
            M_obs = M[start:end]
            i_obs = obs_idx[start:end]

            ## Using GRUObservationCell to update h. Also updating p and loss
            h, losses = self.gru_obs(h, p, X_obs, M_obs, i_obs)
           
            if smoother:
                class_loss_vec[i_obs] += class_criterion(self.classification_model(h[i_obs]),labels[i_obs]).squeeze(1)
                num_evals_vec[i_obs] +=1
            loss_1    = loss_1 + losses.sum()
            p         = self.p_model(h)

            loss_2 = loss_2 + compute_KL_loss(p_obs = p[i_obs], X_obs = X_obs, M_obs = M_obs, logvar=self.logvar)

            if return_path:
                path_t.append(obs_time)
                path_p.append(p)
                path_h.append(h)


        ## after every observation has been processed, propagating until T
        while current_time < T:
            if self.impute is False:
                p = torch.zeros_like(p)
            h = self.gru(p,h)
            p = self.p_model(h)

            counter += 1
            current_time = counter * delta_t
            #Storing the predictions
            if return_path:
                path_t.append(current_time)
                path_p.append(p)
                path_h.append(h)

        loss = loss_1 + self.mixing * loss_2

        if smoother:
            class_loss_vec += class_criterion(self.classification_model(h),labels).squeeze(1)
            class_loss_vec /= num_evals_vec
        
        class_pred = self.classification_model(h)
       
        if return_path:
            if smoother:
                return h, loss, class_pred, np.array(path_t), torch.stack(path_p), torch.stack(path_h), class_loss_vec
            else:
                return h, loss, class_pred, np.array(path_t), torch.stack(path_p), torch.stack(path_h)
        else:
            if smoother:
                return h, loss, class_pred, class_loss_vec
            else:
                return h, loss, class_pred
