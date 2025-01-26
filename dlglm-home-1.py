def dlglm(X,Rx,Y,Ry, covars_r_x, covars_r_y, norm_means_x, norm_sds_x, norm_mean_y, norm_sd_y, learn_r, data_types_x, data_types_x_0, Cs, Cy, early_stop, X_val, Rx_val, Y_val, Ry_val, Ignorable=False, family="Cox", link="log", impute_bs=None,arch="IWAE",draw_miss=True,pre_impute_value=0,n_hidden_layers=2,n_hidden_layers_y=0,n_hidden_layers_r=0,h1=8,h2=8,h3=0,phi0=None,phi=None,train=1,saved_model=None,sigma="elu",bs = 64,n_epochs = 2002,lr=0.001,niws_z=20,M=20,dim_z=5,dir_name=".",trace=False,save_imps=False, test_temp=0.5, L1_weight=0, init_r="default", full_obs_ids=None, miss_ids=None, unbalanced=False):

    weight_y = 1

    if (h2 is None) and (h3 is None):
        h2=h1; h3=h1

    import torch     
    import torch.nn as nn
    import numpy as np
    import numpy_indexed as npi
    import scipy.stats
    import scipy.io
    import scipy.sparse
    import pandas as pd
    import torch.distributions as td
    from torch import nn, optim
    from torch.nn import functional as F
    from torch.distributions import constraints
    from torch.distributions.distribution import Distribution
    from torch.distributions.utils import broadcast_all
    import torch.nn.functional as F
    from torch.autograd import Variable
    import torch.nn.utils.prune as prune
    from collections import OrderedDict
    import os
    import sys
    import h5py
    # Additional imports for Cox model
    from lifelines.utils import concordance_index
    
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    
    torch.cuda.empty_cache()
    
    if np.all(Rx==1) and np.all(Ry==1):
        Ignorable=True

    ids_real = data_types_x=='real'; p_real=np.sum(ids_real)
    ids_count = data_types_x=='count'; p_count=np.sum(ids_count)
    ids_cat = data_types_x=='cat'; p_cat = len(Cs) 
    ids_pos = data_types_x=='pos'; p_pos = np.sum(ids_pos)

    exists_types = [p_real>0, p_count>0, p_pos>0, p_cat>0]   
    print("exists_types (real, count, cat, pos):")
    print(exists_types)
    print("p_real, p_count, p_pos, p_cat:")
    print(str(p_real) + ", " + str(p_count) + ", " + str(p_pos) + ", " + str(p_cat))
    ids_types = [ids_real, ids_count, ids_pos, ids_cat]

    temp_min=torch.tensor(0.5,device="cuda:0",dtype=torch.float64)

    if exists_types[3]:
        temp0 = torch.ones([1], dtype=torch.float64, device='cuda:0')
        temp = torch.ones([1], dtype=torch.float64, device='cuda:0')
    else:
        temp0 = torch.tensor(0.5,device="cuda:0",dtype=torch.float64)  
        temp = torch.tensor(0.5,device="cuda:0",dtype=torch.float64)

    ANNEAL_RATE = torch.tensor(0.003,device="cuda:0",dtype=torch.float64)

    if (family=="Multinomial"):
        C=Cy  
    elif (family=="Cox"):
        C=2  # For survival time and event status
    else:
        C=1

    miss_x = False; miss_y = False
    if np.sum(Rx==0)>0: miss_x = True
    if np.sum(Ry==0)>0: miss_y = True

    covars_miss = None ; covars=False  

    def mse(xhat,xtrue,mask):
        xhat = np.array(xhat)
        xtrue = np.array(xtrue)
        return {'miss':np.mean(np.power(xhat-xtrue,2)[mask<0.5]),'obs':np.mean(np.power(xhat-xtrue,2)[mask>0.5])}

    def cox_mse(xhat, xtrue, mask):
        """MSE for Cox model survival times"""
        xhat = np.array(xhat)
        xtrue = np.array(xtrue) 
        return {'miss':np.mean(np.power(xhat[:,0]-xtrue[:,0],2)[mask<0.5]),
                'obs':np.mean(np.power(xhat[:,0]-xtrue[:,0],2)[mask>0.5])}

    def cox_concordance(time_hat, event_hat, time_true, event_true, mask):
        """Concordance index for Cox model predictions"""
        mask = mask.astype(bool)
        c_obs = concordance_index(time_true[mask], time_hat[mask], event_true[mask]) if np.any(mask) else np.nan
        c_miss = concordance_index(time_true[~mask], time_hat[~mask], event_true[~mask]) if np.any(~mask) else np.nan
        return {'miss': c_miss, 'obs': c_obs}

    def pred_acc(xhat,xtrue,mask,Cs):
        if type(Cs)==int:
            xhat0=xhat; xtrue0=xtrue; mask0=mask
        else:
            xhat0 = np.empty([xhat.shape[0], len(Cs)])
            xtrue0 = np.empty([xtrue.shape[0], len(Cs)])
            mask0 = np.empty([mask.shape[0], len(Cs)])
            for i in range(0,len(Cs)):
                xhat0[:,i] = np.argmax(xhat[:,int(Cs[i]*i):int(Cs[i]*(i+1))], axis=1) + 1
                xtrue0[:,i] = np.argmax(xtrue[:,int(Cs[i]*i):int(Cs[i]*(i+1))], axis=1) + 1
                mask0[:,i] = mask[:,int(Cs[i]*i)]
        return {'miss':np.mean((xhat0==xtrue0)[mask0<0.5]),'obs':np.mean((xhat0==xtrue0)[mask0>0.5])}
    
    # LEVEL 1 - Inside main dlglm function
    xfull = (X - norm_means_x)/norm_sds_x    
    if family=="Gaussian":
        yfull = (Y - norm_mean_y)/norm_sd_y
    elif family=="Cox":
    # For Cox model, normalize survival time but keep event indicator as is
        yfull = np.zeros_like(Y)
        yfull[:,0] = (Y[:,0] - norm_mean_y)/norm_sd_y  # Normalize time
        yfull[:,1] = Y[:,1]  # Keep event indicator unchanged
    else: 
        yfull = Y.astype("float")

    # LEVEL 1  
    if early_stop:
        xfull_val = (X_val - norm_means_x)/norm_sds_x
    if family=="Gaussian":
        yfull_val = (Y_val - norm_mean_y)/norm_sd_y
    elif family=="Cox":
        yfull_val = np.zeros_like(Y_val)
        yfull_val[:,0] = (Y_val[:,0] - norm_mean_y)/norm_sd_y
        yfull_val[:,1] = Y_val[:,1]
    else: 
        yfull_val = Y_val.astype("float")

    # LEVEL 1
    n = xfull.shape[0] 
    p = xfull.shape[1] 
    np.random.seed(1234)

    # LEVEL 1
    bs = min(bs,n)
    if (impute_bs==None): 
        impute_bs = n      
    else: 
        impute_bs = min(impute_bs, n)

    # LEVEL 1  
    xmiss = np.copy(xfull)
    xmiss[Rx==0]=np.nan
    mask_x = np.isfinite(xmiss)

    ymiss = np.copy(yfull)
    ymiss[Ry==0]=np.nan
    mask_y = np.isfinite(ymiss)

    yhat_0 = np.copy(ymiss)
    xhat_0 = np.copy(xmiss)

    # LEVEL 1
    if early_stop:
        xmiss_val = np.copy(xfull_val)
        xmiss_val[Rx_val==0]=np.nan
        mask_x_val = np.isfinite(xmiss_val)
        
        ymiss_val = np.copy(yfull_val)
        ymiss_val[Ry_val==0]=np.nan
        mask_y_val = np.isfinite(ymiss_val)
        
        yhat_0_val = np.copy(ymiss_val)
        xhat_0_val = np.copy(xmiss_val)

    # LEVEL 1  
    if (pre_impute_value == "mean_obs"):
        xhat_0[Rx==0] = np.mean(xmiss[Rx==1],0)
        yhat_0[Ry==0] = np.mean(ymiss[Ry==1],0)
        if early_stop: 
            xhat_0_val[Rx_val==0] = np.mean(xmiss_val[Rx_val==1],0)
            yhat_0_val[Ry_val==0] = np.mean(ymiss_val[Ry_val==1],0)
    elif (pre_impute_value == "mean_miss"):
        xhat_0[Rx==0] = np.mean(xmiss[Rx==0],0)
        yhat_0[Ry==0] = np.mean(ymiss[Ry==0],0)
        if early_stop: 
            xhat_0_val[Rx_val==0] = np.mean(xmiss_val[Rx_val==0],0)
            yhat_0_val[Ry_val==0] = np.mean(ymiss_val[Ry_val==0],0)
    elif (pre_impute_value == "truth"):
        xhat_0 = np.copy(xfull)
        yhat_0 = np.copy(yfull)
        if early_stop: 
            xhat_0_val = np.copy(xfull_val)
            yhat_0_val = np.copy(yfull_val)
    else:
        xhat_0[np.isnan(xmiss)] = pre_impute_value
        yhat_0[np.isnan(ymiss)] = pre_impute_value
        if early_stop: 
            xhat_0_val[np.isnan(xmiss_val)] = pre_impute_value
            yhat_0_val[np.isnan(ymiss_val)] = pre_impute_value

    # LEVEL 1
    init_mse = mse(xfull,xhat_0,mask_x)
    print("Pre-imputation MSE (obs, should be 0): " + str(init_mse['obs']))
    print("Pre-imputation MSE (miss): " + str(init_mse['miss']))

    # LEVEL 1
    prx = np.sum(covars_r_x).astype(int)
    pry = np.sum(covars_r_y).astype(int)
    pr = prx + pry
    if not learn_r: 
        phi=torch.from_numpy(phi).float().cuda()

    # LEVEL 1
    # Define decoder/encoder
    if (sigma=="relu"): 
        act_fun=torch.nn.ReLU()
    elif (sigma=="elu"): 
        act_fun=torch.nn.ELU()
    elif (sigma=="tanh"): 
        act_fun=torch.nn.Tanh()
    elif (sigma=="sigmoid"): 
        act_fun=torch.nn.Sigmoid()

    # LEVEL 1
    p_miss = np.sum(~full_obs_ids)

    # LEVEL 1
    if family=="Gaussian":
        n_params_ym = 2
        n_params_y = 1
    elif family=="Multinomial":
        n_params_ym = C    # probs for each of the K classes
        n_params_y = C
    elif family=="Cox":
        n_params_ym = 2    # time and event status
        n_params_y = 1     # hazard function output
    elif family=="Poisson":
        n_params_ym = 1
        n_params_y = 1

    # LEVEL 1
    n_params_r = p_miss*(miss_x) + 1*(miss_y) # Bernoulli (prob. p features in X) --> 1 if missing in y and not X. #of missing features to model.


    # LEVEL 1
    def network_maker(act_fun, n_hidden_layers, in_h, h, out_h, bias=True, dropout=False, init="orthogonal"):
        # LEVEL 2
        if n_hidden_layers==0:
            layers = [ nn.Linear(in_h, out_h, bias), ]
        elif n_hidden_layers>0:
            layers = [ nn.Linear(in_h , h, bias), act_fun, ]
            for i in range(n_hidden_layers-1):
                layers.append( nn.Linear(h, h, bias), )
                layers.append( act_fun, )
            layers.append(nn.Linear(h, out_h, bias))
        elif n_hidden_layers<0:
            raise Exception("n_hidden_layers must be >= 0")
        
        # LEVEL 2
        if dropout:
            layers.insert(0, nn.Dropout(p=dropout_pct))
        
        # LEVEL 2
        model = nn.Sequential(*layers)
        
        # LEVEL 2 
        def weights_init(layer):
            if init=="normal":
                if type(layer) == nn.Linear: torch.nn.init.normal_(layer.weight, mean=0, std=10)
            elif init=="orthogonal":
                if type(layer) == nn.Linear: torch.nn.init.orthogonal_(layer.weight)
            elif init=="uniform":
                if type(layer) == nn.Linear: torch.nn.init.uniform_(layer.weight, a=-2, b=2)
        model.apply(weights_init)
        
        return model

    # LEVEL 1 
    NNs_xm = {}
    if Ignorable: 
        p2 = p+dim_z+1
    else: 
        p2 = 2*p+dim_z+1

    # LEVEL 1
    init0 = "orthogonal"
    if miss_x:
        if exists_types[0]: 
            NNs_xm['real'] = network_maker(act_fun, n_hidden_layers, p2, h1, 2*p_real, True, False, init0).cuda()
        if exists_types[1]: 
            NNs_xm['count'] = network_maker(act_fun, n_hidden_layers, p2, h1, 2*p_count, True, False, init0).cuda()
        if exists_types[2]: 
            NNs_xm['pos'] = network_maker(act_fun, n_hidden_layers, p2, h1, 2*p_pos, True, False, init0).cuda()
        if exists_types[3]:
            NNs_xm['cat']=[]
            where_ids_cat = np.where(ids_cat)
            for ii in range(0, p_cat):
                NNs_xm['cat'].append(network_maker(act_fun, n_hidden_layers, p2, h1, int(Cs[ii]), True, False, init0).cuda())

    # LEVEL 1
    if Ignorable: 
        p3 = p+1
    else: 
        p3 = 2*p+2

    # LEVEL 1
    # Neural network for Y missing values
    if miss_y:
        if family == "Cox":
            # For Cox model: predict both time and event (2 outputs)
            NN_ym = network_maker(act_fun, n_hidden_layers_y, p3, h1, 2, True, False, init0).cuda()
        else:
            NN_ym = network_maker(act_fun, n_hidden_layers_y, p3, h1, n_params_ym, True, False, init0).cuda()
    else:
        NN_ym = None

    # LEVEL 1
    # Main prediction network
    if family == "Cox":
        # Cox model: output hazard function
        NN_y = network_maker(act_fun, n_hidden_layers_y, p, h2, n_params_y, True, False, init0).cuda()
    else: 
        NN_y = network_maker(act_fun, n_hidden_layers_y, p, h2, n_params_y, True, False, init0).cuda()

    # LEVEL 1
    # Missingness network
    if not Ignorable:
        NN_r = network_maker(act_fun, n_hidden_layers_r, pr, h3, n_params_r, True, False, init0).cuda()
    else:
        NN_r = None

    # LEVEL 1
    if init_r=="alt" and not Ignorable:
        dist = torch.distributions.Uniform(torch.Tensor([-2]), torch.Tensor([2]))
        sh1, sh2 = NN_r[0].weight.shape
        
        custom_weights = (dist.sample([sh1, sh2]).reshape([sh1,sh2])).cuda()
        with torch.no_grad():
            NN_r[0].weight = torch.nn.Parameter(custom_weights)

    # LEVEL 1  
    encoder = network_maker(act_fun, n_hidden_layers, p, h1, 2*dim_z, True, False, init0).cuda()

    # LEVEL 1
    decoders = { }
    if exists_types[0]:
        decoders['real'] = network_maker(act_fun, n_hidden_layers, dim_z, h1, 2*p_real, True, False, init0).cuda()
    if exists_types[1]:
        decoders['count'] = network_maker(act_fun, n_hidden_layers, dim_z, h1, 2*p_count, True, False, init0).cuda()
    if exists_types[2]:
        decoders['pos'] = network_maker(act_fun, n_hidden_layers, dim_z, h1, 2*p_pos, True, False, init0).cuda()
    if exists_types[3]:
        decoders['cat']=[]
        for ii in range(0,p_cat):
            decoders['cat'].append(network_maker(act_fun, n_hidden_layers, dim_z, h1, int(Cs[ii]), True, False, init0).cuda())

    # LEVEL 1
    alpha = torch.ones(1, requires_grad=True, device="cuda:0")

    # LEVEL 1
    def invlink(link="identity"):
        # LEVEL 2
        if link=="identity":
            fx = torch.nn.Identity(0)
        elif link=="log":
            fx = torch.exp
        elif link=="logit":
            fx = torch.nn.Sigmoid()
        elif link=="mlogit":
            fx = torch.nn.Softmax(dim=1)
        return fx

    # LEVEL 1
    def V(mu, alpha, family="Gaussian"):
        # LEVEL 2
        if family=="Gaussian":
            out = alpha*torch.ones([mu.shape[0]]).cuda()
        elif family=="Poisson":
            out = mu
        elif family=="NB":
            out = mu + alpha*torch.pow(mu, 2).cuda()
        elif family=="Binomial":
            out = mu*(1-(mu/n_successes))
        elif family=="Multinomial":
            out = mu*(1-mu)
        elif family=="Cox":
            # For Cox model: variance function for hazard
            out = mu
        return out
    
    # LEVEL 1
    def cox_loss(risk_scores, times, events):
        """
        Calculate Cox partial likelihood loss
        risk_scores: predicted risk scores from model
        times: observed/censored times
        events: event indicators (1 if event, 0 if censored)
        """
        # LEVEL 2
        # Sort by descending time order
        order = torch.argsort(times, descending=True)
        risk_scores = risk_scores[order]
        times = times[order]
        events = events[order]
        
        # LEVEL 2
        # Calculate log partial likelihood
        log_risk = risk_scores
        exp_risk = torch.exp(risk_scores)
        running_sum = torch.zeros_like(times)
        running_sum[0] = exp_risk[0]
        
        # LEVEL 2
        for i in range(1, len(times)):
            running_sum[i] = running_sum[i-1] + exp_risk[i]
        
        # LEVEL 2
        # Loss is negative log partial likelihood
        loss = -torch.sum(events * (log_risk - torch.log(running_sum)))
        return loss

    # LEVEL 1
    def cox_baseline_hazard(risk_scores, times, events):
        """
        Estimate baseline hazard function
        """
        # LEVEL 2
        order = torch.argsort(times)
        risk_scores = risk_scores[order]
        times = times[order]
        events = events[order]
        
        # LEVEL 2
        exp_risk = torch.exp(risk_scores)
        unique_times = torch.unique(times)
        baseline_hazard = torch.zeros_like(unique_times)
        
        # LEVEL 2
        for i, t in enumerate(unique_times):
            at_risk = (times >= t)
            events_at_t = events[times == t]
            if torch.sum(events_at_t) > 0:
                baseline_hazard[i] = torch.sum(events_at_t) / torch.sum(exp_risk[at_risk])
                
        return baseline_hazard, unique_times
    
    # LEVEL 1
    def compute_cindex(risk_scores, times, events):
        """
        Compute concordance index for Cox model predictions
        """
        # LEVEL 2
        total_pairs = 0
        concordant_pairs = 0
        n = len(times)
        
        # LEVEL 2
        for i in range(n):
            if events[i] == 0:  # Skip censored cases
                continue
            for j in range(n):
                if times[i] <= times[j]:  # Skip invalid pairs
                    continue
                # Valid pair found
                total_pairs += 1
                if risk_scores[i] > risk_scores[j]:
                    concordant_pairs += 1
        
        # LEVEL 2
        return concordant_pairs / total_pairs if total_pairs > 0 else float('nan')

    # LEVEL 1
    def compute_brier_score(risk_scores, baseline_hazard, times, events, eval_times):
        """
        Compute Brier score at specified evaluation times
        """
        # LEVEL 2
        n = len(times)
        n_times = len(eval_times)
        brier_scores = torch.zeros(n_times)
        
        # LEVEL 2
        for i, t in enumerate(eval_times):
            survival_probs = torch.exp(-baseline_hazard.cumsum(0).reshape(-1, 1) * 
                                    torch.exp(risk_scores).reshape(1, -1))
            observed = (times > t).float()
            weights = torch.ones(n)  # Can be modified for inverse probability of censoring weights
            brier_scores[i] = torch.mean(weights * (survival_probs - observed) ** 2)
        
        return brier_scores
    # LEVEL 1
    p_z = td.Independent(td.Normal(loc=torch.zeros(dim_z).cuda(),scale=torch.ones(dim_z).cuda()),1)

    # LEVEL 1
    def forward(iota_xfull, iota_yfull, iota_x, iota_y, mask_x, mask_y, batch_size, niw, temp):
        # LEVEL 2: Tiling
        tiled_iota_x = torch.Tensor.repeat(iota_x, [niw, 1])
        tiled_mask_x = torch.Tensor.repeat(mask_x, [niw, 1]) 
        tiled_iota_y = torch.Tensor.repeat(iota_y, [niw, 1])
        tiled_mask_y = torch.Tensor.repeat(mask_y, [niw, 1])
        
        # LEVEL 2: Optional full data tiling
        if not draw_miss:
            tiled_iota_xfull = torch.Tensor.repeat(iota_xfull, [niw, 1])
            tiled_iota_yfull = torch.Tensor.repeat(iota_yfull, [niw, 1])
        else:
            tiled_iota_xfull = None
            tiled_iota_yfull = None
        
        # LEVEL 2: Encoder processing
        out_encoder = encoder(iota_x)
        
        # LEVEL 2: Normal distribution for latent space
        qzgivenx = td.Normal(
            loc=out_encoder[..., :dim_z],
            scale=torch.nn.Softplus()(out_encoder[..., dim_z:(2*dim_z)]) + 0.001
        )
        
        # LEVEL 2: Parameters for latent space
        params_z = {
            'mean': out_encoder[..., :dim_z].reshape([batch_size, dim_z]).detach().cpu().data.numpy(),
            'scale': torch.nn.Softplus()(out_encoder[..., dim_z:(2*dim_z)]).reshape([batch_size, dim_z]).detach().cpu().data.numpy() + 0.001
        }
        
        # LEVEL 2: Sample from latent space
        zgivenx = qzgivenx.rsample([niw]).reshape([-1, dim_z])

        # LEVEL 2: Initialize decoder output containers
        out_decoders = {}
        out_decoders['cat'] = []
        p_xs = {}
        p_xs['cat'] = []
        params_x = {}
        params_x['cat'] = []

        # LEVEL 2: Process real-valued features
        if exists_types[0]:
            out_decoders['real'] = decoders['real'](zgivenx)
            p_xs['real'] = td.Normal(
                loc=out_decoders['real'][..., :p_real],
                scale=torch.nn.Softplus()(out_decoders['real'][..., p_real:(2*p_real)]) + 0.001
            )
            params_x['real'] = {
                'mean': torch.mean(out_decoders['real'][..., :p_real].reshape([niw, batch_size, p_real]), 0).detach().cpu().data.numpy(),
                'scale': torch.mean(torch.nn.Softplus()(out_decoders['real'][..., p_real:(2*p_real)]).reshape([niw, batch_size, p_real]), 0).detach().cpu().data.numpy() + 0.001
            }

        # LEVEL 2: Process count-valued features
        if exists_types[1]:
            out_decoders['count'] = decoders['count'](zgivenx)
            p_xs['count'] = td.Normal(
                loc=out_decoders['count'][..., :p_count],
                scale=torch.nn.Softplus()(out_decoders['count'][..., p_count:(2*p_count)]) + 0.001
            )
            params_x['count'] = {
                'mean': torch.mean(out_decoders['count'][..., :p_count].reshape([niw, batch_size, p_count]), 0).detach().cpu().data.numpy(),
                'scale': torch.mean(torch.nn.Softplus()(out_decoders['count'][..., p_count:(2*p_count)]).reshape([niw, batch_size, p_count]), 0).detach().cpu().data.numpy() + 0.001
            }

        # LEVEL 2: Process positive-valued features
        if exists_types[2]:
            out_decoders['pos'] = decoders['pos'](zgivenx)
            p_xs['pos'] = td.LogNormal(
                loc=out_decoders['pos'][..., :p_pos],
                scale=torch.nn.Softplus()(out_decoders['pos'][..., p_pos:(2*p_pos)]) + 0.001
            )
            params_x['pos'] = {
                'mean': torch.mean(out_decoders['pos'][..., :p_pos].reshape([niw, batch_size, p_pos]), 0).detach().cpu().data.numpy(),
                'scale': torch.mean(torch.nn.Softplus()(out_decoders['pos'][..., p_pos:(2*p_pos)]).reshape([niw, batch_size, p_pos]), 0).detach().cpu().data.numpy() + 0.001
            }

        # LEVEL 2: Process categorical features  
        if exists_types[3]:
            for ii in range(0, p_cat):
                out_decoders['cat'].append(
                    torch.clamp(
                        torch.nn.Softmax(dim=1)(decoders['cat'][ii](zgivenx)),
                        min=0.0001,
                        max=0.9999
                    ).reshape([niw, batch_size, -1]).reshape([niw*batch_size, -1])
                )
                p_xs['cat'].append(
                    td.RelaxedOneHotCategorical(temperature=temp, probs=out_decoders['cat'][ii])
                )
                params_x['cat'].append(
                    torch.mean(out_decoders['cat'][ii].reshape([niw, batch_size, -1]), 0).detach().cpu().data.numpy()
                )

        # LEVEL 2: Prepare flat representations
        xm_flat = torch.Tensor.repeat(iota_x, [niw, 1])
        ym_flat = torch.Tensor.repeat(iota_y, [niw, 1])


        # LEVEL 2: Handling Missing X Features
        if miss_x:
            # LEVEL 3: Initialize containers for missing X features
            outs_NN_xm = {}
            outs_NN_xm['cat'] = []
            qxmgivenxors = {}
            qxmgivenxors['cat'] = []
            params_xm = {}
            params_xm['cat'] = []

            # LEVEL 3: Process real-valued missing features
            if exists_types[0]:
                # LEVEL 4: Determine missingness modeling approach
                if Ignorable:   
                    outs_NN_xm['real'] = NNs_xm['real'](torch.cat([tiled_iota_x, zgivenx, tiled_iota_y], 1))
                else:  
                    outs_NN_xm['real'] = NNs_xm['real'](torch.cat([tiled_iota_x, tiled_mask_x, zgivenx, tiled_iota_y], 1))
                
                # LEVEL 4: Create distribution for missing real features
                qxmgivenxors['real'] = td.Normal(
                    loc=outs_NN_xm['real'][..., :p_real],
                    scale=torch.nn.Softplus()(outs_NN_xm['real'][..., p_real:(2*p_real)]) + 0.001
                )
                params_xm['real'] = {
                    'mean': torch.mean(outs_NN_xm['real'][..., :p_real].reshape([niw, batch_size, p_real]), 0).detach().cpu().data.numpy(),
                    'scale': torch.mean(torch.nn.Softplus()(outs_NN_xm['real'][..., p_real:(2*p_real)]).reshape([niw, batch_size, p_real]), 0).detach().cpu().data.numpy() + 0.001
                }

            # LEVEL 3: Process count-valued missing features
            if exists_types[1]:
                if Ignorable:  
                    outs_NN_xm['count'] = NNs_xm['count'](torch.cat([tiled_iota_x, zgivenx, tiled_iota_y], 1))
                else:  
                    outs_NN_xm['count'] = NNs_xm['count'](torch.cat([tiled_iota_x, tiled_mask_x, zgivenx, tiled_iota_y], 1))
                
                qxmgivenxors['count'] = td.Normal(
                    loc=outs_NN_xm['count'][..., :p_count],
                    scale=torch.nn.Softplus()(outs_NN_xm['count'][..., p_count:(2*p_count)]) + 0.001
                )
                params_xm['count'] = {
                    'mean': torch.mean(outs_NN_xm['count'][..., :p_count].reshape([niw, batch_size, p_real]), 0).detach().cpu().data.numpy(),
                    'scale': torch.mean(torch.nn.Softplus()(outs_NN_xm['count'][..., p_count:(2*p_count)]).reshape([niw, batch_size, p_real]), 0).detach().cpu().data.numpy() + 0.001
                }
                
            # LEVEL 3: Process positive-valued missing features
            if exists_types[2]:
                if Ignorable:   
                    outs_NN_xm['pos'] = NNs_xm['pos'](torch.cat([tiled_iota_x, zgivenx, tiled_iota_y], 1))
                else:  
                    outs_NN_xm['pos'] = NNs_xm['pos'](torch.cat([tiled_iota_x, tiled_mask_x, zgivenx, tiled_iota_y], 1))
                
                qxmgivenxors['pos'] = td.LogNormal(
                    loc=outs_NN_xm['pos'][..., :p_pos],
                    scale=torch.nn.Softplus()(outs_NN_xm['pos'][..., p_pos:(2*p_pos)]) + 0.001
                )
                params_xm['pos'] = {
                    'mean': torch.mean(outs_NN_xm['pos'][..., :p_pos].reshape([niw, batch_size, p_real]), 0).detach().cpu().data.numpy(),
                    'scale': torch.mean(torch.nn.Softplus()(outs_NN_xm['pos'][..., p_pos:(2*p_pos)]).reshape([niw, batch_size, p_real]), 0).detach().cpu().data.numpy() + 0.001
                }

            # LEVEL 3: Process categorical missing features
            if exists_types[3]:
                for ii in range(0, p_cat):
                    if Ignorable:  
                        outs_NN_xm['cat'].append(
                            torch.clamp(
                                torch.nn.Softmax(dim=1)(
                                    NNs_xm['cat'][ii](torch.cat([tiled_iota_x, zgivenx, tiled_iota_y], 1))
                                ), 
                                min=0.0001, 
                                max=0.9999
                            )
                        )
                    else:  
                        outs_NN_xm['cat'].append(
                            torch.clamp(
                                torch.nn.Softmax(dim=1)(
                                    NNs_xm['cat'][ii](torch.cat([tiled_iota_x, tiled_mask_x, zgivenx, tiled_iota_y], 1))
                                ), 
                                min=0.0001, 
                                max=0.9999
                            )
                        )
                    
                    qxmgivenxors['cat'].append(
                        td.RelaxedOneHotCategorical(temperature=temp, probs=outs_NN_xm['cat'][ii])
                    )
                    params_xm['cat'].append(
                        torch.mean(outs_NN_xm['cat'][ii].reshape([niw, batch_size, -1]), 0).detach().cpu().data.numpy()
                    )

            # LEVEL 3: Prepare imputation for missing features
            xm_flat = torch.zeros([M*niw*batch_size, p]).cuda()


            # LEVEL 3: Sampling missing features
            if draw_miss:
                # LEVEL 4: Sample real features
                if exists_types[0]: 
                    xm_flat[:,ids_real] = qxmgivenxors['real'].rsample([M]).reshape([M*niw*batch_size, -1])
                
                # LEVEL 4: Sample count features
                if exists_types[1]:
                    xm_flat[:,ids_count] = qxmgivenxors['count'].rsample([M]).reshape([M*niw*batch_size, -1])
                
                # LEVEL 4: Sample positive features
                if exists_types[2]: 
                    xm_flat[:,ids_pos] = qxmgivenxors['pos'].rsample([M]).reshape([M*niw*batch_size, -1])
                
                # LEVEL 4: Sample categorical features
                if exists_types[3]: 
                    for ii in range(0, p_cat):
                        if ii==0: 
                            C0=0
                            C1=int(Cs[ii])
                        else: 
                            C0=C1
                            C1=C0 + int(Cs[ii])
                        
                        xm_flat[:,(p_real + p_count + p_pos + C0):(p_real + p_count + p_pos + C1)] = qxmgivenxors['cat'][ii].rsample([M]).reshape([M*niw*batch_size, -1])
            else: 
                xm_flat = tiled_iota_xfull

        else: 
            qxmgivenxors = None
            params_xm = None
            xm_flat = torch.Tensor.repeat(iota_x, [M, 1])

        # LEVEL 2: Organize completed (sampled) xincluded for missingness model
        if miss_x:
            xincluded = tiled_iota_x*(tiled_mask_x) + xm_flat*(1-tiled_mask_x)
            mask_included = tiled_mask_x
        else:
            xincluded = iota_x
            mask_included = tiled_mask_x

        # LEVEL 2: Clamp categorical features
        xincluded[:,ids_cat] = torch.clamp(xincluded[:,ids_cat], min=0.0001, max=0.9999)

        # LEVEL 2: Handling Missing Y (Cox Model Specific)
        if miss_y and family == "Cox":
            # LEVEL 3: Determine input for Y missingness network
            if not miss_x:
                if Ignorable:   
                    out_NN_ym = NN_ym(torch.cat([tiled_iota_y, iota_x], 1))
                else:   
                    out_NN_ym = NN_ym(torch.cat([tiled_iota_y, iota_x, mask_x, mask_y], 1))
            elif miss_x:
                if Ignorable:   
                    out_NN_ym = NN_ym(torch.cat([tiled_iota_y, tiled_mask_x*tiled_iota_x + (1-tiled_mask_x)*xm_flat], 1))
                else:   
                    out_NN_ym = NN_ym(torch.cat([tiled_iota_y, tiled_mask_x*tiled_iota_x + (1-tiled_mask_x)*xm_flat, tiled_mask_x, tiled_mask_y], 1))
            
            # LEVEL 3: Create distribution for missing Y (Cox model)
            qymgivenyor = td.Normal(
                loc=out_NN_ym[..., :2],  # First component for time, second for event
                scale=torch.nn.Softplus()(out_NN_ym[..., 2:4]) + 0.001
            )
            
            # LEVEL 3: Compute parameters for Y
            params_ym = {
                'time': torch.mean(
                    torch.mean(out_NN_ym[..., :1].reshape([niw, M**(int(miss_x)), batch_size, 1]), 0), 0
                ).reshape([batch_size, 1]).detach().cpu().data.numpy(),
                'event': torch.mean(
                    torch.mean(torch.sigmoid(out_NN_ym[..., 1:2]).reshape([niw, M**(int(miss_x)), batch_size, 1]), 0), 0
                ).reshape([batch_size, 1]).detach().cpu().data.numpy()
            }

        # LEVEL 2: Handling Y Imputation for Different Models
        if miss_y:
            # LEVEL 3: Default handling for non-Cox models
            if family != "Cox":
                if not miss_x:
                    if Ignorable:  
                        out_NN_ym = NN_ym(torch.cat([iota_y, iota_x], 1))
                    else:   
                        out_NN_ym = NN_ym(torch.cat([iota_y, iota_x, mask_x, mask_y], 1))
                elif miss_x:
                    if Ignorable:   
                        out_NN_ym = NN_ym(torch.cat([tiled_iota_y, tiled_mask_x*tiled_iota_x + (1-tiled_mask_x)*xm_flat], 1))
                    else:   
                        out_NN_ym = NN_ym(torch.cat([tiled_iota_y, tiled_mask_x*tiled_iota_x + (1-tiled_mask_x)*xm_flat, tiled_mask_x, tiled_mask_y], 1))
                
            # LEVEL 3: Distribution and parameter computation for Gaussian/Multinomial
            if family == "Gaussian":
                qymgivenyor = td.Normal(
                    loc=out_NN_ym[..., :1],
                    scale=torch.nn.Softplus()(out_NN_ym[..., 1:2]) + 0.001
                )
                params_ym = {
                    'mean': torch.mean(torch.mean(out_NN_ym[..., :1].reshape([niw, M**(int(miss_x)), batch_size, 1]), 0), 0).reshape([batch_size, 1]).detach().cpu().data.numpy(),
                    'scale': torch.mean(torch.mean(torch.nn.Softplus()(out_NN_ym[..., 1:2]).reshape([niw, M**(int(miss_x)), batch_size, 1]), 0), 0).reshape([batch_size, 1]).detach().cpu().data.numpy() + 0.001
                }
            elif family == "Multinomial":
                qymgivenyor = td.RelaxedOneHotCategorical(temperature=temp, probs=invlink(link)(out_NN_ym))
                params_ym = {
                    'probs': torch.mean(out_NN_ym.reshape([niw, batch_size, -1]), 0).detach().cpu().data.numpy()
                }

            # LEVEL 3: Sampling for missing Y
            if draw_miss: 
                ym = qymgivenyor.rsample([niw])
                ym_flat = ym.reshape([-1, 1])
            else: 
                ym = tiled_iota_yfull
                ym_flat = ym.reshape([-1, 1])

        else:
            qymgivenyor = None
            params_ym = None
            ym_flat = torch.Tensor.repeat(iota_y, [niw, 1])

        # # LEVEL 2: Model-specific Processing
        # if family == "Cox":
        #     # LEVEL 3: Cox model processing
        #     out_NN_y = NN_y(xincluded)
        #     risk_scores = out_NN_y[..., 0]
        #     hazard_ratios = torch.exp(risk_scores)
        
        #     # LEVEL 3: Sort by time for partial likelihood
        #     times = yincluded[:,0]
        #     events = yincluded[:,1]
        #     sorted_idx = torch.argsort(times, descending=True)
        #     sorted_times = times[sorted_idx]
        #     sorted_events = events[sorted_idx]
        #     sorted_risks = risk_scores[sorted_idx]

        # LEVEL 2: Model-specific Processing
        if family == "Cox":
            # LEVEL 3: Cox model processing
            out_NN_y = NN_y(xincluded)
            risk_scores = out_NN_y[..., 0]
            hazard_ratios = torch.exp(risk_scores)
            
            # LEVEL 3: Sort by time for partial likelihood
            times = iota_y[:, 0]
            events = iota_y[:, 1]
            sorted_idx = torch.argsort(times, descending=True)
            sorted_times = times[sorted_idx]
            sorted_events = events[sorted_idx] 
            sorted_risks = risk_scores[sorted_idx]

            # LEVEL 3: Compute cumulative hazard
            pygivenx = td.Normal(
                loc=risk_scores,
                scale=torch.ones_like(risk_scores).cuda()
            )

            baseline_hazard, unique_times = cox_baseline_hazard(sorted_risks, sorted_times, sorted_events)

            # LEVEL 3: Store parameters
            params_y = {
                'risk_scores': torch.mean(
                    torch.mean(risk_scores.reshape([niw, M**(int(miss_x)), batch_size, 1]), 0), 0
                ).reshape([batch_size, 1]).detach().cpu().data.numpy(),
                'hazard_ratios': hazard_ratios.detach().cpu().data.numpy(),
                'baseline_hazard': baseline_hazard.detach().cpu().data.numpy(),
                'unique_times': unique_times.detach().cpu().data.numpy(),
                'sorted_indices': sorted_idx.detach().cpu().data.numpy()
            }
            
            # LEVEL 3: Compute cumulative hazard
            pygivenx = td.Normal(
                loc=risk_scores,
                scale=torch.ones_like(risk_scores).cuda()
            )
            
            baseline_hazard, unique_times = cox_baseline_hazard(risk_scores, times, events)
            
            # LEVEL 3: Store parameters
            params_y = {
                'risk_scores': torch.mean(
                    torch.mean(risk_scores.reshape([niw, M**(int(miss_x)), batch_size, 1]), 0), 0
                ).reshape([batch_size, 1]).detach().cpu().data.numpy(),
                'hazard_ratios': hazard_ratios.detach().cpu().data.numpy(),
                'baseline_hazard': baseline_hazard.detach().cpu().data.numpy(),
                'unique_times': unique_times.detach().cpu().data.numpy(),
                'sorted_indices': sorted_idx.detach().cpu().data.numpy()
            }

        elif family == "Multinomial":
            # LEVEL 3: Multinomial model processing
            probs = invlink(link)(out_NN_y)
            pygivenx = td.OneHotCategorical(probs=probs)
            
            params_y = {
                'probs': torch.mean(
                    torch.mean(probs.reshape([niw, M**(int(miss_x)), batch_size, C]), 0), 0
                ).reshape([batch_size, C]).detach().cpu().data.numpy()
            }

        elif family == "Poisson":
            # LEVEL 3: Poisson model processing
            lambda_y = invlink(link)(out_NN_y[..., 0])
            pygivenx = td.Poisson(rate=lambda_y)
            
            params_y = {
                'lambda': torch.mean(
                    torch.mean(lambda_y.reshape([niw, M**(int(miss_x)), batch_size, 1]), 0), 0
                ).reshape([batch_size, 1]).detach().cpu().data.numpy()
            }

        # # LEVEL 2: Organize covariates for missingness model
        # if covars_r_y == 1:
        #     # LEVEL 3: Include X covariates if specified
        #     if np.sum(covars_r_x) > 0: 
        #         covars_included = torch.cat([xincluded[:,covars_r_x==1], yincluded], 1)
        #     else: 
        #         covars_included = yincluded
        # elif covars_r_y == 0:
        #     # LEVEL 3: Use only X covariates if specified
        #     if np.sum(covars_r_x) > 0: 
        #         covars_included = xincluded[:,covars_r_x==1]

        # LEVEL 2: Organize covariates for missingness model
        if covars_r_y == 1:
            # LEVEL 3: Include X covariates if specified
            if np.sum(covars_r_x) > 0:
                covars_included = torch.cat([xincluded[:, covars_r_x==1], iota_y], 1)
            else:
                covars_included = iota_y
        elif covars_r_y == 0:
            # LEVEL 3: Use only X covariates if specified
            if np.sum(covars_r_x) > 0:
                covars_included = xincluded[:, covars_r_x==1]

        # LEVEL 2: Missingness model processing
        if not Ignorable:
            # LEVEL 3: Prepare missingness network input
            if (covars): 
                out_NN_r = NN_r(torch.cat([covars_included, covars_miss], 1))
            else: 
                out_NN_r = NN_r(covars_included)
            
            # LEVEL 3: Bernoulli distribution for missingness
            prgivenxy = td.Bernoulli(logits=out_NN_r)
            
            # LEVEL 3: Compute missingness probabilities
            params_r = {
                'probs': torch.mean(
                    torch.mean(
                        torch.mean(
                            torch.nn.Sigmoid()(out_NN_r).reshape([M**(int(miss_y)), niw, M**(int(miss_x)), batch_size, n_params_r]), 
                            0
                        ), 0
                    ), 0
                ).reshape([batch_size, n_params_r]).detach().cpu().data.numpy()
            }
        else: 
            prgivenxy = None
            params_r = None

        # LEVEL 2: Return processed variables and parameters
        return xincluded, iota_y, p_xs, qzgivenx, qxmgivenxors, qymgivenyor, pygivenx, prgivenxy, params_xm, params_ym, params_x, params_y, params_r, params_z, zgivenx
        #return xincluded, yincluded, p_xs, qzgivenx, qxmgivenxors, qymgivenyor, pygivenx, prgivenxy, params_xm, params_ym, params_x, params_y, params_r, params_z, zgivenx

    # LEVEL 1
    def compute_loss(iota_xfull, iota_yfull, iota_x, iota_y, mask_x, mask_y, covar_miss, temp):
        # LEVEL 2: Initialize batch variables
        batch_size = iota_x.shape[0]
        tiled_iota_x = torch.Tensor.repeat(iota_x,[niws_z,1]).cuda()
        tiled_iota_y = torch.Tensor.repeat(iota_y,[niws_z,1]).cuda()
        tiled_mask_x = torch.Tensor.repeat(mask_x,[niws_z,1]).cuda()
        tiled_mask_y = torch.Tensor.repeat(mask_y,[niws_z,1]).cuda()

        # LEVEL 2: Handle full data tensors
        if not draw_miss:
            tiled_iota_xfull = torch.Tensor.repeat(iota_xfull,[niws_z,1]).cuda()
            tiled_iota_yfull = torch.Tensor.repeat(iota_yfull,[niws_z,1]).cuda()
        else:
            tiled_iota_xfull = None

        # LEVEL 2: Handle covariates
        if covars:
            tiled_covars_miss = torch.Tensor.repeat(covar_miss,[M,1])
        else:
            tiled_covars_miss = None

        # LEVEL 2: Forward pass
        xincluded, iota_y, p_xs, qzgivenx, qxmgivenxors, qymgivenyor, pygivenx, prgivenxy, params_xm, params_ym, params_x, params_y, params_r, params_z, zgivenx = forward(iota_xfull, iota_yfull, iota_x, iota_y, mask_x, mask_y, batch_size, niws_z, temp)

        # LEVEL 2: Handle multinomial case
        if family=="Multinomial":
            iota_y=torch.nn.functional.one_hot(iota_y.to(torch.int64),num_classes=C).reshape([-1,C])

        # LEVEL 2: Initialize log probability accumulators
        logpx_real = torch.Tensor([0]).cuda()
        logpx_cat = torch.Tensor([0]).cuda()
        logpx_count = torch.Tensor([0]).cuda()
        logpx_pos = torch.Tensor([0]).cuda()
        logqxmgivenxor_real = torch.Tensor([0]).cuda()
        logqxmgivenxor_cat = torch.Tensor([0]).cuda()
        logqxmgivenxor_count = torch.Tensor([0]).cuda()
        logqxmgivenxor_pos = torch.Tensor([0]).cuda()

        # LEVEL 2: Case 1 - Both X and Y missing
        if miss_x and miss_y:
            # LEVEL 3: Compute missingness model log probability
            if not Ignorable:
                all_logprgivenxy = prgivenxy.log_prob(torch.cat([tiled_mask_x[:,miss_ids], tiled_mask_y],1))
                logprgivenxy = torch.sum(all_logprgivenxy,1).reshape([niws_z*M*M,batch_size])
            else:
                all_logprgivenxy = torch.Tensor([0]).cuda()
                logprgivenxy = torch.Tensor([0]).cuda()

            # LEVEL 3: Specific handling for Cox model with both missing
            if family == "Cox":
                # Handle missing survival times and events together
                times = iota_y[:,0]
                events = iota_y[:,1]
                risk_scores = pygivenx.loc
                
                # Handle missing data imputation
                missing_time = ~tiled_mask_y[:,0]
                missing_event = ~tiled_mask_y[:,1]
                imputed_times = qymgivenyor.rsample([niws_z])[:,:,0]
                imputed_events = qymgivenyor.rsample([niws_z])[:,:,1]
                times_complete = torch.where(missing_time, imputed_times, times)
                events_complete = torch.where(missing_event, imputed_events, events)
                
                # Compute Cox metrics
                baseline_hazard, unique_times = cox_baseline_hazard(risk_scores, times_complete, events_complete)
                cox_pl_loss = cox_loss(risk_scores, times_complete, events_complete)
                concordance_idx = compute_cindex(risk_scores, times_complete, events_complete)
                brier_scores = compute_brier_score(
                    risk_scores, baseline_hazard, times_complete, events_complete,
                    eval_times=torch.tensor([1.0, 2.0, 3.0]).cuda()
                )
                all_log_pygivenx = -cox_pl_loss

            # LEVEL 3: Handle other model families
            elif family=="Gaussian":
                all_log_pygivenx = pygivenx.log_prob(iota_y.reshape([-1]))
            else:
                all_log_pygivenx = pygivenx.log_prob(iota_y)

            logpygivenx = all_log_pygivenx.reshape([niws_z*M*M,batch_size])

            # LEVEL 3: Compute log probabilities for feature types (still within miss_x and miss_y case)
            if exists_types[0]:
                logpx_real = torch.sum(p_xs['real'].log_prob(xincluded[:,ids_real]),axis=1).reshape([niws_z*M*M,batch_size])
            if exists_types[1]:
                logpx_count = torch.sum(p_xs['count'].log_prob(xincluded[:,ids_count]),axis=1).reshape([niws_z*M*M,batch_size])
            if exists_types[2]:
                logpx_pos = torch.sum(p_xs['pos'].log_prob(xincluded[:,ids_pos]),axis=1).reshape([niws_z*M*M,batch_size])
            if exists_types[3]:
                for ii in range(0,p_cat):
                    if ii==0:
                        C0=0
                        C1=int(Cs[ii])
                    else:
                        C0=C1
                        C1=C0 + int(Cs[ii])
                    
                    if ii==0:
                        logpx_cat = torch.sum(p_xs['cat'][ii].log_prob(xincluded[:,(p_real + p_count + p_pos + C0):(p_real + p_count + p_pos + C1)].reshape([-1, int(Cs[ii])])),axis=1).reshape([-1,batch_size])
                    else:
                        logpx_cat = logpx_cat + torch.sum(p_xs['cat'][ii].log_prob(xincluded[:,(p_real + p_count + p_pos + C0):(p_real + p_count + p_pos + C1)].reshape([-1, int(Cs[ii])])),axis=1).reshape([-1,batch_size])

            # LEVEL 3: Compute log q(x|.) for each feature type
            if exists_types[0]:
                logqxmgivenxor_real = torch.sum(qxmgivenxors['real'].log_prob(xincluded[:,ids_real].reshape([-1,p_real]))*(1-tiled_mask_x[:,ids_real]),1).reshape([-1,batch_size])
            if exists_types[1]:
                logqxmgivenxor_count = torch.sum(qxmgivenxors['count'].log_prob(xincluded[:,ids_count].reshape([-1,p_count]))*(1-tiled_mask_x[:,ids_count]),1).reshape([niws_z*M*M,batch_size])
            if exists_types[2]:
                logqxmgivenxor_pos = torch.sum(qxmgivenxors['pos'].log_prob(xincluded[:,ids_pos].reshape([-1,p_pos]))*(1-tiled_mask_x[:,ids_pos]),1).reshape([-1,batch_size])
            if exists_types[3]:
                for ii in range(0,p_cat):
                    if ii==0:
                        C0=0
                        C1=int(Cs[ii])
                    else:
                        C0=C1
                        C1=C0 + int(Cs[ii])
                    
                    if ii==0:
                        logqxmgivenxor_cat = (qxmgivenxors['cat'][ii].log_prob(xincluded[:,(p_real + p_count + p_pos + C0):(p_real + p_count + p_pos + C1)].reshape([-1, int(Cs[ii])]))*(1-tiled_mask_x[:,(p_real + p_count + p_pos + C0)])).reshape([-1,batch_size])
                    else:
                        logqxmgivenxor_cat = logqxmgivenxor_cat + (qxmgivenxors['cat'][ii].log_prob(xincluded[:,(p_real + p_count + p_pos + C0):(p_real + p_count + p_pos + C1)].reshape([-1, int(Cs[ii])]))*(1-tiled_mask_x[:,(p_real + p_count + p_pos + C0)])).reshape([-1,batch_size])

            # LEVEL 3: Initialize sums and compute log q(ym|yo,r,xm,xo)
            logpxsum = torch.zeros([M*M,batch_size]).cuda()
            logqxsum = torch.zeros([niws_z*M*M,batch_size]).cuda()
            
            if family == "Cox":
                logqymgivenyor = (qymgivenyor.log_prob(torch.cat([iota_y[:,0].reshape(-1,1), iota_y[:,1].reshape(-1,1)], dim=1))*(1-tiled_mask_y)).reshape([niws_z*M*M,batch_size])
            else:
                logqymgivenyor = (qymgivenyor.log_prob(iota_y)*(1-tiled_mask_y)).reshape([niws_z*M*M,batch_size])
        
        # LEVEL 2: Case 2 - Only X missing
        elif miss_x and not miss_y:
            # LEVEL 3: Compute missingness model log probability
            if not Ignorable:
                all_logprgivenxy = prgivenxy.log_prob(tiled_mask_x[:,miss_ids])
                logprgivenxy = torch.sum(all_logprgivenxy,1).reshape([niws_z*M,batch_size])
            else:
                all_logprgivenxy = torch.Tensor([0]).cuda()
                logprgivenxy = torch.Tensor([0]).cuda()

           
            # LEVEL 3: Compute log probability for y|x
            if family=="Cox":
                times = iota_y[:,0]
                events = iota_y[:,1]
                risk_scores = pygivenx.loc
                
                # Compute Cox metrics with complete Y but missing X
                baseline_hazard, unique_times = cox_baseline_hazard(risk_scores, times, events)
                cox_pl_loss = cox_loss(risk_scores, times, events)
                concordance_idx = compute_cindex(risk_scores, times, events)
                brier_scores = compute_brier_score(
                    risk_scores, baseline_hazard, times, events,
                    eval_times=torch.tensor([1.0, 2.0, 3.0]).cuda()
                )
                all_log_pygivenx = -cox_pl_loss

            elif family=="Multinomial":
                all_log_pygivenx = pygivenx.log_prob(iota_y)
            else:
                all_log_pygivenx = pygivenx.log_prob(iota_y.reshape([-1]))

            # LEVEL 3: Reshape log probability
            logpygivenx = all_log_pygivenx.reshape([niws_z*M,batch_size])

            # LEVEL 3: Compute log probabilities for feature types
            if exists_types[0]:
                logpx_real = torch.sum(p_xs['real'].log_prob(xincluded[:,ids_real]),axis=1).reshape([niws_z*M,batch_size])
            if exists_types[1]:
                logpx_count = torch.sum(p_xs['count'].log_prob(xincluded[:,ids_count]),axis=1).reshape([niws_z*M,batch_size])
            if exists_types[2]:
                logpx_pos = torch.sum(p_xs['pos'].log_prob(xincluded[:,ids_pos]),axis=1).reshape([niws_z*M,batch_size])
            if exists_types[3]:
                for ii in range(0,p_cat):
                    if ii==0:
                        C0=0; C1=int(Cs[ii])
                    else:
                        C0=C1; C1=C0 + int(Cs[ii])
                    if ii==0:
                        logpx_cat = p_xs['cat'][ii].log_prob(xincluded[:,(p_real + p_count + p_pos + C0):(p_real + p_count + p_pos + C1)]).reshape([niws_z*M,batch_size])
                    else:
                        logpx_cat = logpx_cat + p_xs['cat'][ii].log_prob(xincluded[:,(p_real + p_count + p_pos + C0):(p_real + p_count + p_pos + C1)]).reshape([niws_z*M,batch_size])

            # LEVEL 3: Initialize sums
            logpxsum = torch.zeros([niws_z*M,batch_size]).cuda()
            logqxsum = torch.zeros([niws_z*M,batch_size]).cuda()


        # LEVEL 2: Case 3 - Only Y missing
        elif not miss_x and miss_y:
            # LEVEL 3: Compute missingness model log probability
            if not Ignorable:
                all_logprgivenxy = prgivenxy.log_prob(tiled_mask_y)
                logprgivenxy = all_logprgivenxy.reshape([niws_z*M,batch_size])
            else:
                all_logprgivenxy = torch.Tensor([0]).cuda()
                logprgivenxy = torch.Tensor([0]).cuda()

            # LEVEL 3: Handle Y-specific missing data
            if family == "Cox":
                # Log probability for missing survival data
                logqymgivenyor = (qymgivenyor.log_prob(torch.cat([iota_y[:,0].reshape(-1,1), 
                                iota_y[:,1].reshape(-1,1)], dim=1))*(1-tiled_mask_y)).reshape([niws_z*M,batch_size])
                
                # Calculate Cox model metrics
                times = iota_y[:,0]
                events = iota_y[:,1]
                risk_scores = pygivenx.loc
                baseline_hazard, unique_times = cox_baseline_hazard(risk_scores, times, events)
                cox_pl_loss = cox_loss(risk_scores, times, events)
                concordance_idx = compute_cindex(risk_scores, times, events)
                brier_scores = compute_brier_score(
                    risk_scores, baseline_hazard, times, events,
                    eval_times=torch.tensor([1.0, 2.0, 3.0]).cuda()
                )
                all_log_pygivenx = -cox_pl_loss
            else:
                logqymgivenyor = (qymgivenyor.log_prob(iota_y)*(1-tiled_mask_y)).reshape([niws_z*M,batch_size])
                logqxsum = 0
                if family=="Multinomial":
                    all_log_pygivenx = pygivenx.log_prob(iota_y)
                else:
                    all_log_pygivenx = pygivenx.log_prob(iota_y.reshape([-1]))

            # LEVEL 3: Reshape log probability
            logpygivenx = all_log_pygivenx.reshape([niws_z*M,batch_size])

            # LEVEL 3: Compute log probabilities for feature types
            if exists_types[0]:
                logpx_real = torch.sum(p_xs['real'].log_prob(xincluded[:,ids_real]),axis=1).reshape([niws_z*M,batch_size])
            if exists_types[1]:
                logpx_count = torch.sum(p_xs['count'].log_prob(xincluded[:,ids_count]),axis=1).reshape([niws_z*M,batch_size])
            if exists_types[2]:
                logpx_pos = torch.sum(p_xs['pos'].log_prob(xincluded[:,ids_pos]),axis=1).reshape([niws_z*M,batch_size])
            if exists_types[3]:
                for ii in range(0,p_cat):
                    if ii==0:
                        C0=0; C1=int(Cs[ii])
                    else:
                        C0=C1; C1=C0 + int(Cs[ii])
                    if ii==0:
                        logpx_cat = (p_xs['cat'][ii].log_prob(xincluded[:,(p_real + p_count + p_pos + C0):(p_real + p_count + p_pos + C1)])).reshape([niws_z*M,batch_size])
                    else:
                        logpx_cat = logpx_cat + (p_xs['cat'][ii].log_prob(xincluded[:,(p_real + p_count + p_pos + C0):(p_real + p_count + p_pos + C1)])).reshape([niws_z*M,batch_size])

            logpxsum = torch.zeros([M,batch_size]).cuda()

        # LEVEL 2: Final case - No missing data
        else:
            all_logprgivenxy = torch.Tensor([0]).cuda()
            logprgivenxy = torch.Tensor([0]).cuda()
            logqxsum = torch.Tensor([0]).cuda()
            logqymgivenyor = torch.Tensor([0]).cuda()

            # LEVEL 3: Compute log probability based on family
            if family == "Cox":
                times = iota_y[:,0]
                events = iota_y[:,1]
                risk_scores = pygivenx.loc
                
                sorted_idx = torch.argsort(times, descending=True)
                sorted_times = times[sorted_idx]
                sorted_events = events[sorted_idx]
                sorted_risks = risk_scores[sorted_idx]
                
                exp_risks = torch.exp(sorted_risks)
                cum_risks = torch.cumsum(exp_risks, dim=0)
                all_log_pygivenx = sorted_events * (sorted_risks - torch.log(cum_risks))
            elif family=="Multinomial":
                all_log_pygivenx = pygivenx.log_prob(iota_y)
            else:
                all_log_pygivenx = pygivenx.log_prob(iota_y.reshape([-1]))

            logpygivenx = all_log_pygivenx.reshape([niws_z*1*batch_size])

            # LEVEL 3: Compute feature probabilities for no missing data case
            if exists_types[0]:
                logpx_real = torch.sum(p_xs['real'].log_prob(xincluded[:,ids_real]),axis=1).reshape([niws_z*1*batch_size])
            if exists_types[1]:
                logpx_count = torch.sum(p_xs['count'].log_prob(xincluded[:,ids_count]),axis=1).reshape([niws_z*1*batch_size])
            if exists_types[2]:
                logpx_pos = torch.sum(p_xs['pos'].log_prob(xincluded[:,ids_pos]),axis=1).reshape([niws_z*1*batch_size])
            if exists_types[3]:
                for ii in range(0,p_cat):
                    if ii==0:
                        C0=0; C1=int(Cs[ii])
                    else:
                        C0=C1; C1=C0 + int(Cs[ii])
                    if ii==0:
                        logpx_cat = (p_xs['cat'][ii].log_prob(xincluded[:,(p_real + p_count + p_pos + C0):(p_real + p_count + p_pos + C1)].reshape([-1,C]))).reshape([-1])
                    else:
                        logpx_cat = logpx_cat + (p_xs['cat'][ii].log_prob(xincluded[:,(p_real + p_count + p_pos + C0):(p_real + p_count + p_pos + C1)].reshape([-1,C]))).reshape([-1])

            logpxsum = torch.zeros([niws_z*1,batch_size]).cuda()
            logqxsum = torch.zeros([niws_z*1*batch_size]).cuda()

        # LEVEL 2: Sum probability terms
        if not Ignorable:
            sum_logpr = np.sum(logprgivenxy.cpu().data.numpy())
        else:
            sum_logpr = 0

        sum_logpygivenx = np.sum(logpygivenx.cpu().data.numpy())
        sum_logpx = 0

        if exists_types[0]:
            sum_logpx = sum_logpx + np.sum(logpx_real.cpu().data.numpy())
        if exists_types[1]:
            sum_logpx = sum_logpx + np.sum(logpx_count.cpu().data.numpy())
        if exists_types[2]:
            sum_logpx = sum_logpx + np.sum(logpx_pos.cpu().data.numpy())
        if exists_types[3]:
            sum_logpx = sum_logpx + np.sum(logpx_cat.cpu().data.numpy())

        if miss_y:
            sum_logqym = np.sum(logqymgivenyor.cpu().data.numpy())
        else:
            sum_logqym = 0
            logqymgivenyor = torch.Tensor([0]).cuda()

        # LEVEL 2: Initialize sum terms
        if exists_types[0]:
            logpxsum = logpxsum + logpx_real
            logqxsum = logqxsum + logqxmgivenxor_real
        if exists_types[1]:
            logpxsum = logpxsum + logpx_count
            logqxsum = logqxsum + logqxmgivenxor_count
        if exists_types[2]:
            logpxsum = logpxsum + logpx_pos
            logqxsum = logqxsum + logqxmgivenxor_pos
        if exists_types[3]:
            logpxsum = logpxsum + logpx_cat
            logqxsum = logqxsum + logqxmgivenxor_cat

        # LEVEL 2: Compute latent space log probabilities
        logpz = p_z.log_prob(zgivenx).reshape([-1,batch_size])
        logqzgivenx = torch.sum(qzgivenx.log_prob(zgivenx.reshape([-1,batch_size,dim_z])),axis=2).reshape([-1,batch_size])

        # LEVEL 2: Compute final bound based on architecture
        if arch=="VAE":
            neg_bound = -torch.sum(weight_y*logpygivenx + logpxsum + logprgivenxy - logqxsum - logqymgivenyor)
        elif arch=="IWAE":
            neg_bound = -torch.sum(torch.logsumexp(weight_y*logpygivenx + logpxsum + logpz + logprgivenxy - logqzgivenx - logqxsum - logqymgivenyor,0))

        # LEVEL 2: Return results dictionary
        
        return {
        'neg_bound': neg_bound,
        'params_xm': params_xm,
        'params_ym': params_ym,
        'params_x': params_x,
        'params_y': params_y,
        'params_r': params_r,
        'params_z': params_z,
        'sum_logpr': sum_logpr,
        'sum_logpygivenx': sum_logpygivenx,
        'sum_logpx': sum_logpx,
        'sum_logqym': sum_logqym,
        'cox_metrics': {
            'concordance_index': concordance_idx,
            'brier_scores': brier_scores,
            'baseline_hazard': baseline_hazard,
            'unique_times': unique_times
        } if family == "Cox" else None
        }
    

    # LEVEL 1 
    def impute(iota_xfull, iota_yfull, iota_x, iota_y, mask_x, mask_y, covar_miss, niws_z, temp):
        # LEVEL 2
        # Initialize batch variables
        batch_size = iota_x.shape[0]
        tiled_iota_x = torch.Tensor.repeat(iota_x,[niws_z,1]).cuda()
        tiled_iota_y = torch.Tensor.repeat(iota_y,[niws_z,1]).cuda()
        tiled_mask_x = torch.Tensor.repeat(mask_x,[niws_z,1]).cuda()
        tiled_mask_y = torch.Tensor.repeat(mask_y,[niws_z,1]).cuda()

        # LEVEL 2
        # Handle optional full data tiling
        if not draw_miss:
            tiled_iota_xfull = torch.Tensor.repeat(iota_xfull,[niws_z,1]).cuda()
            tiled_iota_yfull = torch.Tensor.repeat(iota_yfull,[niws_z,1]).cuda()
        else:
            tiled_iota_xfull = None
            tiled_iota_yfull = None

        # LEVEL 2  
        # Handle covariates
        if covars:
            tiled_covars_miss = torch.Tensor.repeat(covar_miss,[M,1])
        else:
            tiled_covars_miss = None

        # LEVEL 2
        # Forward pass
        xincluded, iota_y, p_xs, qzgivenx, qxmgivenxors, qymgivenyor, pygivenx, prgivenxy, params_xm, params_ym, params_x, params_y, params_r, params_z, zgivenx = forward(iota_xfull, iota_yfull, iota_x, iota_y, mask_x, mask_y, batch_size, niws_z, temp)

        # LEVEL 2
        # Handle multinomial case
        if family=="Multinomial":
            iota_y=torch.nn.functional.one_hot(iota_y.to(torch.int64),num_classes=C).reshape([-1,C])

        # LEVEL 2
        # Initialize log probability accumulators 
        logpx_real = torch.Tensor([0]).cuda()
        logpx_cat = torch.Tensor([0]).cuda()
        logpx_count = torch.Tensor([0]).cuda()
        logpx_pos = torch.Tensor([0]).cuda()
        logqxmgivenxor_real = torch.Tensor([0]).cuda()
        logqxmgivenxor_cat = torch.Tensor([0]).cuda()
        logqxmgivenxor_count = torch.Tensor([0]).cuda()
        logqxmgivenxor_pos = torch.Tensor([0]).cuda()

        # LEVEL 2
        # Compute log probabilities - Case 1: Both X and Y missing
        if miss_x and miss_y:
            # LEVEL 3
            # Handle missingness model
            if not Ignorable:
                all_logprgivenxy = prgivenxy.log_prob(torch.cat([tiled_mask_x[:,miss_ids], tiled_mask_y],1))
                logprgivenxy = torch.sum(all_logprgivenxy,1).reshape([niws_z*M*M,batch_size])
            else:
                all_logprgivenxy=torch.Tensor([0]).cuda()
                logprgivenxy=torch.Tensor([0]).cuda()

        # LEVEL 2
        # Compute log probabilities - Case 1: Both X and Y missing
        if miss_x and miss_y:
            # LEVEL 3
            # Handle missingness model
            if not Ignorable:
                all_logprgivenxy = prgivenxy.log_prob(torch.cat([tiled_mask_x[:,miss_ids], tiled_mask_y],1))
                logprgivenxy = torch.sum(all_logprgivenxy,1).reshape([niws_z*M*M,batch_size])
            else:
                all_logprgivenxy=torch.Tensor([0]).cuda()
                logprgivenxy=torch.Tensor([0]).cuda()

            # LEVEL 3
            # Compute log probability for y|x
            if family == "Cox":
                # LEVEL 4
                # Handle Cox-specific computations
                times = iota_y[:,0]
                events = iota_y[:,1]
                risk_scores = pygivenx.loc
                
                # LEVEL 4
                # Compute Cox model metrics
                baseline_hazard, unique_times = cox_baseline_hazard(risk_scores, times, events)
                cox_pl_loss = cox_loss(risk_scores, times, events)
                all_log_pygivenx = -cox_pl_loss
            elif family=="Gaussian":
                all_log_pygivenx = pygivenx.log_prob(iota_y.reshape([-1]))
            else:
                all_log_pygivenx = pygivenx.log_prob(iota_y)

            # LEVEL 3
            # Reshape log probability
            logpygivenx = all_log_pygivenx.reshape([niws_z*M*M,batch_size])

            # LEVEL 3
            # Compute log probabilities for feature types
            if exists_types[0]:
                logpx_real = torch.sum(p_xs['real'].log_prob(xincluded[:,ids_real]),axis=1).reshape([niws_z*M*M,batch_size])
            if exists_types[1]:
                logpx_count = torch.sum(p_xs['count'].log_prob(xincluded[:,ids_count]),axis=1).reshape([niws_z*M*M,batch_size])
            if exists_types[2]:
                logpx_pos = torch.sum(p_xs['pos'].log_prob(xincluded[:,ids_pos]),axis=1).reshape([niws_z*M*M,batch_size])

            # LEVEL 3
            # Handle categorical features
            if exists_types[3]:
                for ii in range(0,p_cat):
                    # LEVEL 4
                    # Calculate indices for categorical data
                    if ii==0:
                        C0=0
                        C1=int(Cs[ii])
                    else:
                        C0=C1
                        C1=C0 + int(Cs[ii])
                    
                    # LEVEL 4
                    # Compute log probabilities for categorical features
                    if ii==0:
                        logpx_cat = torch.sum(p_xs['cat'][ii].log_prob(xincluded[:,(p_real + p_count + p_pos + C0):(p_real + p_count + p_pos + C1)].reshape([-1, int(Cs[ii])])),axis=1).reshape([-1,batch_size])
                    else:
                        logpx_cat = logpx_cat + torch.sum(p_xs['cat'][ii].log_prob(xincluded[:,(p_real + p_count + p_pos + C0):(p_real + p_count + p_pos + C1)].reshape([-1, int(Cs[ii])])),axis=1).reshape([-1,batch_size])

            # LEVEL 3
            # Compute log q(x|.) for each feature type
            if exists_types[0]:
                logqxmgivenxor_real = torch.sum(qxmgivenxors['real'].log_prob(xincluded[:,ids_real].reshape([-1,p_real]))*(1-tiled_mask_x[:,ids_real]),1).reshape([-1,batch_size])
            if exists_types[1]:
                logqxmgivenxor_count = torch.sum(qxmgivenxors['count'].log_prob(xincluded[:,ids_count].reshape([-1,p_count]))*(1-tiled_mask_x[:,ids_count]),1).reshape([niws_z*M*M,batch_size])
            if exists_types[2]:
                logqxmgivenxor_pos = torch.sum(qxmgivenxors['pos'].log_prob(xincluded[:,ids_pos].reshape([-1,p_pos]))*(1-tiled_mask_x[:,ids_pos]),1).reshape([-1,batch_size])


            # LEVEL 3
            # Handle categorical features q distributions
            if exists_types[3]:
                for ii in range(0,p_cat):
                    # LEVEL 4
                    # Calculate indices for categorical data
                    if ii==0:
                        C0=0
                        C1=int(Cs[ii])
                    else:
                        C0=C1
                        C1=C0 + int(Cs[ii])
                    
                    # LEVEL 4
                    # Compute log q probabilities for categorical features
                    if ii==0:
                        logqxmgivenxor_cat = (qxmgivenxors['cat'][ii].log_prob(xincluded[:,(p_real + p_count + p_pos + C0):(p_real + p_count + p_pos + C1)].reshape([-1, int(Cs[ii])]))*(1-tiled_mask_x[:,(p_real + p_count + p_pos + C0)])).reshape([-1,batch_size])
                    else:
                        logqxmgivenxor_cat = logqxmgivenxor_cat + (qxmgivenxors['cat'][ii].log_prob(xincluded[:,(p_real + p_count + p_pos + C0):(p_real + p_count + p_pos + C1)].reshape([-1, int(Cs[ii])]))*(1-tiled_mask_x[:,(p_real + p_count + p_pos + C0)])).reshape([-1,batch_size])

            # LEVEL 3
            # Initialize probability sums
            logpxsum = torch.zeros([M*M,batch_size]).cuda()
            logqxsum = torch.zeros([niws_z*M*M,batch_size]).cuda()
            
            # LEVEL 3
            # Compute log q(ym|yo,r,xm,xo)
            if family == "Cox":
                # LEVEL 4
                # Handle Cox model specific y imputation
                logqymgivenyor = (qymgivenyor.log_prob(torch.cat([iota_y[:,0].reshape(-1,1), 
                                iota_y[:,1].reshape(-1,1)], dim=1))*(1-tiled_mask_y)).reshape([niws_z*M*M,batch_size])
            else:
                logqymgivenyor = (qymgivenyor.log_prob(iota_y)*(1-tiled_mask_y)).reshape([niws_z*M*M,batch_size])

        # LEVEL 2
        # Case 2: Only X missing
        elif miss_x and not miss_y:
            # LEVEL 3
            # Handle missingness model
            if not Ignorable:
                all_logprgivenxy = prgivenxy.log_prob(tiled_mask_x[:,miss_ids])
                logprgivenxy = torch.sum(all_logprgivenxy,1).reshape([niws_z*M,batch_size])
            else:
                all_logprgivenxy=torch.Tensor([0]).cuda()
                logprgivenxy=torch.Tensor([0]).cuda()

            # LEVEL 3
            # Compute log probability for y|x
            if family == "Cox":
                # LEVEL 4 
                # Handle Cox model computations
                times = iota_y[:,0]
                events = iota_y[:,1]
                risk_scores = pygivenx.loc
                
                # LEVEL 4
                # Compute Cox metrics
                baseline_hazard, unique_times = cox_baseline_hazard(risk_scores, times, events)
                cox_pl_loss = cox_loss(risk_scores, times, events)
                all_log_pygivenx = -cox_pl_loss
            elif family=="Multinomial":
                all_log_pygivenx = pygivenx.log_prob(iota_y)
            else:
                all_log_pygivenx = pygivenx.log_prob(iota_y.reshape([-1]))

            # LEVEL 3
            # Reshape log probability
            logpygivenx = all_log_pygivenx.reshape([niws_z*M,batch_size])

            # LEVEL 3
            # Compute log probabilities for feature types
            if exists_types[0]:
                logpx_real = torch.sum(p_xs['real'].log_prob(xincluded[:,ids_real]),axis=1).reshape([niws_z*M,batch_size])
            if exists_types[1]:
                logpx_count = torch.sum(p_xs['count'].log_prob(xincluded[:,ids_count]),axis=1).reshape([niws_z*M,batch_size])
            if exists_types[2]:
                logpx_pos = torch.sum(p_xs['pos'].log_prob(xincluded[:,ids_pos]),axis=1).reshape([niws_z*M,batch_size])

            # LEVEL 3
            # Handle categorical features
            if exists_types[3]:
                for ii in range(0,p_cat):
                    # LEVEL 4
                    # Calculate indices for categorical data
                    if ii==0:
                        C0=0
                        C1=int(Cs[ii])
                    else:
                        C0=C1
                        C1=C0 + int(Cs[ii])
                        
                    # LEVEL 4
                    # Compute log probabilities
                    if ii==0:
                        logpx_cat = p_xs['cat'][ii].log_prob(xincluded[:,(p_real + p_count + p_pos + C0):(p_real + p_count + p_pos + C1)]).reshape([niws_z*M,batch_size])
                    else:
                        logpx_cat = logpx_cat + p_xs['cat'][ii].log_prob(xincluded[:,(p_real + p_count + p_pos + C0):(p_real + p_count + p_pos + C1)]).reshape([niws_z*M,batch_size])

            # LEVEL 3
            # Compute log q(x|.) for each feature type 
            if exists_types[0]:
                logqxmgivenxor_real = torch.sum(qxmgivenxors['real'].log_prob(xincluded[:,ids_real].reshape([niws_z*M*batch_size,p_real]))*(1-tiled_mask_x[:,ids_real]),1).reshape([niws_z*M,batch_size])
            if exists_types[1]:
                logqxmgivenxor_count = torch.sum(qxmgivenxors['count'].log_prob(xincluded[:,ids_count].reshape([niws_z*M*batch_size,p_count]))*(1-tiled_mask_x[:,ids_count]),1).reshape([niws_z*M,batch_size])
            if exists_types[2]:
                logqxmgivenxor_pos = torch.sum(qxmgivenxors['pos'].log_prob(xincluded[:,ids_pos].reshape([niws_z*M*batch_size,p_pos]))*(1-tiled_mask_x[:,ids_pos]),1).reshape([niws_z*M,batch_size])

            # LEVEL 3
            # Handle categorical features q distributions
            if exists_types[3]:
                for ii in range(0,p_cat):
                    # LEVEL 4
                    # Calculate indices for categorical data
                    if ii==0:
                        C0=0
                        C1=int(Cs[ii])
                    else:
                        C0=C1
                        C1=C0 + int(Cs[ii])
                    
                    # LEVEL 4
                    # Compute log q probabilities
                    if ii==0:
                        logqxmgivenxor_cat = (qxmgivenxors['cat'][ii].log_prob(xincluded[:,(p_real + p_count + p_pos + C0):(p_real + p_count + p_pos + C1)].reshape([niws_z*M*batch_size,int(Cs[ii])])) * (1-tiled_mask_x[:,(p_real + p_count + p_pos + C0)])).reshape([-1,batch_size])
                    else:
                        logqxmgivenxor_cat = logqxmgivenxor_cat + (qxmgivenxors['cat'][ii].log_prob(xincluded[:,(p_real + p_count + p_pos + C0):(p_real + p_count + p_pos + C1)].reshape([niws_z*M*batch_size,int(Cs[ii])])) * (1-tiled_mask_x[:,(p_real + p_count + p_pos + C0)])).reshape([-1,batch_size])

            # LEVEL 3
            # Initialize probability sums
            logpxsum = torch.zeros([niws_z*M,batch_size]).cuda()
            logqxsum = torch.zeros([niws_z*M,batch_size]).cuda()

        # LEVEL 2
        # Case 3: Only Y missing
        elif not miss_x and miss_y:
            # LEVEL 3
            # Handle missingness model
            if not Ignorable:
                all_logprgivenxy = prgivenxy.log_prob(tiled_mask_y)
                logprgivenxy = all_logprgivenxy.reshape([niws_z*M,batch_size])
            else:
                all_logprgivenxy=torch.Tensor([0]).cuda()
                logprgivenxy=torch.Tensor([0]).cuda()

            # LEVEL 3
            # Handle Y distributions and probabilities
            if family == "Cox":
                # LEVEL 4
                # Handle Cox specific missing Y case
                logqymgivenyor = (qymgivenyor.log_prob(torch.cat([iota_y[:,0].reshape(-1,1), 
                                iota_y[:,1].reshape(-1,1)], dim=1))*(1-tiled_mask_y)).reshape([niws_z*M,batch_size])
                
                times = iota_y[:,0]
                events = iota_y[:,1]
                risk_scores = pygivenx.loc
                
                baseline_hazard, unique_times = cox_baseline_hazard(risk_scores, times, events)
                cox_pl_loss = cox_loss(risk_scores, times, events)
                all_log_pygivenx = -cox_pl_loss
            else:
                logqymgivenyor = (qymgivenyor.log_prob(iota_y)*(1-tiled_mask_y)).reshape([niws_z*M,batch_size])
                logqxsum = 0
                if family=="Multinomial":
                    all_log_pygivenx = pygivenx.log_prob(iota_y)
                else:
                    all_log_pygivenx = pygivenx.log_prob(iota_y.reshape([-1]))

            logpygivenx = all_log_pygivenx.reshape([niws_z*M,batch_size])


            # LEVEL 3
            # Compute log probabilities for feature types
            if exists_types[0]:
                logpx_real = torch.sum(p_xs['real'].log_prob(xincluded[:,ids_real]),axis=1).reshape([niws_z*M,batch_size])
            if exists_types[1]:
                logpx_count = torch.sum(p_xs['count'].log_prob(xincluded[:,ids_count]),axis=1).reshape([niws_z*M,batch_size])
            if exists_types[2]:
                logpx_pos = torch.sum(p_xs['pos'].log_prob(xincluded[:,ids_pos]),axis=1).reshape([niws_z*M,batch_size])
            if exists_types[3]:
                for ii in range(0,p_cat):
                    # LEVEL 4
                    # Calculate indices for categorical data
                    if ii==0:
                        C0=0; C1=int(Cs[ii])
                    else:
                        C0=C1; C1=C0 + int(Cs[ii])
                    
                    # LEVEL 4 
                    # Compute categorical probabilities
                    if ii==0:
                        logpx_cat = (p_xs['cat'][ii].log_prob(xincluded[:,(p_real + p_count + p_pos + C0):(p_real + p_count + p_pos + C1)].reshape([-1,C[ii]]))).reshape([-1,batch_size])
                    else:
                        logpx_cat = logpx_cat + (p_xs['cat'][ii].log_prob(xincluded[:,(p_real + p_count + p_pos + C0):(p_real + p_count + p_pos + C1)].reshape([-1,C[ii]]))).reshape([-1,batch_size])

            # LEVEL 3
            # Initialize sum terms
            logpxsum = torch.zeros([M,batch_size]).cuda()

        # LEVEL 2
        # Case 4: No missing data
        else:
            # LEVEL 3
            # Initialize probability terms
            all_logprgivenxy=torch.Tensor([0]).cuda()
            logprgivenxy=torch.Tensor([0]).cuda()
            logqxsum=torch.Tensor([0]).cuda()
            logqymgivenyor=torch.Tensor([0]).cuda()

            # LEVEL 3
            # Handle different model families with no missing data
            if family == "Cox":
                # LEVEL 4
                # Handle Cox model computations
                times = iota_y[:,0]
                events = iota_y[:,1]
                risk_scores = pygivenx.loc
                
                # LEVEL 4
                # Compute Cox model metrics
                baseline_hazard, unique_times = cox_baseline_hazard(risk_scores, times, events)
                cox_pl_loss = cox_loss(risk_scores, times, events)
                all_log_pygivenx = -cox_pl_loss
            elif family=="Multinomial":
                all_log_pygivenx = pygivenx.log_prob(iota_y)
            else:
                all_log_pygivenx = pygivenx.log_prob(iota_y.reshape([-1]))

            # LEVEL 3
            # Reshape log probability
            logpygivenx = all_log_pygivenx.reshape([niws_z*1*batch_size])

            # LEVEL 3
            # Compute feature probabilities for complete data
            if exists_types[0]:
                logpx_real = torch.sum(p_xs['real'].log_prob(xincluded[:,ids_real]),axis=1).reshape([niws_z*1*batch_size])
            if exists_types[1]:
                logpx_count = torch.sum(p_xs['count'].log_prob(xincluded[:,ids_count]),axis=1).reshape([niws_z*1*batch_size])
            if exists_types[2]:
                logpx_pos = torch.sum(p_xs['pos'].log_prob(xincluded[:,ids_pos]),axis=1).reshape([niws_z*1*batch_size])

            # LEVEL 3
            # Handle categorical features
            if exists_types[3]:
                for ii in range(0,p_cat):
                    # LEVEL 4
                    # Calculate indices
                    if ii==0:
                        C0=0
                        C1=int(Cs[ii])
                    else:
                        C0=C1
                        C1=C0 + int(Cs[ii])


                    # LEVEL 4
                    # Compute categorical probabilities
                    if ii==0:
                        logpx_cat = (p_xs['cat'][ii].log_prob(xincluded[:,(p_real + p_count + p_pos + C0):(p_real + p_count + p_pos + C1)].reshape([-1,C]))).reshape([-1])
                    else:
                        logpx_cat = logpx_cat + (p_xs['cat'][ii].log_prob(xincluded[:,(p_real + p_count + p_pos + C0):(p_real + p_count + p_pos + C1)].reshape([-1,C]))).reshape([-1])

            # LEVEL 3
            # Initialize final sums
            logpxsum = torch.zeros([niws_z*1,batch_size]).cuda()
            logqxsum = torch.zeros([niws_z*1*batch_size]).cuda()

        # LEVEL 2
        # Compute importance weights
        IW = logpxsum + logpygivenx
        
        if not Ignorable:
            IW = IW + logprgivenxy

        # LEVEL 2
        # Handle different missing data cases for weights computation
        if miss_x and miss_y:
            imp_weights = torch.nn.functional.softmax(IW + torch.Tensor.repeat(logpz,[M*M,1]) - torch.Tensor.repeat(logqzgivenx,[M*M,1]) - logqxsum - logqymgivenyor, 0)
        elif miss_x and not miss_y:
            imp_weights = torch.nn.functional.softmax(IW + torch.Tensor.repeat(logpz,[M,1]) - torch.Tensor.repeat(logqzgivenx,[M,1]) - logqxsum, 0)
        elif not miss_x and miss_y:
            imp_weights = torch.nn.functional.softmax(IW + torch.Tensor.repeat(logpz,[M,1]) - torch.Tensor.repeat(logqzgivenx,[M,1]) - logqymgivenyor, 0)
        else:
            imp_weights = torch.ones([1,batch_size])

        # LEVEL 2
        # Compute final imputations
        xm = torch.einsum('ki,kij->ij', imp_weights.float(), xincluded.reshape([-1,batch_size,p]).float())
        xms = xincluded.reshape([-1,batch_size,p])

        # LEVEL 2
        # Handle Y imputations
        if miss_y:
            if family == "Cox":
                # Handle both time and event components
                ym_time = torch.einsum('ki,kij->ij', imp_weights.float(), iota_y[:,0].reshape([-1,batch_size,1]).float())
                ym_event = torch.einsum('ki,kij->ij', imp_weights.float(), iota_y[:,1].reshape([-1,batch_size,1]).float())
                ym = torch.cat([ym_time, ym_event], dim=1)
                yms = iota_y.reshape([-1,batch_size,2])
            else:
                ym = torch.einsum('ki,kij->ij', imp_weights.float(), iota_y.reshape([-1,batch_size,1]).float())
                yms = iota_y.reshape([-1,batch_size,1])
        else:
            ym = iota_y
            yms = iota_y

        # LEVEL 2
        # Return results
        return {'xm': xm, 'ym': ym, 'imp_weights':imp_weights, 'xms': xms.detach(), 'yms': yms.detach()}