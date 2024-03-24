# -*- coding: utf-8 -*-
"""
Created on Thu Nov 23 00:56:50 2023
Changepoint alghorithms experiment
@author: cleiton
"""

import numpy as np
import pandas as pd
import changepoint_module as cm

df = pd.read_pickle('df_series.pkl')
N = len(df)

series_type = ['d_throughput', 'd_rttmean', 'u_throughput', 'u_rttmean']

# Classical methpods - Basic implementations
sequential_ba = [cm.shewhart_ba, cm.ewma_ba, cm.cusum_2s_ba, cm.cusum_wl_ba]

# Classical methods - Proposed implementations
sequential_ps = [cm.shewhart_ps, cm.ewma_ps, cm.cusum_2s_ps, cm.cusum_wl_ps]

pairs = zip(sequential_ba, sequential_ps)
methods = [m for pair in pairs for m in pair]
methods = methods + [cm.vwcd]

# Proposed frametwork commom hyperparameters
w0 = 20                 # phase 1 estimating window size
rl = 4                  # consecutive deviations to consider a changepoint
ka = 6                  # kappa for anomaly
alpha_norm = 0.01       # normality test significace level
alpha_stat = 0.01       # statinarity test significance level
cs_max = 5              # maximum counter for process not stabilized
filt_per = 0.95         # outlier filtering percentil (first window or not. estab.)
max_var = 1.2           # maximum level of variance increasing to consider

# Shewhart hyperparameters
k = 3

# EWMA hyperparameters
lamb = 0.1
kd = 4

# CUSUM hyperparameters
h = 5                   # statistic threshold
delta = 2               # 2S-CUSUM hyperparameter
w1 = 20                 # WL-CUSUM

# VWCD
wv = 20                 # window-size
alpha = beta = 10       # beta-binomial hyperpeparameters
p_thr = 0.8             # threshold probability to an window decide for a changepoint
vote_p_thr = 0.9        # threshold probabilty to decide for a changepoint after aggregation
vote_n_thr = 10         # min. number of votes to decide for a changepoint
y0 = 0.5                # logistic prior hyperparameter
yw = 0.9                # logistic prior hyperparameter
aggreg = 'posterior'    # votes aggregation function

Res = []

for m in methods:
    
    print(f"Processing {m.__name__}")
    
    for n in range(N):
        
        client = df.iloc[n].client
        site = df.iloc[n].site
            
        # Prefixo do arquivo
        prefixo = client + "_" + site + "_"
        
        if m==cm.vwcd and n%10==0:
            print(f'\tprocessing {n+1}/{N} clients-sites')
        
        for s_type in series_type:
            
            # Load the timeseries
            file = prefixo + s_type + ".txt"
            
            y = np.loadtxt(f'../dataset/{file}', usecols=1, delimiter=',')
            
            # Remove possible nan values
            y = y[~np.isnan(y)]
           
            # Maps the kargs for each method
            if m.__name__ == 'shewhart_ba':
                kargs = {'y':y, 'w':w0, 'k':k}
                
            elif m.__name__ == 'shewhart_ps' or m.__name__ == 'shewhart_ps2':
                kargs = {'y':y, 'w':w0, 'k':k, 'rl':rl, 'ka':ka, 
                        'alpha_norm':alpha_norm, 'alpha_stat':alpha_stat, 
                        'filt_per':filt_per, 'max_var':max_var, 
                        'cs_max':cs_max}
            
            elif m.__name__ == 'ewma_ba':
                kargs = {'y':y, 'w':w0, 'kd':kd, 'lamb':lamb}
            
            elif m.__name__ == 'ewma_ps':
                kargs = {'y':y, 'w':w0, 'kd':kd, 'lamb':lamb, 'rl':rl, 'ka':ka, 
                        'alpha_norm':alpha_norm, 'alpha_stat':alpha_stat, 
                        'filt_per':filt_per, 'max_var':max_var, 
                        'cs_max':cs_max}
            
            elif m.__name__ == 'cusum_2s_ba':
                kargs = {'y':y, 'w':w0, 'delta':delta, 'h':h}
            
            elif m.__name__ == 'cusum_2s_ps':
                kargs = {'y':y, 'w':w0, 'delta':delta, 'h':h, 
                         'rl':rl, 'k':k, 'ka':ka, 
                         'alpha_norm':alpha_norm, 'alpha_stat':alpha_stat, 
                         'filt_per':filt_per, 'max_var':max_var, 
                         'cs_max':cs_max}
            
            elif m.__name__ == 'cusum_wl_ba':
                kargs = {'y':y, 'w0':w0, 'w1':w1, 'h':h}
            
            elif m.__name__ == 'cusum_wl_ps':
                kargs = {'y':y, 'w0':w0, 'w1':w1, 'h':h, 
                         'rl':rl, 'k':k, 'ka':ka, 
                         'alpha_norm':alpha_norm, 'alpha_stat':alpha_stat, 
                         'filt_per':filt_per, 'max_var':max_var, 
                         'cs_max':cs_max}
                
            elif m.__name__ == 'vwcd':
                kargs = {'X':y, 'w':wv, 'w0':wv, 'alpha':alpha, 'beta':beta, 
                         'p_thr':p_thr, 'vote_p_thr':vote_p_thr, 
                         'vote_n_thr':vote_n_thr, 'y0':y0, 'yw':yw, 
                         'aggreg':aggreg}
            
            # Call the methods
            num_anom_u = num_anom_l = M0 = S0 = None
            out = m(**kargs)
            if m in sequential_ba:
                CP, elapsed_time = out
            elif m in sequential_ps:
                CP, Anom_u, Anom_l, M0, S0, elapsed_time = out
                num_anom_u = len(Anom_u)
                num_anom_l = len(Anom_l)
            elif m == cm.vwcd:
                CP, M0, S0, elapsed_time = out
            
            # Store the results
            res = {'client':client, 'site':site, 'serie':s_type,
                   'method':m.__name__, 'CP':CP, 'num_cp':len(CP), 
                   'num_anom_u':num_anom_u, 'num_anom_l':num_anom_l,
                   'M0':M0, 'S0':S0, 'elapsed_time':elapsed_time} 

            Res.append(res)

# Dataframe with results
df_res = pd.DataFrame(Res)
pd.to_pickle(df_res, 'df_results.pkl')