# -*- coding: utf-8 -*-
"""
Figura 1a - Shewhart - método proposto
@author: Cleiton Moya de Almeida
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy.stats import shapiro
from statsmodels.tsa.stattools import adfuller
import time

plt.rcParams.update({'font.size': 8, 'axes.titlesize': 8})

# Data loading
file = 'e45f01963c21_gig03_u_throughput.txt'
y = np.loadtxt(f'../dataset/{file}', usecols=1, delimiter=',')[:530]

verbose = True

# Shapiro-Wilk normality test
# H0: normal distribution
def normality_test(y, alpha):
    _,pvalue = shapiro(y)
    return pvalue > alpha

# Augmented Dickey-Fuller test for unitary root (non-stationarity)
# H0: the process has a unit root (non-stationary)
def stationarity_test(y, alpha):
    adf = adfuller(y)
    pvalue = adf[1]
    return pvalue < alpha

# Commom hyperparameters
w = 20              # estimating window size
rl = 4              # consecutive deviations to consider a changepoint
ka = 6              # kappa for anomaly
alpha_norm = 0.01   # normality test significace level
alpha_stat = 0.05   # stationarity test significance level
cs_max = 5          # maximum counter for process not stabilized
filt_per = 0.95     # outlier filtering percentil (first window or not. estab.)
max_var = 1.2       # maximum increased variance allowed to consider stab.

# Shewhart hyperparameter
k = 3               # kappa for statistic deviation

# Auxiliary variables initialization
CP = []             # changepoint list 
Anom_u = []         # up point anomalies list
Anom_l = []         # low point anomalie list
lcp = 0             # last check point (different from changepoint)
win_t0 = 0          # learning window t0
Win_period = []     # stabilization/learning windows
c = 0               # statistic deviation counter
ca_u = 0            # up point up counter
ca_l = 0            # low point anomaly counter
cs = 0              # stabilization counter
Mu0 = []            # phase 1 estimated mu0 at each t
M0_unique = []      # phase 1 estimated mu0 after each changepoint
Sigma0 = []         # phase 1 estimated sigma0
S0_unique = []      # phase 1 estimated sigma0 after each changepoint
U = []              # upper control limit at each t
L = []              # lower control limit at each t

startTime = time.time()
for t, y_t in enumerate(y):

    if (t >= lcp + w):
        
        # At process beginning and after a changepoint, 
        # check if the process is stable before estimating the parameters
        if t==lcp+w:
            yw = y[lcp+1:t+1]
            
            # Shapiro-Wiltker test for normality
            normality = normality_test(yw, alpha_norm)
            
            # Check if the variance level increasing is acceptable
            # If its the first window, accept blindly, but filter possible outliers 
            # before estimating the mu0, s0
            first_window = len(Win_period) == 0
            sw = yw.std(ddof=1)
            if not first_window:
                sa = S0_unique[-1]
                dev_var = abs(sw - sa)/sa
                var_acept =  dev_var <= max_var
            else:
                var_acept = True
                yw = yw[(yw>np.quantile(yw,1-filt_per)) & (yw<np.quantile(yw,filt_per))]
            
            # Stabilization criteria: normality and variance accepted
            stab = normality and var_acept
            
            # If process did not stabilize after cs_max, force the stabilization,
            # but filter possible outliers to estimate mu0, s0
            if stab or cs==cs_max:
                if cs==cs_max:
                    yw = yw[(yw>np.quantile(yw,1-filt_per)) & (yw<np.quantile(yw,filt_per))]
                    if verbose: print(f"n={t}: Considering process stabilized")
                else:
                    if verbose: print(f"n={t}: Process stabilized, lcp={lcp}")
                
                # Phase 1 parameters estimation
                mu0 = yw.mean()
                s0 = yw.std(ddof=1)
                M0_unique.append(mu0)
                S0_unique.append(s0)
                Win_period.append((win_t0,t))
                if verbose: print(f"n={t}: Estimated mu0={mu0}, sigma0={s0}")
                
                # Beyond the non-normality, if the last window was not stationary, 
                # and now the process is normal and statonary, consider a changepoint
                if t != win_t0+w: 
                    lw_stat = stationarity_test(y[lcp-w+1:lcp+1], alpha_stat) # last window is stationary
                    cw_stat = stationarity_test(y[lcp+1:t+1], alpha_stat) # currunt window stationarity
                    print(f"n={t}: Last window stationarity: {lw_stat}; current window stationrity: {cw_stat}; cs={cs}")
                    if not lw_stat and cw_stat and cs!=cs_max:
                        if verbose: print(f"n={t}: Considering t={lcp} a changepoint")
                        CP.append(lcp)
                cs = 0
            else:
                if verbose: print(f"n={t}: Process not stabilized, sw={yw.std(ddof=1)}")
                lcp=lcp+w
                cs = cs+1

        # Lower and upper control limits for deviation
        u = mu0 + k*s0
        l = mu0 - k*s0
        
        # Check for point anomaly (upper and low)
        anom_u = y_t >= mu0 + ka*s0
        anom_l = y_t <= mu0 - ka*s0
        if anom_u:
            Anom_u.append(t)
        if anom_l:
            Anom_l.append(t)
        
        # Check for statistic deviation
        dev = abs(y_t-mu0) >= k*s0
        
        if dev:
            if anom_u:
                ca_u = ca_u+1 
            elif anom_l:
                ca_l = ca_l+1

            c = c+1
            
            if c==rl:
                lcp = t-rl+1    # last check point
                win_t0 = lcp    # learning window t0
                if verbose: print(f'n={t}: Changepoint at t={lcp-1}')
                CP.append(lcp-1)
                if ca_u > 0:
                    Anom_u = Anom_u[:-ca_u]
                if ca_l > 0:
                    Anom_l = Anom_l[:-ca_l]    
                c = 0
                ca_u = 0
                ca_l = 0
        else:
            c = 0
            ca_u = 0
            ca_l = 0
        
    else:
        mu0 = np.nan
        s0 = np.nan
        l = np.nan
        u = np.nan
    
    Mu0.append(mu0)
    Sigma0.append(s0)
    U.append(u)
    L.append(l)
endTime = time.time()
elapsedTime = endTime-startTime


fig, ax = plt.subplots(figsize=(3,2.5))
ax.plot(y, linewidth=0.5)
ax.plot(U, linewidth=0.5, linestyle='--', color='r',  label=r'$\hat{\mu}_0\pm \kappa \hat{\sigma}_0$')
ax.plot(L, linewidth=0.5, linestyle='--', color='r')
ax.set_ylim([0, 500])
ax.set_xlim([0, 600])
ax.grid(linestyle=':')

ymin0, ymax0 = ax.get_ylim()
ax.scatter(Anom_u, y[Anom_u], marker='.', color='r', label='anomalia')
ax.scatter(Anom_l, y[Anom_l], marker='.', color='r')
for j,cp in enumerate(CP):
    if j==0: 
        ax.axvline(cp, color='r', label='change-point')
    else:
        ax.axvline(cp, color='r')

for j,(lcp,wt) in enumerate(Win_period):
    if j==0:
        ax.add_patch(Rectangle((lcp, ymin0), wt-lcp, ymax0-ymin0, color='gray', alpha=0.2, linewidth=0, label='inferência'))
    else:
        ax.add_patch(Rectangle((lcp, ymin0), wt-lcp, ymax0-ymin0, color='grey', alpha=0.2, linewidth=0))
ax.legend(loc='upper center')
ax.set_xlabel('Amostra')
ax.set_ylabel('Vazão de upload (Mbits/s)')
plt.tight_layout()
plt.savefig('pdf/shewhart_proposed.pdf', format='pdf')
