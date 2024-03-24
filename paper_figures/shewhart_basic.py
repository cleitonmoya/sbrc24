# -*- coding: utf-8 -*-
"""
Figura 1a - Shewhart - método básico
@author: Cleiton Moya de Almeida
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import time

plt.rcParams.update({'font.size': 8, 'axes.titlesize': 8})

# Data loading
file = 'e45f01963c21_gig03_u_throughput.txt'
y = np.loadtxt(f'../dataset/{file}', usecols=1, delimiter=',')[:530]

verbose = False

# Hyperparameters
w = 20      # estimating window size
k = 3       # kappa

# Auxiliary variables
CP = []
lcp = 0
dev = False
Mu0 = []
U = []
L = []

startTime = time.time()
for t, y_t in enumerate(y):

    if t >= lcp + w:
        
        if t==lcp+w:
            mu0 = y[lcp:t].mean()
            s0 = y[lcp:t].std()
            if verbose: print(f't={t}: mu0={mu0}, s0={s0}')
        
        # lower and upper control limits
        l = mu0 - k*s0
        u = mu0 + k*s0
        
        # Shewhart statistic deviation checking
        dev = y_t>=u or y_t<=l
        
        if dev:
           lcp = t
           if verbose: print(f't={t}: Changepoint at t={lcp}')
           CP.append(lcp)
           dev = False

    else:
        mu0 = np.nan
        l = np.nan
        u = np.nan
    
    Mu0.append(mu0)
    U.append(u)
    L.append(l)
endTime = time.time()
elapsedTime = endTime-startTime


fig, ax = plt.subplots(figsize=(3,2.5))
ax.plot(y, linewidth=0.5)
ax.plot(Mu0, linewidth=0.5, color='green', label=r'$\hat{\mu}_0$')
ax.plot(U, linestyle='--', linewidth=0.5, color='r', label='$\hat{\mu}_0 \pm \kappa \hat{\sigma}_0$')
ax.plot(L, linestyle='--', linewidth=0.5, color='r')
ax.set_ylim([0, 500])
ax.set_xlim([0, 600])
ax.grid(linestyle=':')

ymin0, ymax0 = ax.get_ylim()
ax.add_patch(Rectangle((0, ymin0), w, ymax0-ymin0, color='gray', alpha=0.2, linewidth=0, label='inferência'))
for j,cp in enumerate(CP):
    if j==0: 
        ax.axvline(cp, color='r', label='change+point')
    else:
        ax.axvline(cp, color='r')
    ax.add_patch(Rectangle((cp, ymin0), w, ymax0-ymin0, color='gray', alpha=0.2, linewidth=0))
ax.legend(loc='upper center')
ax.set_xlabel('Amostra')
ax.set_ylabel('Vazão de upload (Mbits/s)')
plt.tight_layout()
plt.savefig('pdf/shewhart_basic.pdf', format='pdf')
