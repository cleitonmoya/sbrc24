# -*- coding: utf-8 -*-
"""
Plota as figuras 2, 3 e 4
@author: Cleiton Moya de Almeida
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.lines as mlines
from matplotlib.gridspec import GridSpec
from scipy.stats import betabinom

plt.rcParams.update({'font.size': 8, 'axes.titlesize': 8, 'hatch.linewidth':0.5})

series_type = ['d_throughput', 'd_rttmean', 'u_throughput', 'u_rttmean']
series_name = ['Vazão down.', 'Latência down.', 'Vazão up.', 'Latência up.']

clients = ['dca6326b9aa1', 'dca6326b9c99', 'dca6326b9ca8',
       'e45f01359a20', 'e45f01963c21', 'dca6326b9ce4']
client_n = [f'Cliente {n}' for n in range(1,len(clients)+1)]

sites = ['gig01', 'gig02', 'gig03', 'gig04', 
         'gru02', 'gru03', 'gru05','rnp_rj', 'rnp_sp']

methods_ba = ['shewhart_ba', 'ewma_ba', 'cusum_2s_ba', 'cusum_wl_ba']
methods_ps = ['shewhart_ps', 'ewma_ps', 'cusum_2s_ps', 'cusum_wl_ps']
methods_ps2 = ['shewhart_ps', 'ewma_ps', 'cusum_2s_ps', 'cusum_wl_ps', 'vwcd']
methods = methods_ba + methods_ps + ['vwcd']

methods_name = ['Shewhart', 'EWMA', '2S-CUSUM', 'WL-CUSUM', 'VWCD']

hatches = ['', '///']

# Carrega o data-frame de resultados do experimento
df = pd.read_pickle('../experiment/df_results.pkl')


# Computa no úmero de desvios/decrementos de vazão
def dev_abs(M, dev_abs, direction):
    M = np.array(M)
    n_mean = 0
    if len(M) > 1:
        dM = np.diff(M)
        
        if direction == 'dec':
            c2 = dM <= -dev_abs
        else:
            c2 = dM >= dev_abs
        n_mean = c2.sum()
    return n_mean


def method_type(m):
    if m[-2:] == 'ba':
        return 'Basic'
    else:
        return 'Proposed'


def method_name(m):
    if m[:8] == 'cusum_2s':
        return '2S-CUSUM'
    elif m[:8] == 'cusum_wl':
        return 'WL-CUSUM'
    elif m[:4] == 'ewma':
        return 'EWMA'
    elif m[:8] == 'shewhart':
        return 'Shewhart'
    elif m == 'vwcd':
        return 'VWCD'

df['method_name'] = [method_name(m) for m in df['method']]

C0 = np.array([142, 186, 217])/255 # blue
C1 = np.array([255, 190, 134])/255 # orange

#%% Figura 3 - Número de pontos de mudança
df_ = df.groupby(['method'], as_index=False, )['num_cp'].sum()
df_['method_type'] =[method_type(m) for m in df_['method']]
df_['method_name'] =[method_name(m) for m in df_['method']]
df_['id'] = [2, 3, 4, 5, 6, 7, 0, 1, 8]
df_.sort_values(by='id', inplace=True)

fig,ax = plt.subplots(figsize=(3.5,3.5), layout='constrained')

_ = sns.barplot(data=df_, x='method_name', y='num_cp', hue='method_type',
                errorbar=None, palette=[C0, C0], saturation=1, ax=ax, 
                zorder=2, hue_order=['Basic', 'Proposed'])

num_locations = len(df_.method_type.unique())
for i, bar in enumerate(ax.patches):
    if i <=3 or i==9:
       hatch = hatches[1]
    else:
       hatch = hatches[0]
    bar.set_edgecolor('k')
    bar.set_linewidth(0.5)
    bar.set_hatch(hatch)

ax.patches[8].set_x(4-0.2)

ax.set_ylabel("Número de pontos de mudanças")
ax.set_xlabel("")
ax.tick_params(axis='x', labelrotation=20)
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles=handles, labels=['Básico', 'Proposto'], loc='upper right')
ax.set_box_aspect(1)
ax.grid(axis='y', zorder=1, linestyle=':')
plt.savefig('pdf/num_changes.pdf', format='pdf')


#%% Figura 6 - Número de anomalias pontuais por cliente
flierprops = dict( markersize=2, markeredgewidth=0.5)
medianprops = dict(linewidth=1)
fig,ax = plt.subplots(figsize=(7,2), nrows=1, ncols=4, sharey=True, layout='constrained')
for i,s in enumerate(series_type):
    df_ = df[(df['method'].isin(methods_ps)) & (df.serie == s)]
    df_ = df_[['client', 'serie', 'num_anom_u', 'num_anom_l']].melt(id_vars=['client', 'serie'], var_name='anom_type', value_name='num_anom')
    
    ax[i].set_title(series_name[i])
    bar = sns.boxplot(data=df_, x="client", y="num_anom", hue='anom_type', 
                      ax=ax[i], palette = [C0, C1], saturation=1,
                      linewidth=0.5, flierprops=flierprops, zorder=2,
                      medianprops=medianprops)
    
    ax[i].grid(axis='y', linestyle=':')
    
    if i==0:
        bar.legend_.texts[0].set_text('Incremento')
        bar.legend_.texts[1].set_text('Decremento')
        bar.legend_.set_title(None)
        bar.legend_.set_frame_on(True)
        bar.legend_.get_patches()[0].set_hatch(hatches[0])
        bar.legend_.get_patches()[1].set_hatch(hatches[1])
    else:
        ax[i].get_legend().remove()
        
    for j, bar in enumerate(ax[i].patches):
        if j<12:
            hatch = hatches[j//6]
        else:
            hatch = hatches[j-12]  
        bar.set_hatch(hatch)
    
    ax[i].set_xticks(ax[i].get_xticks())
    ax[i].set_xticklabels(range(1,7))
    if i==0:
        ax[i].set_ylabel("Núm. de anomalias pontuais")
        ax[i].set_xlabel("Cliente")
    else:
        ax[i].set_ylabel("")
        ax[i].set_xlabel("")
plt.savefig('pdf/anom_boxplot.pdf', format='pdf')


#%% Figura 7 - Número de pontos de mudança de decremento de vazão de upload 
fig,ax = plt.subplots(nrows=2,ncols=3, sharey=True, figsize=(5,4), layout='constrained')
x_thr = np.arange(1, 101, 1)

for i,c in enumerate(clients):
    ax[i//3, i%3].set_title(f'{client_n[i]}')
    ax[i//3, i%3].grid(linestyle=':')
    
    for j,m in enumerate(methods_ps2):
        
        df_ = df[(df.client==c) & (df.serie==series_type[2]) & (df.method==m)]
        
        M0 = df_.M0.values.tolist()
        M = [sum([dev_abs(m0_list, p, 'dec') 
              for m0_list in M0]) for p in x_thr]
        
        ax[i//3, i%3].plot(x_thr, M, label=f'{methods_name[j]}')
        ax[i//3, i%3].set_xticks(range(0,125,25))
        ax[i//3, i%3].set_xticklabels(range(0,125,25))
        ax[i//3, i%3].set_xlim([0,100])
        ax[i//3, i%3].set_yticks(range(0,30,2))
        ax[i//3, i%3].set_ylim([0,10])
        

        if i==0:
            ax[i//3, i%3].legend()
            ax[i//3, i%3].set_ylabel('Num. de mudanças')
            ax[i//3, i%3].set_xlabel('Decremento (Mbits/s)')
plt.savefig('pdf/ut_decreasing.pdf', format='pdf')


#%% Função auxiliar para figura 4
def get_vwcd_votes(X):
    # Hyperparameters
    w = 20              # window size
    w0 = 20             # window used to estimate the post-change parameters
    alpha = 5           # Beta-binomial hyperp - prior dist. window
    beta = 5            # Beta-binomial hyperp - prior dist. window
    p_thr = 0.8         # threshold probability to an window decide for a changepoint
    pa_thr = 0.9        # threshold probabilty to decide for a changepoint
    vote_n_thr = 10     # min. number of votes to decide for a changepoint
    y0 = 0.5            # logistic prior hyperparameter
    yw = 0.9            # logistic prior hyperparameter
    aggreg = 'mean'     # aggregation function for votes

    # Auxiliary functions
    # Compute the window posterior probability given the log-likelihood and prior
    # using the log-sum-exp trick
    def pos_fun(ll, prior, tau):
        c = np.nanmax(ll)
        lse = c + np.log(np.nansum(prior*np.exp(ll - c)))
        p = ll[tau] + np.log(prior[tau]) - lse
        return np.exp(p)

    # Aggregate a list of votes - compute the posterior probability
    def votes_pos(vote_list, prior_v):
        vote_list = np.array(vote_list)
        prod1 = vote_list.prod()*prior_v
        prod2 = (1-vote_list).prod()*(1-prior_v)
        p = prod1/(prod1+prod2)
        return p

    # Prior probabily for votes aggregation
    def logistic_prior(x, w, y0, yw):
        a = np.log((1-y0)/y0)
        b = np.log((1-yw)/yw)
        k = (a-b)/w
        x0 = a/k
        y = 1./(1+np.exp(-k*(x-x0)))
        return y
    
    def logpdf(x,loc,scale):
        c = 1/np.sqrt(2*np.pi)
        y = np.log(c) - np.log(scale) - (1/2)*((x-loc)/scale)**2
        return y

    # Auxiliary variables
    N = len(X)

    # Prior probatilty for a changepoint in a window - Beta-Binomial
    i_ = np.arange(0,w-3)
    prior_w = betabinom(n=w-4,a=alpha,b=beta).pmf(i_)

    # prior for vot aggregation
    x_votes = np.arange(1,w+1)
    prior_v = logistic_prior(x_votes, w, y0, yw) 

    votes = {i:[] for i in range(N)} # dictionary of votes 
    votes_agg = {}  # aggregated voteylims

    lcp = 0 # last changepoint
    CP = [] # changepoint list
    M0 = [] # list of post-change mean
    S0 = [] # list of post-change standard deviation
    N_votes_tot = np.zeros(N)
    N_votes_ele = np.zeros(N)
    
    for n in range(N):
        if n>=w-1:
            
            # estimate the paramaters (w0 window)
            if n == lcp+w0:
                # estimate the post-change mean and variace
                m_w0 = X[n-w0+1:n+1].mean()
                s_w0 = X[n-w0+1:n+1].std(ddof=1)
                M0.append(m_w0)
                S0.append(s_w0)
            
            # current window
            Xw = X[n-w+1:n+1]
            
            
            LLR_h = []
            for nu in range(1,w-3+1):
                # MLE and log-likelihood for H1
                x1 = Xw[:nu+1] #Xw até nu
                m1 = x1.mean()
                s1 = x1.std(ddof=1)
                if np.round(s1,3) == 0:
                    #if verbose: print(f'n={n}: warning: s1={s1}, using s1=0.001')
                    s1 = 0.001
                logL1 = logpdf(x1, loc=m1, scale=s1).sum()
                
                # MLE and log-likelihood  for H2
                x2 = Xw[nu+1:]
                m2 = x2.mean()
                s2 = x2.std(ddof=1)
                if np.round(s2,3) == 0:
                    #if verbose: print(f'n={n}: warning: s2={s2}, using s2=0.001')
                    s2 = 0.001
                logL2 = logpdf(x2, loc=m2, scale=s2).sum()

                # log-likelihood ratio
                llr = logL1+logL2
                LLR_h.append(llr)

            
            # Compute the posterior probability
            LLR_h = np.array(LLR_h)
            pos = [pos_fun(LLR_h, prior_w, nu) for nu in range(w-3)]
            pos = [np.nan] + pos + [np.nan]*2
            pos = np.array(pos)
            
            # Compute the MAP (vote)
            p_vote_h = np.nanmax(pos)
            nu_map_h = np.nanargmax(pos)
            
            # Store the vote if it meets the hypothesis test threshold
            j = n-w+1+nu_map_h # Adjusted index 
            votes[j].append(p_vote_h)
            
            # Aggregate the votes for X[n-w+1]
            votes_list = votes[n-w+1]
            elegible_votes = [v for v in votes_list if v > p_thr]
            num_votes_tot = len(votes_list)         # number of total votes
            num_votes_ele = len(elegible_votes)     # number of elegible votes
            N_votes_tot[n-w+1] = num_votes_tot
            N_votes_ele[n-w+1] = num_votes_ele
            
            # Decide for a changepoit
            if num_votes_ele >= vote_n_thr:  
                if aggreg == 'posterior':
                    agg_vote = votes_pos(elegible_votes, prior_v[num_votes_ele-1])
                elif aggreg == 'mean':
                    agg_vote = np.mean(elegible_votes)
                votes_agg[n-w+1] = agg_vote
                
                if agg_vote >= pa_thr:
                    lcp = n-w+1 # last changepoint
                    CP.append(lcp)
                    
    return N_votes_ele



#%% Figura 4a
client = clients[2]
site = sites[8]
serie = series_type[3]
file = f'{client}_{site}_{serie}.txt'
y = np.loadtxt(f'../dataset/{file}', usecols=1, delimiter=',')
N = len(y)
num_votes = get_vwcd_votes(y)
num_votes_t = np.where(num_votes)[0]
num_votes_nonzero = num_votes[num_votes_t]

# Formatters
lw = 0.5 # linewidth to plot
unit = 'ms'
markers_dict = {
    'Shewhart' :'x', 
    'EWMA': 's', 
    '2S-CUSUM': '^', 
    'WL-CUSUM': 'v',
    'VWCD': '>'}

# Load the list of changepoints (not empty) for each method
df_ = df[(df.client==client) &
         (df.site==site) &
         (df.serie==serie) &
         (df.CP.str.len() != 0)]

# dict of methods chagepoint list
CP_dict = dict(df_[['method', 'CP']].to_dict(orient='split')['data'])

basic_list = df_[df_.method.isin(methods_ba)].method.unique()
seq_ps_list = df_[df_.method.isin(methods_ps)].method.unique()
methods_name_list = df_.method_name.unique()


fig = plt.figure(figsize=(3.2, 4.5), layout='constrained')

gs = GridSpec(nrows=4, ncols=1, height_ratios=[0.001, 1, 1, 2], 
              hspace=0.7, top=1, bottom=0.08)

ax0 = fig.add_subplot(gs[0])
ax1 = fig.add_subplot(gs[1])
ax2 = fig.add_subplot(gs[2])

subg = gs[3].subgridspec(2, 1, hspace = 0)
ax3 = fig.add_subplot(subg[0])
ax4 = fig.add_subplot(subg[1])
ax4.sharex(ax3)

ax0.axis('off')
xran = np.arange(0,1400,200)
xlim = [0,1200]
yran = np.arange(0,250,50)
ylim = [0,200]

ax = [ax1, ax2, ax3]
for j,_ in enumerate(ax):
    ax[j].xaxis.set_tick_params(labelbottom=True)
    ax[j].set_xticks(xran)
    ax[j].set_xlim(xlim)
    ax[j].set_yticks(yran)

ax1.set_title('Métodos clássicos - implementação básica')
ax1.plot(y, linewidth=lw)
ax1.grid(linestyle=':')
ax1.set_ylabel(unit, fontsize=6)
ax1.tick_params(axis='both', labelsize=6)

y0,y1 = ax1.get_ylim()
y2=0.9*y1
d = (y2-y0)/4
y0 = y0+20
for i,m in enumerate(basic_list):
    CP = CP_dict[m]
    for cp in CP:
        ax1.axvline(cp, color='r', alpha=0.5, linewidth=0.5)
        ax1.plot(cp, y0+i*d, 
                   marker=markers_dict[method_name(m)], 
                   markersize=3, 
                   color='r')

ax2.set_title('Métodos clássicos - proposta')
ax2.plot(y, linewidth=lw)
ax2.grid(linestyle=':')
ax2.set_ylabel(unit, fontsize=6)
ax2.tick_params(axis='both', labelsize=6)
for i,m in enumerate(seq_ps_list):
    CP = CP_dict[m]
    for cp in CP:
        ax2.axvline(cp, color='r', linestyle='-', alpha=0.5, linewidth=0.5)
        ax2.plot(cp, y0+i*d, 
                   marker=markers_dict[method_name(m)], 
                   markersize=3, 
                   color='r')

ax3.set_title('Voting Windows')
ax3.plot(y, linewidth=lw)
ax3.grid(linestyle=':')
ax3.set_ylabel(unit, fontsize=6)
ax3.tick_params(axis='both', labelsize=6)
ax3.tick_params(labelbottom=False)
CP = CP_dict['vwcd']
for cp in CP:
    ax3.axvline(cp, color='r', linestyle='-', alpha=0.5, linewidth=0.5)
    ax3.plot(cp, y0+2*d, 
               marker=markers_dict['VWCD'], 
               markersize=3, 
               color='r')

markerline, stemline, baseline = ax4.stem(num_votes_t, num_votes_nonzero)
plt.setp(markerline, markersize = 3)
plt.setp(baseline, linewidth=0)
plt.setp(stemline, linewidth=0.5)
ax4.set_xlabel('amostras', fontsize=6)
ax4.set_ylabel('Número de votos', fontsize=6, labelpad=7)
ax4.tick_params(axis='both', labelsize=6)
ax4.grid(linestyle=':')

# legenda
lines_leg = [mlines.Line2D([], [], 
                           color='r', 
                           marker=markers_dict[m], 
                           linewidth=0, 
                           markersize=3, 
                           label=m) for m in methods_name_list]

_ = ax0.legend(handles=lines_leg, 
                    loc='upper center',
                    ncol=5, 
                    fontsize=6,
                    handletextpad=0.01,
                    columnspacing=0.5)

plt.savefig('pdf/client3_rnpsp_urtt.pdf', format='pdf')


#%% Figura 4b - Cliente 6, rnp-rj, down throughput
client = clients[5]
site = sites[7]
serie = series_type[0]
file = f'{client}_{site}_{serie}.txt'
y = np.loadtxt(f'../dataset/{file}', usecols=1, delimiter=',')
N = len(y)
num_votes = get_vwcd_votes(y)
num_votes_t = np.where(num_votes)[0]
num_votes_nonzero = num_votes[num_votes_t]

# Formatters
lw = 0.5 # linewidth to plot
unit = 'Mbits/s'
markers_dict = {
    'Shewhart' :'x', 
    'EWMA': 's', 
    '2S-CUSUM': '^', 
    'WL-CUSUM': 'v',
    'VWCD': '>'}

# Load the list of changepoints (not empty) for each method
df_ = df[(df.client==client) &
         (df.site==site) &
         (df.serie==serie) &
         (df.CP.str.len() != 0)]

# dict of methods chagepoint list
CP_dict = dict(df_[['method', 'CP']].to_dict(orient='split')['data'])

basic_list = df_[df_.method.isin(methods_ba)].method.unique()
seq_ps_list = df_[df_.method.isin(methods_ps)].method.unique()
methods_name_list = df_.method_name.unique()

fig = plt.figure(figsize=(3.2, 4.5), layout='constrained')

gs = GridSpec(nrows=4, ncols=1, height_ratios=[0.001, 1, 1, 2], 
              hspace=0.7, top=1, bottom=0.08)

ax0 = fig.add_subplot(gs[0])
ax1 = fig.add_subplot(gs[1])
ax2 = fig.add_subplot(gs[2])

subg = gs[3].subgridspec(2, 1, hspace = 0)
ax3 = fig.add_subplot(subg[0])
ax4 = fig.add_subplot(subg[1])
ax4.sharex(ax3)

ax0.axis('off')
xran = np.arange(0,1400,200)
xlim = [0,1200]
yran = np.arange(0,250,50)
ylim = [0,200]

ax = [ax1, ax2, ax3]
for j,_ in enumerate(ax):
    ax[j].xaxis.set_tick_params(labelbottom=True)
    ax[j].set_xticks(xran)
    ax[j].set_xlim(xlim)

ax1.set_title('Métodos clássicos - implementação básica')
ax1.plot(y, linewidth=lw)
ax1.grid(linestyle=':')
ax1.set_ylabel(unit, fontsize=6)
ax1.tick_params(axis='both', labelsize=6)

y0,y1 = ax1.get_ylim()
y2=0.9*y1
d = (y2-y0)/4
y0 = y0+20
for i,m in enumerate(basic_list):
    CP = CP_dict[m]
    for cp in CP:
        ax1.axvline(cp, color='r', alpha=0.5, linewidth=0.5)
        ax1.plot(cp, y0+i*d, 
                   marker=markers_dict[method_name(m)], 
                   markersize=3, 
                   color='r')
      
ax2.set_title('Métodos clássicos - proposta')
ax2.plot(y, linewidth=lw)
ax2.grid(linestyle=':')
ax2.set_ylabel(unit, fontsize=6)
ax2.tick_params(axis='both', labelsize=6)
for i,m in enumerate(seq_ps_list):
    CP = CP_dict[m]
    for cp in CP:
        ax2.axvline(cp, color='r', linestyle='-', alpha=0.5, linewidth=0.5)
        ax2.plot(cp, y0+i*d, 
                   marker=markers_dict[method_name(m)], 
                   markersize=3, 
                   color='r')

ax3.set_title('Voting Windows')
ax3.plot(y, linewidth=lw)
ax3.grid(linestyle=':')
ax3.set_ylabel(unit, fontsize=6)
ax3.tick_params(axis='both', labelsize=6)
ax3.tick_params(labelbottom=False)
CP = CP_dict['vwcd']
for cp in CP:
    ax3.axvline(cp, color='r', linestyle='-', alpha=0.5, linewidth=0.5)
    ax3.plot(cp, y0+2*d, 
               marker=markers_dict['VWCD'], 
               markersize=3, 
               color='r')

markerline, stemline, baseline = ax4.stem(num_votes_t, num_votes_nonzero)
plt.setp(markerline, markersize = 3)
plt.setp(baseline, linewidth=0)
plt.setp(stemline, linewidth=0.5)
ax4.set_xlabel('amostras', fontsize=6)
ax4.set_ylabel('Número de votos', fontsize=6, labelpad=7)
ax4.tick_params(axis='both', labelsize=6)
ax4.grid(linestyle=':')

# Legenda
lines_leg = [mlines.Line2D([], [], 
                           color='r', 
                           marker=markers_dict[m], 
                           linewidth=0, 
                           markersize=3, 
                           label=m) for m in methods_name_list]

_ = ax0.legend(handles=lines_leg, 
                    loc='upper center',
                    ncol=5, 
                    fontsize=6,
                    handletextpad=0.01,
                    columnspacing=0.5)

plt.savefig('pdf/client6_rnprj_dthr.pdf', format='pdf')