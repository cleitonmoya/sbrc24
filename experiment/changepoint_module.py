# -*- coding: utf-8 -*-
"""
Online changepoint detection module

    - Basic implementation: suffix 'ba'
    - Proposed implementation: suffix 'ps'
    - Functions:
        Shewhart: shewhart_ba, shewhart_ps
        Exponential Weighted Moving Average: ewma_ba, ewma_ps
        Two-sided CUSUM: cusum_2s_ba, cusum_2s_ps
        Window-Limited CUSUM: cusum_wl_ba, cusum_wl_ps
        Voting Windows Changepoint Detection: vwcd
    
@author: Cleiton Moya de Almeida
"""
import numpy as np
from scipy.stats import shapiro, betabinom
from statsmodels.tsa.stattools import adfuller
import time

verbose = False


# Shapiro-Wilk normality test
# H0: normal distribution
def normality_test(y, alpha):
    _, pvalue = shapiro(y)
    return pvalue > alpha


# Augmented Dickey-Fuller test for unitary root (non-stationarity)
# H0: the process has a unit root (non-stationary)
def stationarity_test(y, alpha):
    adf = adfuller(y)
    pvalue = adf[1]
    return pvalue < alpha


# Compute the log-pdf for the normal distribution
# Obs.: the scipy built-in function logpdf does not use numpy and so is inneficient
def logpdf(x,loc,scale):
    c = 1/np.sqrt(2*np.pi)
    y = np.log(c) - np.log(scale) - (1/2)*((x-loc)/scale)**2
    return y


# Compute the log-likelihood value for the normal distribution
# Obs.: the scipy built-in function logpdf does not use numpy and so is inneficient
def loglik(x,loc,scale):
    n = len(x)
    c = 1/np.sqrt(2*np.pi)
    y = n*np.log(c/scale) -(1/(2*scale**2))*((x-loc)**2).sum()
    return y


def shewhart_ba(y, w, k):
    """
    Shewhart - basic implementation
    
    Parameters:
    ----------
    y (numpy array): the input time-series
    w (int): estimating window size
    k (int): number of standard deviations to consider a change-point 

    Returns
    -------
    CP (list): change-points 
    elapsedTime (float): running-time 
    """
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
    
    return CP, elapsedTime


def shewhart_ps(y, w, k, rl, ka, alpha_norm, alpha_stat, filt_per, max_var, cs_max):
    """
    Shewhart - proposed implementation
    
    Parameters:
    ----------
    y (numpy array): the input time-series
    w (int): estimating window size
    k (int): number of standard deviations to consider a deviation
    rl (int): number of consecutives deviation to consider a change-point
    ka (int): number of standard deviations to consider a point-anomaly
    alpha_norm (float): Shapyro-Wilker test significance level
    alpha_stat (float): ADF test significance level
    filt_per (float): outlier filter percentil (first window or not. estab.)
    max_var (float): maximum increased variance allowed to consider stab.
    cs_max (int); maximum counter for process not stabilized

    Returns:
    -------
    CP (list): change-points
    Anom_u (list): upper anomalies
    Anom_l (list): lower anomalies
    M0_unique (list): estimated mean of the segments
    S0_unique (list): estimated standar deviation of the segments
    elapsedTime (float): running-time 
    """
    # Auxiliary variables initialization
    CP = []             # changepoint list 
    Anom_u = []         # up point anomalies list
    Anom_l = []         # low point anomalie list
    lcp = 0             # last checked point
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

        if t >= lcp + w:
            
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
                        if verbose: print(f"n={t}: Process stabilized")
                    
                    # Phase 1 parameters estimation
                    mu0 = yw.mean()
                    s0 = yw.std(ddof=1)
                    M0_unique.append(mu0)
                    S0_unique.append(s0)
                    Win_period.append((win_t0,t))
                    if verbose: print(f"n={t}: Estimated mu0={mu0}, sigma0={s0}")
                    
                    # Beside the non-normality, if the last window was not stationary, 
                    # and now the process is normal and statonary, consider a changepoint
                    if t != win_t0+w \
                        and not stationarity_test(y[lcp-w+1:lcp+1], alpha_stat) \
                            and stationarity_test(y[lcp+1:t+1], alpha_stat) \
                                and cs!=cs_max:
                        if verbose: print(f"n={t}: Considering t={t-w} a changepoint")
                        CP.append(t-w)
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
                    win_t0 = t-rl
                    if verbose: print(f't={t}: Changepoint at t={win_t0}')
                    CP.append(win_t0)
                    lcp = win_t0
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

    return CP, Anom_u, Anom_l, M0_unique, S0_unique, elapsedTime


def ewma_ba(y, w, kd, lamb):
    """
    Exponential Weighted Moving Average (EWMA) - basic implementation
    
    Parameters:
    ----------
    y (numpy array): the input time-series
    w (int): estimating window size
    kd (int): EWMA 'kd' hyperparameter
    lamb (float): EWMA 'lambda' hyperparameter

    Returns
    -------
    CP (list): change-points 
    elapsedTime (float): running-time 
    """
    # Auxiliary variables initialization
    CP = []
    lcp = 0
    Mu0 = []
    Sigma0 = []
    U = []
    L = []
    Z = []

    startTime = time.time()
    for t,y_t in enumerate(y):

        if t >= lcp + w:
            
            # Phase 1 estimation
            if t == lcp+w:
                mu0 = np.mean(y[lcp:t])
                sigma0 = y[lcp:t].std(ddof=0)
                z = mu0# reset the Z statistic
                if verbose: print(f't={t}: mu0={mu0}, sigma0={sigma0}')

            
            # Phase 2 statistic and limits estimation
            z = lamb*y[t] + (1-lamb)*z
            ucl = mu0 + kd*sigma0*np.sqrt((lamb/(2-lamb)))
            lcl = mu0 - kd*sigma0*np.sqrt((lamb/(2-lamb)))
            
            # verifica se há dev do moving range
            dev = z >ucl or z < lcl
            if dev:
                lcp = t
                if verbose: print(f't={t}: Changepoint at t={lcp}')
                CP.append(lcp)

        else:
            ucl = np.nan
            lcl = np.nan
            z = np.nan
            mu0 = np.nan
            sigma0 = np.nan
        
        Z.append(z)    
        U.append(ucl)
        L.append(lcl)
        Mu0.append(mu0)
        Sigma0.append(sigma0)
    endTime = time.time()
    elapsedTime = endTime-startTime

    Z = np.array(Z)
    U = np.array(U)
    L = np.array(L)
    Mu0 = np.array(Mu0)
    Sigma0 = np.array(Sigma0)

    return CP, elapsedTime


def ewma_ps(y, w, kd, lamb, rl, ka, alpha_norm, alpha_stat, filt_per, max_var, cs_max):
    """
    Exponential Weighted Moving Average (EWMA) - basic implementation
    
    Parameters:
    ----------
    y (numpy array): the input time-series
    w (int): estimating window size
    kd (int): EWMA 'kd' hyperparameter
    lamb (float): EWMA 'lambda' hyperparameter
    rl (int): number of consecutives deviation to consider a change-point
    ka (int): number of standard deviations to consider a point-anomaly
    alpha_norm (float): Shapyro-Wilker test significance level
    alpha_stat (float): ADF test significance level
    filt_per (float): outlier filter percentil (first window or not. estab.)
    max_var (float): maximum increased variance allowed to consider stab.
    cs_max (int); maximum counter for process not stabilized

    Returns:
    -------
    CP (list): change-points
    Anom_u (list): upper anomalies
    Anom_l (list): lower anomalies
    M0_unique (list): estimated mean of the segments
    S0_unique (list): estimated standar deviation of the segments
    elapsedTime (float): running-time 
    """
    # Auxiliary variables initialization
    Z = []
    U = []
    L = []
    CP = []
    Anom_u = []         # up point anomalies list
    Anom_l = []         # low point anomalie list
    Mu0 = []
    Sigma0 = []
    Win_period = []
    M0_unique = []      # phase 1 estimated mu0 after each changepoint
    S0_unique = []      # phase 1 estimated sigma0 after each changepoint
    z = np.nan
    za = np.nan
    lcp = 0
    win_t0 = 0          # learning window t0
    c = 0               # sucessive deviation counter
    ca_u = 0            # up point up counter
    ca_l = 0            # low point anomaly counter
    cs = 0              # stabilization counter
    dev = False
    stab = False        # stabilization indicator variable after

    startTime = time.time()
    for t, y_t in enumerate(y):

        if t >= lcp+w:
            
            if t == lcp+w:

                yw = y[lcp+1:t+1]
                sw = yw.std(ddof=1)
                # Shapiro-Wiltker test for normality
                normality = normality_test(yw, alpha_norm)

                # Check if the variance level increasing is acceptable
                # If its the first window, accept blindly, but filter possible
                # outliers before estimating the mu0, sigma0
                first_window = len(Win_period) == 0
                if not first_window:
                    sa = S0_unique[-1]
                    dev_var = abs(sw - sa)/sa
                    var_acept = dev_var <= max_var
                else:
                    var_acept = True
                    yw = yw[(yw>np.quantile(yw,1-filt_per)) & (yw<np.quantile(yw,filt_per))]

                # Stabilization criteria: normality and variance accepted
                stab = normality and var_acept

                # If process did not stabilize after cs_max, force the stabilization,
                # but filter possible outliers to estimate mu0, sigma0
                if stab or cs==cs_max:
                    if cs==cs_max:
                        yw = yw[(yw>np.quantile(yw,1-filt_per)) & (yw<np.quantile(yw,filt_per))]
                        if verbose: print(f"n={t}: Considering process stabilized")
                    else:
                        if verbose: print(f"n={t}: Process stabilized")
     
                    # Phase 1 parameters estimation
                    mu0 = yw.mean()
                    sigma0 = yw.std(ddof=1)
                    M0_unique.append(mu0)
                    S0_unique.append(sigma0)
                    Win_period.append((win_t0,t))
                    z=mu0
                    if verbose: print(f"n={t}: Estimated mu0={mu0}, sigma0={sigma0}")

                    # Beside the non-normality, if the last window was not stationary, 
                    # and now the process is normal and statonary, consider a changepoint
                    if t != win_t0+w \
                        and not stationarity_test(y[lcp-w+1:lcp+1], alpha_stat) \
                            and stationarity_test(y[lcp+1:t+1], alpha_stat) \
                                and cs!=cs_max:
                        if verbose: print(f"n={t}: Considering t={t-w} a changepoint")
                        CP.append(t-w)
                    cs = 0
                else:
                    if verbose: print(f"n={t}: Process not stabilized, normal={normality}, var_acetp={var_acept}")
                    lcp=lcp+w
                    cs = cs+1        

            # Check for point anomaly (upper and low)
            anom_u = y_t >= mu0 + ka*sigma0
            anom_l = y_t <= mu0 - ka*sigma0
            if anom_u:
                Anom_u.append(t)
            if anom_l:
                Anom_l.append(t)

            # EWMA statistic update
            za = z
            z = lamb*y[t] + (1-lamb)*z
            ucl = mu0 + kd*sigma0*np.sqrt((lamb/(2-lamb)))
            lcl = mu0 - kd*sigma0*np.sqrt((lamb/(2-lamb)))

            # check for statistic deviation
            dev = z >ucl or z < lcl
            if dev:
                c=c+1
                if anom_u:
                    ca_u = ca_u+1 
                elif anom_l:
                    ca_l = ca_l+1
                
                # confirms the changepoint and resets the ewma statistic
                if c == rl:
                    win_t0 = t-rl
                    if verbose: print(f't={t}: Changepoint confirmed at t={win_t0}')
                    CP.append(win_t0)
                    lcp = win_t0
                    if ca_u > 0:
                        Anom_u = Anom_u[:-ca_u]
                    if ca_l > 0:
                        Anom_l = Anom_l[:-ca_l] 
                    c = 0
                    ca_u = 0
                    ca_l = 0
                Z.append(z)
                z = za
            else:
                c=0
                ca_u = 0
                ca_l = 0
                Z.append(z)
        else:
            z = np.nan
            ucl = np.nan
            lcl = np.nan
            mu0 = np.nan
            sigma0 = np.nan
            Z.append(z)
            

        U.append(ucl)
        L.append(lcl)
        Mu0.append(mu0)
        Sigma0.append(sigma0)
    endTime = time.time()
    elapsedTime = endTime-startTime

    Z = np.array(Z)
    Mu0 = np.array(Mu0)
    Sigma0 = np.array(Sigma0)
            
    return CP, Anom_u, Anom_l, M0_unique, S0_unique, elapsedTime


def cusum_2s_ba(y, w, delta, h):
    """
    Two-sided CUSUM - basic implementation
    
    Parameters:
    ----------
    y (numpy array): the input time-series
    w (int): estimating window size
    delta (int/float): deviation (in terms of sigma0) to detect
    h (float): statistic threshold (in terms of sigma0)

    Returns
    -------
    CP (list): change-points 
    elapsedTime (float): running-time 
    """
    # Auxiliary variables
    U = []
    L = []
    H = []
    CP = []
    Mu = []
    Sigma = []
    Ut = 0
    Lt = 0
    lcp = 0

    startTime = time.time()
    for t, y_t in enumerate(y):
        
        if t >= lcp+w:
            if Ut is np.nan:
                Ut = 0
                Lt = 0
            
            # Phase 1 parameters updating
            if t==lcp+w:
                mu0 = y[lcp:t].mean()
                sigma0 = y[lcp:t].std(ddof=0)
                Ht = h*sigma0
            
            # Phase 2 CUSUM statitics computing
            Ut = Ut + y_t - mu0 - delta*sigma0/2
            Ut = np.heaviside(Ut,0)*Ut
            Lt = Lt - y_t + mu0 - delta*sigma0/2
            Lt = np.heaviside(Lt,0)*Lt

            # check for statistic deviation
            dev = Ut > Ht or Lt > Ht
            if dev:
                lcp = t    
                if verbose: print(f't={t}: Changepoint at t={lcp}')    
                CP.append(lcp)
                dev = False

        else:
            Ut = np.nan
            Lt = np.nan
            Ht = np.nan
            mu0 = np.nan
            sigma0 = np.nan
            
            
        U.append(Ut)
        L.append(Lt)
        H.append(Ht)
        Mu.append(mu0)
        Sigma.append(sigma0)
    endTime = time.time()
    elapsedTime = endTime-startTime

    U = np.array(U)
    L = np.array(L)
    Mu = np.array(Mu)
    Sigma = np.array(Sigma)

    return CP, elapsedTime


def cusum_2s_ps(y, w, delta, h, rl, ka, alpha_norm, alpha_stat, filt_per, max_var, cs_max):
    """
    Two-sided CUSUM - basic implementation
    
    Parameters:
    ----------
    y (numpy array): the input time-series
    w (int): estimating window size
    delta (int/float): deviation (in terms of sigma0) to detect
    h (float): statistic threshold (in terms of sigma0)
    rl (int): number of consecutives deviation to consider a change-point
    ka (int): number of standard deviations to consider a point-anomaly
    alpha_norm (float): Shapyro-Wilker test significance level
    alpha_stat (float): ADF test significance level
    filt_per (float): outlier filter percentil (first window or not. estab.)
    max_var (float): maximum increased variance allowed to consider stab.
    cs_max (int); maximum counter for process not stabilized

    Returns:
    -------
    CP (list): change-points
    Anom_u (list): upper anomalies
    Anom_l (list): lower anomalies
    M0_unique (list): estimated mean of the segments
    S0_unique (list): estimated standar deviation of the segments
    elapsedTime (float): running-time 
    """
    # Auxiliary variables initialization
    Gu = []
    Gl = []
    H = []
    CP = []
    Anom_u = []         # up point anomalies list
    Anom_l = []         # low point anomalie list
    Mu = []
    Sigma = []
    Win_period = []
    M0_unique = []      # phase 1 estimated mu0 after each changepoint
    S0_unique = []      # phase 1 estimated sigma0 after each changepoint
    gu = 0
    gua = 0
    gl = 0
    gla = 0
    lcp = 0
    win_t0 = 0          # learning window t0
    c = 0               # sucessive outlier counter
    ca_u = 0            # up point up counter
    ca_l = 0            # low point anomaly counter
    stab = False        # stabilization indicator variable after
    cs = 0

    startTime = time.time()
    for t, y_t in enumerate(y):
        
        if t >= lcp+w:
            if gu is np.nan:
                gu = 0
                gl = 0
            
            if t==lcp+w:
                
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
                            if verbose: print(f"n={t}: Process stabilized")
                        
                        # Phase 1 parameters estimation
                        mu0 = yw.mean()
                        sigma0 = yw.std(ddof=1)
                        M0_unique.append(mu0)
                        S0_unique.append(sigma0)
                        Win_period.append((win_t0,t))
                        
                        if verbose: print(f"n={t}: Estimated mu0={mu0}, sigma0={sigma0}")
                        
                        # Beside the non-normality, if the last window was not stationary, 
                        # and now the process is normal and statonary, consider a changepoint
                        if t != win_t0+w \
                            and not stationarity_test(y[lcp-w+1:lcp+1], alpha_stat) \
                                and stationarity_test(y[lcp+1:t+1], alpha_stat) \
                                    and cs!=cs_max:
                            if verbose: print(f"n={t}: Considering t={t-w} a changepoint")
                            CP.append(t-w)
                        cs = 0
                    else:
                        if verbose: print(f"n={t}: Process not stabilized, sw={yw.std(ddof=1)}")
                        lcp=lcp+w
                        cs = cs+1
            
            # control limit for deviation
            ht = h*sigma0 
            
            # CUSUM statistics update
            gua = gu
            gla = gl
            gu = gu + y_t - mu0 - delta*sigma0/2
            gu = np.heaviside(gu,0)*gu
            gl = gl - y_t + mu0 - delta*sigma0/2
            gl = np.heaviside(gl,0)*gl

            # Check for point anomaly (upper and low)
            anom_u = y_t >= mu0 + ka*sigma0
            anom_l = y_t <= mu0 - ka*sigma0
            if anom_u:
                Anom_u.append(t)
            if anom_l:
                Anom_l.append(t)
            
            # check for statistic deviation
            dev = gu > ht or gl > ht
            if dev:
                c=c+1
                if anom_u:
                    ca_u = ca_u+1 
                elif anom_l:
                    ca_l = ca_l+1
                    
                if c == rl: # confirma o changepoint e reinicia o cusum
                    win_t0 = t-rl
                    if verbose: print(f't={t}: Changepoint confirmed at t={win_t0}')
                    CP.append(win_t0)
                    lcp = win_t0
                    if ca_u > 0:
                        Anom_u = Anom_u[:-ca_u]
                    if ca_l > 0:
                        Anom_l = Anom_l[:-ca_l]    
                    c = 0
                    ca_u = 0
                    ca_l = 0
                    
                Gu.append(gu)
                Gl.append(gl)
                gu = gua
                gl = gla
                
            else:
                c=0
                ca_u = 0
                ca_l = 0
                Gu.append(gu)
                Gl.append(gl)

        else:
            gu = np.nan
            gl = np.nan
            ht = np.nan
            mu0 = np.nan
            sigma0 = np.nan
            Gu.append(gu)
            Gl.append(gl)
            
        H.append(ht)
        Mu.append(mu0)
        Sigma.append(sigma0)
    endTime = time.time()
    elapsedTime = endTime-startTime

    Gu = np.array(Gu)
    Gl = np.array(Gl)
    Mu = np.array(Mu)
    Sigma = np.array(Sigma)

    return CP, Anom_u, Anom_l, M0_unique, S0_unique, elapsedTime


def cusum_wl_ba(y, w0, w1, h):
    """
    Window-limited CUSUM - basic implementation
    
    Parameters:
    ----------
    y (numpy array): the input time-series
    w0 (int): pre-change estimating window size
    w1 (int): post-change estimating window size
    h (float): statistic threshold (in terms of sigma0)

    Returns
    -------
    CP (list): change-points 
    elapsedTime (float): running-time 
    """
    # Auxiliary variables
    lcp = 0
    S1 = []
    CP = []
    Mu0 = []
    Sigma0 = []
    Mu1 = []
    H = []
    St = np.nan
    m0 = np.nan
    m1 = np.nan

    startTime = time.time()
    for t, y_t in enumerate(y):
        
        if t >= lcp+w0:

            # Phase 1 parameters learning
            if t == lcp+w0:
                m0 = y[lcp:t].mean()
                s0 = y[lcp:t].std(ddof=1)
                Ht = h*s0
            
            # Phase 2 parameters earning
            m1 = y[t-w1:t].mean()
            s1 = y[t-w1:t].std(ddof=1)
            
            # Phase 2 CUSUM statistic computing
            if St is np.nan:
                St = 0 
            St = np.heaviside(St,0)*St
            St = St + logpdf(y_t, m1, s1) - logpdf(y_t, m0, s0)
            
            # Check for statistic deviation
            dev = St > Ht
            if dev:
                lcp=t
                if verbose: print(f'Changepoint at t={t}')
                CP.append(t)
                dev = False

        else:
            St = np.nan
            m0 = np.nan
            s0 = np.nan
            m1 = np.nan
            s1 = np.nan
            Ht = np.nan
            
        S1.append(St)
        Mu0.append(m0)
        Mu1.append(m1)
        Sigma0.append(s0)
        H.append(Ht)
    endTime = time.time()
    elapsedTime = endTime-startTime

    Mu0 = np.array(Mu0)
    Sigma0 = np.array(Sigma0)
            
    return CP, elapsedTime
    

def cusum_wl_ps(y, w0, w1, h, rl, k, ka, alpha_norm, alpha_stat, filt_per, max_var, cs_max):
    """
    Window-limited CUSUM - basic implementation
    
    Parameters:
    ----------
    y (numpy array): the input time-series
    w0 (int): pre-change estimating window size
    w1 (int): post-change estimating window size
    h (float): statistic threshold (in terms of sigma0)
    rl (int): number of consecutives deviation to consider a change-point
    ka (int): number of standard deviations to consider a point-anomaly
    alpha_norm (float): Shapyro-Wilker test significance level
    alpha_stat (float): ADF test significance level
    filt_per (float): outlier filter percentil (first window or not. estab.)
    max_var (float): maximum increased variance allowed to consider stab.
    cs_max (int); maximum counter for process not stabilized

    Returns:
    -------
    CP (list): change-points
    Anom_u (list): upper anomalies
    Anom_l (list): lower anomalies
    M0_unique (list): estimated mean of the segments
    S0_unique (list): estimated standar deviation of the segments
    elapsedTime (float): running-time 
    """
    # Auxiliary variables
    S = []
    H = []
    CP = []
    Anom_u = []         # up point anomalies list
    Anom_l = []         # low point anomalie list
    Mu0 = []
    Sigma0 = []
    Mu1 = []
    Win_period = []
    M0_unique = []      # phase 1 estimated mu0 after each changepoint
    S0_unique = []      # phase 1 estimated sigma0 after each changepoint
    st = 0              # CUSUM statistic
    Sta = 0             # CUSUM statisitc before deviation
    lcp = 0             # last changepoint
    win_t0 = 0          # learning window t0
    c = 0               # sucessive outlier counter
    ca_u = 0            # up point up counter
    ca_l = 0            # low point anomaly counter
    cs = 0          
    stab = False        # stabilization indicator variable after

    startTime = time.time()
    for t, y_t in enumerate(y):
        
        if t >= lcp+w0:
            
            #if not dev, update mu:
            if t==lcp+w0:
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
                        if verbose: print(f"n={t}: Process stabilized")
                    
                    # Phase 1 parameters estimation
                    mu0 = yw.mean()
                    sigma0 = yw.std(ddof=1)
                    M0_unique.append(mu0)
                    S0_unique.append(sigma0)
                    Win_period.append((win_t0,t))
                    
                    if verbose: print(f"n={t}: Estimated mu0={mu0}, sigma0={sigma0}")
                    
                    # Beside the non-normality, if the last window was not stationary, 
                    # and now the process is normal and statonary, consider a changepoint
                    if t != win_t0+w0 \
                        and not stationarity_test(y[lcp-w0+1:lcp+1], alpha_stat) \
                            and stationarity_test(y[lcp+1:t+1], alpha_stat) \
                                and cs!=cs_max:
                        if verbose: print(f"n={t}: Considering t={t-w0} a changepoint")
                        CP.append(t-w0)
                    cs = 0
                else:
                    if verbose: print(f"n={t}: Process not stabilized, norm={normality}, var_acept={var_acept}, sw={yw.std(ddof=1)}")
                    lcp=lcp+w0
                    cs = cs+1

            # control limit for deviation
            ht = h*sigma0 

            # Phase 2 paramters estimation
            mu1 = y[t-w1:t].mean()
            sigma1 = y[t-w1:t].std(ddof=1)
            
            # CUSUM statistics update
            if st is np.nan:
                st = 0
            Sta = st
            st = np.heaviside(st,0)*st
            st = st + logpdf(y_t, mu1, sigma1) - logpdf(y_t, mu0, sigma0)

            # Check for point anomaly (upper and low)
            anom_u = y_t >= mu0 + ka*sigma0
            anom_l = y_t <= mu0 - ka*sigma0
            if anom_u:
                Anom_u.append(t)
            if anom_l:
                Anom_l.append(t)
            
            # check for statistic deviation
            dev = st > ht
            if dev:
                c=c+1              
                if anom_u:
                    ca_u = ca_u+1 
                elif anom_l:
                    ca_l = ca_l+1
                # confirms the changepoint and resets the cusum statistic
                if c == rl:
                    win_t0 = t-rl
                    if verbose: print(f't={t}: Changepoint confirmed at t={win_t0}')
                    CP.append(win_t0)
                    lcp = win_t0
                    if ca_u > 0:
                        Anom_u = Anom_u[:-ca_u]
                    if ca_l > 0:
                        Anom_l = Anom_l[:-ca_l]  
                    c = 0
                    ca_u = 0
                    ca_l = 0
                    
                S.append(st)
                st = Sta
                
            else:
                c=0
                ca_u = 0
                ca_l = 0
                S.append(st)

        else:
            st = np.nan
            ht = np.nan
            mu0 = np.nan
            mu1 = np.nan
            sigma0 = np.nan
            S.append(st)

        H.append(ht)
        Mu0.append(mu0)
        Mu1.append(mu1)
        Sigma0.append(sigma0)
    endTime = time.time()
    elapsedTime = endTime-startTime

    S = np.array(S)
    Mu0 = np.array(Mu0)
    Mu1 = np.array(Mu1)
    Sigma0 = np.array(Sigma0)
    
    return CP, Anom_u, Anom_l, M0_unique, S0_unique, elapsedTime


def vwcd(X, w, w0, alpha, beta, p_thr, vote_p_thr, vote_n_thr, y0, yw, aggreg):
    """
    Voting Windows Changepoint Detection
    
    Parameters:
    ----------
    X (numpy array): the input time-series
    w (int): sliding window size
    w0 (int): pre-chage estimating window size
    h (float): statistic threshold (in terms of sigma0)
    rl (int): number of consecutives deviation to consider a change-point
    ka (int): number of standard deviations to consider a point-anomaly
    alpha_norm (float): Shapyro-Wilker test significance level
    alpha_stat (float): ADF test significance level
    filt_per (float): outlier filter percentil (first window or not. estab.)
    max_var (float): maximum increased variance allowed to consider stab.
    cs_max (int); maximum counter for process not stabilized

    Returns:
    -------
    CP (list): change-points
    M0 (list): estimated mean of the segments
    S0 (list): estimated standar deviation of the segments
    elapsedTime (float): running-time 
    """
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
    
    alpha=beta

    # Auxiliary variables
    N = len(X)
    #vote_n_thr = np.floor(w*vote_n_thr)

    # Prior probatilty for a changepoint in a window - Beta-B
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

    startTime = time.time()
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
            #for nu in range(w):
                # MLE and log-likelihood for H1
                x1 = Xw[:nu+1] #Xw até nu
                m1 = x1.mean()
                s1 = x1.std(ddof=1)
                if np.round(s1,3) == 0:
                    s1 = 0.001
                logL1 = loglik(x1, loc=m1, scale=s1)
                
                # MLE and log-likelihood  for H2
                x2 = Xw[nu+1:]
                m2 = x2.mean()
                s2 = x2.std(ddof=1)
                if np.round(s2,3) == 0:
                    s2 = 0.001
                logL2 = loglik(x2, loc=m2, scale=s2)

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
            if p_vote_h >= p_thr:
                j = n-w+1+nu_map_h # Adjusted index 
                votes[j].append(p_vote_h)
            
            # Aggregate the elegible votes for X[n-w+1]
            votes_list = votes[n-w+1]
            num_votes = len(votes_list)
            if num_votes >= vote_n_thr:
                if aggreg == 'posterior':
                    agg_vote = votes_pos(votes_list, prior_v[num_votes-1])
                elif aggreg == 'mean':
                    agg_vote = np.mean(votes_list)
                votes_agg[n-w+1] = agg_vote
                
                # Decide for a changepoit
                if agg_vote > vote_p_thr:
                    if verbose: print(f'Changepoint at n={n-w+1}, p={agg_vote}, n={num_votes} votes')
                    lcp = n-w+1 # last changepoint
                    CP.append(lcp)

    endTime = time.time()
    elapsedTime = endTime-startTime
    return CP, M0, S0, elapsedTime