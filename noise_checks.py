# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 21:38:35 2020

@author: mirta
"""
import matplotlib.pyplot as plt 
import numpy  as np
import pickle
from misc_utilities import ma
from scipy.optimize import curve_fit


def func(x, b, c):
     return c*x**(0.001*(-b))

fig, ax = plt.subplots(nrows=1, ncols=6, figsize=(20, 5))
w=20   # smooth out the noise by choosing a moving average window
xcutoff = 9900  # cutoffs for fitting decay curve ansatz
ycutoff = 10
w2=10
measurements = ma(np.arange(10,10000, 20), w)
measurements2 = ma(np.arange(10,10000, 20), 1)
for noise in ['x', 'y', 'z', 'all']:
    try:
        with open("cbias_noise/ampdamp_fixede{}n.pickle".format(noise), "rb") as handle:
            c1n = pickle.load(handle)
    
        with open("cbias_noise/dephase_fixede{}n.pickle".format(noise), "rb") as handle:
            c2n = pickle.load(handle)
    
        with open("cbias_noise/depol_fixede{}n.pickle".format(noise), "rb") as handle:
            c3n = pickle.load(handle)
        with open("cbias_noise/rotationx90_fixede{}n.pickle".format(noise), "rb") as handle:
            c4n = pickle.load(handle)
    
        mc1n, m = ma(c1n, 1)[np.where(ma(c1n, 1) < ycutoff)], measurements2[np.where(ma(c1n, 1) < ycutoff)]
        mc1n, m = mc1n[np.where(m < xcutoff)], m[np.where(m < xcutoff)]    
    
        mc2n = ma(c2n, 1)[np.where(ma(c2n, 1) < ycutoff)]
        mc2n = mc2n[np.where(m < xcutoff)]
    
        mc3n = ma(c3n, 1)[np.where(ma(c3n, 1) < ycutoff)]
        mc3n = mc3n[np.where(m < xcutoff)]
    
        mc4n = ma(c4n, 1)[np.where(ma(c4n, 1) < ycutoff)]
        mc4n = mc4n[np.where(m < xcutoff)]
    
    
        popt, pcov = curve_fit(func, ma(m, w2), ma(mc1n, w2), p0=(1,1), maxfev=6000)
        slope1, a1 = popt[0], popt[1]
    
        popt, pcov = curve_fit(func, ma(m, w2), ma(mc2n, w2), p0=(1,1), maxfev=6000)
        slope2, a2 = popt[0], popt[1]
    
        popt, pcov = curve_fit(func, ma(m, w2), ma(mc3n, w2), p0=(1,1), maxfev=6000)
        slope3, a3 = popt[0], popt[1]
    
        popt, pcov = curve_fit(func, ma(m, w2), ma(mc4n, w2), p0=(1,1), maxfev=6000)
        slope4, a4 = popt[0], popt[1]
    

    
        ax[0].semilogy(measurements, ma(c1n, w), alpha=0.4, label='amp damp noise = {}, slope={:.3E}'.format(noise, slope1, a1))
        ax[1].semilogy(measurements, ma(c2n, w),alpha=0.4, label='dephase noisy = {}, slope={:.3E}'.format(noise, slope2, a2))
        ax[2].semilogy(measurements, ma(c3n, w),alpha=0.4, label='depol noisy = {}, slope={:.3E}'.format(noise, slope3, a3))
        ax[3].semilogy(measurements, ma(c4n, w),alpha=0.4, label='Rotx90 noisy = {}, slope={:.3E}'.format(noise, slope4, a4)) 

        ax[0].semilogy(measurements, func(measurements, slope1, a1), 'b--',alpha=0.3)
        ax[1].semilogy(measurements, func(measurements,slope2, a2), 'b--',alpha=0.3)
        ax[2].semilogy(measurements, func(measurements, slope3, a3), 'b--',alpha=0.3)
        ax[3].semilogy(measurements, func(measurements, slope4, a4), 'b--',alpha=0.3)

    except EOFError as e:
        print("{} for noise={}".format(e, noise))
ax[0].set_xlabel("Measurements 30% noise")
ax[1].set_xlabel("Measurements 30% noise")
ax[2].set_xlabel("Measurements 30% noise")
ax[3].set_xlabel("Measurements 30% noise")

ax[0].set_ylabel("Dnorm")
ax[1].set_ylabel("Dnorm")
ax[2].set_ylabel("Dnorm")
ax[3].set_ylabel("Dnorm")


ax[0].legend(fontsize=10)
ax[1].legend(fontsize=10)
ax[2].legend(fontsize=10)
ax[3].legend(fontsize=10)
