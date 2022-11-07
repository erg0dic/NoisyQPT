# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 18:39:51 2020

@author: mirta
"""
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import numpy as np
import pickle
from misc_utilities import *

#from qpt_oop import *

def func(x, b, c):
     return c* x**(0.001*(-b * x))

fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(50, 10))
w=1
xcutoff = 8000
ycutoff = 2
measurements = ma(np.arange(10,20000, 20), w)
measurements2 = ma(np.arange(10,20000, 20), 1)
for noise in [0,0.1, 0.2, 0.3, 0.4, 0.5]:
    with open("ampdamp_rootlongsimple{}n.pickle".format(int(noise*100)), "rb") as handle:
        c1n = pickle.load(handle)
    with open("dephase_rootlongsimple{}n.pickle".format(int(noise*100)), "rb") as handle:
        c2n = pickle.load( handle)
        
    with open("depol_rootlongsimple{}n.pickle".format(int(noise*100)), "rb") as handle:
        c3n = pickle.load(handle)
    mc1n, m = ma(c1n, 1)[np.where(ma(c1n, 1) < ycutoff)], measurements2[np.where(ma(c1n, 1) < ycutoff)]
    mc1n, m = mc1n[np.where(m < xcutoff)], m[np.where(m < xcutoff)]    
    
    mc2n = ma(c2n, 1)[np.where(ma(c2n, 1) < ycutoff)]
    mc2n = mc2n[np.where(m < xcutoff)]

    mc3n = ma(c3n, 1)[np.where(ma(c3n, 1) < ycutoff)]
    mc3n = mc3n[np.where(m < xcutoff)]
    
    popt, pcov = curve_fit(func, m, mc1n, p0=(1,1), maxfev=6000)
    slope1, a1 = popt[0], popt[1]
    
    popt, pcov = curve_fit(func, m, mc2n, p0=(1,1), maxfev=6000)
    slope2, a2 = popt[0], popt[1]
    
    popt, pcov = curve_fit(func, m, mc3n, p0=(1,1), maxfev=6000)
    slope3, a3 = popt[0], popt[1]
    ax[0].plot(measurements, ma(c1n, w), label='amp damp noise = {}, slope = {:.3E}, a = {:.3E}'.format(noise, slope1, a1))
    ax[1].plot(measurements, ma(c2n, w), label='dephase noisy = {}, slope = {:.3E}, a = {:.3E}'.format(noise, slope2, a2))
    ax[2].plot(measurements, ma(c3n, w), label='depol noisy = {}, slope = {:.3E}, a = {:.3E}'.format(noise, slope3, a3))
    
    ax[0].plot(measurements, func(measurements, slope1, a1), '--',alpha=0.3)
    ax[1].plot(measurements, func(measurements,slope2, a2), '--',alpha=0.3)
    ax[2].plot(measurements, func(measurements, slope3, a3), '--',alpha=0.3)

    ax[0].set_xlabel("Measurements")
    ax[1].set_xlabel("Measurements")
    ax[2].set_xlabel("Measurements")
    ax[0].set_ylabel("Dnorm")
    ax[1].set_ylabel("Dnorm")
    ax[2].set_ylabel("Dnorm")

ax[0].legend()
ax[1].legend()
ax[2].legend()