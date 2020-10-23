import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import errno
import os
from scipy.optimize import curve_fit

def centroid_moments(x, y):
    '''return second moments of coordinates x and y'''
    if len(x) != len(y): 
        raise ValueError('x and y must have the same length!')
    sigma_xy = np.sum((x-x.mean(axis=0))*(y-y.mean(axis=0)), axis=0) / x.shape[0]
    sigma_x_sq = np.var(x, axis=0)
    sigma_y_sq = np.var(y, axis=0)
    return {'delta_xsq_ysq': sigma_x_sq - sigma_y_sq, 'sigma_xy': sigma_xy}

def power_law(t, p, asymptote=0):
    '''
    return a power law at points t, with exponent alpha, amplitude a, and an optional asymptote.
    '''
    if len(p) == 2:
        return p[0] * t**p[1] + asymptote
    elif len(p) == 3:
        return np.array([p[0] if time<p[2] else p[0] * (time-p[2])**p[1] for time in t]) + asymptote
    
def bootstrap_correlation(thing1, thing2, B):
    '''
    Bootstrap a correlation coefficient between thing1 and thing2, sampling B times. 
    '''
    if len(thing1) != len(thing2): raise ValueError('things to correlate must have the same length!')

    idx = range(len(thing1))
    boot_corr_coefs = np.zeros(len(thing1))
    for i in range(B):
        resampled = sklearn.utils.resample(idx, replace=True, n_samples=len(idx)-1)
        boot_corr_coefs[i] = np.corrcoef(thing1[resampled], thing2[resampled], rowvar=False)[0,-1]
    return boot_corr_coefs

def load_dicts(path_list, exp_type=False):
    '''function to load list of dicts and turn them into one big dict
    :path_list: path to dicts to open'''
    primary_dict = {}
    for path in path_list:
        try:
            dict_i = pickle.load(open(path, 'rb'))
        except FileNotFoundError:
            continue
        else:
            if exp_type is not False and exp_type in dict_i.keys():
                dict_i = dict_i[exp_type]
            primary_dict.update(dict_i)
        
    return primary_dict

