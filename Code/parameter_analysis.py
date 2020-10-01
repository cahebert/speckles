import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import errno
import os
from scipy.optimize import curve_fit

class AnalyzeParameters():
    def __init__(self, result_path, info_path, exp_type, source='data'):

        # try to open result dict
        try:
            result_dict = pickle.load(open(result_path, 'rb'))
        except FileNotFoundError:
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), str(result_dict))
        # try to open info dict
        try:
            info_dict = pickle.load(open(info_path, 'rb'))
        except FileNotFoundError:
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), str(info_path))

        self.filters = (562, 832)
        self.source = source

        # double check that the simulations follow this plate scale thing too
        self.scale = {self.filters[0]: .00991, self.filters[1]: .01093,
                      'b': .00991, 'r': .01093}

        # use a sub function to sort through the results /info dict and pull out the info desired
        self.process_results(result_dict, info_dict, exp_type)

    def process_results(self, results, info_dict, exp_type, quiet=True):
        '''run through result dict and output arrays/dataframes for each kind of result'''
        self.exp_type = exp_type

        sub_dict = results[exp_type]
        b_keys = np.sort([k for k in sub_dict.keys() if 'b' in k])
    
        g1 = {'b':[], 'r':[]}
        g2 = {'b':[], 'r':[]}
        size = {'b':[], 'r':[]}
        x = {'b':[], 'r':[]}
        y = {'b':[], 'r':[]}
        
        for bk in b_keys:
            rk = bk.replace('b','r')

            b_errors = np.array([hsm_out.error_message != '' for hsm_out in sub_dict[bk]])
            r_errors = np.array([hsm_out.error_message != '' for hsm_out in sub_dict[rk]])
            
            # check for error messages:
            if (b_errors).any() or (r_errors).any():
                if not quiet: 
                    print(f'dataset {bk} has an HSM error in moments estimation!')
            else:
                # if no errors, put the shapes and sizes of results 
                for k, filt in zip([bk,rk], ['b','r']):
                    g1[filt].append([hsm_out.observed_shape.g1 for hsm_out in sub_dict[k]])
                    g2[filt].append([hsm_out.observed_shape.g2 for hsm_out in sub_dict[k]])
                    size[filt].append([hsm_out.moments_sigma * 2.355 * self.scale[filt] for hsm_out in sub_dict[k]])

                    x[filt].append(info_dict[k]['centroids']['x'] * self.scale[filt])
                    y[filt].append(info_dict[k]['centroids']['y'] * self.scale[filt])

        self.g1 = {k: np.array(l).T for (k,l) in g1.items()}
        self.g2 = {k: np.array(l).T for (k,l) in g2.items()}
        self.g = {k: np.hypot(g1[k], g2[k]).T for k in g1.keys()}
        self.size = {k: np.array(l).T for (k,l) in size.items()}
        self.x = {k: np.array(l).T for (k,l) in x.items()}
        self.y = {k: np.array(l).T for (k,l) in y.items()}

    def correlate_bins(self, parameters=['g1', 'g2', 'g', 'size'], bootstrap=False, B=1000):
        '''return correlation coefficients for the parameter specified. Optional: return bootstrap samples'''
        if self.exp_type == 'acc': 
            raise ValueError("hmm, are you sure you have the right data loaded?")

        for param in parameters:
            bin_dict = {'g1': self.g1, 'g2': self.g2, 'g': self.g, 'size': self.size}[param]

            n_bins = bin_dict['r'].shape[0]
            bin_pairs = [l for k in [[(i,j) for j in range(i,n_bins) if i!=j] for i in range(n_bins)] for l in k]

            try: 
                self.corrs[param] = {}
            except AttributeError:
                self.corrs = {param: {}}

            for filt in ['r','b']:
                if bootstrap:
                    self.corrs[param][filt] = {f'{i}{j}': bootstrap_correlation(bin_dict[filt][i], bin_dict[filt][j], B)
                                               for (i,j) in bin_pairs}
                else:
                    self.corrs[param][filt] = {f'{i}{j}': np.corrcoef(bin_dict[filt][i], bin_dict[filt][j], rowvar=False)[0,-1] 
                                               for (i,j) in bin_pairs}

    def calculate_chromaticity(self):
        '''calculate the chromatic exponent of psf size dependence on wavelength'''
        lam1 = self.filters[0]
        lam2 = self.filters[1]
        # check to make sure the order of the filters is the same in the wavelengths and the sizes!
        if lam1 < lam2:
            b = (np.log(self.size['b']) - np.log(self.size['r'])) / (np.log(lam1) - np.log(lam2))
        elif lam1 > lam2:
            b = (np.log(self.size['r']) - np.log(self.size['b'])) / (np.log(lam1) - np.log(lam2))
        self.b = b

    def calculate_dropoff(self, delay=False, asymptote=False, pts=np.logspace(-1.22,1.79,15), bootstrap=False, B=1000):
        '''
        Fit power law to ellipticity data, return best fit parameters. 
        Options:
        * can fix the asymptotic value to a nonzero value
        * delay in time before the dropoff starts
        '''
        if self.exp_type != 'acc': 
            raise ValueError("hmm, are you sure you have the right data loaded?")

        if not asymptote:
            asymptote = {'b':0, 'r':0}
           
        def power_law_b(t, *p):
            return power_law(t, p, asymptote=asymptote['b'])
        def power_law_r(t, *p):
            return power_law(t, p, asymptote=asymptote['r'])

        if delay:
            p0 = [0.5, -.2, 0]
            bounds = [[-np.inf,-1, 0], [np.inf, 0, 60.]]
            if bootstrap: b_params = np.empty((B, 3))
        else:
            p0=[0.5, -.2]
            bounds=[[-np.inf, -1], [np.inf, 0]]
            if bootstrap: b_params = np.empty((B, 2))

        self.dropoff_params = {}
        for filt, plaw in zip(['b','r'], [power_law_b, power_law_r]):
            if bootstrap:
                for b in range(B):
                    samples = sklearn.utils.resample(ydata, replace=True, n_samples=len(ydata)-1)
                    b_params[b], _ = curve_fit(plaw, xdata=pts, ydata=samples.mean(axis=0), p0=p0, bounds=bounds)
                self.dropoff_params[filt] = b_params
            else:
                self.dropoff_params[filt], _ = curve_fit(plaw, xdata=pts, ydata=self.g[filt].mean(axis=0), p0=p0, bounds=bounds)


    def correlate_g_centroids(self, bootstrap=True, B=1000):
        '''correlate ellipticity with centroid motion. this function works only for full 60s data'''
        if self.exp_type != '60': 
            raise ValueError("hmm, are you sure you have the right data loaded?")

        try: 
            self.corrs['centroid'] = {'g1': {}, 'g2': {}}
        except AttributeError:
            self.corrs = {'centroid': {'g1': {}, 'g2': {}}}

        for p in ['g1', 'g2']:
            for filt in ['b', 'r']:
                y = self.g1[p][filt]
                self.centroid_moms = centroid_moments(self.x[filt], self.y[filt])

                rho_delta = np.corrcoef(y, self.centroid_moms['delta_xsq_ysq'], rowvar=False)[0,-1]
                rho_xy = np.corrcoef(y, self.centroid_moms['sigma_xy'], rowvar=False)[0,-1]

                err_delta = np.std(bootstrap_correlation(y, self.centroid_moms['delta_xsq_ysq'], B))
                err_xy = np.std(bootstrap_correlation(y, self.centroid_moms['sigma_xy'], B))

                self.corrs['centroid'][filt][p] = {'rho_delta': rho_delta, 'rho_xy': rho_xy,
                                                   'err_delta': err_delta, 'err_xy': err_xy}


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


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()

    parser.add_argument("--data_folder", type=str, default='MayHRStars/', 
                        help="path to desired data directory within Zorro (and name for info dict")
    parser.add_argument("--local_path", type=str, help="path to local directory for saving output",
                        default='/Users/clairealice/Documents/research/speckles/intermediates/')
    parser.add_argument('-bins', default=[5, 15, 30, 60, 'acc'])
    parser.add_argument('-masks', default=False, action='store_true')
    args = parser.parse_args()
    
    info_path = os.path.join(args.local_path, f"accepted_info_{args.data_folder.strip('/')}.p")

    if args.masks:
       result_path = os.path.join(args.local_path, f"parameters_{args.data_folder.strip('/')}_wmask.p")
    else:
       result_path = os.path.join(args.local_path, f"parameters_{args.data_folder.strip('/')}.p")

    data = AnalyzeParameters(result_path, info_path, 30)
    # data.correlate_bins()
    # data.calculate_dropoff()
    data.correlate_centroids()


        
