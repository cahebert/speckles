import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import errno
import os

class AnalyzeParameters():
    def __init__(self, result_path, exp_type, source='data'):

        # try to open result dict
        try:
            result_dict = pickle.load(open(result_path, 'rb'))
        except FileNotFoundError:
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), str(fits_path))

        self.filters = (562, 832)
        self.source = source

        # double check that the simulations follow this plate scale thing too
        self.scale = {self.filters[0]: .00991, self.filters[1]: .01093,
                      'b': .00991, 'r': .01093}

        ## use a sub function to sort through the results dict and pull out the info desired
        self.process_results(result_dict, exp_type)

    def process_results(self, results, exp_type, quiet=True):
        '''run through result dict and output arrays/dataframes for each kind of result'''

        sub_dict = results[exp_type]
        b_keys = np.sort([k for k in sub_dict.keys() if 'b' in k])
    
        g1 = {'b':[], 'r':[]}
        g2 = {'b':[], 'r':[]}
        size = {'b':[], 'r':[]}
        
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

        self.g1 = {k: np.array(l) for (k,l) in g1.items()}
        self.g2 = {k: np.array(l) for (k,l) in g2.items()}
        self.g = {k: np.hypot(g1[k], g2[k]) for k in g1.keys()}
        self.size = {k: np.array(l) for (k,l) in size.items()}


    ## need to fill in these below!!

    def correlate_bins(self):

    def calculate_dropoff(self):

    def calculate_chromaticity(self):

    def correlate_centroids(self):


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()

    parser.add_argument("--data_folder", type=str, default='MayHRStars/', 
                        help="path to desired data directory within Zorro (and name for info dict")
    parser.add_argument("--local_path", type=str, 
                        default='/Users/clairealice/Documents/research/speckles/intermediates/', 
                        help="path to local directory for saving output")
    parser.add_argument('-bins', default=[5, 15, 30, 60, 'acc'])
    parser.add_argument('-masks', default=False, action='store_true')
    args = parser.parse_args()
    
    dict_path = os.path.join(args.local_path, 
                             f"accepted_info_{args.data_folder.strip('/')}.p")

    if args.masks:
       result_path = os.path.join(args.local_path, 
                               f"parameters_{args.data_folder.strip('/')}_wmask.p")
    else:
       result_path = os.path.join(args.local_path, 
                             f"parameters_{args.data_folder.strip('/')}.p")

    data = AnalyzeParameters(result_path, 5)


        
