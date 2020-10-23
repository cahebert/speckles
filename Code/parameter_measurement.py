import numpy as np
from astropy.io import fits
import pickle
import errno
import os
import measurement_helper as mhelp

class ExtractParameters():
    def __init__(self, fits_path_list, mask, source='data'):
        '''load the data into the class'''
        if 'r' in fits_path_list[0].split('/')[-1]:
            self.color = 'r'
        elif 'b' in fits_path_list[0].split('/')[-1]:
            self.color = 'b'

        self.source = source
        self.mask = mask

        loaded = mhelp.data_loader(fits_path_list)
        if loaded:
            self.imgs, _ = loaded
        else: 
            return False

        mhelp.subtract_background(self.imgs, exp_mask_dict=self.mask)

    def bin_psf(self, bin_exp, override=True):
        '''
        use this function to bin data into :bin_exp: second chunks
        :override: set whether to ignore imperfect divisions of data into bins
        '''
        residual = bin_exp % .06
        if residual != 0 and override is False:
            raise ValueError(f"Can't perfectly divide data into exposure bins of size {bin_exp}s")
        
        bin_size = int(bin_exp / .06) # in number of exposures
        n_bins = int(self.imgs.shape[0] / bin_size) # number of bins per dataset

        binned = [self.imgs[n * bin_size:(n + 1) * bin_size].sum(axis=0) / bin_size for n in range(n_bins)]
        return binned, bin_exp, False

    def accumulate_psf(self, return_all=False, manual=False):
        '''
        use this function to accumulate psf into frames of log t lengthed exposures, or if 
        :manual: is specfied, into custom accumulationg.
        Toggling :return_all: means accumulating each exposure at a time, result has same length as input
        '''
        
        if manual: indices = manual
        else: indices = [int(np.round(i)) - 1 for i in np.logspace(0, 3, 15)]

        psf = self.imgs[0]
        accumulated = [np.copy(psf)]
                    
        # if return_all, accumulate every exposure
        if return_all:
            for exposure in range(1, self.imgs.shape[0]):
                psf += self.imgs[exposure]
                accumulated.append(np.copy(psf))

            return accumulated, np.linspace(1,1000,1000)
        else:
            for exposure in indices[1:]:
                psf = self.imgs[0:exposure + 1].sum(axis=0)
                accumulated.append(np.copy(psf) / (exposure + 1))
            
            return accumulated, indices, False

    def extract_parameters(self, bins=[5, 15, 30, 60, 'acc']):
        '''
        measure psf parameters for input settings, returns dict of parameters.
        :bins: list of objects that specify what data manipulations to do before measurement
        * A number indicated exposure length of bin
        * str indicates accumulated psf: for now just default indices'''
        parameters = {}
        
        for exps in bins:
            if type(exps)==str:
                accumulated, _, mask = self.accumulate_psf(manual=False)
                parameters[exps] = mhelp.estimate_moments_HSM(accumulated, exp_mask_dict=mask, save_dict={'save':False})
            else:
                binned, _, mask = self.bin_psf(exps)
                parameters[exps] = mhelp.estimate_moments_HSM(binned, exp_mask_dict=mask, save_dict={'save':False})

        return parameters

if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()

    parser.add_argument("--obsdate", type=str, default='20190913', 
                        help="path to desired data directory within Zorro")
    parser.add_argument("--local_path", type=str, 
                        default='/Users/clairealice/Documents/research/speckles/intermediates/', 
                        help="path to local directory for saving output")
    parser.add_argument('--zorro', type=str, default='/Volumes/My Passport/Zorro/', 
                        help="path to main Zorro data directory")
    parser.add_argument('-masks', default=False, action='store_true')
    parser.add_argument('--stars', default='bright', type=str)
    args = parser.parse_args()
    
    data_path = os.path.join(args.zorro, args.obsdate) 
    dict_path = os.path.join(args.local_path, f"accepted_info_{args.obsdate}_{args.stars}.p")
    if args.masks:
       result_path = os.path.join(args.local_path, f"parameters_{args.obsdate}_{args.stars}_wmask.p")
    else:
       result_path = os.path.join(args.local_path, f"parameters_{args.obsdate}_{args.stars}.p")

    # try to open info dict
    try:
        info_dict = pickle.load(open(dict_path, 'rb'))
    except:
        print("warning, no info dict found!")
        raise FileNotFoundError

    if args.stars == 'bright':
        exp_bins = [5, 15, 30, 60, 'acc']
    elif args.stars == 'faint':
        exp_bins = [30, 60]
    # initialize result dict as nested dictionary, with exposure lengths as outermost key
    result_dict = {exps:{} for exps in exp_bins}

    for s in list(info_dict.keys()):
        if args.masks:
            try: mask = info_dict[s]['mask']
            except KeyError: mask = False
        else: mask = False

        print(s)
        data = ExtractParameters(info_dict[s]['fits'], mask)
        out = data.extract_parameters(bins=exp_bins)
        if out: 
            # save results into result_dict 
            for exps in exp_bins:
                result_dict[exps][s] = out[exps]

    try:
        pickle.dump(result_dict, open(result_path, 'wb'))
    # if there's an error, try saving with some random numbers appended 
    # this should probably be only with a specific kind of error
    except:
        pickle.dump(result_dict, open(result_path[:-2] + f'_{np.random.uniform():4f}.p', 'wb'))

