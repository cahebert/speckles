import numpy as np
from astropy.io import fits
import pickle
import errno
import os
import measurement_helper as mhelp

class ExtractParameters():
    def __init__(self, fits_path, mask, source='data'):
        '''load the data into the class'''
        if 'r' in fits_path.split('/')[-1]:
            self.color = 'r'
        elif 'b' in fits_path.split('/')[-1]:
            self.color = 'b'

        self.source = source
        self.mask = mask

        try:
            hdu = fits.open(fits_path)
        except FileNotFoundError:
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), str(fits_path))

        if self.color == 'b': # this is the 562 wavelength
            self.imgs = hdu[0].data.astype('float')
        elif self.color == 'r': # this is the 832 wavelength
            self.imgs = hdu[0].data.astype('float')[:,:,::-1]
        hdu.close()

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

        ### not taking care of masks at all: this assumes that once we bin exposures we don't need to worry.

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

        ## Also not taking care of changing masks yet!!

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

# potentially useful old code to change index of masked exposures (but also wrong cause assumes only one mask?)
    # # make a new mask dict to reflect index changes of accumulating exposures
    # if maskDict:
    #     expid = [i for i in maskDict.keys()][0]
    #     # if accumulating exposures
    #     if numBins is None:
    #         if indices is not None:
    #             maskedExps = np.arange(15)[np.array(indices) >= expid]
    #             newMaskDict = {k: maskDict[expid] for k in maskedExps}
    #         else:
    #             maskedExps = [i for i in np.arange(1000) if i >= expid]
    #             newMaskDict = {k: maskDict[expid] for k in maskedExps}

    #     # if binning exposures
    #     else:
    #         maskedExps = [i for i in np.arange(numBins)*(1000/numBins) if i >= expid]
    #         newMaskDict = {k: maskDict[expid] for k in maskedExps}
    # else:
    #     newMaskDict = None


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()

    parser.add_argument("--data_folder", type=str, default='MayHRStars/', 
                        help="path to desired data directory within Zorro (and name for info dict")
    parser.add_argument("--local_path", type=str, 
                        default='/Users/clairealice/Documents/research/speckles/intermediates/', 
                        help="path to local directory for saving output")
    parser.add_argument('-zorro', type=str, default='/Volumes/My Passport/Zorro/', 
                        help="path to main Zorro data directory")
    parser.add_argument('-bins', default=[5, 15, 30, 60, 'acc'])
    args = parser.parse_args()
    
    data_path = os.path.join(args.zorro, args.data_folder) 
    dict_path = os.path.join(args.local_path, 
                             f"accepted_info_{args.data_folder.strip('/')}.p")
    result_path = os.path.join(args.local_path, 
                               f"parameters_{args.data_folder.strip('/')}.p")

    # try to open info dict
    try:
        info_dict = pickle.load(open(dict_path, 'rb'))
    except:
        print("warning, no info dict found!")
        raise FileNotFoundError

    # list all the files: these are the keys of the info dict
    data_files = list(info_dict.keys())

    ### maybe only take the files that have both filters?? probably should take care of this somewhere else!

    # initialize result dict as nested dictionary, with exposure lengths as outermost key
    result_dict = {exps:{} for exps in args.bins}

    for f in data_files[:5]:
        try:
            mask = info_dict[f]['mask']
        except KeyError:
            mask = False

        print(f)
        data = ExtractParameters(os.path.join(data_path, f), mask)
        out = data.extract_parameters(bins=args.bins)
        if out: 
            # save results into result_dict 
            for exps in args.bins:
                result_dict[exps][f] = out[exps]

    try:
        pickle.dump(result_dict, open(result_path, 'wb'))
    # if there's an error, try saving with some random numbers appended 
    # this should probably be only with a specific kind of error
    except:
        pickle.dump(result_dict, open(result_path[:-2] + f'_{np.random.uniform():4f}.p', 'wb'))

