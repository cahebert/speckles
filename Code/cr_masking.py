import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import pickle

def find_big_gap(n, bins, sigma, i=0):
    '''
    Find the first gap in a histogram (defined by bins and n) which is 
    larger than 2*sigma
    Returns False if the start index is not a gap, otherwise returns
    the indices of the beginning and end of gap 
    '''
    # i is the current index in n 
    if n[i] == 0:
        for j in range(i, len(n)):
            if sum(n[i:j]) == 0:
                if bins[j] - bins[i] > 3*sigma:
                    return i,j
    return False

def scan_image_for_cr(frame):
    '''
    This function implements a cosmic ray cleaning algorithm from Wojtek 
    Pych (arXiv:astro-ph/0311290)
    Given a single image, outputs IDs of affected pixels
    '''
    # bins for histogram go from 0 to saturation limit
    hist_bins = np.linspace(0,65535, 100)
    pixels = frame.flatten()

    n, bins = np.histogram(pixels, bins=hist_bins)
    pix = pixels[pixels<np.mean(pixels)+5*np.std(pixels)]
    sigma = np.std(pix)

    gaps = [j for j in range(len(n)) if n[j]==0]
    if len(gaps):
        # find the first "big" gap with this while loop
        j = 0
        while not find_big_gap(n, bins, sigma, gaps[j]):
            j += 1
            if j == len(gaps):
                return False
        first_gap = find_big_gap(n, bins, sigma, gaps[j])[0]

        # now return IDs of the pixels above this first gap
        return bins[first_gap], np.argwhere(frame>bins[first_gap])
    else: #if there are no gaps in the histogram
        return False

def make_cr_mask(path, fn, mask_dict):
    '''
    run cosmic ray finding on all frames of a dataset. if one is found, add
    the resulting mask as an entry to mask_dict
    '''
    # load in a dataset
    hdu = fits.open(path + fn)
    imgs = hdu[0].data.astype('float')

    # loop over all frames
    for f_index, frame in zip(range(1000), imgs):
        
        frame_out = scan_image_for_cr(frame)
        
        if frame_out:
            threshold, mask_ids = frame_out

            # if mask IDs have anything listed in the first axis, consider 
            # that we do have a mask that we should save to dict
            if mask_ids.shape[0]:
                try:
                    mask_dict[fn][f_index] = mask_ids
                except KeyError:
                    mask_dict[fn] = {f_index: mask_ids}
    return mask_dict

# this part should not be permanent, should really be feeding in whatever list of stars
# are going to be run through a whole analysis (which ideally can be run with one command,
# at least eventually, to reduce any human-introduced errors!!)

if __name__ == '__main__':
    path = '/Volumes/My Passport/Zorro/'

    hrstars = np.loadtxt('./ZorroMayHRStars.txt', dtype='str')
    fns = [f'MayHRStars/S20{h}r.fits.bz2' for h in hrstars] + [f'MayHRStars/S20{h}b.fits.bz2' for h in hrstars]   
    # fn = '20190522/S20190522Z0648r.fits.bz2'
    # fn = '20190522/S20190522Z0792r.fits.bz2'
    # fn = 'MayHRStars/S20190523Z1064r.fits.bz2'
    # fn = 'MayHRStars/S20190522Z0649r.fits.bz2'
    
    dictpath = '/Users/clairealice/Documents/research/speckles/intermediates/mask_3.p'
    
    # try to open the dict!
    try:
        mask_dict = pickle.load(open(dictpath, 'rb'))
    # if it doesn't exist, then intialize
    except:
        mask_dict = {}

    for fn in fns:
        mask_dict = make_cr_mask(path, fn, mask_dict)
        
    try:
        pickle.dump(mask_dict, open(dictpath, 'wb'))
    except:
        pickle.dump(mask_dict, open(dictpath[:-2] + f'_{np.random.uniform():4f}.p', 'wb'))


# need to think about where to define the dict and how to check for overwriting when saving
# and how much of the file number to include

