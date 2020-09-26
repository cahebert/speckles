import numpy as np
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

def flux_test(imgs):
    '''
    check if PSF is wandering off frame using evolution of total flux across series of exposures
    '''
    fluxes = imgs.sum(axis=(1,2))

    
    ## TO DO

    return True
