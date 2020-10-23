import numpy as np
from astropy.io import fits
import pickle
import os


def parse_log_file(obsdate, star_type='both', zpath='/Volumes/My Passport/Zorro/'):
    '''open zorro log file and extract information about the stars.
    Returns a dict of either bright or faint stars or both as indicated by user.
    Faint stars returned only if they have >3*60s of observing time'''
    zpath = zpath + obsdate + '/'
    # list of files in directory to cross reference when making list of file names
    dir_files = [f.split('.')[0] for f in os.listdir(zpath) if os.path.isfile(os.path.join(zpath, f)) and '.bz2' in f]

    file_object  = open(zpath + obsdate + 'zorrolog', "r") 
    contents = file_object.read()

    lines = [c for c in contents.split('\n')]
    split_lines = [[l.strip(' ') for l in h.split('\t')] for h in lines]

    # split lines are, in order:
    # 'star', 'b number', 'r number', 'time', 'b gain', 'r gain', 'coord1', 'coord2', 'obsspeed', 'campaign']
    obj_list = set([s[0] for s in split_lines if s[0]!=''])

    obj_dict = {}

    for obj in obj_list:
        obj_dict[obj + 'b'] = {'fn': [[df for df in dir_files if s[1] in df][0] for s in split_lines if s[0] == obj], 
                               'gain': [s[4] for s in split_lines if s[0] == obj][0],
                               'times': [s[3] for s in split_lines if s[0] == obj],
                               'coords': [s[6] for s in split_lines if s[0] == obj][0],
                               'campaign': [s[-1] for s in split_lines if s[0] == obj][0]}


        obj_dict[obj + 'r'] = {'fn': [[df for df in dir_files if s[2] in df][0] for s in split_lines if s[0] == obj], 
                               'gain': [s[5] for s in split_lines if s[0] == obj][0],
                               'times': [s[3] for s in split_lines if s[0] == obj],
                               'coords': [s[7] for s in split_lines if s[0] == obj][0],
                               'campaign': [s[-1] for s in split_lines if s[0] == obj][0]}

        # {obj: {'fn': [[df for df in dir_files if s[1] in df][0] for s in split_lines if s[0] == obj], 
        #               'gain': [s[4:6] for s in split_lines if s[0] == obj][0],
        #               'times': [s[3] for s in split_lines if s[0] == obj],
        #               'coords': [s[6:8] for s in split_lines if s[0] == obj][0],
        #               'campaign': [s[-1] for s in split_lines if s[0] == obj][0]} for obj in obj_list}

    if star_type == 'bright':
        return {k:d for k,d in obj_dict.items() if 'HR' in k}
    elif star_type == 'faint':
        return {k:d for k,d in obj_dict.items() if 'HR' not in k if len(d['fn'])>3}

def match_filter_pairs(info_dict):
    '''go through the info dict and return version with datasets which have *both* filters accepted'''
    # only include in final save if both filters are accepted!!
    good_r_files = [k for k in info_dict.keys() if 'r' in k]
    good_b_files = [k for k in info_dict.keys() if 'b' in k]
    # all r data that also passed in b filter
    overlap_r_files = [f for f in good_r_files if f.replace('r','b') in good_b_files]
    # all b data that also passed in r filter
    overlap_b_files = [f for f in good_b_files if f.replace('b','r') in good_r_files]
    # want to keep the union of these two above lists
    overlap = overlap_r_files + overlap_b_files

    return {f:info_dict[f] for f in overlap}

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
