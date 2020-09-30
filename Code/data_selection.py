import numpy as np
from astropy.io import fits
import pickle
import errno
import os
import filter_helper as fhelp
import measurement_helper as mhelp

class DataFilter():
    def __init__(self, fits_path, source='data'):
        '''load the data into the class'''
        try:
            hdu = fits.open(fits_path)
        except FileNotFoundError:
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), str(fits_path))
        
        self.source = source
        self.info = {}

        self.imgs = hdu[0].data.astype('float')
        self.header = hdu[0].header
        hdu.close()

    def check_header(self):
        '''make sure that airmass<1.3 and the star matches the correct category'''
        if self.source == 'data':
            airmass_check = self.header['AIRMASS'] < 1.3
            try:
                object_check = 'HR' in self.header['OBJECT']
            except IndexError:
                object_check = 'HR' in self.header['COMMENT'][2]

            if not object_check or not airmass_check:
                return False
            else:
                # if check passes, add some header quantities to self.info dict
                self.info['header'] = {}
                for k in ['TARGRA', 'TARGDEC', 'AIRMASS', 'DATE-OBS', 'OBSTIME']:
                    self.info['header'][k] = self.header[k]
                return True
        else: 
            return True # sim should always pass this!

    def check_center_size(self):
        '''does the dataset pass the size+center cut?'''
        centroids = mhelp.calculate_centroids(self.imgs, subtract=True)
        self.size = mhelp.single_exposure_HSM(np.mean(self.imgs, axis=0)).moments_sigma

        comR = np.sqrt((centroids['x']-128)**2 + (centroids['y']-128)**2).mean()

        center_size_check = comR + self.size * 2.355 > 128
        if center_size_check:
            return False
        else:
            self.info['centroids'] = centroids
            return True

    def check_wander(self):
        '''check whether the PSF leaves the postage stamp'''
        if self.source == 'data':
            fluxes = imgs[1:].sum(axis=(1,2))
            threshold = np.median(fluxes)*.85
            wander_check = (fluxes>threshold).all(): # false if any fluxes are below threshhold
            return wander_check
        else:
            return True # sim will never wander off frame

    def find_cr(self):
        '''find cosmic rays and save them to a mask dictionary'''
        if self.source == 'data':  # sims won't have CRs!
            # loop over all frames
            for f_index, frame in zip(range(1000), self.imgs):
                frame_out = fhelp.scan_image_for_cr(frame)
                
                if frame_out: # would be False if there were no gaps in histogram
                    threshold, mask_ids = frame_out
                    # if mask IDs have a nonzero the first axis, consider that a mask
                    if mask_ids.shape[0]:
                        try: self.info['mask'][f_index] = mask_ids
                        except KeyError: self.info['mask'] = {f_index: mask_ids}
        else: return

    def filter_data(self, cr_scan=False):
        '''run all the data checks. if dataset passes, save self.info'''
        # run header check
        if self.check_header():
            # run center/size cut
            if self.check_center_size():
                # run flux check
                if self.check_wander():
                    if cr_scan: 
                        # find cosmic rays, adds any found to self.info dict
                        self.find_cr()    

                    # if all checks pass, return True
                    return True
                else: 
                    print('I think I wandered off the frame!')
                    return False
            else: return False
        else: return False

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

if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()

    parser.add_argument("--data_folder", type=str, default='MayHRStars/', 
                        help="path to desired data directory within Zorro")
    parser.add_argument("--local_path", type=str, 
                        default='/Users/clairealice/Documents/research/speckles/intermediates/', 
                        help="path to local directory for saving output")
    parser.add_argument('-zorro', type=str, default='/Volumes/My Passport/Zorro/', 
                        help="path to main Zorro data directory")
    parser.add_argument('-masks', default=False, action='store_true')
    args = parser.parse_args()
    
    data_path = os.path.join(args.zorro, args.data_folder) 
    dict_path = os.path.join(args.local_path, 
                        f"accepted_info_{args.data_folder.strip('/')}.p")

    # try to open info dict
    try:
        info_dict = pickle.load(open(dict_path, 'rb'))
    # if it doesn't exist, then intialize
    except:
        info_dict = {}

    # list all the files in the data folder to look through
    data_files = [f.split('.')[0] for f in os.listdir(data_path) if os.path.isfile(os.path.join(data_path, f)) and '.bz2' in f]

    for f in data_files:
        print(f)
        data = DataFilter(os.path.join(data_path, f + '.fits.bz2'))

        if data.filter_data(args.masks):
            info_dict[f] = data.info

    info_dict = match_filter_pairs(info_dict)
    
    try:
        pickle.dump(info_dict, open(dict_path, 'wb'))
    # if there's an error, try saving with some random numbers appended 
    # this should probably be only with a specific kind of error
    except:
        pickle.dump(info_dict, open(dict_path[:-2] + f'_{np.random.uniform():4f}.p', 'wb'))

