import numpy as np
from astropy.io import fits
import pickle
import errno
import os
import filter_helper as fhelp
import measurement_helper as mhelp

class DataFilter():
    def __init__(fits_path, source='data'):
        '''load the data into the class'''
        try:
            hdu = fits.open(fits_path)
        except FileNotFoundError:
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), filename)
        
        self.source = source
        self.fn = fits_path.split('/')[-1]
        self.info = {}

        self.imgs = hdu[0].data.astype('float')
        self.header = hdu[0].header
        hdu.close()

    def check_header():
        '''make sure that airmass<1.3 and the star matches the correct category'''
        if self.source == 'data'
            airmass_check = self.header['AIRMASS'] > 1.3
            try:
                object_check = self.header['COMMENT'][2] == 'HR'
            except IndexError:
                object_check = self.header['OBJECT'] == 'HR'

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

    def check_center_size():
        '''does the dataset pass the size+center cut?'''
        centroids = mhelp.calculate_centroids(self.imgs, subtract=True)
        self.size = mhelp.single_exposure_HSM(np.mean(self.imgs, axis=0)).moments_sigma

        comR = np.sqrt((entroids['x']-128)**2 + (centroids['y']-128)**2).mean()

        if comR + self.size * 2.355 > 128:
            return False
        else:
            self.info['centroids'] = centroids
            return True

    def check_flux():
        '''check whether the PSF leaves the postage stamp'''
        if self.source == 'data':
            fluxes = self.imgs.sum(axis=(1,2))
            ## IN DEV ##
            return True
        else:
            return True # sim will never wander off frame

    def find_cr():
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
        
    def filter_data(info_dict={}):
        '''run all the data checks. if dataset passes, save self.info'''
        # run header check
        if self.check_header():
            # run center/size cut
            if self.check_center_size():
                # run flux check
                if self.check_flux():
                    # find cosmic rays, adds any found to self.info dict
                    self.find_cr()    

                    # if all checks pass, add dataset info to info dict
                    info_dict[self.fn] = self.info

                    return info_dict
                else: return False
            else: return False
        else: return False

