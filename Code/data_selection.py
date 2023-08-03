import numpy as np
from astropy.io import fits
import pickle
import errno
import os
import filter_helper as fhelp
import measurement_helper as mhelp

class DataFilter():
    def __init__(self, fits_path_list, star_type='bright', source='data'):
        '''load the data into the class'''
        # fits_path should be a list of paths (length 1 if bright stars)
        self.source = source
        self.info = {'fits': fits_path_list}
        self.star = star_type

        loaded = mhelp.data_loader(fits_path_list, source)
        if loaded:
            self.imgs, self.header = loaded
        else: 
            raise ValueError('image size was not 256x256')

    def check_header(self):
        '''make sure that airmass<1.3 and the star matches the correct category'''
        if self.source == 'data':
            airmass_check = np.array([h['AIRMASS'] < 1.3 for h in self.header]).all()
            if self.star == 'bright':
                try:
                    object_check = 'HR' in self.header[0]['OBJECT']
                except IndexError:
                    object_check = 'HR' in self.header[0]['COMMENT'][2]
            elif self.star == 'faint':
                object_check = np.array([h['OBJECT']==self.header[0]['OBJECT'] for h in self.header]).all()

            if not object_check or not airmass_check:
                return False
            else:
                # if check passes, add some header quantities to self.info dict
                # for faint stars, just take the info from the first file
                self.info['header'] = {}
                for k in ['TARGRA', 'TARGDEC', 'AIRMASS', 'DATE-OBS', 'OBSTIME']:
                    self.info['header'][k] = self.header[0][k]
                return True
        else: 
            return True # sim should always pass this!

    def check_center_size(self):
        '''does the dataset pass the size+center cut?'''
        centroids = mhelp.calculate_centroids(self.imgs, subtract=True)
        # if this calculation fails, this test fails. 
        if not centroids: return False

        self.size = mhelp.single_exposure_HSM(np.mean(self.imgs, axis=0)).moments_sigma

        comR = np.sqrt((centroids['x']-128)**2 + (centroids['y']-128)**2).mean()

        center_size_check = comR + self.size * 2.355 > 128
        if center_size_check:
            return False
        else:
            self.info['centroids'] = centroids
            return True

    # def check_wander(self):
    #     '''check whether the PSF leaves the postage stamp'''
    #     if self.source == 'data':
    #         return True
    #         # fluxes = self.imgs[1:].sum(axis=(1,2))
    #         # threshold = np.median(fluxes)*.8
    #         # wander_check = sum(fluxes<threshold) > 5 # false if any fluxes are below threshhold
    #         # return wander_check
    #     else:
    #         return True # sim will never wander off frame

    def find_cr(self):
        '''find cosmic rays and save them to a mask dictionary'''
        if self.source == 'data':  # sims won't have CRs!
            # loop over all frames
            for f_index, frame in enumerate(self.imgs):
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
            ## run flux check
            # if self.check_wander():
            # run center/size cut
            if self.check_center_size():
                if cr_scan: 
                    # find cosmic rays, adds any found to self.info dict
                    self.find_cr()    
                return True # if all checks pass, return True
            else:
                print('I did not pass centroid cut!')
                return False
            # else:  
            #     return False
        else: return False


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
    dict_path = os.path.join(args.local_path, 
                        f"accepted_info_{args.obsdate}_{args.stars}.p")

    # try to open info dict
    try:
        info_dict = pickle.load(open(dict_path, 'rb'))
    # if it doesn't exist, then intialize
    except:
        info_dict = {}

    # parse the log file and return a dict of some info
    files_dict = fhelp.parse_log_file(args.obsdate, args.stars, args.zorro)
    for star in files_dict.keys():
        # file names root 
        files = list(files_dict[star]['fn'])
        print(star, files)

        # only want one file for bright stars! so if more, just choose one randomly
        if args.stars == 'bright':
            if len(files)>1:
                files = [np.random.choice(files)]

        # make list of paths for these files
        fits_paths = [os.path.join(data_path, f + '.fits.bz2') for f in files]
        try:
            data = DataFilter(fits_paths, star_type=args.stars)
        except ValueError: # means there was a problem with the file/data wasn't the right size
            continue

        if data.filter_data(args.masks):
            info_dict[star] = data.info

    info_dict = fhelp.match_filter_pairs(info_dict)
    
    try:
        pickle.dump(info_dict, open(dict_path, 'wb'))
    # if there's an error, try saving with some random numbers appended 
    # this should probably be only with a specific kind of error
    except:
        pickle.dump(info_dict, open(dict_path[:-2] + f'_{np.random.uniform():4f}.p', 'wb'))

