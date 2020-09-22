import numpy as np
import galsim
import pickle
from scipy.optimize import curve_fit
import sklearn.utils

def image_com(image, exp_mask=False):
    '''
    Find the center of mass of given image.
    Inputs: ::image:: and optionally ::exp_mask::, exposure mask for the image
    Returns tuple of ints corrsponding to the x, y indices of the center of mass
    '''
    img = np.copy(image)
    if exp_mask:
        if abs(img.min()) > 500:
            subtract_background([img], exp_mask_dict={0:exp_mask})
        # give 0 weight to masked pixels
        img *= exp_mask
    elif abs(img.min()) > 500:
        subtract_background([img])
    indices = np.linspace(0, img.shape[0] - 1, img.shape[0])
    comX = np.dot(img.sum(axis=1), indices) / img.sum()
    comY = np.dot(img.sum(axis=0), indices) / img.sum()
    return int(comX), int(comY)

def image_fwhm(img, x, y):
    '''
    Given an image and coordinates x,y (e.g. CoM coordinates), find
    the approximate FWHM of the distribution. 
    Returns: FWHM of image, averaged between x and y slices.
    '''
    img = np.copy(img)
    if abs(img.min())>500:
        subtract_background([img])

    fwhm = 0
    for coord_j, slice_i in zip([y,x], [img[x,:], img[:,y]]):
        # find what's above/below the half max
        hm_rel = slice_i - slice_i.max()/2

        # find where we change from above to below
        dCoord = np.sign(hm_rel[:-1]) - np.sign(hm_rel[1:])
        right = np.where(dCoord>0)[0]
        left = np.where(dCoord<0)[0]

        # check for the derivative actually working
        if len(right) == 0 and len(left) == 0:
            print('something is fishy! check image subtraction')
        if len(right) == 0:
            if len(left) != 0:
                # if right side doesn't cross zero, make fwhm twice the left to COM distance
                fwhm += 2 * abs(left[0] - coord_j)
        elif len(left) == 0:
            if len(right) != 0:
                # if left side doesn't cross zero, make fwhm twice the right to COM distance
                fwhm += 2 * abs(right[-1] - coord_j)
        else:
            # if neither side is zero, difference is FWHM
            fwhm += abs(right[-1] - left[0])
        
    # return averaged FWHM
    return fwhm / 2

def subtract_background(img_series, accumulated=False, exp_mask_dict=False):
    '''
    Subtract a flat background from a given series of images. Uses slices from the four sides of an image 
    and uses the mean of whichever has the smallest variance 
    NOTES: 
    * Modifies the input series img_series in place.
    * no currently implemented way to handle masks differently for accumulated exposures
    Input:
    :img_series: list of images to be background subtracted
    :accumulated: optional, whether the images are single exposures
    :exp_mask_dict: optional, dictionary of exposures with associated pixel masks
    '''
    if exp_mask_dict:
        flags = list(exp_mask_dict.keys())
        # if accumulated: 
            # something should happen here, like flags are all exposures and mask is combined?
            # but that doesn't work in the scenario that there are a ton of masked pixels in the center of PSF
    else:
        flags = []
        
    for i in range(len(img_series)):
        img = np.copy(img_series[i])
        nx, ny = img.shape
        if nx <= 15:
            mask_size = 1
        else:
            mask_size = 10
        
        side_slices = np.zeros((4, mask_size, nx))
        if i in flags: 
            img = img * exp_mask_dict[i]

        side_slices[0] = img[:mask_size, :]
        side_slices[1] = img[nx - mask_size:nx, :]
        side_slices[2] = img[:, :mask_size].T
        side_slices[3] = img[:, ny - mask_size:ny].T

        min_var = np.argmin([side_slices[j][side_slices[j]!=0].flatten().var() for j in range(4)])
        img_series[i] -= np.mean(side_slices[min_var][side_slices[min_var]!=0])

        # if nx <= 15:
        #     mask_size = 1
        # else:
        #     mask_size = 10
        # sideSlices = np.zeros((4, mask_size, nx))
        # if flag is not None and accumulated or i == flag:
            # maskedExposure = img * exp_mask
            # sideSlices[0] = maskedExposure[:mask_size, :]
            # sideSlices[1] = maskedExposure[nx - mask_size:nx, :]
            # sideSlices[2] = maskedExposure[:, :mask_size].T
            # sideSlices[3] = maskedExposure[:, ny - mask_size:ny].T
            #don't include masked pixels in the variance!
        #     minVariance = np.argmin([sideSlices[j][sideSlices[j]!=0].flatten().var() for j in range(4)])
                            
        # else:
        #     sideSlices[0] = img[:mask_size, :]
        #     sideSlices[1] = img[nx - mask_size:nx, :]
        #     sideSlices[2] = img[:, :mask_size].T
        #     sideSlices[3] = img[:, ny - mask_size:ny].T
            
        # minVariance = np.argmin(sideSlices.var(axis=(1,2)))
        # img -= np.mean(sideSlices[minVariance][sideSlices[minVariance]!=0])

def single_exposure_HSM(img, exp_mask=False, subtract=True, max_iters=400,
                      max_ashift=75, max_amoment=5.0e5, strict=True):
    if subtract:
        img = np.copy(img)
        exp_mask_dict = {0: exp_mask} if exp_mask else False
        subtract_background([img], exp_mask_dict=exp_mask_dict)
    
    # guesstimate center and size of PSF as start for HSM
    comx, comy = image_com(img)
    fwhm = image_fwhm(img, comx, comy)
    guestimateSig = fwhm / 2.355

    badPix = galsim.Image(1 - exp_mask, xmin=0, ymin=0) if exp_mask else False

    # make GalSim image of the exposure
    new_params = galsim.hsm.HSMParams(max_amoment=max_amoment, 
                                      max_ashift=max_ashift,
                                      max_mom2_iter=max_iters)
    galImage = galsim.Image(img, xmin=0, ymin=0)

    # run HSM adaptive moments with initial sigma guess
    try:
        hsmOut = galsim.hsm.FindAdaptiveMom(galImage, 
                                            hsmparams=new_params,
                                            badpix=badPix,
                                            guess_sig=guestimateSig,
                                            guess_centroid=galsim.PositionD(comx, comy),
                                            strict=strict)
    except RuntimeError:
        return False
        
    # return HSM output
    return hsmOut


def calculate_centroids(imgs, N=200, subtract=False):
    '''calculate the centroids of the image series using HSM
    ::N:: sum 1000/N images together to calculate the centroid on'''
    step = int(1000/N)
    centroids = np.zeros((2, N))
    
    for i in range(N):
        fit = single_exposure_HSM(img=np.sum(imgs[i*step:(i+1)*step], axis=0)/step, 
                                  subtract=subtract, max_iters=25000, 
                                  max_ashift=200, max_amoment=1e7)
        if fit == False: return False
        centroids[:,i] = [fit.moments_centroid.x, fit.moments_centroid.y]
    
    return {'x': centroids[0], 'y': centroids[1]}

def estimate_moments_HSM(images, exp_mask_dict=False, save_dict={'save':True, 'path':None}, 
                         strict=False, subtract=False, max_iters=800, 
                         max_ashift=100, max_amoment=5.0e5):
    '''
    Estimate the moments of the PSF images using HSM.
    TO DO:
    - fit third moment of PSF as well?
    '''
    # length of sequence to fit
    N = len(images)

    fitResults = []

    for i in range(N):
        # try to assign exp_mask to the entry in exp_mask_dict
        try:
            exp_mask = exp_mask_dict[i]
        except TypeError:
            # no exp_mask_dict object => this dataset has no masks.
            exp_mask = False
        except KeyError:
            # no mask for exposure i => set to False.
            exp_mask = False

        # fit the image(s)
        hsmOut = single_exposure_HSM(images[i], exp_mask=exp_mask, max_iters=max_iters, 
                                     subtract=subtract, strict=strict, 
                                     max_ashift=max_ashift, max_amoment=max_amoment)

        if hsmOut == False: return False
        # put results in a list
        fitResults.append(hsmOut)
        
    if save_dict['save'] == True:
        try: 
            # Save HSM output list to a pickle file.
            with open(save_dict['path'], 'wb') as file:
                pickle.dump(fitResults, file)
        except FileNotFoundError:
            print('Save file not found. Try adding/checking path in save_dict')
        return True
    else:
        return fitResults
    
