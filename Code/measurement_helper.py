import numpy as np
import galsim
import pickle
from scipy.optimize import curve_fit
import sklearn.utils
import errno
import os
from astropy.io import fits
import lmfit

def data_loader(fits_path, source):
    '''load fits and extract image and header data'''
    imgs = np.zeros((len(fits_path)*1000, 256, 256))
    header = []
    for i, f in enumerate(fits_path):
        try:
            hdu = fits.open(f)
        except FileNotFoundError:
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), str(f))
        data = hdu[0].data.astype('float')

        if data.shape[1] != 256 or data.shape[2] != 256:
            return False

        if 'r' in f: # this is the 832 wavelength
            data = data[:,:,::-1]
        
        if source=='data':
            # convert to electron counts
            head = hdu[0].header
            gain = head['EMGAIN'] / head['PREAMP']
            if gain != 0: data /= gain 
        else:
            data *= 1e7
        imgs[i*1000:(i+1)*1000] = data
        header.append(hdu[0].header)
        hdu.close()

    return imgs, header

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
    try:
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
    except IndexError:
        return False
        
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
            img[exp_mask_dict[i][:,0],exp_mask_dict[i][:,1]] = 0

        side_slices[0] = img[:mask_size, :]
        side_slices[1] = img[nx - mask_size:nx, :]
        side_slices[2] = img[:, :mask_size].T
        side_slices[3] = img[:, ny - mask_size:ny].T

        min_var = np.argmin([side_slices[j][side_slices[j]!=0].flatten().var() for j in range(4)])
        img_series[i] -= np.mean(side_slices[min_var][side_slices[min_var]!=0])


def single_exposure_HSM(img, exp_mask=False, subtract=True, max_iters=400,
                      max_ashift=75, max_amoment=5.0e5, strict=True):
    if subtract:
        img = np.copy(img)
        exp_mask_dict = {0: exp_mask} if exp_mask else False
        subtract_background([img], exp_mask_dict=exp_mask_dict)
    
    # guesstimate center and size of PSF as start for HSM
    comx, comy = image_com(img)
    fwhm = image_fwhm(img, comx, comy)
    if not fwhm: guestimateSig = 50
    else: guestimateSig = fwhm / 2.355

    badPix = galsim.Image(1 - exp_mask, xmin=0, ymin=0) if exp_mask else None

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


def calculate_centroids(imgs, step=5, subtract=False):
    '''calculate the centroids of the image series using HSM
    ::N:: sum N images together to calculate the centroid on'''
    N = int(imgs.shape[0]/step)
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
            exp_mask = None
        except KeyError:
            # no mask for exposure i => set to None.
            exp_mask = None

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
    
#########

def zorro_psf_model(p, scale=.01, noise=False, profile='Kolmogorov'):
    '''
    params should include:
    - PSF flux
    - size (fwhm) !!units: arcsec
    - x,y centroid offset from center of image !!units: pixels
    - g1,g2 shear 
    - background level
    '''
    if profile=='Kolmogorov':
        gauss = galsim.Kolmogorov(flux=p['flux'], fwhm=p['fwhm'])
    elif profile=='Gaussian':
        gauss = galsim.Gaussian(flux=p['flux'], fwhm=p['fwhm'])
    shear_gauss = gauss.shear(g1=p['g1'], g2=p['g2'])
    shift_shear_gauss = shear_gauss.shift(0,0)
        
    img = shift_shear_gauss.drawImage(nx=256, ny=256, 
                                      scale=scale, 
                                      offset=(p['x'],p['y']), 
                                      use_true_center=False,
                                      dtype=np.float64)
    
    if noise: img.addNoise(galsim.PoissonNoise(sky_level=p['background']))
    
    return p['background'] + img.array

def fit_profile_moments(data, scale, profile='Kolmogorov', subtract=False, exp_mask=None):
    if subtract:
        data = np.copy(data)
        exp_mask_dict = {0: exp_mask} if exp_mask else False
        subtract_background([data], exp_mask_dict=exp_mask_dict)
        
    badPix = 1 - exp_mask if exp_mask else np.zeros(data.shape)

    def model_residual(p, data, **args):
        resid = data - zorro_psf_model(p, args['scale'], profile=profile)
        out = resid / np.sqrt(data)
        out[data==0] = 0
        out[badPix==1] = 0
        return out.flatten()

    def construct_params(fwhm=.5, x=0, y=0):
        p = lmfit.parameter.Parameters()
        pdict = {'flux': lmfit.parameter.Parameter(name='flux', value=np.sum(data), min=np.sum(data)*1e-1, max=np.sum(data)*1e2),
                 'fwhm': lmfit.parameter.Parameter(name='fwhm', value=fwhm, min=.1, max=2.5), 
                 'x': lmfit.parameter.Parameter(name='x', value=x, min=-100, max=100), 
                 'y': lmfit.parameter.Parameter(name='y', value=y, min=-100, max=100), 
                 'g1': lmfit.parameter.Parameter(name='g1', value=0, min=-.5, max=.5), 
                 'g2': lmfit.parameter.Parameter(name='g2', value=0, min=-.5, max=.5),
                 'background': lmfit.parameter.Parameter(name='background', value=0, min=-100, max=np.min(data)+100)}
        for k,v in pdict.items():
            p[k] = v
        return p

    params = construct_params(fwhm=.5)
    lmout = lmfit.minimize(model_residual, params, args=[data], kws={'scale':scale})
    
    return lmout


def param_fit(images, scale, exp_mask_dict=False, save_dict={'save':True, 'path':None}, subtract=False):
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
            exp_mask = None
        except KeyError:
            # no mask for exposure i => set to None.
            exp_mask = None

        # fit the image(s)
        mom_result = fit_profile_moments(images[i], scale, exp_mask=exp_mask, subtract=subtract)

        if mom_result.message != 'Fit succeeded.': 
            print("fit did not succeed?!")
            return False
        # put results in a list
        fitResults.append(mom_result)
        
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
