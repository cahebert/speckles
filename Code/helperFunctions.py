import numpy as np
from matplotlib import pyplot as plt
import galsim

def makeMask(image, maskSize, maskCenter=False):
    # define a mask
    nx, ny = np.shape(image)
    mask = np.zeros((nx, ny))
    
    # if desired, find the ~centroid
    if maskCenter is True:
        (x, y) = np.where(image == np.max(image))
        maskCenter = (x.mean() - nx / 2, y.mean() - ny / 2)
#         print(f'centroid is approximately at: {maskCenter}')
    else:
        maskCenter = (0, 0)
        
    # make sure centered mask is within the image limits
    if maskCenter[0] + maskSize[0] >= 0:
        xlEdge = int(maskCenter[0] + maskSize[0])
    else: 
        xlEdge = 0
    if nx - maskSize[0] + maskCenter[0] <= nx:
        xrEdge = int(nx - maskSize[0] + maskCenter[0])
    else: 
        xrEdge = nx   
    
    if maskCenter[1] + maskSize[1] >= 0:
        ylEdge = int(maskCenter[1] + maskSize[1])
    else: 
        ylEdge = 0
    if nx - maskSize[1] + maskCenter[1] <= ny:
        yuEdge = int(ny - maskSize[1] + maskCenter[1])
    else: 
        yuEdge = ny

    # define the mask
    mask[:xlEdge, :] = 1
    mask[xrEdge:nx, :] = 1
    mask[:, :ylEdge] = 1
    mask[:, yuEdge:ny] = 1
    
    return mask
    
def demonstrateCenteringMask(image, maskSize, plot=False, pScale=.01, nExp=1):
    
    # define a mask
    mask = makeMask(image, maskSize)
    if plot:
        plt.imshow(mask*image)
        plt.title('mask')
        plt.show()

    # apply to image, find the std of the background 
    back = (mask*image).ravel()
    back = back[back!=0]    
    std = np.std(back)
    
    # do a 2 sigma clipping on the distribution of background pixels
    twoSigma = np.mean(back) + 2 * std
    clippedStd = np.std(back[back <= twoSigma])
    
    if plot:
        print(f'standard deviation: {std:.2f}' +
              f'\n2 sigma clipped standard deviation: {clippedStd:.2f}')
        bins = np.linspace(int(min(back)), int(max(back)), int(max(back) - min(back)))
        plt.hist(back, bins=bins)
        plt.axvline(twoSigma, color='yellow')
        plt.ylabel('counts', fontsize=12)
        plt.xlabel('pixel intensity', fontsize=12)
        plt.show()
    
    # define centered mask
    centeredMask = makeMask(image, maskSize, maskCenter=True)
    if plot:
        plt.imshow(centeredMask*image)
        plt.title('centered mask')
        plt.show()    

    # apply to image, find std of centered background
    centeredBack = (centeredMask*image).ravel()
    centeredBack = centeredBack[centeredBack!=0]
    centeredStd = np.std(centeredBack)
    
    # do a 2 sigma clipping on the distribution of background pixels
    twoSigma = np.mean(centeredBack) + 2 * centeredStd
    clippedCenteredStd = np.std(centeredBack[centeredBack <= twoSigma])
    
    if plot:
        print(f'standard deviation, centered: {centeredStd:.2f}' +
              f'\n2 sigma clipped standard deviation, centered: {clippedCenteredStd:.2f}')
        bins = np.linspace(
            int(min(centeredBack)), int(max(centeredBack)), 
            int(max(centeredBack) - min(centeredBack)))
        plt.hist(centeredBack, bins=bins)
        plt.axvline(twoSigma, color='yellow')
        plt.ylabel('counts', fontsize=12)
        plt.xlabel('pixel intensity', fontsize=12)
        plt.show()

        
def estimateBackground(image, maskSize, maskCenter=True, plot=False, pScale=.01, nExp=1):
    '''
    Estimates the background of image.
    Parameters:
    ----------
    ...
    '''
    # define centered mask
    centeredMask = makeMask(image, maskSize, maskCenter=True)
    if plot:
        plt.imshow(centeredMask*image)
        plt.title('centered mask')
        plt.show()    

    # apply to image, find std of centered background
    centeredBack = (centeredMask*image).ravel()
    centeredBack = centeredBack[centeredBack!=0]
    centeredStd = np.std(centeredBack)
    
    # do a 2 sigma clipping on the distribution of background pixels
    twoSigma = np.mean(centeredBack) + 2 * centeredStd
    clippedCenteredStd = np.std(centeredBack[centeredBack <= twoSigma])
    
    if plot:
        print(f'standard deviation, centered: {centeredStd:.2f}' +
              f'\n2 sigma clipped standard deviation, centered: {clippedCenteredStd:.2f}')
        bins = np.linspace(
            int(min(centeredBack)), int(max(centeredBack)), 
            int(max(centeredBack) - min(centeredBack)))
        plt.hist(centeredBack, bins=bins)
        plt.axvline(twoSigma, color='yellow')
        plt.ylabel('counts', fontsize=12)
        plt.xlabel('pixel intensity', fontsize=12)
        plt.show()
    
    # return standard deviation of this clipped background
    return clippedCenteredStd

def imageCOM(image):
    '''
    Find the center of mass of given image.
    Returns:
    ========
    Tuple of ints corrsponding to the indices of the center of mass
    '''
    img = np.copy(image)
    subtractBackground([img])
    indices = np.linspace(0, img.shape[0]-1, img.shape[0])
    comX = np.dot(img.sum(axis=1), indices) / img.sum()
    comY = np.dot(img.sum(axis=0), indices) / img.sum()
    return int(np.rint(comX)), int(np.rint(comY))

def imageFWHM(img, x, y):
    '''
    Given an image and coordinates x,y (e.g. CoM coordinates), find
    the approximate FWHM of the distribution. To output the FWHM of 
    the PSF, input the CoM coordinates.
    Returns:
    ========
    FWHM of image, averaged between x and y slices.
    '''
    img = np.copy(img)
    subtractBackground([img])

    # take a slice through x
    sliceX = img[x,:]
    # find what's above/below the half max
    tempX = sliceX - sliceX.max()/2
    
    # find where we change from above to below
    dX = np.sign(tempX[:-1]) - np.sign(tempX[1:])
    right = np.where(dX>0)[0]
    left = np.where(dX<0)[0]
    
    # check for the derivative actually working
    if len(right) == 0:
        if len(left) != 0:
            # if right side doesn't cross zero, make fwhm twice the left to COM distance
            fwhmX = 2 * abs(left[0] - y)
    elif len(left) == 0:
        if len(right) != 0:
            # if left side doesn't cross zero, make fwhm twice the right to COM distance
            fwhmX = 2 * abs(right[-1] - y)
    else:
        # if neither side is zero, difference is FWHM
        fwhmX = abs(right[-1] - left[0])
    
    # repeat for y direction
    sliceY = img[:,y]
    tempY = sliceY - sliceY.max()/2
    
    dY = np.sign(tempY[:-1]) - np.sign(tempY[1:])
    right = np.where(dY>0)[0]
    left = np.where(dY<0)[0]
    if len(right) == 0:
        if len(left) != 0:
            # if right side doesn't cross zero, make fwhm twice the left to COM distance
            fwhmY = 2 * abs(left[0] - x)
    elif len(left) == 0:
        if len(right) != 0:
            # if left side doesn't cross zero, make fwhm twice the right to COM distance
            fwhmY = 2 * abs(right[-1] - x)
    else:
        # if neither side is zero, difference is FWHM
        fwhmY = abs(right[-1] - left[0])
    
    # return averaged FWHM
    return (fwhmX + fwhmY) / 2

def spatialBinToLSST(image, n=14):
    '''
    Given an image, return a spatially binned image size 14x14
    TO DO?
    - potentially have functionality for an image that isn't 256x256
    '''
    image = image[2:-2, 2:-2]
    newIm = np.empty((n, n))
    for i in range(n):
        for j in range(n):
            newIm[i, j] = image[18 * i:18 * (i + 1), 18 * j:18 * (j + 1)].sum()
    return newIm

def singleExposureHSM(img, maxIters=400, max_ashift=75, max_amoment=5.0e5):
    comx, comy = imageCOM(img)
    fwhm = imageFWHM(img, comx, comy)
    guestimateSig = fwhm / 2.355

    # make GalSim image of the exposure
    new_params = galsim.hsm.HSMParams(max_amoment=max_amoment, 
                                      max_ashift=max_ashift,
                                      max_mom2_iter=maxIters)
    galImage = galsim.Image(img, xmin=0, ymin=0)
    # run HSM adaptive moments with initial sigma guess
    speckleMoments = galsim.hsm.FindAdaptiveMom(galImage, hsmparams=new_params,
                                                guess_sig=guestimateSig,
                                                guess_centroid=galsim.PositionD(comx, comy))

    # tuple of results
    return speckleMoments.observed_shape.g1, speckleMoments.observed_shape.g2, speckleMoments.moments_sigma


def singleExposureKolmogorov(image, pScale, sBack, nExp):
    '''
    Given an image, find the best fit Kolmogorov profile parameters.
    Takes in:
    - pScale: pixel scale of the image
    - sBack: the background count fluctuation
    - nExp: the number of exposures (e.g. !=1 for accumulated exposures)
    Returns:
    - lmfit minimizer result
    '''
    # to avoid weird data type bug with lmfit:
    if image.dtype != '>f4':
        image = np.array(image, dtype='>f4')

    # Initialize Parameters object
    params = Parameters()
    params.add('p_scale', value=pScale, vary=False)  # pixel scale is fixed
    params.add('g1', value=.02, min=-.6, max=.6)
    params.add('g2', value=.02, min=-.6, max=.6)
    params.add('hlr', value=.4, min=0., max=.8)
    # find max and set as first centroid guess
    m = np.max(image)
    (y, x) = np.where(image == m)
    # HERE is where to change the parameter bounds/guesses
    params.add('offsetX', value=x.mean() - np.shape(image)[0] / 2)
    # , min=x.mean() - 200, max=x.mean() - 20)
    params.add('offsetY', value=y.mean() - np.shape(image)[0] / 2)
    # , min=y.mean() - 200, max=y.mean() - 20)

    # use sum of pixels without minimum value as first flux guess
    dataMin = np.min(image)
    dataTotal = (
        np.sum(image) - dataMin * np.shape(image)[0] * np.shape(image)[1])
    params.add(
        'flux', value=dataTotal)#, min=dataTotal / 100, max=dataTotal * 5)
    params.add('background', value=dataMin)
    # , min=dataMin - 500, max=dataMin + 800)

    def chiSqrKolmogorov(p):
        kIm = imageKolmogorov(p, image.shape)
        err = sBack**2 + abs(image - p['background'])
        return np.sqrt(nExp / err) * (kIm - image)

    return Minimizer(chiSqrKolmogorov, params).minimize(method='leastsq')


def imageKolmogorov(P, shape):
    """
    Creates an image of a Kolmogorov PSF given Parameters class P, which holds:
        - the half light radius (in arcsec)
        - shear ellipticities g1 and g2
        - pixel scale
        - offset in x and y from image center
        - flux
        - background level
    """
    k = galsim.Kolmogorov(half_light_radius=P['hlr'], flux=P['flux'])
    k = k.shear(g1=P['g1'], g2=P['g2'])
    kIm = k.drawImage(
        nx=shape[1], ny=shape[0], scale=P['p_scale'],
        offset=(P['offsetX'], P['offsetY'])).array
    return kIm + P['background']

def subtractBackground(imgSeries):
    for img in imgSeries:
        nx, ny = [int(np.ceil(i/50)) for i in img.shape]
        mask = makeMask(img, (nx, ny), maskCenter=True)
        maskedExposure = img * mask
        background = maskedExposure[maskedExposure!=0].flatten()
        img -= background.mean()
        
def accumulateExposures(sequence, pScale, subtract=True, indices=None, numBin=None):
    '''
    Accumulates exposures (or bins data) for given sequence, returns result
    '''
    N = len(sequence)
    if numBin is None:
        psf = sequence[0].astype(np.float32)
        if pScale == 0.2:
            accumulatedPSF = [spatialBinToLSST(psf)]
        else:
            accumulatedPSF = [np.copy(psf)]

        if indices is None:
            for exposure in range(1, N):
                if pScale == 0.2:
                    temp = spatialBinToLSST(sequence[exposure])
                    psf += temp
                else:
                    psf += sequence[exposure].astype(np.float32)
                accumulatedPSF.append(np.copy(psf))

            if subtract:
                subtractBackground(accumulatedPSF)

            return [accumulatedPSF[i] / (i + 1) for i in range(N)]

        # if a list of indices is given, return accumulated PSFs for those only
        else:
            # this is definitely not optimal, should be reusing sum as I go
            for exposure in indices[1:]:
                psf = sequence[0:exposure + 1].sum(axis=0).astype(np.float32)
                if pScale == 0.2:
                    accumulatedPSF.append(spatialBinToLSST(psf) / exposure)
                else:
                    accumulatedPSF.append(np.copy(psf) / exposure)
            
            if subtract:
                subtractBackground(accumulatedPSF)

            return accumulatedPSF

    else:
        assert N % float(numBin) == 0, \
            'Number of requested bins does not divide length of dataset'
        binSize = N / numBin

        binnedPSF = [
            np.sum(sequence[n * binSize:(n + 1) * binSize], axis=0) / binSize
            for n in range(numBin)]

        if subtract:
            subtractBackground(accumulatedPSF)
        
        return binnedPSF
