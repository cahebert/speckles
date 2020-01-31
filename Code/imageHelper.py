import numpy as np
from matplotlib import pyplot as plt
import galsim
import pickle
from scipy.optimize import curve_fit
import sklearn.utils


def imageCOM(image, expMask=None):
    '''
    Find the center of mass of given image.
    Returns:
    ========
    Tuple of ints corrsponding to the x, y indices of the center of mass
    '''
    img = np.copy(image)
    if expMask is not None:
        if abs(img.min()) > 500:
            subtractBackground([img], expMaskDict={0:expMask})
        # give 0 weight to masked pixels
        img *= expMask
    elif abs(img.min()) > 500:
        subtractBackground([img])
    indices = np.linspace(0, img.shape[0] - 1, img.shape[0])
    comX = np.dot(img.sum(axis=1), indices) / img.sum()
    comY = np.dot(img.sum(axis=0), indices) / img.sum()
    return int(comX), int(comY)

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
    if abs(img.min())>500:
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
    if len(right) == 0 and len(left) == 0:
        print('something is fishy! check image subtraction')
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

def spatialBinToLSST(image, expMask=None, n=14):
    '''
    Given an image, return a spatially binned image size 14x14
    TO DO?
    - potentially have functionality for an image that isn't 256x256
    '''
    image = image[2:-2, 2:-2]
    newIm = np.empty((n, n))
    
    if expMask is not None:
        image = np.ma.array(data=image, mask=1-expMask[2:-2, 2:-2])

    for i in range(n):
        for j in range(n):
            newIm[i,j] = image[18 * i:18 * (i + 1), 18 * j:18 * (j + 1)].mean()

    return newIm

def subtractBackground(imgSeries, accumulated=False, expMaskDict=None, method='slice', center='new'):
    '''
    Subtract a flat background from a given series of images. Uses slices from the four sides of an image 
    and uses the mean of whichever has the smallest variance 
    Input:
    ======
    imgSeries: list of images to be background subtracted
    accumulated: optional, whether the images are sinlge exposures
    expMaskDict: optional, dictionary of exposures with associated pixel masks
    method: specify method for finding the background value. 
            'slice' means finding edge slice with lowest variance, and 
            'mask' is applying a mask based on the PSF position.
    center: if using the 'mask' method, how do you center the mask and decide on edge values? 
            'new' method centers mask but doesn't move the mask edges further than 10 pixels from the edge, 
            'old' method moves it arbitrarily far.
    Output:
    =======
    None. Modifies the input series imgSeries in place.
    Notes:
    ======
    If passing just a single image in, key value in expMaskDisk must reflect the position of the image in the input sequence, not the original series. 
    '''
    if expMaskDict is not None:
        # not accumulated: only flag the exposure in the dict
        flag = list(expMaskDict.keys())[0]
        assert len(list(expMaskDict.keys())) == 1,'Found more than one mask in dict. Beware: code is not built for this!'
        pixelMask = expMaskDict[flag]
    else: 
        flag = None
        
    for i in range(len(imgSeries)):
        img = imgSeries[i]
        
        if method == 'slice':
            nx, ny = img.shape
            if nx <= 15:
                mask_size = 1
            else:
                mask_size = 10
            sideSlices = np.zeros((4, mask_size, nx))
            if flag is not None and accumulated or i == flag:
                maskedExposure = img * pixelMask
                sideSlices[0] = maskedExposure[:mask_size, :]
                sideSlices[1] = maskedExposure[nx - mask_size:nx, :]
                sideSlices[2] = maskedExposure[:, :mask_size].T
                sideSlices[3] = maskedExposure[:, ny - mask_size:ny].T
                #don't include masked pixels in the variance!
                minVariance = np.argmin([sideSlices[j][sideSlices[j]!=0].flatten().var() for j in range(4)])
                                
            else:
                sideSlices[0] = img[:mask_size, :]
                sideSlices[1] = img[nx - mask_size:nx, :]
                sideSlices[2] = img[:, :mask_size].T
                sideSlices[3] = img[:, ny - mask_size:ny].T
                
            minVariance = np.argmin(sideSlices.var(axis=(1,2)))
            img -= np.mean(sideSlices[minVariance][sideSlices[minVariance]!=0])


def accumulateExposures(sequence, maskDict=None, subtract=True, indices=None, numBins=None, overRide=False):
    '''
    Accumulates exposures (or bins data) for given sequence, returns result
    '''
    N = len(sequence)
    sequence = np.copy(sequence)
    
    if subtract:
        subtractBackground(sequence, maskDict=maskDict, accumulated=True)
    
    # make a new mask dict to reflect index changes of accumulating exposures
    if maskDict is not None:
        expid = [i for i in maskDict.keys()][0]
        # if accumulating exposures
        if numBins is None:
            if indices is not None:
                maskedExps = np.arange(15)[np.array(indices) >= expid]
                newMaskDict = {k: maskDict[expid] for k in maskedExps}
            else:
                maskedExps = [i for i in np.arange(1000) if i >= expid]
                newMaskDict = {k: maskDict[expid] for k in maskedExps}

        # if binning exposures
        else:
            maskedExps = [i for i in np.arange(numBins)*(1000/numBins) if i >= expid]
            newMaskDict = {k: maskDict[expid] for k in maskedExps}
    else:
        newMaskDict = None
    
    if numBins is None:
        psf = sequence[0].astype(np.float64)
        accumulatedPSF = [np.copy(psf)]
                    
        # if no indices are specified, accumulate every exposure
        if indices is None:
            for exposure in range(1, N):
                psf += sequence[exposure].astype(np.float64)
                accumulatedPSF.append(np.copy(psf))

            return [accumulatedPSF[i] / (i + 1) for i in range(N)], newMaskDict

        # if a list of indices is given, return accumulated PSFs for those only
        else:
            # this is definitely not optimal, should be reusing sum as I go
            for exposure in indices[1:]:
                psf = sequence[0:exposure + 1].sum(axis=0).astype(np.float64)
                accumulatedPSF.append(np.copy(psf) / (exposure+1))
            
            return accumulatedPSF, newMaskDict

    # if binning data
    else:
        residual = N % float(numBins)
        if residual != 0 and overRide is False:
            raise ValueError(f'Number of requested bins ({numBins}) does not divide length of dataset ({N})')
        
        binSize = int(N / numBins)

        binnedPSF = [
            sequence[n * binSize:(n + 1) * binSize].sum(axis=0) / binSize
            for n in range(numBins)]

        return binnedPSF, newMaskDict


def singleExposureHSM(img, expMask=None, subtract=True, maxIters=400,
                      max_ashift=75, max_amoment=5.0e5, strict=True):
    if subtract:
        img = np.copy(img)
        expMaskDict = {0: expMask} if expMask is not None else None
        subtractBackground([img], expMaskDict=expMaskDict)
    
    # guesstimate center and size of PSF as start for HSM
    comx, comy = imageCOM(img)
    fwhm = imageFWHM(img, comx, comy)
    guestimateSig = fwhm / 2.355

    badPix = galsim.Image(1 - expMask, xmin=0, ymin=0) if expMask is not None else None

    # make GalSim image of the exposure
    new_params = galsim.hsm.HSMParams(max_amoment=max_amoment, 
                                      max_ashift=max_ashift,
                                      max_mom2_iter=maxIters)
    galImage = galsim.Image(img, xmin=0, ymin=0)

    # run HSM adaptive moments with initial sigma guess
    try:
        hsmOut = galsim.hsm.FindAdaptiveMom(galImage, hsmparams=new_params,
                                        badpix=badPix,
                                        guess_sig=guestimateSig,
                                        guess_centroid=galsim.PositionD(comx, comy),
                                        strict=strict)
    except RuntimeError:
        return False
        
    # return HSM output
    return hsmOut

def calculateCentroids(imgSeries, N=200, subtract=False):
    step = int(1000/N)
    centroids = np.zeros((2, N))
    
    for i in range(N):
        fit = singleExposureHSM(img=np.sum(imgSeries[i*step:(i+1)*step], axis=0)/step, 
                                subtract=subtract, maxIters=25000, 
                                max_ashift=200, max_amoment=1e7)
        if fit == False: return False
        centroids[:,i] = [fit.moments_centroid.x, fit.moments_centroid.y]
    
    return {'x': centroids[0], 'y': centroids[1]}
    
def estimateMomentsHSM(images, maskDict=None, saveDict={'save':True, 'path':None}, 
                       strict=False, subtract=False, maxIters=800, 
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
        # try to assign expMask to the entry in maskDict
        try:
            expMask = maskDict[i]
        except TypeError:
            # no maskDict object => this dataset has no masks.
            expMask = None
        except KeyError:
            # no mask for exposure i => set to None.
            expMask = None

        # fit the image(s)
        hsmOut = singleExposureHSM(images[i], expMask=expMask, maxIters=maxIters, 
                                   subtract=subtract, strict=strict, 
                                   max_ashift=max_ashift, max_amoment=max_amoment)

        if hsmOut == False: return False
        # put results in a list
        fitResults.append(hsmOut)
        
    if saveDict['save'] == True:
        try: 
            # Save HSM output list to a pickle file.
            with open(saveDict['path'], 'wb') as file:
                pickle.dump(fitResults, file)
        except FileNotFoundError:
            print('Save file not found. Try adding/checking path in saveDict')
        return True
    else:
        return fitResults
    
# def makeMask(image, maskSize, maskCenter=False, center='new'):
#     # define a mask
#     nx, ny = np.shape(image)
#     mask = np.zeros((nx, ny))
    
#     # if desired, find the ~centroid
#     if maskCenter is True:
#         (x, y) = np.where(image == np.max(image))
#         maskCenter = (x.mean() - nx / 2, y.mean() - ny / 2)
# #         print(f'centroid is approximately at: {maskCenter}')
#     else:
#         maskCenter = (0, 0)
        
#     if center == 'new':
#         if maskCenter[0] > maskSize[0]:
#             xlEdge = maskSize[0]
#             xrEdge = nx
#         elif maskCenter[0] < maskSize[0]:
#             xlEdge = 0
#             xrEdge = nx - maskSize[0]
#         else:
#             xlEdge = int(maskCenter[0] + maskSize[0])
#             xrEdge = int(nx - maskSize[0] + maskCenter[0])

#         if maskCenter[1] > maskSize[1]:
#             ylEdge = maskSize[1]
#             yuEdge = ny
#         elif maskCenter[1] < maskSize[1]:
#             ylEdge = 1
#             yuEdge = ny - maskSize[1]
#         else:
#             ylEdge = int(maskCenter[1] + maskSize[1])
#             yuEdge = int(ny - maskSize[1] + maskCenter[1])

#     if center == 'old':
#             # make sure centered mask is within the image limits
#         if maskCenter[0] + maskSize[0] >= 0:
#             xlEdge = int(maskCenter[0] + maskSize[0])
#         else: 
#             xlEdge = 0
#         if nx - maskSize[0] + maskCenter[0] <= nx:
#             xrEdge = int(nx - maskSize[0] + maskCenter[0])
#         else: 
#             xrEdge = nx   

#         if maskCenter[1] + maskSize[1] >= 0:
#             ylEdge = int(maskCenter[1] + maskSize[1])
#         else: 
#             ylEdge = 0
#         if nx - maskSize[1] + maskCenter[1] <= ny:
#             yuEdge = int(ny - maskSize[1] + maskCenter[1])
#         else: 
#             yuEdge = ny

#     # define the mask
#     mask[:xlEdge, :] = 1
#     mask[xrEdge:nx, :] = 1
#     mask[:, :ylEdge] = 1
#     mask[:, yuEdge:ny] = 1
    
#     return mask
    
# def demonstrateCenteringMask(image, maskSize, plot=False, pScale=.01, nExp=1):
    
#     # define a mask
#     mask = makeMask(image, maskSize)
#     if plot:
#         plt.imshow(mask*image)
#         plt.title('mask')
#         plt.show()

#     # apply to image, find the std of the background 
#     back = (mask*image).ravel()
#     back = back[back!=0]    
#     std = np.std(back)
    
#     # do a 2 sigma clipping on the distribution of background pixels
#     twoSigma = np.mean(back) + 2 * std
#     clippedStd = np.std(back[back <= twoSigma])
    
#     if plot:
#         print(f'standard deviation: {std:.2f}' +
#               f'\n2 sigma clipped standard deviation: {clippedStd:.2f}')
#         bins = np.linspace(int(min(back)), int(max(back)), int(max(back) - min(back)))
#         plt.hist(back, bins=bins)
#         plt.axvline(twoSigma, color='yellow')
#         plt.ylabel('counts', fontsize=12)
#         plt.xlabel('pixel intensity', fontsize=12)
#         plt.show()
    
#     # define centered mask
#     centeredMask = makeMask(image, maskSize, maskCenter=True)
#     if plot:
#         plt.imshow(centeredMask*image)
#         plt.title('centered mask')
#         plt.show()    

#     # apply to image, find std of centered background
#     centeredBack = (centeredMask*image).ravel()
#     centeredBack = centeredBack[centeredBack!=0]
#     centeredStd = np.std(centeredBack)
    
#     # do a 2 sigma clipping on the distribution of background pixels
#     twoSigma = np.mean(centeredBack) + 2 * centeredStd
#     clippedCenteredStd = np.std(centeredBack[centeredBack <= twoSigma])
    
#     if plot:
#         print(f'standard deviation, centered: {centeredStd:.2f}' +
#               f'\n2 sigma clipped standard deviation, centered: {clippedCenteredStd:.2f}')
#         bins = np.linspace(
#             int(min(centeredBack)), int(max(centeredBack)), 
#             int(max(centeredBack) - min(centeredBack)))
#         plt.hist(centeredBack, bins=bins)
#         plt.axvline(twoSigma, color='yellow')
#         plt.ylabel('counts', fontsize=12)
#         plt.xlabel('pixel intensity', fontsize=12)
#         plt.show()

        
# def estimateBackground(image, maskSize, maskCenter=True, plot=False, pScale=.01, nExp=1):
#     '''
#     Estimates the background of image.
#     Parameters:
#     ----------
#     ...
#     '''
#     # define centered mask
#     centeredMask = makeMask(image, maskSize, maskCenter=True)
#     if plot:
#         plt.imshow(centeredMask*image)
#         plt.title('centered mask')
#         plt.show()    

#     # apply to image, find std of centered background
#     centeredBack = (centeredMask*image).ravel()
#     centeredBack = centeredBack[centeredBack!=0]
#     centeredStd = np.std(centeredBack)
    
#     # do a 2 sigma clipping on the distribution of background pixels
#     twoSigma = np.mean(centeredBack) + 2 * centeredStd
#     clippedCenteredStd = np.std(centeredBack[centeredBack <= twoSigma])
    
#     if plot:
#         print(f'standard deviation, centered: {centeredStd:.2f}' +
#               f'\n2 sigma clipped standard deviation, centered: {clippedCenteredStd:.2f}')
#         bins = np.linspace(
#             int(min(centeredBack)), int(max(centeredBack)), 
#             int(max(centeredBack) - min(centeredBack)))
#         plt.hist(centeredBack, bins=bins)
#         plt.axvline(twoSigma, color='yellow')
#         plt.ylabel('counts', fontsize=12)
#         plt.xlabel('pixel intensity', fontsize=12)
#         plt.show()
    
#     # return standard deviation of this clipped background
#     return clippedCenteredStd


# def accumulateExposures(sequence, maskDict=None, subtract=True, indices=None, numBins=None):
#     '''
#     Accumulates exposures (or bins data) for given sequence, returns result
#     '''
#     N = len(sequence)
#     sequence = np.copy(sequence)
    
#     if subtract:
#         subtractBackground(sequence, maskDict=maskDict, accumulated=True)
    
#     # make a new mask dictionary (if a current one exists and if using DSSI pixels) 
#     # to reflect the index changes of accumulating or binning the exposures
#     if maskDict is not None and if pixelScale != 0.2:
#         expid = [i for i in maskDict.keys()][0]
#         # if accumulating exposures
#         if numBins is None:
#             if indices is not None:
#                 maskedExps = np.arange(15)[np.array(indices) >= expid]
#                 newMaskDict = {(maskDict[expid], k) for k in maskedExps}
#             else:
#                 maskedExps = [i for i in np.arange(1000) if i >= expid]
#                 newMaskDict = {(maskDict[expid], k) for k in maskedExps}

#         # if binning exposures
#         else:
#             maskedExps = [i for i in np.arange(numBins)*(1000/numBins) if i >= expid]
#             newMaskDict = {(maskDict[expid], k) for k in maskedExps}
#     else:
#         newMaskDict = None
    
#     if numBins is None:
#         psf = sequence[0].astype(np.float64)
#         if pixelScale == 0.2:
#             accumulatedPSF = [spatialBinToLSST(psf)]
#         else:
#             accumulatedPSF = [np.copy(psf)]
                    
#         # if no indices are specified, accumulate every exposure
#         if indices is None:
#             for exposure in range(1, N):
#                 if pixelScale == 0.2:
#                     if maskDict is not None:
#                         temp = spatialBinToLSST(sequence[exposure], expMask=maskDict[exposure])
#                     else:
#                         temp = spatialBinToLSST(sequence[exposure])
#                     psf += temp
#                 else:
#                     psf += sequence[exposure].astype(np.float64)
#                 accumulatedPSF.append(np.copy(psf))

#             return [accumulatedPSF[i] / (i + 1) for i in range(N)], newMaskDict

#         # if a list of indices is given, return accumulated PSFs for those only
#         else:
#             # this is definitely not optimal, should be reusing sum as I go
#             for exposure in indices[1:]:
#                 psf = sequence[0:exposure + 1].sum(axis=0).astype(np.float64)
#                 if pixelScale == 0.2:
#                     if maskDict is not None:
#                         accumulatedPSF.append(spatialBinToLSST(psf, expMask=maskDict[exposure]) / (exposure+1))
#                     else:
#                         accumulatedPSF.append(spatialBinToLSST(psf) / (exposure+1))
#                 else:
#                     accumulatedPSF.append(np.copy(psf) / (exposure+1))
            
#             return accumulatedPSF, newMaskDict

#     else:
#         assert N % float(numBins) == 0, \
#             'Number of requested bins does not divide length of dataset'
#         binSize = N / numBin

#         binnedPSF = [
#             np.sum(sequence[n * binSize:(n + 1) * binSize], axis=0) / binSize
#             for n in range(numBins)]
        
#         if pixelScale == 0.2:
#             for i in numBins:
#                 binnedPSF[i] = spatialBinToLSST(binnedPSF[i])
            
#         return binnedPSF, newMaskDict