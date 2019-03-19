import numpy as np
from matplotlib import pyplot as plt

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