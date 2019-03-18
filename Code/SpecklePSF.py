import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
import galsim
import pandas as pd
from lmfit import Minimizer, Parameters

# basedir = '/Users/clairealice/Documents/Research/Burchat/SpeckleAnalysis/'

gains = {'025': {'a': 14.9, 'b': 12.8}, '234': {'a': 11.21, 'b': 11.21},
         '258': {'a': 11.21, 'b': 11.21}, '484': {'a': 14.9, 'b': 12.8},
         '663': {'a': 2.56, 'b': 11.21}, '693': {'a': 12.8, 'b': 12.8},
         '809': {'a': 11.21, 'b': 12.8}, '1039': {'a': 2.03, 'b': 2.56},
         '1262': {'a': 11.21, 'b': 11.21}}
# '1237': {'a': 32.96,'b': 265.83}

backgrounds = {'025': {'a': 30.85, 'b': 53.57},
               '234': {'a': 42.74, 'b': 139.74},
               '258': {'a': 69.39, 'b': 144.92},
               '484': {'a': 33.65, 'b': 61.42},
               '663': {'a': 32.13, 'b': 88.25},
               '693': {'a': 73.12, 'b': 81.39},
               '809': {'a': 39.08, 'b': 59.07},
               '1039': {'a': 26.04, 'b': 99.80},
               '1262': {'a': 84.10, 'b': 140.63}}
# '1237': {'a': 32.96,'b': 265.83}


class SpeckleSeries():
    '''
    Initialize class and load exposures from file
    - filenumber is the number in the filename if data, or seed if simultation
    - source is data vs simulation
    - basedir is location of data/sim folders
    - pScale is pixel scale
    - numBin is None for accumulated PSFs, number of desired bins otherwise
    '''

    def __init__(self, fileNumber, source, baseDir,
                 pScale=0.01, numBin=None, fitPts=11):
        # Define filenames for a and b fPlters according to data or simulation.
        if source == 'data':
            self.aFile = f'DSSIData/img_a_{fileNumber}.fits'
            self.bFile = f'DSSIData/img_b_{fileNumber}.fits'
        elif source == 'sim':
            self.aFile = f'simulations/sim_a_1000frames_s{fileNumber}.fits'
            self.bFile = f'simulations/sim_b_1000frames_s{fileNumber}.fits'

        self.source = source
        self.baseDir = baseDir
        self.pScale = pScale
        self.numBin = numBin

        # gain and background dictionaries for a and b filters
        self.gain = gains[str(fileNumber)]
        self.background = backgrounds[str(fileNumber)]
        if pScale == 0.2:
            # divide background std by sqrt number of subpixels
            self.background.update((k, v / 18)
                                   for (k, v) in self.background.items())

        # PSF plotting helpers
            self.ticks = [0, 3.25, 6.5, 9.75, 13]
        else:
            self.ticks = [0, 64, 128, 192, 256]

        self.tickLabels = [0, .7, 1.4, 2.1, 2.8]

        self.loadMinimalExposures()

    def loadMinimalExposures(self, fitPts=11):
        '''
        Try and open .fits files of cumulative/binned PSFs for both a+b filters
        If those don't exist, run adding/binning function on instantaneous PSFs
        numBin is None for accumulated PSF, and the number of bins otherwise
        TO DO:
        add check: if pScale==0.2 the loaded data should be 14/14
        '''
        # define filenames according to choice of accumulated/binned PSFs
        aFilename = self.aFile.split('.')[0]
        bFilename = self.bFile.split('.')[0]

        if self.pScale == 0.2:
            aFilename += '_LSSTpix'
            bFilename += '_LSSTpix'

        if self.numBin is None:
            aFilename += '_cumulative_11.fits'
            bFilename += '_cumulative_11.fits'

            # find indices of the images we'll want to fit later
            # take images that double the number of exposures
            self.indices = [
                int(np.round(i)) - 1 for i in np.logspace(0, 3, fitPts)]
            # make sure there are no duplicate indices:
            assert len([1 for x in self.indices
                        if list(self.indices).count(x) != 1]) == 0,\
                        'Please reduce fitPts'

        else:
            aFilename += '_%ibins.fits' % self.bins
            bFilename += '_%ibins.fits' % self.bins

        # try opening said files - if they don't exist, process data
        # just one try statement bc they should always be created in pairs
        try:
            ahdulist = fits.open(self.baseDir + aFilename)
            bhdulist = fits.open(self.baseDir + bFilename)
        except FileNotFoundError:
            # exception: load the raw data
            ahdulist = fits.open(self.baseDir + self.aFile)
            a = ahdulist[0].data
            ahdulist.close()

            bhdulist = fits.open(self.baseDir + self.bFile)
            b = bhdulist[0].data
            bhdulist.close()

            # if inst PSFs are raw data, flip b filter data to correct optics
            if self.source == 'data':
                b = np.fliplr(b)

            # run the accumulation/binning and save to object
            if self.numBin is None:
                self.aSeq = accumulateExposures(a, self.pScale, self.indices)
                self.bSeq = accumulateExposures(b, self.pScale, self.indices)
            else:
                self.indices = None
                self.aSeq = accumulateExposures(a, self.pScale, self.numBin)
                self.bSeq = accumulateExposures(b, self.pScale, self.numBin)
        else:
            # if no exception raised, use existing processed data:
            if self.numBin is None:
                self.aSeq = ahdulist[0].data
                self.bSeq = bhdulist[0].data
            else:
                self.aSeq = ahdulist[0].data
                self.bSeq = bhdulist[0].data

                ahdulist.close()
                bhdulist.close()

    def loadAllExposures(self):
        '''
        Try and open .fits files of cumulative/binned PSFs for both a+b filters
        If those don't exist, run adding/binning function on instantaneous PSFs
        numBin is None for accumulated PSF, and the number of bins otherwise
        '''
        # define filenames according to choice of accumulated/binned PSFs
        aFilename = self.aFile.split('.')[0]
        bFilename = self.bFile.split('.')[0]

        if self.pScale == 0.2:
            aFilename += '_LSSTpix'
            bFilename += '_LSSTpix'

        if self.numBin is None:
            aFilename += '_cumulative.fits'
            bFilename += '_cumulative.fits'
        else:
            aFilename += '_%ibins.fits' % self.bins
            bFilename += '_%ibins.fits' % self.bins

        # try opening said files - if they don't exist, process data
        # just one try statement bc they should always be created in pairs
        try:
            ahdulist = fits.open(self.baseDir + aFilename)
            bhdulist = fits.open(self.baseDir + bFilename)
        except FileNotFoundError:
            # exception: load the raw data
            ahdulist = fits.open(self.baseDir + self.aFile)
            a = ahdulist[0].data
            ahdulist.close()

            bhdulist = fits.open(self.baseDir + self.bFile)
            b = bhdulist[0].data
            bhdulist.close()

            # if inst PSFs are raw data, flip b filter data to correct optics
            if self.source == 'data':
                b = np.fliplr(b)

            # run the accumulation/binning and save to object
            if self.numBin is None:
                self.aAccumulated = accumulateExposures(a, self.pScale)
                self.bAccumulated = accumulateExposures(b, self.pScale)
            else:
                self.aBinned = accumulateExposures(a, self.pScale, numBin=self.numBin)
                self.bBinned = accumulateExposures(b, self.pScale, numBin=self.numBin)
        else:
            # if no exception raised, use existing processed data:
            if self.numBin is None:
                self.aAccumulated = ahdulist[0].data
                self.bAccumulated = bhdulist[0].data
            else:
                self.aBinned = ahdulist[0].data
                self.bBinned = bhdulist[0].data

                ahdulist.close()
                bhdulist.close()

    def fitExposures(self, fitPts=11):
        '''
        Fit object data (specify binned/accumulated PSFs) to Kolmogorov profile
        Saves the best fit parameters in a dataframe.
        TO DO:
        - save the fits!
        - try different fit methods: moments?
        - implement possiblitiy of a Von Karman fit
        - fit third moment of PSF as well!
        '''
        # length of sequence to fit
        N = len(self.aSeq)

        # if binned, define N_exp
        if self.numBin is not None:
            self.indices = None
            # N_exp is input to chi squared calculation
            N_exp = 1000 / N
        else:
            # make sure self.indices exists
            assert self.indices is not None,\
                'Uh oh, expected self.indices for accumulated PSF data'
        # indices = [int(np.round(i)) - 1 for i in np.logspace(0, 3, fitPts)]

        aFitResults = []
        bFitResults = []

        aFitResiduals = []
        bFitResiduals = []

        for i in range(N):
            if self.indices is not None:  # i.e. if data is accumulated
                N_exp = self.indices[i] + 1

            # fit the a filter image
            aParamRes = fitSingleExposure(self.aSeq[i] * self.gain['a'],
                                          self.pScale, self.background['a'],
                                          nExp=N_exp)
            aParam = aParamRes.params
            # make sure the fit converged well
            try:
                aErr = np.sqrt(np.diag(aParamRes.covar))
            except AttributeError:
                import lmfit
                lmfit.report_fit(aParamRes)
                print('check parameter bounds for a: too close to fit values?')
            # put results in a dataframe -> append to sequence list
            aResDict = {'g1': aParam['g1'].value, 's_g1': aErr[0],
                        'g2': aParam['g2'].value, 's_g2': aErr[1],
                        'hlr': aParam['hlr'].value, 's_hlr': aErr[2],
                        'x': aParam['offsetX'].value, 's_offsetX': aErr[3],
                        'y': aParam['offsetY'].value, 's_offsetY': aErr[4],
                        'flux': aParam['flux'], 's_flux': aErr[5],
                        'background': aParam['background'],
                        's_background': aErr[6],
                        'r_chi_sqr': aParamRes.redchi}

            aFitResiduals.append(aParamRes.residual)

            aFitResults.append(pd.DataFrame(data=aResDict, index=[0]))

            # fit the b filter image
            bParamRes = fitSingleExposure(self.bSeq[i] * self.gain['b'],
                                          self.pScale, self.background['b'],
                                          nExp=N_exp)
            bParam = bParamRes.params
            # make sure the fit converged well
            try:
                bErr = np.sqrt(np.diag(bParamRes.covar))
            except ValueError:
                import lmfit
                lmfit.report_fit(bParamRes)
                print('check parameter bounds for b: too close to fit values?')
            # results of fit to a dataframe, append to sequence list
            bResDict = {'g1': bParam['g1'].value, 's_g1': bErr[0],
                        'g2': bParam['g2'].value, 's_g2': bErr[1],
                        'hlr': bParam['hlr'].value, 's_hlr': bErr[2],
                        'x': bParam['offsetX'].value, 's_offsetX': bErr[3],
                        'y': bParam['offsetY'].value, 's_offsetY': bErr[4],
                        'flux': bParam['flux'].value, 's_flux': bErr[5],
                        'background': bParam['background'].value,
                        's_background': bErr[6],
                        'r_chi_sqr': bParamRes.redchi}

            bFitResiduals.append(aParamRes.residual)

            bFitResults.append(pd.DataFrame(data=bResDict, index=[0]))

        # concatenate all the dataframes together into one for each filter
        self.aFits = pd.concat(aFitResults, ignore_index=True)
        self.bFits = pd.concat(bFitResults, ignore_index=True)

        self.aFitResiduals = aFitResiduals
        self.bFitResiduals = bFitResiduals

    def plotResidual(self, filt, frm=None, paramsDict={}, figSize=(14, 12),
                     saveName=None):
        '''
        Plot data, model, and residual images.
        TO DO:
        - not sure if this way of plotting colorbars will work, not sure how to
          add them to each axis. Can even make on cbar for all -- not sure
        - indicate accumulated exposure time of the frame displayed
        - optional grid?
        '''
        plotParamsDict = {'origin': 'lower', 'interpolation': 'nearest'}
        plotParamsDict.update(paramsDict)

        # if no specific frame is selected, a random one is chosen
        if frm is None:
            frm = np.random.randint(0, 11)

        if filt == 'a':
            data = self.gain['a'] * self.aSeq[frm]
            shape = np.shape(data)
            fit = self.aFits.loc[frm]
        elif filt == 'b':
            data = self.gain['b'] * self.bSeq[frm]
            shape = np.shape(data)
            fit = self.bFits.loc[frm]
        else:
            raise Exception('filter must be either "a" (692nm) or "b" (880nm)')

        k = galsim.Kolmogorov(half_light_radius=fit['hlr'], flux=fit['flux'])
        k = k.shear(g1=fit['g1'], g2=fit['g2'])
        model = k.drawImage(
            nx=shape[1], ny=shape[0], scale=self.pScale,
            offset=(fit['x'], fit['y'])).array + fit['background']

        chiMap = self.aFitResiduals[frm].reshape(shape)

        normConst = np.amax([data, model])

        fig, ((axD, axM), (axR, axC)) = plt.subplots(2, 2, figsize=figSize)

        imD = axD.imshow(data / normConst, **plotParamsDict, vmax=1)
        fig.colorbar(imD, ax=axD)
        imM = axM.imshow(model / normConst, **plotParamsDict, vmax=1)
        fig.colorbar(imM, ax=axM)
        imR = axR.imshow((data - model) / normConst, **plotParamsDict)
        fig.colorbar(imR, ax=axR)
        imC = axC.imshow(chiMap, **plotParamsDict)
        fig.colorbar(imC, ax=axC)

        for ax in [axD, axM, axR, axC]:
            ax.set_xticks(self.ticks)
            ax.set_yticks(self.ticks)
            ax.set_xticklabels(self.tickLabels)

        axD.set_ylabel('[arcsec]', fontsize=12)
        axD.set_title("(Normalized) Data", fontsize=14)
        axD.set_yticklabels(self.tickLabels)

        axM.set_title("Kolmogorov Model", fontsize=14)
        axM.set_yticklabels([])

        axR.set_ylabel('[arcsec]', fontsize=12)
        axR.set_title("Residual", fontsize=14)
        axR.set_yticklabels(self.tickLabels)

        axC.set_title("$\chi$ map", fontsize=14)
        axC.set_yticklabels([])

        if saveName is not None:
            plt.savefig(saveName + '.png', bbox_to_inches='tight')

        plt.show()

    def saveFitParams(self, path=None):
        '''
        Save pandas dataFrames containing fit parameters to pickke.
        If specified, the save to directory located at 'path', or
        default to baseDir/fit_pickles
        '''
        # Save to particular path if given. Else, save to .p folder
        if path is None:
            path = self.baseDir + 'fit_pickles/'

        saveA = path + '/' + self.aFile.split('.')[0].split('/')[-1]
        saveB = path + '/' + self.bFile.split('.')[0].split('/')[-1]

        if self.pScale == 0.2:
            saveA += '_LSSTpix'
            saveB += '_LSSTpix'

        if self.indices is not None:
            saveA += '_cumulative.p'
            saveB += '_cumulative.p'
        else:
            saveA += '_binned.p'
            saveB += '_binned.p'

        self.aFits.to_pickle(saveA)
        self.bFits.to_pickle(saveB)

    def saveToFits(self, path=None):
        '''
        Save accumulated/binned images to FITS file.
        If specified, the save to directory located at 'path', or
        default to baseDir/DSSIData
        TO DO
        - add checks that the accumulated/binned files exist before trying
            to save them
        '''
        if path is None:
            path = self.baseDir + 'DSSIData/'

        saveA = path + '/' + self.aFile.split('.')[0].split('/')[-1]
        saveB = path + '/' + self.bFile.split('.')[0].split('/')[-1]

        if self.pScale == 0.2:
            saveA += '_LSSTpix'
            saveB += '_LSSTpix'

        if self.indices is not None:
            hduA = fits.PrimaryHDU(self.aAccumulated)
            hduB = fits.PrimaryHDU(self.bAccumulated)

            saveA += '_cumulative.fits'
            saveB += '_cumulative.fits'
        else:
            hduA = fits.PrimaryHDU(self.aBinned)
            hduB = fits.PrimaryHDU(self.bBinned)

            saveA += '_binned.fits'
            saveB += '_binned.fits'

        hduA.writeto(saveA)
        hduB.writeto(saveB)


####################
# HELPER FUNCTIONS #
####################


def accumulateExposures(sequence, pScale, indices=None, numBin=None):
    '''
    Accumulates exposures (or bins data) for given sequence, returns result
    '''
    N = len(sequence)
    if numBin is None:
        psf = spatialBinToLSST(sequence[0].astype(np.float32))
        if pScale == 0.2:
            accumulatedPSF = [psf]
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

            return [accumulatedPSF[i] / (i + 1) for i in range(N)]

        # if a list of indices is given, return accumulated PSFs for those only
        else:
            # this is defniitely not optimal, should be reusing sum as I go
            for exposure in indices[1:]:
                psf = sequence[0:exposure].sum(axis=0).astype(np.float32)
                if pScale == 0.2:
                    accumulatedPSF.append(spatialBinToLSST(psf) / exposure)
                else:
                    accumulatedPSF.append(np.copy(psf) / exposure)

            return accumulatedPSF

    else:
        assert N % float(numBin) == 0, \
            'Number of requested bins does not divide length of dataset'
        binSize = N / numBin

        binnedPSF = [
            np.sum(sequence[n * binSize:(n + 1) * binSize], axis=0) / binSize
            for n in range(numBin)]

        return binnedPSF


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


def fitSingleExposure(image, pScale, sBack, nExp):
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
