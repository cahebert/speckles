import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
import galsim
import pandas as pd
from lmfit import Minimizer, Parameters
import pickle
import helperFunctions as helper

# basedir = '/Users/clairealice/Documents/Research/Burchat/SpeckleAnalysis/'

# gains = {'025': {'a': 14.9, 'b': 12.8}, '234': {'a': 11.21, 'b': 11.21},
#          '258': {'a': 11.21, 'b': 11.21}, '484': {'a': 14.9, 'b': 12.8},
#          '663': {'a': 2.56, 'b': 11.21}, '693': {'a': 12.8, 'b': 12.8},
#          '809': {'a': 11.21, 'b': 12.8}, '1039': {'a': 2.03, 'b': 2.56},
#          '1262': {'a': 11.21, 'b': 11.21}}
# '1237': {'a': 32.96,'b': 265.83}

# backgrounds = {'025': {'a': 30.85, 'b': 53.57},
#                '234': {'a': 42.74, 'b': 139.74},
#                '258': {'a': 69.39, 'b': 144.92},
#                '484': {'a': 33.65, 'b': 61.42},
#                '663': {'a': 32.13, 'b': 88.25},
#                '693': {'a': 73.12, 'b': 81.39},
#                '809': {'a': 39.08, 'b': 59.07},
#                '1039': {'a': 26.04, 'b': 99.80},
#                '1262': {'a': 84.10, 'b': 140.63}}
# '1237': {'a': 32.96,'b': 265.83}

hlrSigmaconversion = .01270655

class SpeckleSeries():
    '''
    Initialize class and load exposures from file
    - filenumber is the number in the filename if data, or seed if simultation
    - source is data vs simulation
    - basedir is location of data/sim folders
    - pScale is pixel scale
    - numBin is None for accumulated PSFs, number of desired bins otherwise
    '''

    def __init__(self, fileNumber, source, baseDir, subtract=True,
                 pScale=0.01, numBin=None, fitPts=15):
        # Define filenames for a and b fPlters according to data or simulation.
        if source == 'data':
            self.aFile = f'rawSpeckles/img_a_{fileNumber}.fits'
            self.bFile = f'rawSpeckles/img_b_{fileNumber}.fits'
        elif source == 'sim':
            self.aFile = f'simulations/sim_a_1000frames_s{fileNumber}.fits'
            self.bFile = f'simulations/sim_b_1000frames_s{fileNumber}.fits'

        self.source = source
        self.baseDir = baseDir
        self.pScale = pScale
        self.numBin = numBin
        self.subtract = subtract

        # pixel mask dictionary
        with open('./pixelMasks.p', 'rb') as file:
            maskDict = pickle.load(file)
            maskDict = dict(maskDict)
        if self.aFile.split('/')[-1] in maskDict.keys():
            self.aMask = maskDict[self.aFile.split('/')[-1]]
        else:
            self.aMask = None
        if self.bFile.split('/')[-1] in maskDict.keys():
            self.bMask = maskDict[self.bFile.split('/')[-1]]
        else:
            self.bMask = None
        
        # gain and background dictionaries for a and b filters
        with open('./eConversionWithGain.p', 'rb') as file:
            aduConvertWithGain = pickle.load(file)
        self.gain = dict(
            (k.split('_')[1], v) for k, v in dict(aduConvertWithGain).items()
            if fileNumber in k)
        self.background = {'a': 33.7, 'b': 33.7} #backgrounds[str(fileNumber)]
        if pScale == 0.2:
            # divide background std by sqrt number of subpixels
            self.background.update((k, v / 18)
                                   for (k, v) in self.background.items())

        # PSF plotting helpers
            self.ticks = [0, 3.25, 6.5, 9.75, 13]
        else:
            self.ticks = [0, 64, 128, 192, 256]

        self.tickLabels = [0, .7, 1.4, 2.1, 2.8]

        self.loadMinimalExposures(fitPts)

    def loadMinimalExposures(self, fitPts=15, save=False):
        '''
        Try and open .fits files of cumulative/binned PSFs for both a+b filters
        If those don't exist, run adding/binning function on instantaneous PSFs
        numBin is None for accumulated PSF, and the number of bins otherwise
        TO DO:
        add check: if pScale==0.2 the loaded data should be 14/14
        '''
        # define filenames according to choice of accumulated/binned PSFs
        aFilename = 'rawSpeckles/accumulated/' + self.aFile.split('.')[0].split('/')[-1]
        bFilename = 'rawSpeckles/accumulated/' + self.bFile.split('.')[0].split('/')[-1]

        if self.pScale == 0.2:
            aFilename += '_LSSTpix'
            bFilename += '_LSSTpix'

        if self.numBin is None:
            aFilename += f'_cumulative_{fitPts}.fits'
            bFilename += f'_cumulative_{fitPts}.fits'

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
            a = ahdulist[0].data.astype(np.float64)
            ahdulist.close()

            bhdulist = fits.open(self.baseDir + self.bFile)
            b = bhdulist[0].data.astype(np.float64)
            bhdulist.close()

            # if inst PSFs are raw data, flip a filter data to correct optics
            if self.source == 'data':
                a = a[:,:,::-1]

            # run the accumulation/binning and save to object
            if self.numBin is None:
                self.aSeq = helper.accumulateExposures(a, 
                                                       self.pScale, 
                                                       expMaskDict=self.aMask,
                                                       subtract=self.subtract,
                                                       indices=self.indices)
                self.bSeq = helper.accumulateExposures(b, 
                                                       self.pScale,
                                                       expMaskDict=self.bMask,
                                                       subtract=self.subtract,
                                                       indices=self.indices)
            else:
                self.indices = None
                self.aSeq = helper.accumulateExposures(a, 
                                                       self.pScale, 
                                                       expMaskDict=self.aMask,
                                                       subtract=self.subtract,
                                                       numBin=self.numBin)
                self.bSeq = helper.accumulateExposures(b, 
                                                       self.pScale, 
                                                       expMaskDict=self.bMask,
                                                       subtract=self.subtract,
                                                       numBin=self.numBin)

            # save sequences to file
            hduA = fits.PrimaryHDU(self.aSeq)
            hduB = fits.PrimaryHDU(self.bSeq)

            hduA.writeto(self.baseDir + aFilename)
            hduB.writeto(self.baseDir + bFilename)
                
        else:
            # if no exception raised, use existing processed data:
            if self.numBin is None:
                self.aSeq = ahdulist[0].data.astype(np.float64)
                self.bSeq = bhdulist[0].data.astype(np.float64)
            else:
                self.aSeq = ahdulist[0].data.astype(np.float64)
                self.bSeq = bhdulist[0].data.astype(np.float64)

                ahdulist.close()
                bhdulist.close()


                
    def fitExposures(self, fitMethod='hsm', fitPts=15, 
                     maxIters=400, max_ashift=75, max_amoment=5.0e5, 
                     savePath=None):
        '''
        Fit loaded exposures to extract PSF parameters. Call appropriate class function
        based on fitMethod. Save if desired.
        '''
        if fitMethod == 'hsm':
            self.estimateMomentsHSM(fitPts=fitPts, 
                                    maxIters=maxIters, 
                                    max_ashift=max_ashift, 
                                    max_amoment=max_amoment)
        
        elif fitMethod == 'kolmogorov':
            self.kolmogorovFitExposures()
        
        else:
            print('Please enter valid method: choose hsm or kolmogorov.')
        
        if savePath is not None:
            # pass fitMethod to be included in file name.
            self.saveFitParams(fitMethod, path=savePath)
        else:
            print('beware, your fits have not been saved to file')
            
    def estimateMomentsHSM(self, maxIters=400, max_ashift=75, max_amoment=5.0e5, fitPts=15):
        '''
        Estimate the moments of the PSF images using HSM.
        TO DO:
        - fit third moment of PSF as well?
        '''
        # length of sequence to fit
        N = len(self.aSeq)

        aFitResults = []
        bFitResults = []

        for i in range(N):
            expMask = None
            if self.aMask is not None:
                if i in self.aMask.keys():
                    expMask = self.aMask[i]
            # fit the a filter image
            aParams = helper.singleExposureHSM(self.aSeq[i], 
                                               expMask = expMask,
                                               maxIters = maxIters, 
                                               max_ashift = max_ashift, 
                                               max_amoment = max_amoment)

            # put results in a dataframe -> append to sequence list
            aResDict = {'g1': aParams[0], 'g2': aParams[1], 'hlr': aParams[2] * hlrSigmaconversion,
                        'x': aParams[3].x, 'y': aParams[3].y}
            aFitResults.append(pd.DataFrame(data=aResDict, index=[0]))

            expMask = None
            if self.bMask is not None:
                if i in self.bMask.keys():
                    expMask = self.bMask[i]
            # fit the b filter image
            bParams = helper.singleExposureHSM(self.bSeq[i], 
                                               expMask = expMask,
                                               maxIters = maxIters, 
                                               max_ashift = max_ashift, 
                                               max_amoment = max_amoment)

            # put results in a dataframe -> append to sequence list
            bResDict = {'g1': bParams[0], 'g2': bParams[1], 'hlr': bParams[2] * hlrSigmaconversion,
                        'x': bParams[3].x, 'y': bParams[3].y}
            bFitResults.append(pd.DataFrame(data=bResDict, index=[0]))


        # concatenate all the dataframes together into one for each filter
        self.aFits = pd.concat(aFitResults, ignore_index=True)
        self.bFits = pd.concat(bFitResults, ignore_index=True)
            

#     def plotResidual(self, filt, frm=None, paramsDict={}, figSize=(14, 12),
#                      saveName=None):
#         '''
#         Plot data, model, and residual images.
#         ONLY call this if data has been fit to Kolmogorov, otherwise class won't have models to plot
#         TO DO:
#         - not sure if this way of plotting colorbars will work, not sure how to
#           add them to each axis. Can even make on cbar for all -- not sure
#         - indicate accumulated exposure time of the frame displayed
#         - optional grid?
#         '''
#         plotParamsDict = {'origin': 'lower', 'interpolation': 'nearest'}
#         plotParamsDict.update(paramsDict)

#         # if no specific frame is selected, a random one is chosen
#         if frm is None:
#             frm = np.random.randint(0, 11)

#         if filt == 'a':
#             data = self.gain['a'] * self.aSeq[frm]
#             shape = np.shape(data)
#             fit = self.aFits.loc[frm]
#         elif filt == 'b':
#             data = self.gain['b'] * self.bSeq[frm]
#             shape = np.shape(data)
#             fit = self.bFits.loc[frm]
#         else:
#             raise Exception('filter must be either "a" (692nm) or "b" (880nm)')

#         k = galsim.Kolmogorov(half_light_radius=fit['hlr'], flux=fit['flux'])
#         k = k.shear(g1=fit['g1'], g2=fit['g2'])
#         model = k.drawImage(
#             nx=shape[1], ny=shape[0], scale=self.pScale,
#             offset=(fit['x'], fit['y'])).array + fit['background']

#         chiMap = self.aFitResiduals[frm].reshape(shape)

#         normConst = np.amax([data, model])

#         fig, ((axD, axM), (axR, axC)) = plt.subplots(2, 2, figsize=figSize)

#         imD = axD.imshow(data / normConst, **plotParamsDict, vmax=1)
#         fig.colorbar(imD, ax=axD)
#         imM = axM.imshow(model / normConst, **plotParamsDict, vmax=1)
#         fig.colorbar(imM, ax=axM)
#         imR = axR.imshow((data - model) / normConst, **plotParamsDict)
#         fig.colorbar(imR, ax=axR)
#         imC = axC.imshow(chiMap, **plotParamsDict)
#         fig.colorbar(imC, ax=axC)

#         for ax in [axD, axM, axR, axC]:
#             ax.set_xticks(self.ticks)
#             ax.set_yticks(self.ticks)
#             ax.set_xticklabels(self.tickLabels)

#         axD.set_ylabel('[arcsec]', fontsize=12)
#         axD.set_title("(Normalized) Data", fontsize=14)
#         axD.set_yticklabels(self.tickLabels)

#         axM.set_title("Kolmogorov Model", fontsize=14)
#         axM.set_yticklabels([])

#         axR.set_ylabel('[arcsec]', fontsize=12)
#         axR.set_title("Residual", fontsize=14)
#         axR.set_yticklabels(self.tickLabels)

#         axC.set_title("$\chi$ map", fontsize=14)
#         axC.set_yticklabels([])

#         if saveName is not None:
#             plt.savefig(saveName + '.png', bbox_to_inches='tight')

#         plt.show()

    def saveFitParams(self, fitMethod, path=None):
        '''
        Save pandas dataFrames containing fit parameters to pickke.
        If specified, the save to directory located at 'path', or
        default to baseDir/fit_pickles
        '''
        # Save to particular path if given. Else, save to .p folder
        if path is None:
            path = self.baseDir + 'fit_pickles/'

        saveA = path + '/' + fitMethod + '_' + self.aFile.split('.')[0].split('/')[-1]
        saveB = path + '/' + fitMethod + '_' + self.bFile.split('.')[0].split('/')[-1]

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
        default to baseDir/accumulatedSpeckles/
        TO DO
        - add checks that the accumulated/binned files exist before trying
            to save them
        '''
        if path is None:
            path = self.baseDir + 'accumulated/'

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
