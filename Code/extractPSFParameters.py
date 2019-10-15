import numpy as np
from astropy.io import fits
import galsim
import pandas as pd
from lmfit import Minimizer, Parameters
import pickle
import imageHelper as imgHelper

# assumptions:
# 1. Only input numbers of files of useable data (i.e. already tossed files that are have e.g. blurred frames)
# 2. Have already run some rudimentary CoM and FWHM estimation, stored in paths args.COM and args.FWHM (one each for a and b filters) 
# 3. Pixel masks (if any) are a dict stored in the args.pixelMask path
# 4. There exists directories needed for args.savePath that are formattable according to (in order): pixel size binning, color, filename, number of PSFs 


def extractPSFParameters(args):
    # load in fileNames (these should just be file numbers not complete names!)
    inputFileNumbers = np.loadtxt(args.baseDir + args.fileNumbers, delimiter=',', dtype='str')
    
    # load in CoM and FWHM files   
    with open(args.baseDir + args.com.format('B'), 'rb') as file:
        comB = pickle.load(file)
    with open(args.baseDir + args.fwhm.format('B'), 'rb') as file:
        fwhmB = pickle.load(file)

    with open(args.baseDir + args.com.format('A'), 'rb') as file:
        comA = pickle.load(file)
    with open(args.baseDir + args.fwhm.format('A'), 'rb') as file:
        fwhmA = pickle.load(file)
    
    # use these to make an array of accepted datasets
    acceptedFileNumbers = [fn for fn in inputFileNumbers 
                           if comA[f'img_a_{fn}.fits'] + fwhmA[f'img_a_{fn}.fits'] < 128 
                           and comB[f'img_b_{fn}.fits'] + fwhmB[f'img_b_{fn}.fits'] < 128] 
        
    # if source is data, load in pixelMasks and gain conversions
    if args.source == 'data':
        with open(args.baseDir + args.masks, 'rb') as file:
            pixelMasks = pickle.load(file)
    
        with open(args.baseDir + args.gain, 'rb') as file:
            gains_e = dict(pickle.load(file))
    else:
        pixelMasks = None
    
    # define a dict to store the centroid calculated later
    centroidDict = {'a':{}, 'b':{}}

    for fileNumber in acceptedFileNumbers:
        for color in ['a', 'b']:
            # load in raw speckle data from fits file
            hdu = fits.open(args.dataDir + args.fileNameFormat.format(color, fileNumber))
            data = hdu[0].data.astype('float64')
            hdu.close()
            
            # apply gain and subtract background
            if args.source == 'data':
                data *= gains_e[fileNumber]
                if color == 'a':
                    data = data[:,:,::-1]
                # if source is data, will input a pixelMask file. Is there anything for this dataset?
                try:
                    maskDict = pixelMasks[f'img_{color}_{fileNumber}.fits']
                except:
                    maskDict = None
            else: 
                maskDict = None

            imgHelper.subtractBackground(data, maskDict)
            
            # calculate 200 centroids throughout the dataset, each on a stack of 5 exposures
#             centroidDict[color][fileNumber] = imgHelper.calculateCentroids(data, N=200, subtract=True)
        
            ## accumulate stacked PSFs for DSSI pixels
            dssi_series = []
            
            # 12x5s stacks
            dssi_series.append( imgHelper.accumulateExposures(sequence=data, numBins=12,
                                                              subtract=False, 
                                                              maskDict=maskDict,
                                                              overRide=True) )
#             # 4x15s stacks
#             dssi_series.append( imgHelper.accumulateExposures(sequence=data, numBins=4,
#                                                            subtract=False, 
#                                                            maskDict=maskDict) )
#             # 2x30s stacks
#             dssi_series.append( imgHelper.accumulateExposures(sequence=data, numBins=2,
#                                                            subtract=False, 
#                                                            maskDict=maskDict) )
#             # full 60s stack: save 15 images at 2^N exposure time
#             indices = [int(np.round(i)) - 1 for i in np.logspace(0, 3, 15)]

#             dssi_series.append( imgHelper.accumulateExposures(sequence=data, 
#                                                            subtract=False, 
#                                                            maskDict=maskDict,
#                                                            indices=indices) )

            ## accumulate for LSST pixels   
            # if no mask for dataset, easy: just spatially bin the already processed sequences
            if maskDict is None:
                lsst_series = [
                    [np.array([imgHelper.spatialBinToLSST(img) for img in dssi_series[i][0]]), 0] 
                    for i in range(len(dssi_series))]

            # if there are masks, then have to spatially bin first (using masks) and then reprocess.
            else: 
                lsst_data = np.array([imgHelper.spatialBinToLSST(data[i]) for i in range(1000)])
                for i in maskDict.keys():
                    lsst_data[i] = imgHelper.spatialBinToLSST(data[i], expMask=maskDict[i])
                    
                lsst_series = []
                # 12x5s stacks
                lsst_series.append( imgHelper.accumulateExposures(sequence=lsst_data, numBins=12,
                                                               subtract=False, overRide=True) )
#                 # 4x15s stacks
#                 lsst_series.append( imgHelper.accumulateExposures(sequence=lsst_data, numBins=4,
#                                                                subtract=False) )
#                 # 2x30s stacks
#                 lsst_series.append( imgHelper.accumulateExposures(sequence=lsst_data, numBins=2,
#                                                                subtract=False) )
#                 # full 60s stack
#                 lsst_series.append( imgHelper.accumulateExposures(sequence=lsst_data, indices=indices,
#                                                                subtract=False) )
                
            names = ['12']#, '4', '2', '15']
            ## extract PSF parameters
            for i in range(len(names)):
                # for DSSI
                savePathDSSI = args.baseDir + args.savePath.format('DSSI', color, fileNumber, names[i])
                imgHelper.estimateMomentsHSM(dssi_series[i][0], maskDict=dssi_series[i][1], 
                                          saveDict={'save':True, 'path':savePathDSSI})
                # for LSST
                savePathLSST = args.baseDir + args.savePath.format('LSST', color, fileNumber, names[i])
                imgHelper.estimateMomentsHSM(lsst_series[i][0], max_ashift=15, 
                                          saveDict={'save':True, 'path':savePathLSST},
                                          strict=False)

#     # save the dict of all centroid fits to a pickle file
#     with open(args.baseDir + 'Fits/centroids.p', 'wb') as file:
#         pickle.dump(centroidDict, file)

    
    
if __name__ == '__main__':
    from argparse import ArgumentParser, RawDescriptionHelpFormatter
    parser = ArgumentParser()

    parser.add_argument("--dataDir", type=str, default='/global/cscratch1/sd/chebert/rawSpeckles/', 
                        help="Path to data directory")
    parser.add_argument("--baseDir", type=str, default='/global/homes/c/chebert/SpecklePSF/', 
                        help="Path to base directory for e.g. code and saving")
    parser.add_argument("--savePath", type=str, default='Fits/{}pixels/{}Filter/img{}_{}psfs.p', 
                        help="Path to save, from baseDir")
    
    parser.add_argument("--masks", type=str, default='Code/pixelMasks.p', 
                        help="Path to pixel mask file, from baseDir directory")
    parser.add_argument("--gain", type=str, default='Code/eConversionWithGain.p', 
                        help="Path to gain file, from baseDir directory")
    parser.add_argument("--fileNumbers", type=str, default='Code/fileNumbers.txt', 
                        help="Path to file numbers file, from baseDir directory")

    parser.add_argument("--com", type=str, default='Fits/centerOfMass{}.p', 
                        help="Path to CoM file, from baseDir directory")
    parser.add_argument("--fwhm", type=str, default='Fits/fwhm{}.p', 
                        help="Path to FWHM file, from baseDir directory")

    parser.add_argument("--source", type=str, default='data', help='Is the input data or simulation')
    parser.add_argument("--fileNameFormat", type=str, default='img_{}_{}.fits',
                       help='Format for data files, from dataDir directory')

    args = parser.parse_args()

    extractPSFParameters(args)