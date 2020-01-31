import numpy as np
from astropy.io import fits
import galsim
import pandas as pd
from lmfit import Minimizer, Parameters
import pickle
import imageHelper as imgHelper

# assumptions:
# 1. Only input numbers of files of useable data (i.e. already tossed files that are have e.g. blurred frames)
# 2. Pixel masks (if any) are a dict stored in the args.pixelMask path
# 3. There exists directories needed for args.savePath that are formattable according to (in order): pixel size binning, color, filename, number of PSFs 


def extractPSFParameters(args):
    # load in fileNames (these should just be file numbers not complete names!)
    inputFileNumbers = np.loadtxt(args.baseDir + args.fileNumbers.format(args.source), 
                                  delimiter=',', dtype='str')
                                  
    if args.source == 'Zorro':
        acceptedFileNumbers = inputFileNumbers
#         with open(args.baseDir + args.masks, 'rb') as file:
#             pixelMasks = pickle.load(file)        
    else:
        pixelMasks = None
        acceptedFileNumbers = [f.strip(' ') for f in inputFileNumbers]
    
    # try opening centroidDict. If it doesn't exist, make a new one.
    # centroidDict will store the centroids calculated later
    try:
        with open(args.baseDir + f'Fits/{args.source}centroids.p', 'rb') as file:
            centroidDict = pickle.load(file)
    except FileNotFoundError:
        centroidDict = {c:{} for c in args.filters}
    
    for fileNumber in acceptedFileNumbers:
        for color in args.filters:
            if args.source == 'Zorro': colorL = 'b' if color==562 else 'r' 
            if args.source == 'sim': colorL = 562 if color==562 else 832
                
            # load in raw speckle data from fits file
            try:
                hdu = fits.open(args.dataDir + args.fileNameFormat.format(fileNumber, colorL))
                data = hdu[0].data.astype('float64')
                header = hdu[0].header
                hdu.close()
            except FileNotFoundError:
                print(f'file {fileNumber}{colorL} not found! looked for it at: {args.dataDir +
                        args.fileNameFormat.format(fileNumber, colorL)}')
            except IOError:
                print(f'hmm, something weird happened when opening {fileNumber}{colorL}.')
                    
            if args.source == 'Zorro':
                if color == 832:
                    data = data[:,:,::-1]
                try:
                    maskDict = pixelMasks[f'img_{color}_{fileNumber}.fits']
                except:
                    maskDict = None
                # if source is data, will input a pixelMask file. Is there anything for this dataset?
                try:
                    maskDict = pixelMasks[f'img_{color}_{fileNumber}.fits']
                except:
                    maskDict = None
                
            else: 
                maskDict = None

            # call helper to subtract background counts from image data
            imgHelper.subtractBackground(data, maskDict)
            
            # calculate 200 centroids throughout the dataset, each on a stack of 5 exposures
            centroid_out = imgHelper.calculateCentroids(data, N=200, subtract=True)
            
            # check that centroid ran correctly. if not, don't save it to dict and delete any existing entry
            if centroid_out is False:
                print(f'HSM moments estimation had an error for PSF number {names[i]} in file {fileNumber}')
                try:
                    del centroidDict[color][fileNumber]
                except:
                    continue
            else: 
                # this will overwrite any old value of centroid if this dict was read in
                centroidDict[color][fileNumber] = centroid_out
        
            # accumulate stacked PSFs for speckle pixels
            speckle_series = []
            
            # 12x5s stacks
            speckle_series.append( imgHelper.accumulateExposures(sequence=data, numBins=12,
                                                              subtract=False, 
                                                              maskDict=maskDict,
                                                              overRide=True) )
            # 4x15s stacks
            speckle_series.append( imgHelper.accumulateExposures(sequence=data, numBins=4,
                                                              subtract=False, 
                                                              maskDict=maskDict) )
            # 2x30s stacks
            speckle_series.append( imgHelper.accumulateExposures(sequence=data, numBins=2,
                                                              subtract=False, 
                                                              maskDict=maskDict) )
            # full 60s stack: save 15 images at 2^N exposure time
            indices = [int(np.round(i)) - 1 for i in np.logspace(0, 3, 15)]

            speckle_series.append( imgHelper.accumulateExposures(sequence=data, 
                                                              subtract=False, 
                                                              maskDict=maskDict,
                                                              indices=indices) )

            if args.lsstpix:
                ## accumulate for LSST pixels   
                # if no mask for dataset, easy: just spatially bin the already processed sequences
                if maskDict is None:
                    lsst_series = [
                        [np.array([imgHelper.spatialBinToLSST(img) for img in speckle_series[i][0]]), 0] 
                        for i in range(len(speckle_series))]

                # if there are masks, then have to spatially bin first (using masks) and then reprocess.
                else: 
                    lsst_data = np.array([imgHelper.spatialBinToLSST(data[i]) for i in range(1000)])
                    for i in maskDict.keys():
                        lsst_data[i] = imgHelper.spatialBinToLSST(data[i], expMask=maskDict[i])

                    lsst_series = []
                    # 12x5s stacks
                    lsst_series.append( imgHelper.accumulateExposures(sequence=lsst_data, numBins=12,
                                                                   subtract=False, overRide=True) )
                    # 4x15s stacks
                    lsst_series.append( imgHelper.accumulateExposures(sequence=lsst_data, numBins=4,
                                                                   subtract=False) )
                    # 2x30s stacks
                    lsst_series.append( imgHelper.accumulateExposures(sequence=lsst_data, numBins=2,
                                                                   subtract=False) )
                    # full 60s stack
                    lsst_series.append( imgHelper.accumulateExposures(sequence=lsst_data, indices=indices,
                                                                   subtract=False) )
                
            names = ['12', '4', '2', '15']
            ## extract PSF parameters
            for i in range(len(names)):
                # for Speckle
                savePathSpeckle = args.baseDir + args.savePath.format(args.source, color, 
                                                                      fileNumber, names[i])
                success = imgHelper.estimateMomentsHSM(speckle_series[i][0], maskDict=speckle_series[i][1], 
                                              saveDict={'save':True, 'path':savePathSpeckle})
                if not success:
                    print(f'HSM moments estimation had an error for PSF number {names[i]} in file {fileNumber}')
                    
                if args.lsstpix:
                    # for LSST
                    savePathLSST = args.baseDir + args.savePath.format('LSST', color, fileNumber, names[i])
                    imgHelper.estimateMomentsHSM(lsst_series[i][0], max_ashift=15, 
                                              saveDict={'save':True, 'path':savePathLSST},
                                              strict=False)

    # save the dict of all centroid fits to a pickle file
    with open(args.baseDir + f'Fits/{args.source}centroids.p', 'wb') as file:
        pickle.dump(centroidDict, file)

#     if args.source == 'Zorro':
#         # save the dict of all header fits to a pickle file
#         with open(args.baseDir + f'Code/{args.source}headers.p', 'wb') as file:
#             pickle.dump(headerInfo, file)
    
    
if __name__ == '__main__':
    from argparse import ArgumentParser, RawDescriptionHelpFormatter
    parser = ArgumentParser()

    parser.add_argument("--dataDir", type=str, default='/global/cscratch1/sd/chebert/rawSpeckles/', 
                        help="Path to data directory")
    parser.add_argument("--baseDir", type=str, default='../', 
                        help="Path to base directory for e.g. code and saving")
    parser.add_argument("--savePath", type=str, default='Fits/{}/{}Filter/img{}_{}psfs.p', 
                        help="Path to save, from baseDir")
    
    parser.add_argument("--masks", type=str, default='Code/{}pixelMasks.p', 
                        help="Path to pixel mask file, from baseDir directory")
    parser.add_argument("--gain", type=str, default='Code/eConversionWithGain.p', 
                        help="Path to gain file, from baseDir directory")
    parser.add_argument("--fileNumbers", type=str, default='Code/{}fileNumbers.txt', 
                        help="Path to file numbers file, from baseDir directory")

    parser.add_argument("--source", type=str, default='Zorro', help='Is the input data or simulation')
    parser.add_argument("--fileNameFormat", type=str, default='img_{}_{}.fits',
                       help='Format for data files, from dataDir directory')

    parser.add_argument("--filters", type=tuple, default=(562,832),
                       help='Two filters for the data')
    parser.add_argument("--lsstpix", type=bool, default=False,
                       help='Whether to extract parameters from images binned to LSST pixels')

    args = parser.parse_args()

    extractPSFParameters(args)