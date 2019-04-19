import galsim
from matplotlib import pyplot as plt
plt.style.use(['dark_background'])
%matplotlib inline

import numpy as np
from SpecklePSF import SpeckleSeries
import pickle
import helperFunctions as helper
import pandas as pd

scratchdir = '/global/cscratch1/sd/chebert/'
saveDir = './../Fits/hsmFits'

# find all image files in the scratch directory
fileNames = ! ls /global/cscratch1/sd/chebert/rawSpeckles/ | grep 'img' 
fileNames = [f for f in fileNames if f not in ['img_a_004.fits', 'img_a_388.fits', 'img_a_389.fits',
                                               'img_b_004.fits', 'img_b_388.fits', 'img_b_389.fits']]
                                               
for file in fileNames:
    fileNumber = file.split('.')[0].split('_')[-1]
    test = SpeckleSeries(fileNumber, 'data', scratchdir)
    test.fitExposures(fitMethod='hsm', 
                      maxIters=10000, 
                      max_amoment=5.0e6, 
                      max_ashift=120, 
                      savePath=saveDir)