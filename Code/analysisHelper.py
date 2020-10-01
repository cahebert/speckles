import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
import sklearn.utils
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
from astropy.io import fits
import pickle
    
# def filterData(fileNames, headerFile, centroidFile, filterOutliers=False, fwhmWeight=1):
#     '''
#     Function to filter data: accept files that are:
#     - in the bright star catalog 
#     - satisfy a centroid and fwhm size cut
#     '''
#     # load in dict of centroids
#     with open(centroidFile, 'rb') as file:
#         centroidDict = pickle.load(file)

#     # load in dict of headers
#     with open(headerFile, 'rb') as file:
#         headerDict = pickle.load(file)
        
#     accepted = []
#     for fN in fileNames:
#         # throw out a few on purpose:
#         if fN == '190716Z0672' or fN == '190619Z0190': 
#             continue
            
#         if filterOutliers:
#             if fN == '190619Z0122' or fN == '190619Z0130':
#                 continue

#         # is object in bright star catalog
#         try:
#             if headerDict[fN]['object'][:2] != 'HR':
#                 continue
#         except KeyError:
#             print(f'{fN} was not in headerDict!')
#             continue
            
#         # is airmass < 1.3
#         try:
#             if headerDict[fN]['AIRMASS'] > 1.3:
#                 continue
#         except KeyError:
#             print(f'{fN} was not in headerDict!')
#             continue
            
#         # load in HSM fit file
#         with open('../Fits/{}/filter{}/{}_{}psfs.p'.format('Zorro', 562, fN, '15'), 'rb') as file:
#             hsmResult = pickle.load(file)

#         # check HSM result checks out:
#         if (np.array([hsmResult[j].error_message != '' for j in range(int(15))])).any():
#             continue

#         # check for centroid + size
#         fwhm = hsmResult[-1].moments_sigma * 2.355
#         try:
#             comR = np.sqrt((centroidDict[562][fN]['x']-128)**2 + (centroidDict[562][fN]['y']-128)**2).mean()
#         except KeyError:
#             print(f'{fN} was not in centroiDict!')
#             continue

#         if fwhm*fwhmWeight + comR > 128:
#             continue
            
#         accepted.append(fN)
        
#     return accepted


def plotRvT(ax1, ax2, psfN, color, goodSeeing, badSeeing, colors, goodBoot=None, badBoot=None,
            ylims=[0,.9], alpha=1):
    '''
    Meant as a helper function to be called from the analysis class. 
    Plot the correlation functions of g1 and g2, with the data split between good and bad seeing
    Inputs:
    - 2 axis instances
    - number of psf bins (expect either '4' or '12')
    - color: which filter you want to plot
    - goodSeeing and badSeeing, dicts containing the correlation coefficients.
    '''
    if psfN == '12':
        pairs = [[i for i in goodSeeing['g1'][color].keys() if i[0] == str(j) or (i=='1011' and j==10)] 
                 for j in range(11)]
        distances = [[pairs[i][j] for i in range(len(pairs)-j)] for j in range(11)]
        
        ptsG = np.arange(11)
        if color > 800: ptsG = [i + .2 for i in ptsG]
        ptsB = [i + 0 for i in ptsG]
    
    elif psfN == '4':
        distances = ['01', '12', '23', '02', '13', '03']
        ptsG = [0,0,0,1,1,2]
        if color > 800: ptsG = [i + .1 for i in ptsG]
        ptsB = [i + .05 for i in ptsG]
        
    else:
        assert False, 'please specify one of "4" or "12" for psfN!'
    
    # plot g1 and g2 on axes 1 and 2 respectively. 
    for i in range(2):
        param = ['g1', 'g2'][i]
        ax = [ax1, ax2][i]
        fmt = 'o' if color <800 else '^'

        if psfN == '4':
            ax.errorbar(ptsG, [goodSeeing[param][color][j] for j in distances], 
                        yerr = [np.std(goodBoot[param][color][j]) for j in distances],
                        fmt=fmt, color=colors[0], capsize=2, alpha=alpha)
            ax.errorbar(ptsB, [badSeeing[param][color][j] for j in distances],
                        yerr = [np.std(badBoot[param][color][j]) for j in distances],
                        fmt=fmt, color=colors[1], capsize=2, alpha=alpha)
            ax.set_xticks([0,1,2])
            ax.set_xticklabels(['15','30','45'])
        if psfN == '12':
            ax.errorbar(ptsG, [np.mean([goodSeeing[param][color][j] for j in sl]) for sl in distances], 
                        yerr=[np.std([goodSeeing[param][color][j] for j in sl]) for sl in distances],
                        fmt=fmt, color=colors[0], capsize=2, alpha=alpha)
            ax.errorbar(ptsB, [np.mean([badSeeing[param][color][j] for j in sl]) for sl in distances],
                        yerr=[np.std([badSeeing[param][color][j] for j in sl]) for sl in distances],
                        fmt=fmt, color=colors[1], capsize=2, alpha=alpha)
            ax.set_xticks([0,2,4,6,8,10])
            ax.set_xticklabels(['5','15','25','35','45','55'])
            
        if param == 'g1': ax.set_ylabel(r'$\rho$', fontsize=12)
        else: ax.set_yticklabels([])
            
        ax.set_xlabel('$\Delta$ t [s]', fontsize=12)
        ax.set_title('g$_1$' if param == 'g1' else 'g$_2$', fontsize=12)
        ax.set_ylim(ylims)
    return ax1, ax2

def imagePSF(fileN, save, filters=(562, 832), expTime=[0,1000], source='zorro',
             filePath='/global/cscratch1/sd/chebert/rawSpeckles/img_{}_{}.fits'):
    '''
    produce (and optionally save) an image of the 60s (or other) integrated PSF for data number fileN
    '''
    if source == 'dssi':
        cL1 = 'a'
        cL2 = 'b'
    elif source == 'zorro':
        cL1 = 'b' 
        cL2 = 'r'
    elif source == 'sim':
        cL1 = filters[0]
        cL2 = filters[1]

    hdu = fits.open(filePath.format(cL1, fileN))
    data1 = hdu[0].data
    hdu.close()
    hdu = fits.open(filePath.format(cL2, fileN))
    if source == 'sim': data2 = hdu[0].data
    else: data2 = hdu[0].data[:,:,::-1]
    hdu.close()
    
    plt.figure(figsize=(6,3))
    ax=plt.subplot(121)
    plt.imshow(data1[expTime[0]:expTime[1]].mean(axis=0), origin='lower', cmap='plasma')
    plt.xticks([0, 64, 128, 192, 256], [0, .625, 1.25, 1.874, 2.5])
    plt.yticks([0, 64, 128, 192, 256], [0, .625, 1.25, 1.874, 2.5])
    plt.ylabel('[arcsec]')
    plt.xlabel('[arcsec]')
    ax.text(5, 235, f'{filters[0]}nm', color='gold', fontsize=12)
    ax=plt.subplot(122)
    plt.imshow(data2[expTime[0]:expTime[1]].mean(axis=0), origin='lower', cmap='plasma')
    plt.xticks([0, 64, 128, 192, 256], [0, .625, 1.25, 1.874, 2.5])
    plt.yticks([0, 64, 128, 192, 256], [])
    ax.text(5, 235, f'{filters[1]}nm', color='gold', fontsize=12)
    plt.xlabel('[arcsec]')
    
    plt.tight_layout()
    if save:
        plt.savefig(f'../Plots/psfImage_{fileN}.png', bbox_to_inches='tight', dpi=200)
    plt.show()
    
def pearsonEllipse(pearson, ax, label, mean_x, mean_y, scale_x, scale_y, edgecolor, ellipseArgs):
    '''
    plotting helper function: adds a correlation ellipse to given axis instance. 
    Inputs:
    - a pearson correlation coefficient 
    - axis instance ax
    - means/scales of the x and y data
    - color for the ellipse
    - optional dict of additional arguments for the ellipse
    '''
    x_radius = np.sqrt(1 + pearson)
    y_radius = np.sqrt(1 - pearson)
    # define ellipse with given customization
    ellipse = Ellipse((0, 0), width=x_radius * 2, height=y_radius * 2, facecolor='None', 
                      edgecolor=edgecolor, label=label, **ellipseArgs)
    # transform ellipse by data means and standard deviation
    transf = transforms.Affine2D().rotate_deg(45) \
                                  .scale(scale_x, scale_y) \
                                  .translate(mean_x, mean_y)
    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)

# def corrDict(thing, parameter, filters=(562, 832), bootstrap=False, B=1000, N=61):
#     '''
#     Calculate correlation coefficients for PSF parameters
#     Can calculate these using bootstrap samples of original data
#     '''
#     if parameter == 'size':
#         nVars = thing[filters[0]][parameter].shape[1]
#         pairs = [l for k in [[(i,j) for j in range(i,nVars) if i!=j] for i in range(nVars)] for l in k]
#         if bootstrap:
#             if nVars == 2:
#                 corrDict = {c: bootstrapCorr(thing[c]['size'][:,0], thing[c]['size'][:,1], B, N)
#                                for c in filters}
#             else:
#                 corrDict = {c: {f'{i}{j}': bootstrapCorr(thing[c]['size'][:,i], thing[c]['size'][:,j], B, N)
#                                 for (i,j) in pairs} for c in filters}
#         else:
#             if nVars == 2:
#                 corrDict = {c: np.corrcoef(thing[c]['size'][:,0], thing[c]['size'][:,1], rowvar=False)[0,-1]
#                                for c in filters}
#             else:
#                 corrDict = {c: {f'{i}{j}': np.corrcoef(thing[c]['size'][:,i], thing[c]['size'][:,j], 
#                                                           rowvar=False)[0,-1]
#                                 for (i,j) in pairs} for c in filters}

#     elif parameter == 'ellipticity':
#         nVars = thing[filters[0]]['g1'].shape[1]
#         pairs = [l for k in [[(i,j) for j in range(i,nVars) if i!=j] for i in range(nVars)] for l in k]
#         if bootstrap:
#             if nVars == 2:
#                 corrDict = {ellipticity:
#                                {c: bootstrapCorr(thing[c][ellipticity][:,0], thing[c][ellipticity][:,1], B, N)
#                                for c in filters} for ellipticity in ['g1', 'g2']}
#             else:
#                 corrDict = {ellipticity:
#                                {c: {f'{i}{j}': bootstrapCorr(thing[c][ellipticity][:,i], 
#                                                              thing[c][ellipticity][:,j], B, N)
#                                 for (i,j) in pairs} for c in filters} for ellipticity in ['g1', 'g2']}
#         else:
#             if nVars == 2:
#                 corrDict = {ellipticity:
#                                {c: np.corrcoef(thing[c][ellipticity][:,0], 
#                                                thing[c][ellipticity][:,1], rowvar=False)[0,-1]
#                                for c in filters} for ellipticity in ['g1', 'g2']}
#             else:
#                 corrDict = {ellipticity:
#                                {c: {f'{i}{j}': np.corrcoef(thing[c][ellipticity][:,i], 
#                                                            thing[c][ellipticity][:,j], rowvar=False)[0,-1]
#                                 for (i,j) in pairs} for c in filters} for ellipticity in ['g1', 'g2']}
#     return corrDict
        
# def bootstrapCorr(thing1, thing2, B, N=61):
#     '''
#     Bootstrap a correlation coefficient between thing1 and thing2, sampling B times. Dataset length N.
#     '''
#     idx = range(len(thing1))
#     samples = []
#     for i in range(B):
#         resampledIdx = bootstrap(idx, N=N)
#         samples.append(np.corrcoef(thing1[resampledIdx], thing2[resampledIdx], rowvar=False)[0,-1])
#     return samples
    
# def bootstrap(thing, N=61):
#     '''
#     resample dataset thing of length N
#     '''
#     return sklearn.utils.resample(thing, replace=True, n_samples=N)
    
# def powerLaw(t, p, asymptote=0):
#     '''
#     return a power law at points t, with exponent alpha, amplitude a, and an optional asymptote.
#     '''
#     if len(p) == 2:
#         return p[0] * t**p[1] + asymptote
#     elif len(p) == 3:
#         return np.array([p[0] if time<p[2] else p[0] * (time-p[2])**p[1] for time in t]) + asymptote
    
# def fitDropoff(ellipticity, pts=np.logspace(-1.22,1.79,15), expectedAsymptote=None, delay=False):  
#     '''
#     Fit a powerlaw to ellipticity data and return the best fit parameters. 
#     Optionally can:
#     - fix the asymptotic value to a nonzero value
#     - have a delay in time before the dropoff starts
#     '''
#     if expectedAsymptote is None:
#         expectedAsymptote = np.zeros(2)
       
#     def powerLaw1(t, *p):
#         return powerLaw(t, p, asymptote=expectedAsymptote[0])
#     def powerLaw2(t, *p):
#         return powerLaw(t, p, asymptote=expectedAsymptote[1])

#     if delay:
#         fitParams = np.zeros((2,3))
#         p0 = [0.5, -.2, 0]
#         bounds = [[-np.inf,-1, 0], [np.inf, 0, 60.]]
#     else:
#         fitParams = np.zeros((2,2))
#         p0=[0.5, -.2]
#         bounds=[[-np.inf, -1], [np.inf, 0]]
        
#     for i in range(2):
#         if i == 0:
#             fun = powerLaw1
#         else: 
#             fun = powerLaw2
#         fitParams[i], _ = curve_fit(fun, xdata=pts, ydata=ellipticity[i], p0=p0, bounds=bounds)
    
#     return fitParams
    
# def bootstrapDropoff(ellipticity, B=100, pts=np.logspace(-1.22,1.79,15), expectedAsymptote=None, delay=False):
#     '''
#     Bootstrap ellipticity dropoff: return set of B best fit parameters from bootstrap samples drawn from data
#     '''
#     N = ellipticity.shape[1]
#     if delay:
#         bootstrapParameters = np.empty((2, B, 3))
#     else:
#         bootstrapParameters = np.empty((2, B, 2))
#     for b in range(B):
#         samples = [bootstrap(ellipticity[i]) for i in range(2)]
#         y = np.mean(samples, axis=1)
#         bootstrapParameters[:, b, :] = fitDropoff(y, expectedAsymptote=expectedAsymptote, delay=delay)
#     return bootstrapParameters

def addExpTimeAxis(fig, subplotN, fntsize=12, label=True, tickLabels=True):
    '''
    plotting helper: add a log time axis (for plotting accumulating ellipticity data)
    '''
    logAx = fig.add_subplot(subplotN, label="2", frame_on=False)
    logAx.set_yticks([])
    if label: 
        logAx.set_xlabel('exposure time [sec]', fontsize=fntsize)
    logAx.set_xscale('log')
    logAx.set_xlim((0.055,.068*1000))
    logAx.set_xticks([.06, 1, 10, 60])
    if tickLabels: 
        logAx.set_xticklabels([.06, 1, 10, 60])
    else:
        logAx.set_xticklabels([])
    return logAx

def makeBoxPlot(fig, subplotN, data, mainColor, meanColor, xLabel=True, hline=True, fliers=False):
    '''
    Plotting helper function: add a box plot of data onto subplotN of fig instance. 
    Configured s.t. whiskers hold 2sigma of the data.
    Specify colors mainColor and meanColor for the box face and mean markers respectively
    '''
    ax = fig.add_subplot(subplotN)
    if hline:
        plt.axhline(0, color='gray', linewidth=1, alpha=.75)

    bp = ax.boxplot(data, whis=[15.9,84.1], showmeans=True, meanline=True,
                    meanprops={'color':meanColor, 'linewidth':7}, 
                    boxprops={'linewidth':1, 'color':mainColor, 'facecolor':mainColor, 'alpha':0.8}, 
                    medianprops={'linewidth':0},
                    sym='', widths=.2, patch_artist=True)
    ax.set_xticks([])
    ax.set_xticklabels([])
    logAx = addExpTimeAxis(fig, subplotN, label=xLabel, tickLabels=xLabel)

    for element in ['whiskers', 'caps']:
        plt.setp(bp[element], color=mainColor, linewidth=2, alpha=.8)
    if fliers: plt.setp(bp['fliers'], alpha=.75, ms=3, markeredgecolor=mainColor)
    return ax, logAx, bp
