import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
import sklearn.utils
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
from astropy.io import fits
    
def plotRvT(ax1, ax2, psfN, color, goodSeeing, badSeeing, colors, goodBoot=None, badBoot=None,
            ylims=[0,.9], alpha=1):
    '''
    Meant as a helper function to be called from the analysis class. 
    Plot the correlation functions of g1 and g2, with the data split between good and bad seeing
    Inputs:
    - 2 axis instances
    - number of psf bins (expect either '4' or '12')
    - color ('a' or 'b') corresponding to which filter you want to plot
    - goodSeeing and badSeeing, dicts containing the correlation coefficients.
    '''
    if psfN == '12':
        pairs = [[i for i in goodSeeing['g1']['a'].keys() if i[0] == str(j) or (i=='1011' and j==10)] 
                 for j in range(11)]
        distances = [[pairs[i][j] for i in range(len(pairs)-j)] for j in range(11)]
        
        ptsG = np.arange(11)
        if color == 'b': ptsG = [i + .2 for i in ptsG]
        ptsB = [i + 0 for i in ptsG]
    
    elif psfN == '4':
        distances = ['01', '12', '23', '02', '13', '03']
        ptsG = [0,0,0,1,1,2]
        if color == 'b': ptsG = [i + .1 for i in ptsG]
        ptsB = [i + .05 for i in ptsG]
        
    else:
        assert False, 'please specify one of "4" or "12" for psfN!'
    
    # plot g1 and g2 on axes 1 and 2 respectively. 
    for i in range(2):
        param = ['g1', 'g2'][i]
        ax = [ax1, ax2][i]
        fmt = '^' if color=='a' else 'o'

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

def imagePSF(fileN, save, filePath='/global/cscratch1/sd/chebert/rawSpeckles/img_{}_{}.fits'):
    '''
    produce (and optionally save) an image of the 60s integrated PSF for data number fileN
    '''
    hdu = fits.open(filePath.format('a', fileN))
    dataA = hdu[0].data[:,:,::-1]
    hdu.close()
    hdu = fits.open(filePath.format('b', fileN))
    dataB = hdu[0].data
    hdu.close()
    
    plt.figure(figsize=(6,3))
    ax=plt.subplot(121)
    plt.imshow(dataA.mean(axis=0), origin='lower', cmap='plasma')
    plt.xticks([0, 64, 128, 192, 256], [0, .7, 1.4, 2.1, 2.8])
    plt.yticks([0, 64, 128, 192, 256], [0, .7, 1.4, 2.1, 2.8])
    plt.ylabel('[arcsec]')
    plt.xlabel('[arcsec]')
    ax.text(5, 235, '692nm', color='gold', fontsize=12)
    ax=plt.subplot(122)
    plt.imshow(dataB.mean(axis=0), origin='lower', cmap='plasma')
    plt.xticks([0, 64, 128, 192, 256], [0, .7, 1.4, 2.1, 2.8])
    plt.yticks([0, 64, 128, 192, 256], [])
    ax.text(5, 235, '880nm', color='gold', fontsize=12)
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

def corrDict(thing, parameter, bootstrap=False, B=1000, N=61):
    '''
    Calculate correlation coefficients for PSF parameters
    Can calculate these using bootstrap samples of original data
    '''
    if parameter == 'size':
        nVars = thing['a'][parameter].shape[1]
        pairs = [l for k in [[(i,j) for j in range(i,nVars) if i!=j] for i in range(nVars)] for l in k]
        if bootstrap:
            if nVars == 2:
                corrDict = {c: bootstrapCorr(thing[c]['size'][:,0], thing[c]['size'][:,1], B, N)
                               for c in ['a', 'b']}
            else:
                corrDict = {c: {f'{i}{j}': bootstrapCorr(thing[c]['size'][:,i], thing[c]['size'][:,j], B, N)
                                for (i,j) in pairs} for c in ['a', 'b']}
        else:
            if nVars == 2:
                corrDict = {c: np.corrcoef(thing[c]['size'][:,0], thing[c]['size'][:,1], rowvar=False)[0,-1]
                               for c in ['a', 'b']}
            else:
                corrDict = {c: {f'{i}{j}': np.corrcoef(thing[c]['size'][:,i], thing[c]['size'][:,j], 
                                                          rowvar=False)[0,-1]
                                for (i,j) in pairs} for c in ['a', 'b']}

    elif parameter == 'ellipticity':
        nVars = thing['a']['g1'].shape[1]
        pairs = [l for k in [[(i,j) for j in range(i,nVars) if i!=j] for i in range(nVars)] for l in k]
        if bootstrap:
            if nVars == 2:
                corrDict = {ellipticity:
                               {c: bootstrapCorr(thing[c][ellipticity][:,0], thing[c][ellipticity][:,1], B, N)
                               for c in ['a', 'b']} for ellipticity in ['g1', 'g2']}
            else:
                corrDict = {ellipticity:
                               {c: {f'{i}{j}': bootstrapCorr(thing[c][ellipticity][:,i], 
                                                             thing[c][ellipticity][:,j], B, N)
                                for (i,j) in pairs} for c in ['a', 'b']} for ellipticity in ['g1', 'g2']}
        else:
            if nVars == 2:
                corrDict = {ellipticity:
                               {c: np.corrcoef(thing[c][ellipticity][:,0], 
                                               thing[c][ellipticity][:,1], rowvar=False)[0,-1]
                               for c in ['a', 'b']} for ellipticity in ['g1', 'g2']}
            else:
                corrDict = {ellipticity:
                               {c: {f'{i}{j}': np.corrcoef(thing[c][ellipticity][:,i], 
                                                           thing[c][ellipticity][:,j], rowvar=False)[0,-1]
                                for (i,j) in pairs} for c in ['a', 'b']} for ellipticity in ['g1', 'g2']}
    return corrDict
        
def bootstrapCorr(thing1, thing2, B, N=61):
    '''
    Bootstrap a correlation coefficient between thing1 and thing2, sampling B times. Dataset length N.
    '''
    idx = range(len(thing1))
    samples = []
    for i in range(B):
        resampledIdx = bootstrap(idx, N=N)
        samples.append(np.corrcoef(thing1[resampledIdx], thing2[resampledIdx], rowvar=False)[0,-1])
    return samples
    
def bootstrap(thing, N=61):
    '''
    resample dataset thing of length N
    '''
    return sklearn.utils.resample(thing, replace=True, n_samples=N)
    
def powerLaw(t, alpha, a, asymptote=0):
    '''
    return a power law at points t, with exponent alpha, amplitude a, and an optional asymptote.
    '''
    return a * t**alpha + asymptote
    
def fitDropoff(ellipticity, pts=np.logspace(-1.22,1.79,15), expectedAsymptote=None):  
    '''
    Fit a powerlaw to ellipticity data and return the best fit parameters. 
    Optionally can fix the asymptotic value to a nonzero value.
    '''
    if expectedAsymptote is None:
        expectedAsymptote = np.zeros(2)
        
    def powerLawA(t, alpha, a):
        return powerLaw(t, alpha, a, expectedAsymptote[0])
    def powerLawB(t, alpha, a):
        return powerLaw(t, alpha, a, expectedAsymptote[1])
    
    fitParams = np.zeros((2,2))
    for i in range(2):
        if i == 0:
            fun = powerLawA
        else: 
            fun = powerLawB
        fitParams[i], _ = curve_fit(fun, xdata=pts, ydata=ellipticity[i], p0=[-.2, 0.5], 
                                    bounds=[[-1,-np.inf], [0, np.inf]])
    return fitParams
    
def bootstrapDropoff(ellipticity, B=100, pts=np.logspace(-1.22,1.79,15), expectedAsymptote=None):
    '''
    Bootstrap ellipticity dropoff: return set of B best fit parameters from bootstrap samples drawn from data
    '''
    N = ellipticity.shape[1]
    bootstrapParameters = np.empty((2, B, 2))
    for b in range(B):
        samples = [bootstrap(ellipticity[i]) for i in range(2)]
        y = np.mean(samples, axis=1)
        bootstrapParameters[:, b, :] = fitDropoff(y, expectedAsymptote=expectedAsymptote)
    return bootstrapParameters

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
