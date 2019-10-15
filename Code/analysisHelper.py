import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
import sklearn.utils
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms

def pearsonEllipse(pearson, ax, label, mean_x, mean_y, scale_x, scale_y, edgecolor, ellipseArgs):
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

def corrDict(thing, parameter, bootstrap=False, B=1000, N=62):
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
        
def bootstrapCorr(thing1, thing2, B, N=62):
    idx = range(len(thing1))
    samples = []
    for i in range(B):
        resampledIdx = bootstrap(idx, N=N)
        samples.append(np.corrcoef(thing1[resampledIdx], thing2[resampledIdx], rowvar=False)[0,-1])
    return samples
    
def bootstrap(thing, N=62):
    return sklearn.utils.resample(thing, replace=True, n_samples=N)
    
def powerLaw(t, alpha, a, asymptote=0):
    return a * t**alpha + asymptote
    
def fitDropoff(ellipticity, pts=np.logspace(-1.22,1.79,15), expectedAsymptote=None):  
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
    N = ellipticity.shape[1]
    bootstrapParameters = np.empty((2, B, 2))
    for b in range(B):
        samples = [bootstrap(ellipticity[i]) for i in range(2)]
        y = np.mean(samples, axis=1)
        bootstrapParameters[:, b, :] = fitDropoff(y, expectedAsymptote=expectedAsymptote)
    return bootstrapParameters

def addExpTimeAxis(fig, subplotN, fntsize=12, label=True, tickLabels=True):
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
