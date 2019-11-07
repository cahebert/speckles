import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D
import analysisHelper as helper
import seaborn

class psfParameters():
    '''
    Class to load and analyze PSF parameters extracted from speckle images
    '''
    def __init__(self, source, 
                 fileNumbers='Code/{}fileNumbers.txt', 
                 baseDir='/global/homes/c/chebert/SpecklePSF/', 
                 filters=[692, 880],
                 plottingColors=['royalblue','darkorange']):
        '''
        Initialize an instance of this class for analyzing HSM outpute for speckle images.
        This class is associated with datasets from two wavelengths at a given pixel scale
        '''
        self.filters = filters
        self.source = source
        
        if source=='Zorro':
            self.scale = {self.filters[0]: .00992, self.filters[1]: .01095}
        else:
            self.scale = {self.filters[0]: .011, self.filters[1]: .011}
            
        self.base = baseDir
        self.fileNumbers = np.loadtxt(self.base + fileNumbers.format(self.source), delimiter=',', dtype='str')
        
        self.parameters = {psfN:{} for psfN in ['2', '4', '12', '15']}
        
        self.eDropoff = {}
        self.bootstrapE = {}
        
        self.R = {psfN:{} for psfN in ['2', '4', '12']}
        self.bootstrapR = {psfN:{} for psfN in ['2', '4', '12']}
        
        # plotting settings
        self.colors = {self.filters[0]: plottingColors[0], self.filters[1]: plottingColors[1]}
            
    def loadParameterSet(self, psfN, filePath=None):
        '''
        Load in a set of HSM parameters corresponding to a particular pixel size and data bins. 
        filePath is path (from self.base) of (formattable) name of pickle file containing HSM outputs.
        '''
        if filePath is None:
            if self.source=='Zorro':
                filePath='Fits/{}/filter{}/{}_{}psfs.p'
            elif self.source=='DSSI':
                filePath='Fits/{}/{}Filter/img{}_{}psfs.p'
        for color in self.filters:
            self.parameters[psfN][color] = {} 
            
            shears = np.zeros((2, len(self.fileNumbers), int(psfN)))
            sizes = np.zeros((len(self.fileNumbers), int(psfN)))
            rho4s = np.zeros((len(self.fileNumbers), int(psfN)))
            
            unusedFiles = 0
            for i in range(len(self.fileNumbers)):
                try:
                    with open(self.base + filePath.format(self.source, color, self.fileNumbers[i], psfN), 'rb') as file:
                        hsmResult = pickle.load(file)
                except FileNotFoundError:
                    unusedFiles += 1
                    print(f'file {self.fileNumbers[i]} was not found!')
                    # if fileNotFound, assume the dataset was rejected (may not be the best way of doing this)
                    continue
                    
                if (np.array([hsmResult[j].error_message != '' for j in range(int(psfN))])).any():
                    # do something 
                    print(f'dataset {self.fileNumbers[i]} has an HSM error in moments estimation!')
                else:
                    shears[:, i-unusedFiles, :] = np.array([[hsmResult[j].observed_shape.g1, 
                                                             hsmResult[j].observed_shape.g2] 
                                                            for j in range(int(psfN))]).T
                    sizes[i-unusedFiles, :] = np.array([hsmResult[j].moments_sigma for j in range(int(psfN))])
                    rho4s[i-unusedFiles, :] = np.array([hsmResult[j].moments_rho4 for j in range(int(psfN))])
            
            self.parameters[psfN][color]['g1'] = shears[0]
            self.parameters[psfN][color]['g2'] = shears[1]
            self.parameters[psfN][color]['g'] = np.sqrt(shears[0]**2 + shears[1]**2)
            self.parameters[psfN][color]['size'] = sizes * 2.355 * self.scale[color]
            self.parameters[psfN][color]['rho4'] = rho4s
            
            if unusedFiles != 0:
                print(f'beware, {unusedFiles} files (for filter {color}) were not loaded in correctly!')
            
    def loadAllParameters(self, filePath=None):
        '''
        Load all parameter sets by calling self.loadParameterSet()
        '''
        for psfN in ['2', '4', '12', '15']:
            self.loadParameterSet(psfN, filePath=filePath)
        
    def analyzeBinnedParameters(self, B=1000):
        '''
        For all binned data, compute parameter correlations coefficients and (if not 5s bins) bootstrap errors
        '''
        for psfN in ['2', '4', '12']:
            if self.filters[0] not in self.parameters[psfN].keys():
                self.loadParameterSet(psfN)
            
        for psfN in ['2', '4', '12']:
            # Compute correlation coefficients for g1 and g2 in both filters
            self.R[psfN] = helper.corrDict(self.parameters[psfN], filters=self.filters, parameter='ellipticity')
            
            if psfN != '12':
                # Bootstrap correlation coefficients for g1 and g2 in both filters
                self.bootstrapR[psfN] = helper.corrDict(self.parameters[psfN], filters=self.filters, 
                                                        parameter='ellipticity', bootstrap=True, B=B)
            
            # Repeat for size parameter
            self.R[psfN]['size'] = helper.corrDict(self.parameters[psfN], filters=self.filters, parameter='size')
            if psfN != '12':
                self.bootstrapR[psfN]['size'] = helper.corrDict(self.parameters[psfN], filters=self.filters, 
                                                                parameter='size', bootstrap=True, B=B)

            
    def plot30sParameters(self, psfN, alpha=0.6, fontsize=12, plotArgs=None, limits=(-.18,.11),
                             figsize=(11,4), save=False, ellipse=False, ellipseArgs=None):
        '''
        Plot 30s PSF parameters and their correlation ellipses.
        '''
        # plot correlation of 30s PSFs
        plt.figure(figsize=figsize)
        ax1 = plt.subplot(131)
        ax2 = plt.subplot(132)
        ax3 = plt.subplot(133)
        
        for param in ['g1', 'g2', 'size']:
            ax = ax1 if param == 'g1' else ax2 if param == 'g2' else ax3
            for color in self.filters:
                # scatter plot the ellipticity component values
                x = self.parameters['2'][color][param][:,0]
                y = self.parameters['2'][color][param][:,1]
                marker = 'o' if color==self.filters[1] else '^'
                ax.plot(x, y, marker, color=self.colors[color], alpha=alpha, **plotArgs)
                
                if ellipse:
                    # sigma of the bootstrap correlation coefficients
                    sigma_plot = np.std(self.bootstrapR['2'][param][color])
                    # add ellipse + add label with r +/- above sigma
                    helper.pearsonEllipse(self.R['2'][param][color], ax, 
                                           fr"$\rho$={self.R['2'][param][color]:.2f}$\pm${sigma_plot:.2f}",
                                           x.mean(), y.mean(), x.std(), y.std(),
                                           edgecolor=self.colors[color], ellipseArgs=ellipseArgs)
                    
            ax.set_xlabel('FWHM (30s) [arcsec]' if param=='size' else param + ' (30s)', fontsize=fontsize)
            ax.set_ylabel('FWHM (next 30s) [arcsec]' if param=='size' else param + ' (next 30s)', fontsize=fontsize)
            lims = ax.axis(option='equal')
            lims = np.min(lims), np.max(lims)
            ax.set(xlim=lims, ylim=lims)
            ax.legend(frameon=False)
            
        plt.tight_layout()
        if save:
            plt.savefig(f'../Plots/{self.source}/Results/30sParameters.png', bbox_to_inches='tight', dpi=200)
            plt.close()
        else:
            plt.show()

            
    def plotEComps(self, figsize=(10,5), limits=[-.28,.24], fontsize=12, save=False):
        '''
        Scatter plot PSF ellipticity components against each other for 4 exposure times.
        Illustration of the clouds of parameters shrinking with exposure time
        '''
        try:
            e1 = np.array([self.parameters['15'][self.filters[0]]['g1'], self.parameters['15'][self.filters[1]]['g1']])
            e2 = np.array([self.parameters['15'][self.filters[0]]['g2'], self.parameters['15'][self.filters[1]]['g2']])
        except KeyError:
            print("Make sure you've loaded in the correct dataset!")
            
        times = [.06, 1, 14, 60]
        idx = [0, 6, 11, 14, 0, 6, 11, 14]
        
        fig = plt.figure(figsize=figsize)
        for j in range(1,9):
            if j<=4:
                color = self.filters[0]
                k = 0
            else:
                color = self.filters[1]
                k = 1
            a = fig.add_subplot(2, 4, j)

            # plot g1 vs g2 
            a.plot(e1[k,:,idx[j-1]], e2[k,:,idx[j-1]], 'o', ms=4, alpha=0.65, color=self.colors[color])

            if j not in [1,5]: a.set_yticks([])
            if k != 1: 
                a.set_xticks([])
                a.set_title(str(times[j-1]) + ' sec')

            a.set_ylim(limits), a.set_xlim(limits)
            a.axhline(0, linestyle='--', color='gray'), a.axvline(0, linestyle='--', color='gray')

            if j in [4,8]:
                legend_elements = [Line2D([0], [0], color=self.colors[color], lw=0, marker='o', label=f'{color}nm')]
                plt.legend(frameon=False,handles=legend_elements)
        fig.text(0.5, 0.00, '$g_1$', fontsize=12, ha='center', va='center')
        fig.text(0.00, 0.5, '$g_2$', fontsize=12, ha='center', va='center', rotation='vertical')
        plt.tight_layout();
        if save: 
            plt.savefig(f'../Plots/{self.source}/Results/ellipticityComponents.png', dpi=200, bbox_inches='tight')
            plt.close(fig)
    
    def analyzeEMag(self, Nboot=1000, expectedAsymptote=None, save=False, plot=True, delay=False, overlay=False):
        '''
        Fit the ellipticity dropoff power law, and bootstrap for uncertainties. 
        Optional parameter expectedAsymptote can force a nonzero asymptotic value of |g|. 
            This should be None, or a tuple of values for filter a and b respectively.
        Optional: delay parameter, includes a time delay before the ellipticity drop-off begins.
        '''
        if self.filters[0] not in self.parameters['15'].keys():
            self.loadParameterSet('15')

        ellipticity = np.array([self.parameters['15'][self.filters[0]]['g'], self.parameters['15'][self.filters[1]]['g']])
        if expectedAsymptote:
                expectedAsymptote = np.array([np.sqrt(self.parameters['15'][self.filters[0]]['g1'][:,-1].mean()**2 +
                                                      self.parameters['15'][self.filters[0]]['g2'][:,-1].mean()**2),
                                              np.sqrt(self.parameters['15'][self.filters[1]]['g1'][:,-1].mean()**2 +
                                                      self.parameters['15'][self.filters[1]]['g2'][:,-1].mean()**2)])
            
        self.eDropoff['zero'] = helper.fitDropoff(ellipticity.mean(axis=1), delay=delay)
        # bootstrap the zero asymptote case, save the sample parameter values to dict
        self.bootstrapE['zero'] = helper.bootstrapDropoff(ellipticity, B=Nboot, delay=delay)
        
        # if desired, do the same with asymptote = g60s
        if expectedAsymptote is not None:
            self.eDropoff['g60'] = helper.fitDropoff(ellipticity.mean(axis=1),
                                                     expectedAsymptote=expectedAsymptote,
                                                     delay=delay)
            self.bootstrapE['g60'] = helper.bootstrapDropoff(ellipticity, B=Nboot, 
                                                             expectedAsymptote=expectedAsymptote,
                                                             delay=delay)
        
        # if user wants the plotted results:
        if plot:
            self.plotEMag(ellipticity, save=save, expectedAsymptote=expectedAsymptote, delay=delay, overlay=overlay)
        return
    
    def plotEMag(self, ellipticity=None, figsize=(8, 7), save=False, fontsize=12, overlay=False,
                 limits=(-0.01,.165), pairAlpha=0.15, expectedAsymptote=None, delay=False):
        '''
        Plot ellipciticy dropoff computed using analyzeEMag above
        '''
        # load some data, check it's in the class
        if self.filters[0] not in self.parameters['15'].keys():
            self.loadParameterSet('15')
        
        # check that the bootstrap estimates have been calculated in order to quote uncertainty
        try:
            paramStdsZero = self.bootstrapE['zero'].std(axis=1)
            if expectedAsymptote is not None: 
                paramStdsG60 = self.bootstrapE['g60'].std(axis=1)
        except KeyError:
            "Perform the bootstrap before you plot the results!"
        
        fig = plt.figure(1, figsize=figsize)
        pts = np.logspace(-1.22,1.79,15)

        for i in range(2):
            color = self.filters[i]
            # plot the box plot of ellipticity data 
            ax, logAx, bp = helper.makeBoxPlot(fig, 211 if i==0 else 212, 
                                               self.parameters['15'][color]['g'], 
                                               mainColor=self.colors[color], 
                                               meanColor='#0a481e' if i==0 else 'sienna', 
                                               xLabel=False if i==0 else True, 
                                               hline=False)
            # plot the power law
            logAx.plot(pts, 
                       helper.powerLaw(pts, self.eDropoff['zero'][i]), 
                       color='gray', alpha=0.75, 
                       label=r'$\alpha_0={:.2f} \pm {:.2f}$'.format(self.eDropoff['zero'][i,1], 
                                                                    paramStdsZero[i,1]))

            # do the same for the other asymptote if desired
            if expectedAsymptote is not None: 
                logAx.plot(pts, expectedAsymptote[i] * np.ones(15), 
                           ':', color=self.colors[color], alpha=0.5)
                logAx.plot(pts, 
                           helper.powerLaw(pts, self.eDropoff['g60'][i], asymptote=expectedAsymptote[i]), 
                           '--', color='gray', alpha=0.75, 
                           label=r'$\alpha_{{g60}}={:.2f} \pm {:.2f}$'.format(self.eDropoff['g60'][i,1], 
                                                                              paramStdsG60[i,1]))

                if overlay:
                    ps = np.copy(self.eDropoff['g60'][i])
                    ps[1] = -0.5
                    logAx.plot(pts, helper.powerLaw(pts, ps, asymptote=expectedAsymptote[i]), color='m')
                
            # make some legends
            if i == 0:
                ax.legend([bp["boxes"][0], bp["whiskers"][0], bp["means"][0]], 
                      ['inner 50th percentile', 'inner +/- 1$\sigma$', 'means'], frameon=False)
            logAx.legend(frameon=False, loc='upper center')

            ax.set_ylabel('|g|', fontsize=fontsize)
            ax.set_ylim(limits)
            logAx.set_ylim(limits)
            
        plt.tight_layout()
        if save:
            plt.savefig(f'../Plots/{self.source}/Results/ellipticityMag.png', dpi=200, bbox_to_inches='tight')
            plt.close(fig)
            

        # pair plot of bootstrap fit parameters
        # define some colormaps and other plotting settings
        r = 'YlOrBr'
        b = 'Greys'
        s = 8
        expected_args = {'linestyles':'dashed', 'linewidths':2, 'alpha':1, 'n_levels':5}
        zero_args={'linewidths':2, 'alpha':0.75, 'n_levels':5}

        # loop over colors: plot parameter fits from zero asymptote, and g60 asymptote if indicated.
        for i in range(2):
            if i == 0:
                # first entry is color, second is bootstrap sample, and third is parameter
                g = seaborn.JointGrid(x=self.bootstrapE['zero'][i][:,1], 
                                      y=self.bootstrapE['zero'][i][:,0],
                                      height=5);
            if i == 1:
                g.x = self.bootstrapE['zero'][i][:,1];
                g.y = self.bootstrapE['zero'][i][:,0];

            g = g.plot_joint(seaborn.kdeplot, cmap=b if i == 0 else r, **zero_args);
            g = g.plot_joint(plt.scatter, color=self.colors[self.filters[i]], alpha=pairAlpha, s=s);
            g = g.plot_marginals(seaborn.kdeplot, color=self.colors[self.filters[i]]);

            if expectedAsymptote is not None:
                g.x = self.bootstrapE['g60'][i][:,1];
                g.y = self.bootstrapE['g60'][i][:,0];
                g = g.plot_joint(seaborn.kdeplot, cmap=b if i == 0 else r, **expected_args);
                g = g.plot_joint(plt.scatter, color=self.colors[self.filters[i]], alpha=pairAlpha, s=s);
                g = g.plot_marginals(seaborn.kdeplot, color=self.colors[self.filters[i]], linestyle='--');  

        # only put the legend on if there are two different parameter sets
        if expectedAsymptote is not None:
            legend_elements = [Line2D([0], [0], color='gray', lw=2, linestyle='-', label='0'),
                               Line2D([0], [0], color='gray', lw=2, linestyle='--', label='$g_{60s}$')]
            g.ax_joint.legend(frameon=False, handles=legend_elements, 
                              fontsize=12, title='asymptote fixed at:', loc='best')
        g.set_axis_labels(r'exponent $\alpha$', 'amplitude a', fontsize=12);
        if save:
            g.savefig(f'../Plots/{self.source}/Results/cornerDropoffParams.png', dpi=200, bbox_to_inches='tight')
        return
    
    def plotCentroids(self, centroidFile='../Fits/{}/centroids.p', save=False, alpha=.85,
                      figsize=(8,7), fontsize=12, ms=5, B=1000, labelPos=(0.04,.92)):
        '''
        Generate a plot of the impact of centroid motion on PSF parameters.
        '''
        # load centroid dict
        try:
            with open(centroidFile.format(self.source), 'rb') as file:
                centroidDict = pickle.load(file)
        except FileNotFoundError:
            print('Please make sure the file exists where you think it does!')
        
        # load in centroid and save their x and y second moments
        self.centroidSigma = {}
        self.centroidVar = {}
        for color in self.filters:
            x = np.array([centroid['x'] for (fileN, centroid) in centroidDict[color].items() if fileN in self.fileNumbers])
            y = np.array([centroid['y'] for (fileN, centroid) in centroidDict[color].items() if fileN in self.fileNumbers])
            self.centroidSigma[color] = [np.sum((x-x.mean(axis=1)[:,None])**2, axis=1)/x.shape[1] - 
                                         np.sum((y-y.mean(axis=1)[:,None])**2, axis=1)/y.shape[1],
                                         np.sum((x-x.mean(axis=1)[:,None])*(y-y.mean(axis=1)[:,None]), axis=1)/x.shape[1]]
            self.centroidVar[color] = np.sqrt(np.sum((x-x.mean(axis=1)[:,None])**2, axis=1)/x.shape[1] + 
                                              np.sum((y-y.mean(axis=1)[:,None])**2, axis=1)/y.shape[1])
        
        fig = plt.figure(figsize=figsize)
        grid = plt.GridSpec(2,2, wspace=.075, hspace=.075, top=.94)
        fig.suptitle('impact of PSF centroid second moments on ellipticity', fontsize=fontsize+2)
        
        for i in range(2):
            param = ['g1','g2'][i]
            y_a = self.parameters['15'][self.filters[0]][param][:,-1]
            y_b = self.parameters['15'][self.filters[1]][param][:,-1]
                        
            for j in [0,1]:
                x_a = self.centroidSigma[self.filters[0]][j]*self.scale[self.filters[0]]
                x_b = self.centroidSigma[self.filters[1]][j]*self.scale[self.filters[1]]
                
                # calculate correlation coefficients
                rho_a = np.corrcoef(y_a, x_a, rowvar=False)[0,-1]
                rho_b = np.corrcoef(y_b, x_b, rowvar=False)[0,-1]

                # bootstrap uncertainties for these 
                err_a = np.std(helper.bootstrapCorr(y_a, x_a, B=B))
                err_b = np.std(helper.bootstrapCorr(y_b, x_b, B=B))

                ax = plt.subplot(grid[i,j])
                # plot g_i vs difference of second moments for both filters
                ax.plot(x_a, y_a, 'o', ms=ms, color=self.colors[self.filters[0]], alpha=alpha)
                ax.plot(x_b, y_b, 'o', ms=ms, color=self.colors[self.filters[1]], alpha=alpha)
            
                # add text with correlation coefficients + bootstrapped errors (color coded text)
                labelA = fr'$\rho$={rho_a:.2f}$\pm${err_a:.2f}'
                labelB = fr'$\rho$={rho_b:.2f}$\pm${err_b:.2f}'
                ax.text(labelPos[0], labelPos[1], labelA, 
                        color=self.colors[self.filters[0]], transform=ax.transAxes, fontsize=12)
                ax.text(labelPos[0], labelPos[1]-.075, labelB, 
                        color=self.colors[self.filters[1]], transform=ax.transAxes, fontsize=12)

                # add lines through along 0s
                ax.axhline(0, linestyle='--', color='lightgray')
                ax.axvline(0, linestyle='--', color='lightgray')
                
                # axis labels and ticks
                if i==0 and j==0: 
                    ax.set_ylabel('g$_1$', fontsize=fontsize)
                    ax.set_xticklabels([]), ax.set_xticks([])
                if i==0 and j==1: 
                    ax.set_xticklabels([]), ax.set_xticks([])
                    ax.set_yticklabels([]), ax.set_yticks([])
                if i==1 and j==0:
                    ax.set_ylabel('g$_2$', fontsize=fontsize)
                    ax.set_xlabel('$\sigma_x^2$ - $\sigma_y^2$ [arcsec$^2$]', fontsize=fontsize)
                if i==1 and j==1:
                    ax.set_yticklabels([]), ax.set_yticks([])
                    ax.set_xlabel('$\sigma_{xy}^2$  [arcsec$^2$]', fontsize=fontsize)

        if save:
            plt.savefig(f'../Plots/{self.source}/Results/centroidSpread.png', bbox_to_inches='tight', dpi=200)
            plt.close()
        plt.show()
        
        # plot impact on PSF parameters
        plt.figure(figsize=(4.5,6))

        a = plt.subplot(211)
        for color in self.filters:
            plt.plot(self.centroidVar[color]*self.scale[color], self.parameters['15'][color]['size'][:,-1], 
                     'o', alpha=alpha, ms=ms, color=self.colors[color], label=f'{color}nm')
#         plt.plot(np.sqrt(self.centroidSigmas['b']['x']**2 + self.centroidSigmas['b']['y']**2), 
#                  self.parameters['15']['b']['size'][:,-1], 
#                  'o', alpha=alpha, ms=ms, color=self.col['b'], label='880nm')
        a.tick_params(labelbottom=False)
        plt.ylabel('FWHM [arcsec]', fontsize=fontsize)
        plt.legend(loc=4)

        plt.subplot(212)
        for color in self.filters:
            plt.plot(self.centroidVar[color]*self.scale[color], self.parameters['15'][color]['g'][:,-1], 
                 'o', alpha=alpha, ms=ms, color=self.colors[color], label=f'{color}nm')
#         plt.plot(np.sqrt(self.centroidSigmas['b']['x']**2 + self.centroidSigmas['b']['y']**2), 
#                  np.sqrt(self.parameters['15']['b']['g1'][:,-1]**2+self.parameters['15']['b']['g2'][:,-1]**2), 
#                  'o', alpha=alpha, ms=ms, color=self.col['b'], label='880nm')
        plt.ylabel('|g|', fontsize=fontsize)
        plt.xlabel('$\sqrt{\sigma_x^2 + \sigma_y^2}$ [arcsec]', fontsize=fontsize)
        
        plt.tight_layout()
        if save:
            plt.savefig(f'../Plots/{self.source}/Results/centroidSpreadImpact.png', bbox_to_inches='tight', dpi=200)
            plt.close()
        plt.show()
        
    def chromaticityPlots(self, figsize=(7,4), color='darkcyan', ms=5, save=False):
        '''
        Plot the values of the chromatic exponent against PSF size.
        '''
        size1 = self.parameters['15'][self.filters[0]]['size'][:,-1]
        size2 = self.parameters['15'][self.filters[1]]['size'][:,-1]
        lam1 = self.filters[0]
        lam2 = self.filters[1]
        b = (np.log(size1) - np.log(size2)) / (np.log(lam1) - np.log(lam2))
        self.b = b
        
        plt.figure(figsize=figsize)
        grid = plt.GridSpec(1, 4, wspace=0, hspace=0)

        plt.subplot(grid[:,:3])
        plt.plot(size1, b, 'o', color=color, ms=ms)
        plt.xlabel('FWHM at {}nm [arcsec]'.format(self.filters[0]), fontsize=12)
        plt.ylabel('b', fontsize=12)
        # Kolmogorov prediction: sizeA = (lamA/lamB)^b * sizeB
        plt.axhline(-0.2, linestyle='--', color='darkgrey', linewidth=1, label='Kolmogorov turbulence')
        plt.legend(frameon=False, fontsize=12)
#         plt.xticks([.4,.6,.8,1.,1.2, 1.])

        ax = plt.subplot(grid[:,3:])
        plt.axhline(b.mean(), linestyle='--', color=color)
        plt.hist(b, histtype='step', bins=10, color=color, orientation="horizontal")
        plt.hist(b, bins=10, color=color, alpha=0.15, orientation="horizontal")
        ax.text(2, b.mean()+.02, f'$-${abs(b.mean()):.1f}$\pm${b.std():.2f}', fontsize=12, color=color)
        ax.text(2, b.mean()+.02, f'$-${abs(b.mean()):.1f}$\pm${b.std():.2f}', 
                fontsize=12, alpha=0.75, color='darkslategrey')
        plt.yticks([],[])
        plt.axhline(-0.2, linestyle='--', color='darkgrey', linewidth=1, label='Kolmogorov turbulence')

        plt.tight_layout()
        if save:
            plt.savefig(f'../Plots/{self.source}/Results/chromaticity.png', bbox_to_inches='tight', dpi=200)
        plt.show()
        
        
    def plotCorrelations(self, psfN, nSplit, nSplit2=None, save=False, 
                         ylims=None, figsize=(10.5,3.5), colors=None, alpha=1):
        '''
        Plot correlation function for data bins. 
        Specify 5 or 15s bins, and how many data points to include in each seeing split
        '''
        # check that data is loaded
        if self.filters[0] not in self.parameters[psfN].keys():
            print('loading in correct dataset...')
            self.loadParameterSet(psfN=psfN)

        if colors is None:
            colors=['steelblue','darkorange']
            
        # sort datasets by size (small-big) using average size between filters
        idxSort = np.argsort(self.parameters[psfN][self.filters[0]]['size'].mean(axis=1))
        
        # save size at which we will split the data, for use later
        sizeSplit = self.parameters[psfN][self.filters[0]]['size'].mean(axis=1)[idxSort[nSplit]]
        
        if nSplit2 is not None:
            sizeSplit2 = self.parameters[psfN][
                self.filters[0]]['size'].mean(axis=1)[idxSort[nSplit2]]
        else: 
            nSplit2 = nSplit
            sizeSplit2 = sizeSplit
            
        # split data into good and bad seeing samples, and compute correlation coefficients
        goodSeeing = {c: {ellipticity: self.parameters[psfN][c][ellipticity][idxSort[:nSplit]] 
                          for ellipticity in ['g1','g2']} for c in self.filters}
        badSeeing = {c: {ellipticity: self.parameters[psfN][c][ellipticity][idxSort[nSplit2:]] 
                         for ellipticity in ['g1','g2']} for c in self.filters}
        goodRs = helper.corrDict(goodSeeing, 'ellipticity', filters=self.filters)
        badRs = helper.corrDict(badSeeing, 'ellipticity', filters=self.filters) 
        
        if psfN == '4':
            goodRBoot = helper.corrDict(goodSeeing, 'ellipticity', bootstrap=True, filters=self.filters, N=nSplit)
            badRBoot = helper.corrDict(badSeeing, 'ellipticity', bootstrap=True, 
                                       filters=self.filters, N=len(self.fileNames)-nSplit2)
           
        # set up figure
        plt.figure(figsize=figsize)
        ax1, ax2 = plt.subplot(121), plt.subplot(122)

        # plot 15s binned correlations using helper function
        if psfN == '4':
            if ylims is None: ylims=[0,1]

            helper.plotRvT(ax1, ax2, '4', self.filters[0], goodRs, badRs, colors,
                           goodBoot=goodRBoot, badBoot=badRBoot, alpha=alpha, ylims=ylims)
#             helper.plotRvT(ax1, ax2, '4', self.filters[1], goodRs, badRs, colors,
#                            goodBoot=goodRBoot, badBoot=badRBoot, alpha=alpha, ylims=ylims);
        
        # or plot 5s correlations
        if psfN == '12':
            if ylims is None: ylims=[-.2,1]
            
            helper.plotRvT(ax1, ax2, '12', self.filters[0], goodRs, badRs, colors, alpha=alpha, ylims=ylims)
#             helper.plotRvT(ax1, ax2, '12', self.filters[1], goodRs, badRs, colors, alpha=alpha, ylims=ylims);

        # add legends to axes
        leg1 = [Line2D([0], [0], color=colors[0], alpha=0.8, lw=0, marker='o', label=f'< {sizeSplit:.2f}arcsec'),
                Line2D([0], [0], color=colors[1], alpha=0.8, lw=0, marker='o', label=f'> {sizeSplit2:.2f}arcsec')]
        ax1.legend(frameon=False, handles=leg1, title=f'seeing at {self.filters[0]}nm:', fontsize=11)
        
#         leg2 = [Line2D([0], [0], color='gray', lw=0, marker='^', label=f'{self.filters[0]}nm')]#,
#                 Line2D([0], [0], color='gray', lw=0, marker='o', label=f'{self.filters[1]}nm')]
#         ax2.legend(frameon=False, handles=leg2);
        
        plt.tight_layout()
        # save if desired.
        if save:
            plt.savefig(f'../Plots/{self.source}/Results/correlation_{int(60/int(psfN))}sbins_{sizeSplit:.2f}"cut.png',
                        bbox_to_inches='tight', dpi=200);
        else: 
            plt.show()
