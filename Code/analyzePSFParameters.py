import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D
import analysisHelper as helper
import seaborn
from palettable.cmocean.sequential import Amp_20

class psfParameters():
    '''
    Class to load and analyze PSF parameters extracted from speckle images
    '''
    def __init__(self, source, fileNumbers='Code/fileNumbers.txt', N=62, 
                 baseDir='/global/homes/c/chebert/SpecklePSF/', size='FWHM'):
        self.source = source
        self.base = baseDir
        self.fileNumbers = np.loadtxt(self.base + fileNumbers, delimiter=',', dtype='str')
        self.N = N
        self.size = size
        
        self.parameters = {psfN:{pix:{} for pix in ['LSST', 'DSSI']} for psfN in ['2', '4', '15']}
        self.eDropoff = {'DSSI': {}, 'LSST': {}}
        self.bootstrapE = {'DSSI': {}, 'LSST': {}}
        self.bootstrapR = {psfN:{'DSSI': {}, 'LSST': {}} for psfN in ['2', '4', '15']}
        self.R = {psfN:{pix:{} for pix in ['LSST', 'DSSI']} for psfN in ['2', '4']}
        
        # plotting settings
#         self.fontsize = 12
        self.col = {'a': 'royalblue', 'b': 'indianred'}
        
        self.tickLabels = [0, .7, 1.4, 2.1, 2.8]
        self.ticksLSST = [0, 3.25, 6.5, 9.75, 13]
        self.ticksDSSI = [0, 64, 128, 192, 256]
            
    def loadParameterSet(self, psfN, pix, filePath='Fits/{}pixels/{}Filter/img{}_{}psfs.p'):

        for color in ['a', 'b']:
            self.parameters[psfN][pix][color] = {} 
            
            shears = np.zeros((2, self.N, int(psfN)))
            sizes = np.zeros((self.N, int(psfN)))
            rho4s = np.zeros((self.N, int(psfN)))
            
            unusedFiles = 0
            for i in range(len(self.fileNumbers)):
                try:
                    with open(self.base + filePath.format(pix, color, self.fileNumbers[i], psfN), 'rb') as file:
                        hsmResult = pickle.load(file)
                except FileNotFoundError:
                    unusedFiles += 1
                    # if fileNotFound, assume the dataset was rejected (may not be the best way of doing this)
                    continue
                    
                if (np.array([hsmResult[j].error_message != '' for j in range(int(psfN))])).any():
                    # do something 
                    print(f'dataset {self.fileNumbers[i]} has an HSM error in moments estimation!')
                else:
                    shears[:, i-unusedFiles, :] = np.array([[hsmResult[j].observed_shape.g1, 
                                                             hsmResult[j].observed_shape.g2] for j in range(int(psfN))]).T
                    sizes[i-unusedFiles, :] = np.array([hsmResult[j].moments_sigma for j in range(int(psfN))])
                    rho4s[i-unusedFiles, :] = np.array([hsmResult[j].moments_rho4 for j in range(int(psfN))])
            
            self.parameters[psfN][pix][color]['g1'] = shears[0]
            self.parameters[psfN][pix][color]['g2'] = shears[1]
            if self.size=='FWHM':
                self.parameters[psfN][pix][color]['size'] = sizes * 2.355
            elif self.size=='HLR':
                self.parameters[psfN][pix][color]['size'] = sizes * 1.163
            else:
                self.parameters[psfN][pix][color]['size'] = sizes
                self.size = 'sigma (HSM)'
            self.parameters[psfN][pix][color]['rho4'] = rho4s
            
    def loadAllParameters(self, filePath='Fits/{}pixels/{}Filter/img{}_{}psfs.p'):
        
        for psfN in ['2', '4', '15']:
            for pix in ['DSSI', 'LSST']:
                self.loadParameterSet(psfN, pix)
        
    def analyzeBinnedParameters(self, pix, Nboot=1000):
        for psfN in ['2', '4']:
            if 'a' not in self.parameters[psfN][pix].keys():
                self.loadParameterSet(psfN, pix)
            
        for psfN in ['2', '4']:
            # Compute correlation coefficients for g1 and g2 in both filters
            self.R[psfN][pix] = helper.corrDict(self.parameters[psfN][pix], parameter='ellipticity')
            # Bootstrap correlation coefficients for g1 and g2 in both filters
            self.bootstrapR[psfN][pix] = helper.corrDict(self.parameters[psfN][pix], 
                                                         parameter='ellipticity', bootstrap=True)
            
            # Repeat for size parameter
            self.R[psfN][pix]['size'] = helper.corrDict(self.parameters[psfN][pix], parameter='size')
            self.bootstrapR[psfN][pix]['size'] = helper.corrDict(self.parameters[psfN][pix], 
                                                                 parameter='size', bootstrap=True)

            
    def plotBinnedParameters(self, pix, psfN, alpha=0.6, fontsize=12, limits=(-.18,.11), 
                             figsize=(11,4), save=False, ellipse=False, ellipseArgs=None):

        # plot correlation of 30s PSFs
        plt.figure(figsize=figsize)
        ax1 = plt.subplot(131)
        ax2 = plt.subplot(132)
        ax3 = plt.subplot(133)
        
        for param in ['g1', 'g2', 'size']:
            ax = ax1 if param == 'g1' else ax2 if param == 'g2' else ax3
            for color in ['a', 'b']:
                # scatter plot the ellipticity component values
                x = self.parameters['2'][pix][color][param][:,0]
                y = self.parameters['2'][pix][color][param][:,1]
                ax.plot(x, y, 'o', color=self.col[color], alpha=alpha)
                
                if ellipse:
                    # sigma of the bootstrap correlation coefficients
                    sigma_plot = np.std(self.bootstrapR['2'][pix][param][color])
                    # add ellipse + add label with r +/- above sigma
                    helper.pearsonEllipse(self.R['2'][pix][param][color], ax, 
                                           fr"$\rho$={self.R['2'][pix][param][color]:.2f}$\pm${sigma_plot:.2f}",
                                           x.mean(), y.mean(), x.std(), y.std(),
                                           edgecolor=self.col[color], ellipseArgs=ellipseArgs)
                    
            ax.set_xlabel(self.size + ' (30s) [pixels]' if param=='size' else param + ' (30s)', fontsize=fontsize)
            ax.set_ylabel(self.size + ' (next 30s) [pixels]' if param=='size' else param + ' (next 30s)', fontsize=fontsize)
            lims = ax.axis(option='equal')
            lims = np.min(lims), np.max(lims)
            ax.set(xlim=lims, ylim=lims)
            ax.legend(frameon=False)
            
        plt.tight_layout()
        if save:
            plt.savefig(f'../Plots/Results/30sParameters_{pix}.png', bbox_to_inches='tight', dpi=200)
            plt.close()
        else:
            plt.show()

            
    def plotEComps(self, pix, figsize=(10,5), limits=[-.28,.24], fontsize=12, save=False):
        try:
            e1 = np.array([self.parameters['15'][pix]['a']['g1'], self.parameters['15'][pix]['b']['g1']])
            e2 = np.array([self.parameters['15'][pix]['a']['g2'], self.parameters['15'][pix]['b']['g2']])
        except KeyError:
            print("Make sure you've loaded in the correct dataset!")
            
        times = [.06, 1, 14, 60]
        idx = [0, 6, 11, 14, 0, 6, 11, 14]
        fig = plt.figure(figsize=figsize)
        for j in range(1,9):
            if j<=4:
                color = 'a'
                k = 0
            else:
                color = 'b'
                k = 1
            a = fig.add_subplot(2, 4, j)

            # plot g1 vs g2 
            a.plot(e1[k,:,idx[j-1]], e2[k,:,idx[j-1]], 'o', ms=4, alpha=0.65, color=self.col[color])

            if j not in [1,5]: a.set_yticks([])
            if k != 1: 
                a.set_xticks([])
                a.set_title(str(times[j-1]) + ' sec')

            a.set_ylim(limits), a.set_xlim(limits)
            a.axhline(0, linestyle='--', color='gray')
            a.axvline(0, linestyle='--', color='gray')

            if j in [4,8]:
                label = '692nm' if color=='a' else '880nm'
                legend_elements = [Line2D([0], [0], color=self.col[color], lw=0, marker='o', label=label)]
                plt.legend(frameon=False,handles=legend_elements)
        fig.text(0.5, 0.00, '$g_1$', fontsize=12, ha='center', va='center')
        fig.text(0.00, 0.5, '$g_2$', fontsize=12, ha='center', va='center', rotation='vertical')
        plt.tight_layout();
        if save: 
            plt.savefig(f'../Plots/Results/ellipticityComponents_{pix}.png', dpi=200, bbox_inches='tight')
            plt.close(fig)
    
    def analyzeEMag(self, pix, Nboot=1000, expectedAsymptote=None, save=False, plot=True):
        try:
            ellipticity = np.array([np.sqrt(self.parameters['15'][pix]['a']['g1']**2 +
                                            self.parameters['15'][pix]['a']['g2']**2),
                                    np.sqrt(self.parameters['15'][pix]['b']['g1']**2 +
                                            self.parameters['15'][pix]['b']['g2']**2)])
            
            if expectedAsymptote:
                expectedAsymptote = np.array([np.sqrt(self.parameters['15'][pix]['a']['g1'][:,-1].mean()**2 +
                                                      self.parameters['15'][pix]['a']['g2'][:,-1].mean()**2),
                                              np.sqrt(self.parameters['15'][pix]['b']['g1'][:,-1].mean()**2 +
                                                      self.parameters['15'][pix]['b']['g2'][:,-1].mean()**2)])

        except KeyError:
            print("Make sure you've loaded in the correct dataset!")
            
        self.eDropoff[pix]['zero'] = helper.fitDropoff(ellipticity.mean(axis=1))
        
        # bootstrap the zero asymptote case, save the sample parameter values to dict
        self.bootstrapE[pix]['zero'] = helper.bootstrapDropoff(ellipticity, B=Nboot)
        
        # if desired, do the same with asymptote = g60s
        if expectedAsymptote is not None:
            self.eDropoff[pix]['g60'] = helper.fitDropoff(ellipticity.mean(axis=1),
                                                          expectedAsymptote=expectedAsymptote)
            
            self.bootstrapE[pix]['g60'] = helper.bootstrapDropoff(ellipticity, B=Nboot, 
                                                                  expectedAsymptote=expectedAsymptote)
        
        # if user wants the plotted results:
        if plot:
            self.plotEMag(pix, ellipticity, save=save, expectedAsymptote=expectedAsymptote)
        return
    
    def plotEMag(self, pix, ellipticity=None, figsize=(8, 7), save=False, fontsize=12, 
                 limits=[-0.01,.165], pairAlpha=0.15, expectedAsymptote=None):
        # load some data, check it's in the class
        try:
            paramsAZero = self.bootstrapE[pix]['zero'] 
            if expectedAsymptote is not None:
                paramsAG60 = self.bootstrapE[pix]['g60']
        except KeyError:
            "Perform the bootstrap before you plot the results!"

        if ellipticity is None:
            try:
                ellipticity = np.array([np.sqrt(self.parameters['15'][pix]['a']['g1']**2 +
                                                self.parameters['15'][pix]['a']['g2']**2),
                                        np.sqrt(self.parameters['15'][pix]['b']['g1']**2 +
                                                self.parameters['15'][pix]['b']['g2']**2)])
            except KeyError:
                print("Make sure you've loaded in the correct dataset!")
        
        fig = plt.figure(1, figsize=figsize)
        pts = np.logspace(-1.22,1.79,15)

        paramStdsZero = paramsAZero.std(axis=1)
        if expectedAsymptote is not None: 
            paramStdsG60 = paramsAG60.std(axis=1)

        
        # filter a
        for i in range(2):
            color = ['a', 'b'][i]
            # plot the box plot of ellipticity data 
            ax, logAx, bp = helper.makeBoxPlot(fig, 211 if i==0 else 212, 
                                               ellipticity[i], 
                                               mainColor=self.col[color], 
                                               meanColor='navy' if i==0 else 'maroon', 
                                               xLabel=False if i==0 else True, 
                                               hline=False)
            # plot the power law
            logAx.plot(pts, 
                       helper.powerLaw(pts, self.eDropoff[pix]['zero'][i,0], 
                                            self.eDropoff[pix]['zero'][i,1]), 
                       color='gray', alpha=0.75, 
                       label=r'$\alpha_0={:.2f} \pm {:.2f}$'.format(self.eDropoff[pix]['zero'][i,0], 
                                                                    paramStdsZero[i,0]))

            # do the same for the other asymptote if desired
            if expectedAsymptote is not None: 
                logAx.plot(pts, expectedAsymptote[i] * np.ones(15), 
                           ':', color=self.col[color], alpha=0.5)
                logAx.plot(pts, 
                           helper.powerLaw(pts, self.eDropoff[pix]['g60'][i,0], 
                                                self.eDropoff[pix]['g60'][i,1], expectedAsymptote[i]), 
                           '--', color='gray', alpha=0.75, 
                           label=r'$\alpha_{{g60}}={:.2f} \pm {:.2f}$'.format(self.eDropoff[pix]['g60'][i,0], 
                                                                              paramStdsG60[i,0]))

            # make some legends
            if color == 'a':
                ax.legend([bp["boxes"][0], bp["whiskers"][0], bp["means"][0]], 
                      ['inner 50th percentile', 'inner +/- 1$\sigma$', 'means'], frameon=False)
            logAx.legend(frameon=False, loc='upper center')

            ax.set_ylabel('|g|', fontsize=fontsize)
            ax.set_ylim(limits)
            logAx.set_ylim(limits)
        plt.tight_layout()
        if save:
            plt.savefig(f'../Plots/Results/ellipticityMag_{pix}.png', dpi=200, bbox_to_inches='tight')
            plt.close(fig)
            

        # pair plot of bootstrap fit parameters
        # define some colormaps and other plotting settings
        r = Amp_20.mpl_colormap
        b = 'Blues'
        s = 8
        expected_args = {'linestyles':'dashed', 'linewidths':2, 'alpha':1, 'n_levels':5}
        zero_args={'linewidths':2, 'alpha':0.75, 'n_levels':5}

        # loop over colors: plot parameter fits from zero asymptote, and g60 asymptote if indicated.
        for i in range(2):
            if i == 0:
                # first entry is color, second is bootstrap sample, and third is parameter
                g = seaborn.JointGrid(x=self.bootstrapE[pix]['zero'][i][:,0], 
                                      y=self.bootstrapE[pix]['zero'][i][:,1],
                                      height=5);
            if i == 1:
                g.x = self.bootstrapE[pix]['zero'][i][:,0];
                g.y = self.bootstrapE[pix]['zero'][i][:,1];

            g = g.plot_joint(seaborn.kdeplot, cmap=b if i == 0 else r, **zero_args);
            g = g.plot_joint(plt.scatter, color=self.col['a' if i==0 else 'b'], alpha=pairAlpha, s=s);
            g = g.plot_marginals(seaborn.kdeplot, color=self.col['a' if i==0 else 'b']);

            if expectedAsymptote is not None:
                g.x = self.bootstrapE[pix]['g60'][i][:,0];
                g.y = self.bootstrapE[pix]['g60'][i][:,1];
                g = g.plot_joint(seaborn.kdeplot, cmap=b if i == 0 else r, **expected_args);
                g = g.plot_joint(plt.scatter, color=self.col['a' if i==0 else 'b'], alpha=pairAlpha, s=s);
                g = g.plot_marginals(seaborn.kdeplot, color=self.col['a' if i==0 else 'b'], linestyle='--');  

        # only put the legend on if there are two different parameter sets
        if expectedAsymptote is not None:
            legend_elements = [Line2D([0], [0], color='gray', lw=2, linestyle='-', label='0'),
                               Line2D([0], [0], color='gray', lw=2, linestyle='--', label='$g_{60s}$')]
            g.ax_joint.legend(frameon=False, handles=legend_elements, 
                              fontsize=12, title='asymptote fixed at:', loc='best')
        g.set_axis_labels(r'exponent $\alpha$', 'amplitude a', fontsize=12);
        if save:
            g.savefig(f'../Plots/Results/cornerDropoffParams_{pix}.png', dpi=200, bbox_to_inches='tight')
            g.close()
        
        return
    
    def plotCentroids(self, centroidFile='../Fits/centroids.p', save=False, figsize=(7.5,6), fontsize=12, adjust=False, cmap='BrBG', s=15):
        pix = 'DSSI'
        try:
            with open(centroidFile, 'rb') as file:
                centroidDict = pickle.load(file)
        except FileNotFoundError:
            print('Please make sure the file exists where you think it does!')
        
        self.centroidSigmas = {'a':{}, 'b':{}}
        
        for (name, color) in [('x','a'),('x','b'),('y','a'),('y','b')]:
            x = np.array([centroid[name] for (fileN, centroid) in centroidDict[color].items()])
            self.centroidSigmas[color][name] = np.sqrt(np.sum((x-x.mean(axis=1)[:,None])**2, axis=1)/x.shape[1])
        
        a_lims = [np.min([self.centroidSigmas['a']['x'], self.centroidSigmas['a']['y']]),
                  np.max([self.centroidSigmas['a']['x'], self.centroidSigmas['a']['y']])]
        b_lims = [np.min([self.centroidSigmas['b']['x'], self.centroidSigmas['b']['y']]),
                  np.max([self.centroidSigmas['b']['x'], self.centroidSigmas['b']['y']])]

        plt.figure(figsize=figsize)
        
        for i in range(1,5):
            color = ['a', 'b', 'a' ,'b'][i-1]
            plt.subplot(2,2,i)
            g = self.parameters['15'][pix][color]['g1' if i<3 else 'g2'][:,-1]
            if adjust and i == 2:
                g -= g.mean()
            plt.scatter(self.centroidSigmas[color]['x'], self.centroidSigmas[color]['y'], 
                        alpha=0.8, s=s, c=g, cmap=cmap,vmin=-np.max(np.abs(g)), vmax=np.max(np.abs(g)))
            cb = plt.colorbar()
            if color=='a':
                plt.plot(a_lims, a_lims, '--', color='lightgray')
            else: 
                plt.plot(b_lims, b_lims, '--', color='lightgray')
            
            if i%2==1: 
                plt.ylabel('centroid $\sigma_y$ [pix]', fontsize=fontsize)
            if i<3: 
                cb.ax.set_title('g$_1$')
                if adjust and i==2:
                    cb.ax.set_title('g$_1$ (adjusted)')
                plt.title('692 nm' if color=='a' else '880 nm')
            else: 
                cb.ax.set_title('g$_2$')
                plt.xlabel('centroid $\sigma_x$ [pix]', fontsize=fontsize)
        plt.tight_layout()
        if save:
            plt.savefig('../Plots/Results/centroidSpread.png')
            plt.close()
        plt.show()
        
        # plot impact on PSF parameters
        plt.figure(figsize=(4,5.25))

        a = plt.subplot(211)
        plt.plot(np.sqrt(self.centroidSigmas['a']['x']**2 + self.centroidSigmas['a']['y']**2), 
                 self.parameters['15'][pix]['a']['size'][:,-1], 
                 'o', alpha=0.75, ms=4, color=self.col['a'], label='692nm')
        plt.plot(np.sqrt(self.centroidSigmas['b']['x']**2 + self.centroidSigmas['b']['y']**2), 
                 self.parameters['15'][pix]['b']['size'][:,-1], 
                 'o', alpha=0.75, ms=4, color=self.col['b'], label='880nm')
        a.tick_params(labelbottom=False)
        plt.ylabel(f'{self.size} [pixel]', fontsize=fontsize)
        plt.legend(loc=4)

        plt.subplot(212)
        plt.plot(np.sqrt(self.centroidSigmas['a']['x']**2 + self.centroidSigmas['a']['y']**2), 
                 np.sqrt(self.parameters['15'][pix]['a']['g1'][:,-1]**2+self.parameters['15'][pix]['a']['g2'][:,-1]**2), 
                 'o', alpha=0.75, ms=4, color=self.col['a'], label='692nm')
        plt.plot(np.sqrt(self.centroidSigmas['b']['x']**2 + self.centroidSigmas['b']['y']**2), 
                 np.sqrt(self.parameters['15'][pix]['b']['g1'][:,-1]**2+self.parameters['15'][pix]['b']['g2'][:,-1]**2), 
                 'o', alpha=0.75, ms=4, color=self.col['b'], label='880nm')
        plt.ylabel('|g|', fontsize=fontsize)
        plt.xlabel('centroid $\sigma$ [pixels]', fontsize=fontsize)
        
        plt.tight_layout()
        if save:
            plt.savefig('../Plots/Results/centroidSpreadImpact.png', dpi=200)
            plt.close()
        plt.show()