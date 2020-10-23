import numpy as np
import matplotlib.pyplot as pyplot

def scatterg1g2(pobj, filt, axes, limits=[-.28,.24], fontsize=12, labelson=True):
    '''
    Scatter plot PSF ellipticity components against each other for 4 exposure times.
    Illustration of the clouds of parameters shrinking with exposure time
    '''

    g1 = pobj.g1[filt]
    g2 = pobj.g2[filt]
         
    times = [.06, 1, 14, 60]
    idx = [0, 6, 11, 14]

    if len(axes) != 4: raise ValueError('please pass in a list of 4 ax objects.')
    
    for i, a in enumerate(axes):
        # plot g1 vs g2 
        a.plot(g1[:,idx[i]], g2[:,idx[i]], 'o', ms=4, alpha=0.65, color=pobj.fdict[filt])

        if i: a.set_yticks([])
        if labelson: 
            a.set_xticks([])
            a.set_title(str(times[i]) + ' sec')

        a.set_ylim(limits), a.set_xlim(limits)
        a.axhline(0, linestyle='--', color='gray')
        a.axvline(0, linestyle='--', color='gray')

        if i == 3:
            legend_elements = [Line2D([0], [0], color=self.colors[color], lw=0, marker='o', label=f'{color}nm')]
            plt.legend(frameon=False,handles=legend_elements)
    a[0].set_ylabel('$g_2$', fontsize=fontsize)
    a[2].set_ylabel('$g_1$', fontsize=fontsize, position=(0,0))

def scatter30s(pobj, axes, pltcol, alpha=0.6, fontsize=12, plotArgs=None, limits=(-.18,.11),
                         ellipse=False, ellipseArgs=None):
    '''
    Plot 30s PSF parameters and their correlation ellipses.
    '''
    if len(axes) != 3: raise ValueError('please pass in list of 3 ax objects.')

    for i, param in enumerate([pobj.g1, pobj.g2, pobj.size]):
        for color in pobj.fdict.keys:
            # scatter plot the ellipticity component values
            x = param[color][:,0]
            y = param[color][:,1]
            marker = 'o' if color=='r' else '^'
            axes[i].plot(x, y, marker, color=pltcol[color], alpha=alpha, **plotArgs)
            
            ### below this line is old code, still needs converting!!
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


# def pearsonEllipse(pearson, ax, label, mean_x, mean_y, scale_x, scale_y, edgecolor, ellipseArgs):
#     '''
#     plotting helper function: adds a correlation ellipse to given axis instance. 
#     Inputs:
#     - a pearson correlation coefficient 
#     - axis instance ax
#     - means/scales of the x and y data
#     - color for the ellipse
#     - optional dict of additional arguments for the ellipse
#     '''
#     x_radius = np.sqrt(1 + pearson)
#     y_radius = np.sqrt(1 - pearson)
#     # define ellipse with given customization
#     ellipse = Ellipse((0, 0), width=x_radius * 2, height=y_radius * 2, facecolor='None', 
#                       edgecolor=edgecolor, label=label, **ellipseArgs)
#     # transform ellipse by data means and standard deviation
#     transf = transforms.Affine2D().rotate_deg(45) \
#                                   .scale(scale_x, scale_y) \
#                                   .translate(mean_x, mean_y)
#     ellipse.set_transform(transf + ax.transData)
#     return ax.add_patch(ellipse)

# def addExpTimeAxis(fig, subplotN, fntsize=12, label=True, tickLabels=True):
#     '''
#     plotting helper: add a log time axis (for plotting accumulating ellipticity data)
#     '''
#     logAx = fig.add_subplot(subplotN, label="2", frame_on=False)
#     logAx.set_yticks([])
#     if label: 
#         logAx.set_xlabel('exposure time [sec]', fontsize=fntsize)
#     logAx.set_xscale('log')
#     logAx.set_xlim((0.055,.068*1000))
#     logAx.set_xticks([.06, 1, 10, 60])
#     if tickLabels: 
#         logAx.set_xticklabels([.06, 1, 10, 60])
#     else:
#         logAx.set_xticklabels([])
#     return logAx

# def makeBoxPlot(fig, subplotN, data, mainColor, meanColor, xLabel=True, hline=True, fliers=False):
#     '''
#     Plotting helper function: add a box plot of data onto subplotN of fig instance. 
#     Configured s.t. whiskers hold 2sigma of the data.
#     Specify colors mainColor and meanColor for the box face and mean markers respectively
#     '''
#     ax = fig.add_subplot(subplotN)
#     if hline:
#         plt.axhline(0, color='gray', linewidth=1, alpha=.75)

#     bp = ax.boxplot(data, whis=[15.9,84.1], showmeans=True, meanline=True,
#                     meanprops={'color':meanColor, 'linewidth':7}, 
#                     boxprops={'linewidth':1, 'color':mainColor, 'facecolor':mainColor, 'alpha':0.8}, 
#                     medianprops={'linewidth':0},
#                     sym='', widths=.2, patch_artist=True)
#     ax.set_xticks([])
#     ax.set_xticklabels([])
#     logAx = addExpTimeAxis(fig, subplotN, label=xLabel, tickLabels=xLabel)

#     for element in ['whiskers', 'caps']:
#         plt.setp(bp[element], color=mainColor, linewidth=2, alpha=.8)
#     if fliers: plt.setp(bp['fliers'], alpha=.75, ms=3, markeredgecolor=mainColor)
#     return ax, logAx, bp
