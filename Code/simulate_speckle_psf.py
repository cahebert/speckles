import galsim
import psfws
import numpy as np
from astropy.io import fits
import pandas as pd
import pickle
from pynverse import inversefunc

def r0_from_vk(fwhm, L0):
    '''
    given FWHM values (in arcsec), return the r0 values (in cm) as predicted 
    by Van Karman turbulence with the input L0
    '''
    def fwhm_vankarman(r0, L0):
        kolm_term = (0.976 * .5) / (4.85 * r0)
        vk_correction = np.sqrt(1 - 2.183 * (r0 / L0)**(0.356)) 
        return kolm_term * vk_correction    

    if type(fwhm) == np.ndarray:
        r0 = np.array([inversefunc(fwhm_vankarman, fwhm[i], 
                                   args=(L0[i]), domain=[.0001,1]) for i in range(500)])
    else:
        r0 = inversefunc(fwhm_vankarman, fwhm, args=(L0), domain=[.0001,1])

    return r0

def draw_atmosphere_params(args):
    '''
    draw inputs for galsim.Atmosphere simulation based on: 
    Tokovinin 2006 OTP model, seeing from LSST site characterization, and NOAA GFS model winds
    Inputs
    ======
    - random_state to seed generation (default is None)
    - save: where to save the dict of parameters, except if None (default)
    '''
    rng = np.random.default_rng(args.rnd_seed)
    
    ws = psfws.ParameterGenerator()
    index = ws.data_gl.index[:300]
    pt = rng.choice(index)
    
#     choose alt and az
    az = rng.uniform() * 360
    with open('./ZorroHRHeaders_05ThruMid07.p','rb') as file:
        headers = pickle.load(file)
    alt=-1
    while alt < 50:
        i = rng.choice([k for k in headers.keys()])
        airmass = headers[i]['AIRMASS']
        zenith = np.arccos(1/airmass) * 180/np.pi
        alt = 90-zenith
    
    params = ws.get_parameters(pt, skycoord=True, alt=alt, az=az)

    h = [h-ws.h0 for h in params['h']]
    h[0] += 0.2
    out = {'altitude': h}
    out['r0_weights'] = params['j']
    out['speed'] = params['speed'].flatten()
    out['direction'] = [i*galsim.degrees for i in params['phi'].flatten()]

    # use random seed to seed a gaussian random number generator
    gd = galsim.GaussianDeviate(galsim.BaseDeviate(args.rnd_seed))

    # outer scale, use Josh's ImSim numbers (truncated log normal, median at 25m):
    L0 = 0
    while L0 < 10.0 or L0 > 100:
        L0 = np.exp(gd() * .6 + np.log(25.))
    out['L0'] = [L0]

    # set parameters of log-normal seeing distribution:
    # from the LSST SRD (or similar document)
    s = .452
    mu = -.5174
    
    # set an r0 value
    fwhm = np.exp(gd() * s + mu)

    targetFWHM = fwhm * airmass**0.6 * (args.color/500.)**(-0.3)
    out['r0_500'] = [r0_from_vk(targetFWHM, L0)]

    if args.save_params_path is not None:
        saveout = out.copy()
        saveout['alt'] = alt
        saveout['az'] = az
        saveout['airmass'] = airmass
        with open(args.save_params_path, 'wb') as file:
            pickle.dump(saveout, file)
        
    ## or maybe store results in a dict that I can unzip/whatever directly into Atmosphere
    return out

def simulate_speckle_psf(args):
    '''
    Returns an atmospheric simulation (Van Karman) to replicate Zorro data. 
    Inputs
    ======
    - color: wavelength of the filter desired
    - rnd_seed: random seed for the number generators
    - scale: screen scale for the turbulence layer. TBD what the default should be for robust results
    - total_time: total exposure time (default: 60s)
    - params_save_path: path to save the parameters, except if None
    - exp_time: time for each exposure. Default is .06s
    Returns
    =======
    an array of simulated short exposure (60ms) PSFs
    '''
    # pixel scale depends on color!
    if args.color == 562:
        pixel_scale = .00992
    else:
        pixel_scale = .01095
    
    # GS quantities
    diameter = 8.1
    obscuration = 1.024/8.1
    nx, ny = 256, 256
    
    # draw random values of r0, r0_weights, altitude, speed, and direction
    atm_args = draw_atmosphere_params(args)
    
    rng = galsim.BaseDeviate(args.rnd_seed)

    # fix screen size according to 'wrap' method from appendix test
    screen_size = max(atm_args['speed']) * args.total_time / 3
    
    # number of exposures to fit in the total time
    N = int(args.total_time / args.exp_time) 
    dead_time = 4./1000
        
    # make the atmosphere object:
    atm = galsim.Atmosphere(screen_size=screen_size, rng=rng, **atm_args)
    
    if args.scale != 'default':
        for layer in atm:
            layer.screen_scale = args.scale * layer.screen_scale

    # for each exposure in the series, draw a PSF of exposure length exp_time
    psf_series = np.zeros((N, nx, ny))
    for n in range(N):
        psf = atm.makePSF(lam=args.color, t0=n*(args.exp_time + dead_time), exptime=args.exp_time, 
                          diam=diameter, obscuration=obscuration)
        
        psf_series[n] = psf.drawImage(nx=nx, ny=ny, scale=pixel_scale).array
        
    if args.save_psfs_path is not None:
        hdu = fits.PrimaryHDU(psf_series)
        hdu.writeto(args.save_psfs_path)  

    return psf_series
    

if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser() 
    
    # define the input parameters needed
    # Only color and save_psfs_path are required 
    parser.add_argument("--color", type=int, default=562, help='wavelength (in nm) of the desired image. Required.')
    parser.add_argument("--save_psfs_path", type=str, default=None, help='path to save psf series result')
    # optional inputs below:
    parser.add_argument("--exp_time", type=float, default=0.06,
                        help="Exposure time per frame (in seconds).  Default: 60ms")
    parser.add_argument("--scale", type=float, default=0.9, 
                        help='scale (in fractions of r0_500) for the phase screens. Default: 0.75')
    parser.add_argument("--total_time", type=float, default=60, help='total imaging time')
    parser.add_argument("--rnd_seed", type=int, default=None, help='random seed')
    parser.add_argument("--save_params_path", type=str, default=None, help='path to save input params, if any')
    args = parser.parse_args()

    simulate_speckle_psf(args)

