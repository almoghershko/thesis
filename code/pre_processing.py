# -*- coding: utf-8 -*-
"""
Created on Tue Mar 23 19:45:59 2021

@author: AlmogHershko
"""

import numpy as np
import scipy.signal
from astropy.io import fits
from scipy.interpolate import interp1d
import sfdmap
import os
import pathlib

# dust model for dereden_spectrum
wls_x = np.array([ 2600,  2700,  4110,  4670,  5470,  6000, 12200, 26500])
a_l_x = np.array([ 6.591,  6.265,  4.315,  3.806,  3.055,  2.688,  0.829,  0.265])
f_interp = interp1d(wls_x, a_l_x, kind="cubic")

# loading SFD map
file_path = pathlib.Path(__file__).parent.absolute()
ebvMap = sfdmap.SFDMap(os.path.join(file_path,'sfddata-master'))

def getUrl(g):
    """
    Creates the URL string for a given fits file.
    
    Parameters
    ----------
    g : pandas dataseries
        a row from the table. contains a single galaxy.

    Returns
    -------
    url : string
        the url for the fits file.
    """
    
    # Extract identifiers
    plate = str(g['plate']).zfill(4)
    mjd = str(g['mjd'])
    fiberid = str(g['fiberid']).zfill(4)
    run2d = str(g['run2d'])
    
    # Build url
    filename = '-'.join(['spec',plate,mjd,fiberid]) + '.fits'
    SDSSpath = 'sas/dr16/sdss/spectro/redux/' + run2d + '/spectra/lite/' + plate + '/'
    url = 'https://data.sdss.org/' + SDSSpath + filename
    
    return url

def dereden_spectrum_i(wl, spec, E_bv):
    """
    Dereddens a spectrum based on the given extinction_g value and Fitzpatric99 model
    IMPORTANT: the spectrum should be in the observer frame (do not correct for redshift)

    :param wl: 10 ** loglam
    :param spec: flux
    :param E_bv: m.ebv(ra,dec)
    :return:
    """

    a_l_all = f_interp(wl)
    #E_bv = extinction_g / 3.793
    A_lambda = E_bv * a_l_all
    C = 10 ** (A_lambda / 2.5)

    return spec * C, C

def de_redshift(wave, z):
    """
    Switch to rest frame wave length

    :param wave: wl in observer frame
    :param z: Red shift
    :return: wl in rest frame
    """
    wave = wave / (1 + z)
    return wave

def calcEbv_i(g, ebvMap, i):
    return i, ebvMap.ebv(g['ra'], g['dec'])

def extend_ivar(ivar, wl, wl_grid, C):
    """
    Extending the variance inverse by minimum operations to match the
    interpolation + median filter operations performed on the spectra.

    Parameters
    ----------
    ivar : numpy 1-D array
        variance inverse array from the fits file.
    wl : numpy 1-D array
        original wavelengths grid of the fits file.
    wl_grid : numpy 1-D array
        the wavelengths grid for interpolation.

    Returns
    -------
    ivar : numpy 1-D array
        extended variance inverse array.

    """
    
    # de-reddening
    ivar = ivar / (C ** 2)
    
    # adding zeros to the ivar before and after the real array to make sure
    # that ivar will be zeros outside the real wavelength array
    wl = np.append([0], wl)
    ivar = np.append([0], ivar)
    wl = np.append(wl, np.inf)
    ivar = np.append(ivar, 0)
    
    # extending for the interpolation
    j = 0
    I = np.zeros(shape=(wl_grid.shape[0],2))
    for i in range(len(wl_grid)):
        
        while wl_grid[i] >= wl[j+1]:
            j += 1
        
        if wl_grid[i] <= wl[j]:
            I[i,0] = j
            I[i,1] = j
        else: # wl_grid[i] > wl[j] and wl_grid[i] < wl[j+1]
            I[i,0] = j
            I[i,1] = j+1
    
    ivar = np.min(ivar[I.astype(int)], axis=1)
    
    # extending for the median filter
    #ivar = sp.ndimage.minimum_filter(ivar, size=5, mode='nearest')
    
    return ivar

def download_spectrum(g, pre_proc=False, wl_grid=None):

    url = getUrl(g)
    
    j = 0
    while j < 10:
        try:
    
            # download
            with fits.open(url, memmap=False, cache=False) as hdulist:
                data = hdulist[1].data    
            # Get flux values and wavelengths
            spec = data['flux']
            wl = np.array(10 ** data['loglam'], dtype=float)
            ivar = data['ivar']
            # Mark pixels where sky residual emission is known to be high
            ivar[np.bitwise_and(wl > 5565, wl < 5590)] = 0

            if pre_proc:
                # Deredenning spectrum
                _, ebv = calcEbv_i(g, ebvMap, 0)
                spec, C = dereden_spectrum_i(wl, spec, ebv)
                # De-redshift spectrum
                wl = de_redshift(wl, g['z'])
                # interpolate to grid and apply median filter (width of 2.5 angstroms)
                spec = np.interp(wl_grid, wl, spec, left=np.nan, right=np.nan)
                spec = scipy.signal.medfilt(spec, 5)            
                # Normalize spectra by median
                med_flux = np.nanmedian(spec)
                spec = np.divide(spec, med_flux)
                # extending and normalizing the inverse variance
                ivar = extend_ivar(ivar, wl, wl_grid, C)
                # updating wavelength vec
                wl = wl_grid
            
            return spec.astype(np.float32), wl.astype(np.float32), ivar.astype(np.float32)
                
        # If failed, try to re-download the .fits file and try again (up to 10 times)
        except Exception as e:
            j += 1
            if j == 10:
                print('  * Failed to get data *  \t' + 'URL: ' + url + '\t Error: ' + str(e))
                return np.array([]), np.array([]), np.array([])