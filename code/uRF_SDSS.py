#!/usr/bin/env python2.7

import pandas as pd
import os
import urllib.request
import numpy as np
from scipy.interpolate import interp1d
from scipy import signal
from astropy.io import fits
#from sklearn.impute import SimpleImputer
import sfdmap
import multiprocessing as mp
from functools import partial
from time import time, sleep
from itertools import compress
import warnings
from joblib import Parallel, delayed
# from numba import jit

# Global variables
global_specs = np.nan
leafs = np.nan
realLeafs = np.nan

# dust model for dereden_spectrum
wls_x = np.array([ 2600,  2700,  4110,  4670,  5470,  6000, 12200, 26500])
a_l_x = np.array([ 6.591,  6.265,  4.315,  3.806,  3.055,  2.688,  0.829,  0.265])
f_interp = interp1d(wls_x, a_l_x, kind="cubic")


def getFitsFiles(gs, dbPath):

    """
    Downloads .fits files from SDSS dr14 for the galaxies specified in the gs DataFrame to dbPath (parallel version)
    :param gs: Pandas Dataframe: Containing the fields 'plate', 'mjd', 'fiberid', run2d'
    :param dbPath: String: Path to dB folder (where 'sas' folder is located).
    :param fitsFolder: String: fitsFolder name inside dbPath to place the .fits file
    :return: None.
    """

    if __name__ == 'uRF_SDSS':      # Wrap Parallelized code for Windows OS
        print ('*** Getting  %d fits files ****' % (len(gs)))
        start = time()

        # Create fits' DB folder, if needed
        if not os.path.exists(dbPath):
            os.makedirs(dbPath)

        # Download .fits files parallelly
        pool = mp.Pool()
        f = partial(getFitFile_i, gs, dbPath)
        res = pool.map(f, range(len(gs)))
        nFailed=sum(res)

        end = time()
        print('  Done! Finished downloading .fits files\t')
        print('  Failed to download: ' + str(nFailed))
        print('  Time: ' + str(round(end - start, 2)) + ' [s]')


def getFitFile_i(gs, dbPath, i):
    """
    Downloading the ith .fits file in the gs DataFrame to dbPath.
    :param gs: Pandas Dataframe: Containing the fields 'plate', 'mjd', 'fiberid', run2d'
    :param dbPath: String: Path to dB folder (where 'sas' folder is located).
    :param i: Int: The index of the galaxy's .fits file to download
    :return: 0, if succeeded to download. 1 if failed.
    """

    # Extract identifiers
    obj = gs.iloc[i]
    plate = str(obj['plate']).zfill(4)
    mjd = str(obj['mjd'])
    fiberid = str(obj['fiberid']).zfill(4)
    run2d = str(obj['run2d'])

    # Try & Except a failed Download
    try:
        getFitsFile(plate, mjd, fiberid, run2d, dbPath)
        return 0
    except Exception as e:
        print('Failed to DL: ' + 'https://data.sdss.org/' + 'sas/dr14/sdss/spectro/redux/' + run2d + '/spectra/lite/' + plate + '/' + '-'.join(['spec',plate,mjd,fiberid]) + '.fits')
        print(str(e))
        return 1


def getFitsFile(plate, mjd, fiberid, run2d, dbPath):

    """
    Download a .fits files by plate,mjd,fiberid & run2d to dBPath
    :param plate: String (4 digits): Plate number
    :param mjd: String: mjd
    :param fiberid: String (4 digits): fiber ID
    :param run2d: String: run2d
    :param dbPath: String: Path to dB folder (where 'sas' folder is located).
    :return: None.
    """

    # Calculate paths
    filename = '-'.join(['spec',plate,mjd,fiberid]) + '.fits'
    SDSSpath = 'sas/dr14/sdss/spectro/redux/' + run2d + '/spectra/lite/' + plate + '/'
    url = 'https://data.sdss.org/' + SDSSpath + filename
    folderPath = dbPath + SDSSpath
    dest = folderPath + filename

    # Create folder for the .fits file, if needed
    if not os.path.exists(folderPath):
        os.makedirs(folderPath)

    # Try to download the .fits file 10 times
    # A file under 60kB is considered a failure
    if not(os.path.isfile(dest)):
        urllib.request.urlretrieve(url, dest)

    j = 0
    while j < 10 and (os.path.getsize(dest) < 60000):
        os.remove(dest)
        urllib.request.urlretrieve(url, dest)
        j += 1
        sleep(1)

    if (os.path.getsize(dest) < 60000):
        print('Size is: ' + str(os.path.getsize(dest)) + ' '),
        os.remove(dest)
        raise Exception('Spectra wasn\'t found')


def calcEbv(gs, dbPath):

    """
    :param gs: Pandas DataFrame
    :param dbPath: String: Path to DB folder
    :return: numpy array 1-D: E(B-V) values for each galaxy in the gs DataFrame
    """
    if __name__ == 'uRF_SDSS':  # Wrap Parallelized code for Windows OS
        # Get dust map
        ebvMap = sfdmap.SFDMap(dbPath + '\\sfddata-master')

        # Calculate E(b-v) for each galaxy parallelly
        res = Parallel(n_jobs=-1, verbose=1)(
            delayed(calcEbv_i)(gs.iloc[i], ebvMap, i) for i in range(len(gs)))

        res = sorted(res, key=lambda x: x[0])

        return np.array([x[1] for x in res])


def calcEbv_i(g, ebvMap, i):
    return i, ebvMap.ebv(g['ra'], g['dec'])


def fitsToSpecs(gs, dbPath, nanRemovalThreshold=0.15):

    """
    :param gs: Pandas DataFrame
    :param dbPath: String: Path to DB folder
    :param nanRemovalThreshold: Float (range: (0,1)): Spectra which over nanRemovalThreshold of their spectrum are
            missing values will be discarded
    :return:
            gs: Pandas DataFrame: The remaining galaxies
            specs: numpy array 2-D: The spectra
            grid: numpy array 1-D: The wavelengths grid
            specobjids: numpyarray 1-D: specobjids from SpecObjAll table
    """

    if __name__ == 'uRF_SDSS':

        print('*** Getting Spectra from %d .fits files ***' % len(gs))
        print ('  Getting data from .fits files... ')
        start = time()

        # Get spectra from .fits files
        res = Parallel(n_jobs=-1, verbose=5)(
            delayed(fitsToSpecs_i)(gs.iloc[i], dbPath, i) for i in range(len(gs)))
        print('    Done!')

        # Sort & Filter only good results (non-empty arrays for wl,spec)
        print('  Sorting and filtering results... ')
        res = sorted(res, key=lambda x: x[0])
        goodRes = [len(val[2]) > 0 for val in res]
        gs = gs[goodRes]
        gs.index = range(len(gs))
        res = list(compress(res, goodRes))

        specobjids = []
        wls = []
        specsList = []
        for val in res:
            specobjids.append(val[1])
            wls.append(val[2])
            specsList.append(val[3])

        # Make sure that the galaxy DataFrame matches the specobjids
        assert np.array_equal(gs.specobjid, specobjids), 'Data mismatch'
        print('    Done! Succeeded to get data from %d .fits files' % sum(goodRes))

        # Calculate wavelengths grid
        print('  Calculating grid... ')
        grid = calcGrid(wls)
        print('    Done!')

        # Place spectra on grid & smooth
        print('  Manipulating spectra... ')

        res = Parallel(n_jobs=-1, verbose=5)(
            delayed(fitsToSpec_manipulate_i)(grid, wls[i], specsList[i], i) for i in range(len(gs)))

        res = sorted(res, key=lambda x: x[0])
        specs = np.array([x[1] for x in res])

        print('    Done!')

        print('  Removing spectra with many NaNs... '),
        maxNans = nanRemovalThreshold * len(grid)
        properNanCountMask = [np.sum(np.isnan(spec)) < maxNans for spec in list(specs)]

        specs = specs[properNanCountMask]
        specobjids = list(compress(specobjids, properNanCountMask))
        gs = gs[properNanCountMask]
        gs.index = range(len(gs))
        assert np.array_equal(gs.specobjid, specobjids), 'Data mismatch'
        print('Done! %d spectra left... ' % len(gs))

        # Normalize spectra by median
        gs['spec_median'] = np.nanmedian(specs, 1)
        specs = np.divide(specs.T, np.array(gs.spec_median)).T

        # Impute NaNs by median
        print('  Imputing NaNs... ',)
        specs = SimpleImputer(missing_values=np.nan, strategy='median').fit_transform(specs)
        print('Done!')

        end = time()
        print('  Done! Finished getting data from .fits files\t'),
        print('Time: ' + str(round(end - start, 2)) + ' [s]')
        return gs, specs, grid, specobjids


def fitsToSpecs_i(g, dbPath, i):
    """
    :param g: Pandas Seris: A galaxy
    :param dbPath: String: Path to DB folder
    :param i: The index of the galaxy in the gs DataFrame
    :return:
            i: The index of the galaxy in the gs DataFrame
            specobjid: Galaxy's specobjid from SpecObjAll table
            wl: numpy array 1-D: The wavelengths in Angstrom corresponding the flux (Empty if couldn't retrieve spectrum)
            spec: numpy array 1-D: flux values at each of the wavelengths (Empty if couldn't retrieve spectrum)
    """

    # Calculate galaxy's identifiers
    plate = str(g['plate']).zfill(4)
    mjd = str(g['mjd'])
    fiberid = str(g['fiberid']).zfill(4)
    run2d = str(g['run2d'])

    # Calculate path to .fits file
    filename = '-'.join(['spec', plate, mjd, fiberid]) + '.fits'
    SDSSpath = 'sas/dr14/sdss/spectro/redux/' + run2d + '/spectra/lite/' + plate + '/'

    dest = dbPath + SDSSpath + filename

    # Try to obtain the spectrum from the .fits file 10 times
    warnings.filterwarnings("error")
    j = 0
    while j < 10:
        try:
            # Get data from .fits file
            hdulist = fits.open(dest, memmap=False)
            data = hdulist[1].data
            specobjid = int(hdulist[2].data['specobjid'].item())
            hdulist.close()

            # Make sure the file matches the galaxy desired
            assert specobjid == g['specobjid'], 'Files do not match galaxies dataframe'

            # Get flux values and wavelengths
            spec = data['flux']
            wl = np.array(10 ** data['loglam'], dtype=float)

            # Remove pixels where sky is known to be high
            spec[np.bitwise_and(wl > 5565, wl < 5590)] = np.nan

            # Remove large relative uncertainties (where std>flux)
            ivar = np.copy(data['ivar'])
            ivar[ivar == 0] = np.nan
            var = np.divide(1., ivar)
            var[np.isnan(ivar)] = np.inf
            var = np.nan_to_num(var)
            spec_comp = np.nan_to_num(spec)
            highVarMask = np.greater(np.sqrt(var), spec_comp)
            spec[highVarMask] = np.nan

            # Deredenning spectrum
            ebv = g['ebv']
            spec = dereden_spectrum(wl, spec, ebv)

            # De-redshift spectrum
            wl = de_redshift(wl, g['z'])

            # Return result
            return i, specobjid, wl, spec

        # If failed, try to re-download the .fits file and try again (up to 10 times)
        except Exception as e:
            j += 1
            if not(os.path.isfile(dest)):
                getFitsFile(plate, mjd, fiberid, run2d, dbPath)
            else:
                os.remove(dest)
                getFitsFile(plate, mjd, fiberid, run2d, dbPath)
            if j == 10:
                print('  * Failed to get data *  \t' + 'Path: ' + dest + '\t Size: ' + str(os.path.getsize(dest))\
                      + '\t Error: ' + str(e))
                return i, g['specobjid'], np.array([]), np.array([])


def fitsToSpec_manipulate_i(grid, wl, spec, i):

    """
    :param grid: numpy array 1-D: Wavelengths mutual grid
    :param wl: numpy array 1-D: Current wavelengths
    :param spec: numpy array 1-D: Flux values for the current wavelengths
    :param i: int: Galaxy's index in the gs DataFrame
    :return:
            i: int: Galaxy's index in the gs DataFrame
            spec: numpy array 1-D: Smoothed Flux values at the grid wavelengths
    """

    # Treat warnings as errors
    warnings.filterwarnings("error")

    # Return the smoothed Flux values at the grid wavelengths
    try:
        return i, medfilt(same_grid_single(grid, wl, spec))
    except:
        return i, np.repeat(np.nan, len(grid))


def remove_cont(specs, grid, poly_order=5, bin_size=600):

    """
    :param specs: numpy array 2-D: The spectra
    :param grid: numpy array 1-D: The wavelengths grid
    :param poly_order: int: The order of the fitted polynomial
    :param bin_size: int: Number of wavelengths to include for each sample
    :return:
            spec_no_cont: numpy array 2-D: The spectra after its continuum was removed
            poly_coefs: numpy array 2-D: The polynomial coefficients for each spectrum
    """
    if __name__ == 'uRF_SDSS':  # Wrap Parallelized code for Windows OS
        # Define a global specs variable for an efficient parallelized computation
        global global_specs
        global_specs = np.copy(specs)

        # Spectra dimensions
        m, n = global_specs.shape

        # Calculate the wavelengths corresponding to the spectrum's samples
        spec_med_wl = [np.median(grid[j * bin_size:(j + 1) * bin_size]) for j in range(int(len(grid) / bin_size))]
        spec_med_wl = [np.median(grid[0:100])] + spec_med_wl + [np.median(grid[-100:])]

        # Calculate the spectrum with no continuum parallelly
        res = Parallel(n_jobs=1)(   # Change the number of jobs to -1 to use all available cores (for larger samples)
            delayed(remove_cont_i)(grid, spec_med_wl, poly_order, bin_size, i) for i in range(m))
        specs_no_cont = np.zeros(global_specs.shape)
        poly_coefs = np.zeros((m, poly_order + 1))
        for tup in res:
            i = tup[0]
            poly_coefs[i] = tup[1]
            specs_no_cont[i] = tup[2]

        return np.array(specs_no_cont, dtype=np.float16), poly_coefs


def remove_cont_i(grid, spec_med_wl, poly_order, bin_size, i):

    """
    :param grid: numpy array 1-D: The wavelengths grid
    :param spec_med_wl: numpy array 1-D: The wavelengths corresponding to the spectrum's samples
    :param poly_order: int: The order of the fitted polynomial
    :param bin_size: int: Number of wavelengths to include for each sample
    :param i: int: The index of the galaxy in the gs DataFrame
    :return:
            i: int: The index of the galaxy in the gs DataFrame
            poly_i_coef: numpy array 1-D: The polynomial coefficients used for the fit
            spec_no_cont: numpy array 1-D: The spectrum after continuum removal
    """
    # Choose the spectrum matching the ith galaxy
    spec = global_specs[i]

    # Sample the continuum according to the bin size (as well as 2 sample at a 100 bin-size at the edges)
    spec_med = [np.median(spec[j * bin_size:(j + 1) * bin_size]) for j in range(int(len(grid) / bin_size))]
    spec_med = [np.median(spec[:100])] + spec_med + [np.median(spec[-100:])]

    # Fit a polynomial of order poly_order to the samples
    poly_i_coef = np.polyfit(spec_med_wl, spec_med, poly_order)
    poly_i = np.poly1d(poly_i_coef)

    # Divide the flux by the polynomial values for every wavelength
    spec_no_cont = spec / poly_i(grid)

    return i, poly_i_coef, spec_no_cont


def calcGrid(wls):
    wls_min = [np.min(wl) for wl in wls]
    wls_max = [np.max(wl) for wl in wls]

    # wls_min = [np.min(wls[i][np.invert(np.isnan(specsList[i]))]) for i in xrange(len(wls))]
    # wls_max = [np.max(wls[i][np.invert(np.isnan(specsList[i]))]) for i in xrange(len(wls))]

    grid_min = np.percentile(wls_min, 99)
    grid_max = np.percentile(wls_max, 1)

    grid = np.arange(grid_min, grid_max, 0.5)


    print('    Grid min wavelength: %.2f [A]' % np.min(grid))
    print('    Grid max wavelength: %.2f [A]' % np.max(grid_max))

    print('    Number of features: %d' % len(grid))
    # grid = np.linspace(grid_min, grid_max, 10000)
    # grid = np.linspace(grid_min, grid_max, np.median([len(x) for x in wls]))

    return grid


def dereden_spectrum(wl, spec, E_bv):
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

    return spec * 10 ** (A_lambda / 2.5)


def de_redshift(wave, z):
    """
    Switch to rest frame wave length

    :param wave: wl in observer frame
    :param z: Red shift
    :return: wl in rest frame
    """
    wave = wave / (1 + z)
    return wave


def same_grid_single(wave_common, wave_orig, spec_orig):
    """
    Putting a single spectrum on the common wavelength grid

    :param wave_common: wl vector
    :param wave_orig:
    :param spec_orig:
    :return:
    """
    spec = np.interp(wave_common, wave_orig, spec_orig, left=np.nan, right=np.nan)

    return spec


def medfilt(spec, size=7):

    return signal.medfilt(spec, size)


def return_synthetic_data(X):
    """
    The function returns a matrix with the same dimensions as X but with synthetic data
    based on the marginal distributions of its features
    """
    features = len(X[0])
    X_syn = np.zeros(X.shape)

    for i in range(features):
        obs_vec = X[:,i]
        syn_vec = np.random.choice(obs_vec, len(obs_vec)) # here we chose the synthetic data to match the marginal distributions of the real data
        X_syn[:,i] += syn_vec

    return X_syn


def merge_work_and_synthetic_samples(X_real, X_syn):

    """
    The function merges the data into one sample, giving the label "1" to the real data and label "2" to the synthetic data
    """

    # build the labels vector
    Y = np.ones(len(X_real))
    Y_syn = np.ones(len(X_syn)) * 2

    Y_total = np.concatenate((Y, Y_syn)) # Classes vector
    X_total = np.concatenate((X_real, X_syn)) # Merged array
    return X_total, Y_total


def calcDisMat(rf, X_real):

    """
    :param rf: Random forest classifier object
    :param X_real: (Number of obs. ,Number of feat.) Matrix
    :return: (Number of obs., Number of obs.) Distance matrix (1 - Similarity matrix)

    *** Parallel Version ***

    The function builds the similarity matrix for the real observations X_syn based on the random forest we've trained.
    The matrix is normalized so that the highest similarity is 1 and the lowest is 0

    This function counts only leafs in which the object is classified as a "real" object,
    meaning that more than half of the observations at a leaf are "real".
    """

    if __name__ == 'uRF_SDSS':

        # Number of Observations
        n = len(X_real)

        # leafs: A row represents a galaxy, a column represents a tree (estimator).
        #        A cell's value is the index of the leaf where a galaxy ended up at a specific tree.
        print('  Applying forest to results... ')
        global leafs
        global realLeafs
        leafs = np.array(rf.apply(X_real), dtype=np.int16)

        # Make sure we are not out of the int16 limits
        print('  Max leaf index: ')
        print(np.max(leafs))

        # realLeafs: A row represents a galaxy, a column represents a tree (estimator).
        #            A cell's value is True if most of the galaxies are real, and False otherwise.
        print('  Calculating real leafs... ')
        # Same structure as leafs: 1 if real, 0 else
        estList = rf.estimators_
        n_est = len(estList)

        realLeafs = Parallel(n_jobs=-1, verbose=1)(
            delayed(realLeaf_i)(estList[i].tree_.value, leafs[:, i]) for i in range(n_est))
        realLeafs = np.array(realLeafs, dtype=bool).T

        # We calculate the similarity matrix based on leafs & realLeafs
        print('  Calculating similarity matrix... ')
        sim_mat = np.zeros((n, n), dtype=np.float16)

        # It is suggested to run the same parallelization using the multiprocessing package and test which is faster
        # pool = mp.Pool()
        # f = partial(dist_i_mp)
        # res = pool.map(dist_i_mp, range(n))
        res = Parallel(n_jobs=-1, verbose=1)(delayed(dist_i)(i, leafs[i:, :], leafs[i], realLeafs[i:, :], realLeafs[i]) for i in range(n))

        del leafs, realLeafs
        for tup in res:
            i = tup[1]
            sim_mat[i:, i] = tup[0]

        # Symmetrisize the similarity matrix
        sim_mat = np.array(sim_mat, dtype=np.float16)
        sim_mat += sim_mat.T - np.diag(np.diagonal(sim_mat))

        # dissimilarity=1-similarity (Dij=1-Sij)
        return 1 - sim_mat


# @jit
def realLeaf_i(val, cleafs):
    cRealLeafs = np.zeros(cleafs.shape)
    for i in range(len(cleafs)):
        cleaf = val[cleafs[i]][0]
        cRealLeafs[i] = cleaf[0] > cleaf[1]
    return np.array(cRealLeafs, np.int8)


def dist_i(i, a, b, c, d):
    """
    :param i: int: The index of the galaxy
    :param a: leafs[i:, :]
    :param b: leafs[i]
    :param c: realLeafs[i:, :]
    :param d: realLeafs[i]
    :return: Returns the last (n-i) rows of the ith column in the similarity matrix
    """

    # Count the number of leafs where galaxies are at the same leafs (a==b) and classified as real (c)
    mutLeafs = np.logical_and((a == b), c).sum(1)

    # ount the number of leafs where galaxies are classified as real
    mutRealLeafs = np.logical_and(d, c).sum(1, dtype=np.float64)

    return np.divide(mutLeafs, mutRealLeafs, dtype=np.float16), i


def dist_i_mp(i):
    """
    It is compatible with the multiprocessing commented out in the calcDisMat function.
    :param i: int: The index of the galaxy
    :return: Returns the last (n-i) rows of the ith column in the similarity matrix.
    """
    mutLeafs = np.logical_and((leafs[i:, :] == leafs[i]), realLeafs[i:, :], dtype=np.int16).sum(1)
    mutRealLeafs = np.logical_and(realLeafs[i], realLeafs[i:, :], dtype=np.int16).sum(1, dtype=np.float64)

    return np.divide(mutLeafs, mutRealLeafs, dtype=np.float16), i


def get_leaf_sizes(rf, X):

    """
    :param rf: Random forest classifier object
    :param X: The data we would like to classify and return its leafs' sizes
    :return: leafs_size: A list with all of the leafs sizes for each leaf an observation at X ended up at
    """

    apply_mat = rf.apply(X)
    leafs_sizes = []
    for i, est in enumerate(rf.estimators_):
        real_leafs = np.unique(apply_mat[:, i])
        for j in range(len(real_leafs)):
            leafs_sizes.append(est.tree_.value[real_leafs[j]][0].sum())
    return leafs_sizes


def make_clickable(val):
    return '<a href="{}">{}</a>'.format(val,val)