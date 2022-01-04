# -*- coding: utf-8 -*-
"""
Created on Tue Mar 23 19:54:34 2021

@author: AlmogHershko
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
import pathlib
import os
import pandas as pd

# Loading emission and absorption lines
file_path = pathlib.Path(__file__).parent.absolute()
eLines_path = os.path.join(file_path,'SDSS_emission_lines.txt')
aLines_path = os.path.join(file_path,'SDSS_absorption_lines.txt')
eLines = pd.read_csv(eLines_path, sep='\t', encoding = "utf", header=None)
aLines = pd.read_csv(aLines_path, sep='\t', encoding = "utf", header=None)

def split_spec(spec, wl, ivar):
    ivar_diff = np.append(np.array([0]),np.diff((ivar==0).astype(int)))
    split_indices = np.where(ivar_diff)[0]
    spec_splits = np.split(spec, split_indices)
    wl_splits = np.split(wl, split_indices)
    ivar_splits = np.split(ivar, split_indices)
    i_splits = [i for i,split in enumerate(ivar_splits) if split[0]!=0]
    spec_splits = [spec_splits[i] for i in i_splits]
    wl_splits = [wl_splits[i] for i in i_splits]
    ivar_splits = [ivar_splits[i] for i in i_splits]
    return spec_splits, wl_splits, ivar_splits

def plot_spectrum(spec, wl, ivar, plot_lines=False, log_wl=False, med=False, g=None, title=None):
    
    # converting to log scale
    if log_wl:
        wl = np.log10(wl)
    
    # setting a y-axis quanta for plotting (used for the lines plotting)
    spec_vals = spec[~np.isnan(spec)]
    dy = (max(spec_vals)-min(spec_vals))/40
    
    # creating a figure
    fig, ax = plt.subplots(figsize=(6,8))
    
    # splitting the spectrum where ivar=0 and plotting the partial spectra
    spec_splits, wl_splits, ivar_splits = split_spec(spec, wl, ivar)
    for spec_i, wl_i, ivar_i in zip(spec_splits, wl_splits, ivar_splits):
        
        # plotting the spectrum
        ax.plot(wl_i, spec_i, 'k', linewidth=1)
        
        # plotting the median if requested
        if med:
            spec_i_med = signal.medfilt(spec_i, 5)
            ax.plot(wl_i, spec_i_med, 'r', linewidth=0.5)
            #ax.plot(wl_i, sp.ndimage.gaussian_filter1d(spec_i_med, 3), 'orange', linewidth=1)
        
        # plotting the error
        err = 1/np.sqrt(ivar_i)
        ax.fill_between(wl_i, spec_i - err, spec_i + err, color='grey', alpha=0.5)
        
        # plotting the lines
        if plot_lines:
        
            # Adding the absorption lines
            for i, row in aLines.iterrows():
                wl_line = row[1]
                label_line = row[0]
                if (wl_line >= wl_i[0] and wl_line <= wl_i[-1]):
                    f_line = spec_i[np.abs(wl_i-wl_line).argmin()]
                    ax.annotate(
                        label_line,
                        xy=(wl_line, f_line-dy),
                        xytext=(wl_line, f_line-3*dy),
                        arrowprops=dict(arrowstyle="-", color='red'),
                        horizontalalignment="center",
                        fontsize=8
                    )
        
            # Adding the emission lines
            for i, row in eLines.iterrows():
                wl_line = row[1]
                label_line = row[0]
                if (wl_line >= wl_i[0] and wl_line <= wl_i[-1]):
                    f_line = spec_i[np.abs(wl_i-wl_line).argmin()]
                    ax.annotate(
                        label_line,
                        xy=(wl_line, f_line+dy),
                        xytext=(wl_line, f_line+3*dy),
                        arrowprops=dict(arrowstyle="-", color='blue'),
                        horizontalalignment="center",
                        fontsize=8
                    )
        
    # marking where ivar=0
    for wl1,wl2 in zip(wl_splits[:-1],wl_splits[1:]):
        ax.axvspan(wl1[-1], wl2[0], facecolor='0.5', alpha=0.5)
        
    # setting axes labels
    font = {'family': 'Times New Roman',
            'style':  'italic',
            'size': 16
            }
    ax.set_xlabel(r'wavelength [$\mathdefault{\AA}$]', fontdict=font)
    ax.set_ylabel(r'$\mathdefault{f_{\lambda} [10^{-17}erg/cm^{2}/s/\AA]}$', fontdict=font)
    
    # adding title
    title_lines = []
    if title is not None:
        title_lines.append(title)
    if g is not None:
        title_lines.append('RA=%.3f, Dec=%.3f, Plate=%d, Fiber=%d, MJD=%d' % (g.ra, g.dec, g.plate, g.fiberid, g.mjd))
        title_lines.append('medianSNR=%.3f, z=%.3f' % (g.snMedian, g.z))
    if len(title_lines)>0:
        ax.set_title('\n'.join(title_lines), fontdict=font)
    
    # returning the figure and the axes
    return fig, ax