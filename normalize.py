import numpy as np
from rebin import find_common_idx_range
from scipy.optimize import curve_fit
import time

def func(x,a,b):
    return a*x + b
    
def normalize(wave,spec1,spec2,WL_bands,coeff_constraint=0.2):
    '''
    Given two spectra, and wavelength bands to normalize over, fits the spectra with an additive and multiplicative term. As described in
    Eracleous et al. 2012.
    
    f'_lambda=a f_lambda + b_lambda
    
    "Where a and b_lambda are constants, determined by minimizing the differences between the SDSS and post-SDSS spectra in selected regions of the broad-line profiles and the adjacent continuum."
    
    Added constraints to avoid ~0 a values
    
    params:
        wave - wavelength vector
        spec1 - first spectrum
        spec2 - second spectrum
        WL_bands - list of lists of len 2, each entry is [minWL,maxWL] for one band
        coeff_constraint - lower limit for a, used to prevent ~0 a values
    returns:
        a - slope of fit
        b - intercept of fit
    ex:
        a,b=normalize(spec1,spec2,[[4600,4650],[5100,5250],[4770,4930]])
        spec1_norm=a*spec1+b
    '''
    
    # Determine indices to include in fit
    #time0=time.time()
    idx_keep=np.asarray([])
    for i in WL_bands:
        idx_min,idx_max=find_common_idx_range(wave,i[0],i[1])
        idx_keep=np.append(idx_keep,np.arange(idx_min,idx_max))
        
    idx_keep=idx_keep.astype(int)
        
    spec1_cut=np.take(spec1,idx_keep)
    spec2_cut=np.take(spec2,idx_keep)
    
    idx1_nancut=np.isnan(spec1_cut)   
    idx2_nancut=np.isnan(spec2_cut)
    
    #idx_nancut should have length spec1_cut and be true if idx1_nancut[i] is true or if idx2_nancut[i] is true
    idx_nancut=np.logical_or(idx1_nancut,idx2_nancut)
    spec1_cut=np.delete(spec1_cut,idx_nancut)
    spec2_cut=np.delete(spec2_cut,idx_nancut)
    
    #time1=time.time()
    #print("Slicing Time in Normalize: {:.4f} s".format(time1-time0))
    
    # This function is constrained to force a>coeff_constraint
    #time0=time.time()
    popt_cons, _ = curve_fit(func, spec1_cut, spec2_cut, bounds=([coeff_constraint, -np.inf], [np.inf, np.inf]))
    #print(popt_cons)
    #time1=time.time()
    #print("Curve Fit Time in Normalize: {:.4f} s".format(time1-time0))
    
    #p=np.polyfit(spec1_cut,spec2_cut,deg=1)
    a=popt_cons[0]
    b=popt_cons[1]
    
    return a,b



def normalize_both(wave,spec1,spec2,WL_bands):
    '''
    given two spectra, rebinned to a common wavelength vector, fits both spectra with first order fits
    
    params:
        wave - wavelength vector
        spec1 - first spectrum
        spec2 - second spectrum
        WL_bands - list of lists of len 2, each entry is [minWL,maxWL] for one band
    returns:
        m1 - slope of fit for spec1
        b1 - intercept of fit for spec1
        m2 - slope of fit for spec2
        b2 - intercept of fit for spec2
    example:
        m1,b1,m2,b2=normalize_both(wave,spec1,spec2,[[4600,4650],[5100,5250]])
        spec1_norm=spec1-(m1*wave+b1)
        spec2_norm=spec2-(m2*wave+b2)
    '''
    
    # Determine indices to include in fit
    idx_keep=np.asarray([])
    for i in WL_bands:
        idx_min,idx_max=find_common_idx_range(wave,i[0],i[1])
        idx_keep=np.append(idx_keep,np.arange(idx_min,idx_max))
        
    idx_keep=idx_keep.astype(int)    

    spec1_cut=spec1[idx_keep]
    spec2_cut=spec2[idx_keep]
    
    wave_cut=wave[idx_keep]
    
    p1=np.polyfit(wave_cut,spec1_cut,deg=1)
    p2=np.polyfit(wave_cut,spec2_cut,deg=1)
    
    m1=p1[0]
    b1=p1[1]
    m2=p2[0]
    b2=p2[1]
    
    return m1,b1,m2,b2


def normalize_emission_lines(wave,spec1,spec2,WL_band):
    
    '''
    given two spectra (that should already be continuum subtracted and rebinned to a common wavelength vector),
    normalizes the two spectra with only a multiplicative factor. Finds f such that spec1=f*spec2.
    
    params:
        wave - wavelength vector
        spec1 - first spectrum
        spec2 - second spectrum
        WL_band - list of len 2, [minWL,maxWL] for emission line
    returns:
        f - multiplicative factor to match spec2 to spec1
    example:
        f=normalize_emission_lines(wave,spec1_norm,spec2_norm,[4770,4930])
        spec2_norm_match=f*spec2_norm
    '''
    
    
    idx_min,idx_max=find_common_idx_range(wave,WL_band[0],WL_band[1])
    
    spec1_cut=spec1[idx_min:idx_max]
    spec2_cut=spec2[idx_min:idx_max]

    def mult(x,factor):
        return factor*x
    
    popt, pcov = curve_fit(mult, spec2_cut, spec1_cut)
        
    return popt[0]
    