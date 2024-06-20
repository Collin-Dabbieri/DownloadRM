import numpy as np

def find_common_idx_range(wave,minWL,maxWL):
    '''
    Given a list, a minimum value, and a maximum value, provides the minimum and maximum indices that live within the defined range.
    Note since array[start:stop] will not include index stop, idx_max will be the first index outside of the defined range.
    '''
    idx_min=0
    for i in range(len(wave)):
        if wave[i]>minWL:
            idx_min=i
            break

    # if max(wave)=maxWL, then if statement below will never trigger
    # in that case, idx_max=len(wave)
    idx_max=len(wave)
    for i in range(len(wave)):
        if wave[i]>maxWL:
            idx_max=i
            break

    return idx_min,idx_max

def rebin(wave1,wave2,spec1,spec2,err_spec1=None,err_spec2=None):
    '''
    Given two spectra, interpolates onto common wavelength bins in a way that preserves the total intensity of both spectra.
    If errors are provided, also interpolates errors in the same way.

    params:
        wave1 - list of wavelength values for spectrum 1
        wave2 - list of wavelength values for spectrum 2
        spec1 - list of intensity values for spectrum 1
        spec2 - list of intensity values for spectrum 2
        err_spec1 - list of error values for spectrum 1
        err_spec2 - list of error values for spectrum 2

    if err_spec1 is None and err_spec2 is None returns:
        wave - common wavelength bins
        spec1_interp - intensity values for spectrum 1 on common wavelength bins
        spec2_interp - intensity values for spectrum 2 on common wavelength bins

    if err_spec1 and err_spec2 are not None returns:
        wave
        spec1_interp
        err_spec1_interp - error values for spectrum 1 on common wavelength bins
        spec2_interp
        err_spec2_interp - error values for spectrum 2 on common wavelength bins

    if only err_spec1 is None returns:
        wave
        spec1_interp
        spec2_interp
        err_spec2_interp

    if only err_spec2 is None returns:
        wave
        spec1_interp
        err_spec1_interp
        spec2_interp

    ex:
        wave,spec1_interp,spec2_interp=jitterfitter.rebin(wave1,wave2,spec1,spec2)

    '''

    # assign number of points for common bins
    numpoints=np.amin(np.asarray([len(wave1),len(wave2)]))

    # minimum wavelength is the lowest shared wavelength
    minWL=np.amax(np.asarray([min(wave1),min(wave2)]))
    # maximum wavelength is the highest shared wavelength
    maxWL=np.amin(np.asarray([max(wave1),max(wave2)]))

    # create common wavelength bins
    wave=np.linspace(minWL,maxWL,numpoints)

    # calculate integrated intensity of spectra
    # this should only include points within [minWL,maxWL]
    spec1_minidx,spec1_maxidx=find_common_idx_range(wave1,minWL,maxWL)
    spec2_minidx,spec2_maxidx=find_common_idx_range(wave2,minWL,maxWL)
    spec1_integral=np.trapz(y=spec1[spec1_minidx:spec1_maxidx],x=wave1[spec1_minidx:spec1_maxidx])
    spec2_integral=np.trapz(y=spec2[spec2_minidx:spec2_maxidx],x=wave2[spec2_minidx:spec2_maxidx])

    # interpolate spectra onto common WL bins
    spec1_interp_nonorm=np.interp(wave,wave1,spec1)
    spec2_interp_nonorm=np.interp(wave,wave2,spec2)
    if err_spec1 is not None:
        err_spec1_interp_nonorm=np.interp(wave,wave1,err_spec1)
    if err_spec2 is not None:
        err_spec2_interp_nonorm=np.interp(wave,wave2,err_spec2)

    # calculate integrated intensity of interpolated spectra
    spec1_interp_integral=np.trapz(y=spec1_interp_nonorm,x=wave)
    spec2_interp_integral=np.trapz(y=spec2_interp_nonorm,x=wave)

    # normalize interpolated spectra so that total integrated intensity is the same
    spec1_norm=spec1_integral/spec1_interp_integral
    spec1_interp=spec1_interp_nonorm*spec1_norm
    spec2_norm=spec2_integral/spec2_interp_integral
    spec2_interp=spec2_interp_nonorm*spec2_norm
    
    if err_spec1 is not None:
        err_spec1_interp=err_spec1_interp_nonorm*spec1_norm
    if err_spec2 is not None:
        err_spec2_interp=err_spec2_interp_nonorm*spec2_norm


    if ((err_spec1 is None)&(err_spec2 is None)):
        return wave,spec1_interp,spec2_interp
    elif err_spec1 is None:
        return wave,spec1_interp,spec2_interp,err_spec2_interp
    elif err_spec2 is None:
        return wave,spec1_interp,err_spec1_interp,spec2_interp
    else:
        return wave,spec1_interp,err_spec1_interp,spec2_interp,err_spec2_interp
