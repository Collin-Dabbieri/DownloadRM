# This script uses DR16Q_v4.fits to grab the plate, MJD, and fiber of all publicly available RM quasar spectra through DR16
# For each spectrum, it executes a wget command to download the spectrum and saves it to the specified directory
# Within that directory, it will create a separate directory for each object containing all of the spectra for that object


############### IMPORTANT ############################################################################ 
# sfdmap may not work with modern numpy, you may have to make a slight alteration to the source code
# one random line of code uses 'np.int' which is depreciated, you have to replace that with 'int'
# for me (using venv) I edited ~/Documents/venv/lib/python3.9/site-packages/sfdmap.py
# in line 125 and line 127 in _bilinear_interpolate
# change y0 = yfloor.astype(np.int) to y0 = yfloor.astype(int)
# change x0 = xfloor.astype(np.int) to x0 = xfloor.astype(int)

########################### USER SPECIFIED VARIABLES ######################################################################################
###########################################################################################################################################

savedir='/Users/dabbiecm/RM_Spectra/' #directory where we want to save the spectra
DR16Q_path='/Users/dabbiecm/DataProducts/DR16Q_v4.fits' #path to DR16Q fits file
make_unred_z=True # whether to generate a binary file with rest-frame, Milky Way dereddened data
delete_fits=True # whether to delete the fits files after generating unred_z files
sfdpath='./sfddata-master' # path to SFD fits files directory (only needed if make_unred_z is True)

################## IMPORT #################################################################################################################
###########################################################################################################################################
import numpy as np
from astropy.io import fits
import pickle
import os
if make_unred_z:
    import sfdmap
    from ccm_unred import ccm_unred
    m=sfdmap.SFDMap(sfdpath)

################## FUNCTIONS ##############################################################################################################
###########################################################################################################################################

def get_rm_idx(rm_bitmask_dr16q):
    # bitmask that's relevant for RM is 'ANCILLARY_TARGET2'
    # bit values of 54 or 55 indicate RM (https://www.sdss4.org/dr17/algorithms/ancillary/reverberation-mapping/)
    rmflag_dr16q=np.array([])
    for flag in rm_bitmask_dr16q:
        if (flag==-1):
            rmflag_dr16q=np.append(rmflag_dr16q,0)
        elif ((flag & 2**54) != 0):
            rmflag_dr16q=np.append(rmflag_dr16q,1)
        elif ((flag & 2**55) != 0):
            rmflag_dr16q=np.append(rmflag_dr16q,1)
        else:
            rmflag_dr16q=np.append(rmflag_dr16q,0)

    return rmflag_dr16q

def get_spec_filename(plate,mjd,fiber):
    plate_str=str(plate).zfill(4) # turn '45' into '0045'
    mjd_str=str(mjd).zfill(5)
    fiber_str=str(fiber).zfill(4)

    filename='spec-'+plate_str+'-'+mjd_str+'-'+fiber_str+'.fits'
    return filename

def get_spec_names(indexes):
    '''
    Takes indexes from DR16Q and grabs all of the spectra names for ALL of the observations, using PLATE_DUPLICATE, etc.
    This function also assumes that DR16Q is loaded into memory

    params:
        indexes - list of indexes in DR16Q for objects you're interested in
    returns:
        specnames - list of lists of strings, each entry in specnames is a list with the filenames of each spectrum for that object
    '''
    specnames=[]
    for i in indexes:
        specnames_iter=[]

        # first, get the primary spectrum
        plate=DR16Q['PLATE'][i]
        mjd=DR16Q['MJD'][i]
        fiber=DR16Q['FIBERID'][i]
        filename=get_spec_filename(plate,mjd,fiber)
        specnames_iter.append(filename)
        if DR16Q['NSPEC'][i]>0: #NSPEC is actually the number of repeat spectra, not total number of spectra
            #Next, get the repeat spectra if they exist
            num_repeats=DR16Q['NSPEC'][i]
            for j in range(num_repeats):
                plate=DR16Q['PLATE_DUPLICATE'][i][j]
                mjd=DR16Q['MJD_DUPLICATE'][i][j]
                fiber=DR16Q['FIBERID_DUPLICATE'][i][j]
                filename=get_spec_filename(plate,mjd,fiber)
                specnames_iter.append(filename)
        specnames.append(specnames_iter)
    return specnames
                

def download_spectrum(filename,savedir):
    '''
    given a filename, searches the different potential DR16Q directories where that file might live and uses wget to download the spectrum
    DR16 stores the spectra in 4 subdirectories, each subdirectory covers an overlapping range in plate values
    https://data.sdss.org/sas/dr16/sdss/spectro/redux/103/spectra/lite/ 1960-2912
    https://data.sdss.org/sas/dr16/sdss/spectro/redux/104/spectra/lite/ 2640-3480
    https://data.sdss.org/sas/dr16/sdss/spectro/redux/26/spectra/lite/ 0266-3006
    https://data.sdss.org/sas/dr16/sdss/spectro/redux/v5_13_0/spectra/lite/ 3523-11704

    below these directories are a directory for each plate, and then the spectra, ex.
    https://data.sdss.org/sas/dr16/sdss/spectro/redux/103/spectra/lite/1960/spec-1960-12345-1234.fits

    When wget reaches a path that does not exist, it will skip it and move on.
    So for each filename generate all possible paths, it will skip the ones that aren't there.

    params:
        filename - single filename of the spectrum
        savedir - path to directory for saving spectrum
    '''

    # filename can be spec-XXXX-XXXXX-XXXX.fits
    # or spec-XXXXX-XXXXX-XXXX.fits
    # need to extract the plate for directory searching
    if len(filename)==25:
        #plate has 4 digits
        assert(filename[4]=='-'),"file {:s} is weird".format(filename) #make sure '-' is in the correct place
        assert(filename[9]=='-'),"file {:s} is weird".format(filename)
        plate=int(filename[5:9])
    elif len(filename)==26:
        #plate has 5 digits
        assert(filename[4]=='-'),"file {:s} is weird".format(filename)
        assert(filename[10]=='-'),"file {:s} is weird".format(filename)
        plate=int(filename[5:10])
    else:
        raise ValueError("Filename of length {:.0f} is the wrong length".format(len(filename)))

    plate_str=str(plate).zfill(4) # turn '45' into '0045'

    # Now that we have the plate we know which directories it can live in
    paths=[] #this will be the list of all potential paths for this filename
    if ((plate>=1960)&(plate<=2912)):
        path='https://data.sdss.org/sas/dr16/sdss/spectro/redux/103/spectra/lite/'
        path+=plate_str+'/'
        path+=filename
        paths.append(path)

    if ((plate>=2640)&(plate<=3480)):
        path='https://data.sdss.org/sas/dr16/sdss/spectro/redux/104/spectra/lite/'
        path+=plate_str+'/'
        path+=filename
        paths.append(path)

    if ((plate>=266)&(plate<=3006)):
        path='https://data.sdss.org/sas/dr16/sdss/spectro/redux/26/spectra/lite/'
        path+=plate_str+'/'
        path+=filename
        paths.append(path)

    if ((plate>=3523)&(plate<=11704)):
        path='https://data.sdss.org/sas/dr16/sdss/spectro/redux/v5_13_0/spectra/lite/'
        path+=plate_str+'/'
        path+=filename
        paths.append(path)

    # Now paths contains all potential locations for this filename, only one of them should exist
    # Use wget to download the filename
    # when wget is passed a path that does not exist, it will skip it and move on

    command='wget --directory-prefix='
    command+=savedir+' ' #add directory to save files to
    for i in paths:
        command+=i+' '

    #now run wget command
    os.system(command)

def generate_unred_z(specname,ra,dec,z,filedir,savepath,delete_fits=True):
    '''
    takes a list of paths to SDSS fits files, unreddens each file, shifts to rest frame and saves out in binary form
    replaces spec-1234-12345-1234.fits with spec-1234-12345-1234_unred_z.npy

    params:
        specname - name of SDSS fits file. ex. 'spec-1234-12345-1234.fits'
        ra - decimal right ascension
        dec - decimal declination
        z - redshift
        filedir - directory where files are stored. ex. '~/Spectra/'
        savepath - directory for saving processed spectra ex. '~/Spectra/'
        delete_fits - True if you want to delete the fits file when the new file is created

    '''

    file_exists=True
    try:
        temp=fits.open(filedir+specname)
    except:
        print("File {:s} not downloaded".format(specname))
        file_exists=False

    if file_exists:

        loglam=temp[1].data['loglam']
        flux=temp[1].data['flux']
        ivar=temp[1].data['ivar']
        lam=np.power(10,loglam)

        sigma=np.zeros(len(ivar))
        for j in range(len(sigma)):
            if ivar[j]!=0.:
                sigma[j]=np.sqrt(1./ivar[j])
            else:
                sigma[j]=0

        ebv=m.ebv(ra,dec)
        flux_unred=ccm_unred(lam,flux,ebv)
        sigma_unred=ccm_unred(lam,sigma,ebv)

        WL_rest=lam/(1+z)
        INT_rest=flux_unred*(1+z)
        ERR_rest=sigma_unred*(1+z)
        savedict={}
        savedict['WL']=WL_rest
        savedict['INT']=INT_rest
        savedict['ERR']=ERR_rest

        fbase=specname[:-5] #remove .fits

        filename=savepath+fbase+'_unred_z.npy'

        np.save(filename,savedict)

        if delete_fits:
            #now remove the fits file to save space
            command='rm '+filedir+specname
            os.system(command)


def normalize_oiii(WL1,INT1,ERR1,WL2,INT2,ERR2,norm_bands=[[4980,5030]]):
    wave,spec1,err1,spec2,err2=rebin(WL1,WL2,INT1,INT2,ERR1,ERR2)
    a,b=normalize(wave,spec1,spec2,norm_bands)
    spec1_norm=a*spec1+b
    spec2_norm=spec2
    err1_norm=a*err1
    err2_norm=err2

    return wave,spec1_norm,err1_norm,spec2_norm,err2_norm


########################### MAIN LOOP #####################################################################################################
###########################################################################################################################################

if __name__=='__main__':

    temp=fits.open(DR16Q_path)
    DR16Q=temp[1].data
    rm_bitmask_dr16q=DR16Q['ANCILLARY_TARGET2']
    #print(rm_bitmask_dr16q[:200])
    # bit values of 54 or 55 indicates RM (https://www.sdss4.org/dr17/algorithms/ancillary/reverberation-mapping/)

    ####################
    # you should only need to generate rm_indexes once, afterward you can just load it in with the pickle commands below
    ####################
    #rmflag_dr16q=get_rm_idx(rm_bitmask_dr16q) # indexed the same as DR16Q, 0 if not rm, 1 if rm
    #rm_indexes=np.where(rmflag_dr16q==1)[0] #indexes in DR16Q for rm objects
    #picklefile = open('rm_indexes.pkl', 'ab')
    #pickle.dump(rm_indexes, picklefile)
    #picklefile.close()
    picklefile = open('rm_indexes.pkl', 'rb')
    rm_indexes = pickle.load(picklefile)
    print(len(rm_indexes))
    
    ####################
    # you should only need to generate specnames once, afterward you can just load it in with the pickle commands below
    ####################
    #specnames=get_spec_names(rm_indexes)
    #picklefile = open('specnames.pkl', 'ab')
    #pickle.dump(specnames, picklefile)                    
    #picklefile.close()
    picklefile = open('specnames.pkl', 'rb')
    specnames = pickle.load(picklefile)

    for i in rm_indexes[:20]:
        print(DR16Q['NSPEC'][i])

    num_objects=len(specnames)
    # we're going to create a subdirectory for each RM quasar, the name of that directory will be the SDSS jname
    for i in range(num_objects):
        dr16_index=rm_indexes[i]
        ra=DR16Q['RA'][dr16_index]
        dec=DR16Q['DEC'][dr16_index]
        z=DR16Q['Z'][dr16_index]
        jname=DR16Q['SDSS_NAME'][dr16_index]
        path=savedir+'J'+jname+'/'
        cmd='mkdir '+path
        os.system(cmd)

        num_spectra=len(specnames[i])
        for j in range(num_spectra):
            # download the spectrum into the appropriate savedir
            download_spectrum(filename=specnames[i][j],savedir=path)
            if make_unred_z:
                generate_unred_z(specname=specnames[i][j],ra=ra,dec=dec,z=z,filedir=path,savepath=path,delete_fits=delete_fits)
        
        


###########################################################################################################################################