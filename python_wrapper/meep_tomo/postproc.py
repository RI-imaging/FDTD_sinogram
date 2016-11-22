#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division, print_function, unicode_literals


import h5py
import numpy as np
import os
from scipy import interpolate
import warnings

from . import meta
 
def CropPML(arr, crop):
    """
        Crops the PML of thickness `crop` from each end of each axis.
        Works in 2d and 3d.
    """
    a=arr.copy().squeeze()
    shape = a.shape

    if len(shape) == 3:
        (l1, l2, l3) = a.shape
        b = a[crop:l1-crop, crop:l2-crop, crop:l3-crop]
    elif len(shape) == 2:
        (l1, l2) = a.shape
        b = a[crop:l1-crop, crop:l2-crop]
    elif len(shape) == 1:
        b = a[crop:len(a)-crop]
    else:
        raise NotImplementedError(
                  "Cannot handle array of shape {}.".format(len(shape)))
    return b


def GetDielecticStructure(DIR, filename="eps-000000.00.h5", cut=0,
                          outd=None, outf=None, savepng=False):
                          
    """
       Given a .h5 file that contains a dielectric structure, return the
       2- or 3d array of refractive index.
    """
    D = h5py.File(os.path.join(DIR,filename),'r')
    
    # eps = n^2
    n = np.sqrt(np.array(D.items()[0][1]))
    
    D.close()
    
    if cut != 0:
        n = CropPML(n,cut)

    if savepng:
        from . import plot
        if outf is None:
            outf = filename[:-3]
        
        if outd is None:
            outd = DIR
        
        if outf[-4:] != ".png":
            outf += ".png"

        # remember to do this someplace else:
        if len(n.shape) == 2:
            plot.arr2im(n, scale=True).save(os.path.join(outd,outf))
        elif len(n.shape) == 3:
            plot.saveslice_xyz(n, outf, outd)
        else:
            raise NotImplementedError(
                  "Cannot handle array of shape {}.".format(len(n.shape)))
    # return RI
    return n




def GetFieldAtLDFromCenter(h5file, lD=0):
    """ Obtain cross-section of field from h5 file

    Get the EM field at a distance lD [px] from the center of the
    FDTD domain in direction of propagation. 
    
    The direction of propagation is assumed to be the last axis
    for 2D and 3D data.
    
    lD is the axial position in pixels from the center of the image
    in pixels. For even images, the center is the right hand pixel
    of the actual center.
    """
    if not isinstance(lD, int):
        warnings.warn("Integer: Setting lD from {} to {}.".format(lD, int(lD)))
        lD = int(lD)
    # Open the file
    Ex = h5py.File(h5file,'r')

    try:
        Exr = Ex["ez.r"]
    except KeyError:
        try:
            Exr = Ex["ex.r"]
        except KeyError:
            try:
                Exr = Ex["ey.r"]
            except KeyError:
                raise Exception("Could not find real components of field.")
    try:
        Exi = Ex["ez.i"]
    except KeyError:
        try:
            Exi = Ex["ex.i"]
        except KeyError:
            try:
                Exi = Ex["ey.i"]
            except KeyError:            
                warnings.warn("Could not find imag. components of field."+
                          "\nSetting it to zero")
                Exi = np.zeros(Exr.shape)


    #Exr, Exi = np.squeeze(Ex.items()[1][1]), np.squeeze(Ex.items()[0][1])

    ls = list(Exr.shape).count(1)

    if len(Exr.shape) -ls == 2:
        xsh = Exr.shape[0]
        psh = Exr.shape[1]
        pos = int(np.floor(psh * 0.50)) + lD
        if pos+1 > psh:
            warnings.warn("lD is probably too large")
        Field = (Exr[:,pos] + 1j*Exi[:,pos])
    elif len(Exr.shape) -ls == 3:
        xsh = Exr.shape[0]
        ysh = Exr.shape[1]
        psh = Exr.shape[2]
        pos = np.int(np.floor(psh * 0.50)) + lD
        #print("tool_FDTD.py GETEXATLINEFROMCENTER: CHECK POSTION")
        if pos+1 > psh:
            warnings.warn("lD is probably too large")
        Field = (Exr[:,:,pos] + 1j*Exi[:,:,pos])
    else:
        raise NotImplementedError(
                  "Cannot handle array of shape {}.".format(len(Exr.shape)))
    Ex.close()
    return np.transpose(np.squeeze(Field))


def GetFieldAtLineFromCenter(DIR, exdata="ex-002173.91.h5", lD=0):
    return GetFieldAtLDFromCenter(os.path.join(DIR,exdata), lD=lD)


def GetFieldFromDIR(DIR, timestep=-1, return_filename=False):
    """ Get complex field y-component of h5 files in DIR
        
        Files are sorted and by default the last one is returned.
    """
    files=list()
    for f in os.listdir(DIR):
        if f.startswith("ey") and f.endswith(".h5"):
            files.append(f)
        if f.startswith("ez") and f.endswith(".h5"):
            files.append(f)
    files.sort()

    E = h5py.File(os.path.join(DIR,files[timestep]),'r')
    
    BEzr = E.items()[1][1]
    BEzi = E.items()[0][1]
    
    CE = BEzr[:] + 1j*BEzi[:]
    
    E.close()
    return CE


def GetH5files(DIR, prefix="eps"):
    """ Filenames of field components
    
    If DIR does not contain any h5 fields, then returns all h5 fields
    in subdirectories that start with `prefix`.
    """
    DIR = os.path.realpath(DIR)
    files = os.listdir(DIR)
    ffil = list()
    for f in files:
        if f.endswith(".h5") and not f.startswith("eps"):
            ffil.append(os.path.join(DIR,f))
    ffil.sort()
    if len(ffil) != 0:
        return ffil
    else:
        # go through subdirs
        ffil = list()
        for df in files:
            if (df.startswith(prefix) and 
                os.path.isdir(os.path.join(DIR,df))):
                df = os.path.join(DIR,df)
                sfiles = os.listdir(df)
                for f in sfiles:
                    if f.endswith(".h5") and not f.startswith("eps"):
                        ffil.append(os.path.join(df,f))
    ffil.sort()
    return ffil


def get_info_medium_ri(DIR):
    BGDIR, dirlist = meta.GetDirlistSimulation(DIR)
    info = meta.GetInfoFromFolder(dirlist[-1])
    nm =  info["Medium RI"]
    return nm
    
def get_info_pml_px(DIR):
    BGDIR, dirlist = meta.GetDirlistSimulation(DIR)
    info = meta.GetInfoFromFolder(dirlist[-1])
    pxcut =  info["PML thickness [wavelengths]"]
    res = meta.get_info_resolution(DIR)
    return int(np.ceil(pxcut*res))

def get_info_axial_obj_size(DIR):
    BGDIR, dirlist = meta.GetDirlistSimulation(DIR)
    info = meta.GetInfoFromFolder(dirlist[-1])
    aos =  info["Axial object size [wavelengths]"]
    return aos


def InterpolateFields(a, newsize, info=None):
    """ Interpolate two-dimensional complex field on square grid.
    
        newsize : int
            size of the new square array
            
        info : dictionary
            info dictionary from InfoSimulation will be updated and
            returned
    """
    INTERPOLATION = newsize
    Field = a
    size = Field.shape[0]
    print("...Interpolating complex data from {}px to {}px.".format(
                    size, INTERPOLATION))
    x = np.arange(size)
    pr = interpolate.RectBivariateSpline(x,x,Field.real, kx=1, ky=1)
    pi = interpolate.RectBivariateSpline(x,x,Field.imag, kx=1, ky=1)
    xn = np.linspace(0, size-1, INTERPOLATION, endpoint=True)
    NewEx = pr(xn,xn) + 1j*pi(xn,xn)

    #arr2im(NewEx.real/np.max(NewEx.real)*255).save("a.bmp")
    #arr2im(Exfield.real/np.max(Exfield.real)*255).save("b.bmp")
    #arr2im(NewEx.imag/np.max(NewEx.imag)*255).save("aimag.bmp")
    #arr2im(Exfield.imag/np.max(Exfield.imag)*255).save("bimag.bmp")

    del Field
    Field = NewEx
    
    ##### This is very important:
    ##### If we rescale the image, we need to change the parameters
    ##### accordingly:
    smaller = INTERPOLATION/(size)
    # less pixels are needed for a micrometer
    # pixel size increases
    if info is not None:
        info["effective pixel size [um]"] *= 1/smaller
        # there are less pixels -> everything in px decreases
        for key in info.keys():
            if key.endswith("[px]"):
                info[key] *= smaller
        return Field, info
    else:
        warnings.warn("Info data not edited, because not in argument!")
        return Field



def resizeref(ref, resize):
    """
        resize cubic reference image
    """
    if len(ref.shape) != 3:
        warnings.warn("Only USE 3D data! ERROR will come.")
        
    (sx, sy, sz) = ref.shape
    

    # 2D interpolation
    twodintp = np.zeros((sx, resize, resize))
    xn = np.linspace(0, ref.shape[0], resize, endpoint=True)
    for i in range(sx):
        x = np.arange(sx)
        pr = interpolate.RectBivariateSpline(x,x,ref[i], kx=1, ky=1)
        twodintp[i] = pr(xn,xn)
    
    med = np.transpose(twodintp, [1,0,2])
    
    out = np.zeros((resize, resize, resize))
    # 1D interpolation
    for i in range(resize):
        y = np.arange(sx)
        z = np.arange(resize)
        pr = interpolate.RectBivariateSpline(y,xn,med[i], kx=1, ky=1)
        out[i] = pr(xn,xn)

    return np.transpose(out, [1,0,2])


