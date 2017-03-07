#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division, print_function, unicode_literals

import h5py
import numpy as np
import os
from scipy import interpolate
import unwrap
import warnings

from . import meep



def crop_arr(arr, crop):
    """Crops a border frame from an array

    Parameters
    ----------
    arr: 2d or 3d ndarray
        The volume from which to crop the frames
    crop: int
        The thickness of the frame in pixels

    Returns
    -------
    cropped: 2d or 3d ndarray
        The cropped array
    """
    a=arr.copy().squeeze()
    shape = a.shape
    if crop:
        if len(shape) == 3:
            b=a[crop:-crop, crop:-crop, crop:-crop]
        elif len(shape) == 2:
            b=a[crop:-crop, crop:-crop]
        elif len(shape) == 1:
            b=a[crop:-crop]
        else:
            raise NotImplementedError(
                      "Cannot handle array of shape {}.".format(len(shape)))
    else:
        b=a
    return b


def get_field_at_ld(h5file, ld=0):
    """ Obtain cross-section of field from h5 file

    Get the EM field at a distance ld [px] from the center of the
    FDTD domain in direction of propagation. 
    
    The direction of propagation is assumed to be the last axis
    for 2D and 3D data.
    
    `ld` is the axial position in pixels from the center of the image
    in pixels. For even images, the center is the right hand pixel
    of the actual center.
    """
    if not isinstance(ld, int):
        if ld != int(ld):
            warnings.warn("Setting lD from {} to {}.".format(ld, int(ld)))
        ld = int(ld)

    # Open the file
    with h5py.File(h5file,'r') as fdem:
        try:
            freal = fdem["ez.r"]
        except KeyError:
            try:
                freal = fdem["ex.r"]
            except KeyError:
                try:
                    freal = fdem["ey.r"]
                except KeyError:
                    raise Exception("Could not find real components of field.")
        try:
            fimag = fdem["ez.i"]
        except KeyError:
            try:
                fimag = fdem["ex.i"]
            except KeyError:
                try:
                    fimag = fdem["ey.i"]
                except KeyError:            
                    warnings.warn("Could not find imag. components of field."+
                                  "\nSetting it to zero")
                    fimag = np.zeros(fimag.shape)

        ls = list(freal.shape).count(1)
    
        if len(freal.shape)-ls == 2:
            psh = freal.shape[1]
            pos = int(np.floor(psh*.5)) + ld
            if pos+1 > psh:
                warnings.warn("`ld` is probably too large")
            Field = (freal[:,pos] + 1j*fimag[:,pos])
        elif len(freal.shape)-ls == 3:
            psh = freal.shape[2]
            pos = np.int(np.floor(psh*.5)) + ld
            if pos+1 > psh:
                warnings.warn("`ld` is probably too large")
            Field = (freal[:,:,pos] + 1j*fimag[:,:,pos])
        else:
            msg="Cannot handle array of shape {}.".format(freal.shape)
            raise NotImplementedError(msg)

    return np.transpose(np.squeeze(Field))


def get_field_h5files(sdir, prefix_dirs="ph"):
    """Return names of field h5 files in a directory
    
    Parameters
    ----------
    sdir: str
        Path to the search directory
    prefix_dirs: str
        If no matching files are found in sdir, search 
        subdirectories whose name starts with this string. 
    
    Returns
    -------
    files: list of str
        Paths to the found h5 files
    
    Notes
    -----
    If DIR does not contain any h5 fields, then returns all h5 fields
    in subdirectories that start with `prefix`.
    
    This method ignores h5 files of the eps structure, i.e. h5 files
    starting with "eps" are ignored.
    """
    sdir = os.path.realpath(sdir)
    files = os.listdir(sdir)
    ffil = []
    for f in files:
        if f.endswith(".h5") and not f.startswith("eps"):
            ffil.append(os.path.join(sdir,f))
    ffil.sort()
    if len(ffil):
        return ffil
    else:
        # go through subdirs
        for df in files:
            if (df.startswith(prefix_dirs) and 
                os.path.isdir(os.path.join(sdir,df))):
                df = os.path.join(sdir,df)
                sfiles = os.listdir(df)
                for f in sfiles:
                    if f.endswith(".h5") and not f.startswith("eps"):
                        ffil.append(os.path.join(df,f))
    ffil.sort()
    return ffil


def get_ri_structure(path, crop=0, outd=None, outf=None, savepng=False):
    """Obtain the 3D refractive index from an .h5 file
    
    Parameters
    ----------
    path: str
        Path to the .h5 file or a folder containing it. In the second
        case, we will look for a file similar to "eps-000000.00.h5".

    Returns
    -------
    ri: ndarray
        The 2D or 3D refractive index structure
    """
    if os.path.isdir(path):
        files = [ f for f in  os.listdir(path) if f.endswith(".h5") ]
        files = [ f for f in  files if f.startswith("eps-") ]
        files.sort()
        assert len(files), "No eps stucture files found in {}".format(path)
        path = os.path.join(path, files[0])
        
    with h5py.File(path,'r') as h5fd:
        # eps = n^2
        n = np.sqrt(np.array(h5fd.items()[0][1]))
    
    n = crop_arr(n, crop)

    # return RI
    return n



def get_sim_info(sim_path):
    """Get simulation parameters from C++ phantom file

    Parameters
    ----------
    sim_path: str
        Path to a .cpp file (e.g. "ph_phantom_2d_0.0000000000.cpp")
        or to a directory containing such a file.
    
    Returns
    -------
    info: dict
        Parsed information as dict
    
    Notes
    -----
    This method uses the original C++ file that is e.g. copied to the
    simulation folder after the simulation.
    """
    if os.path.isdir(sim_path):
        files = [ f for f in os.listdir(sim_path) if f.endswith(".cpp") ]
        files.sort()
        if len(files):
            sim_path = os.path.join(sim_path, files[0])
        else:
            raise OSError("No .cpp files in {}!".format(sim_path))
    
    info = meep.get_phantom_kwargs(sim_path)
    
    wl = info["wavelength"]
    for kk in list(info.keys()):
        if kk.endswith(("size", "_a", "_b", "_c", "_x", "_y", "_z")):
            info[kk+" [px]"] = info[kk]*wl
            info[kk+" [wavelengths]"] = info[kk]
            info.pop(kk)

    return info


def interpolate_field(field, newsize, info=None, verbose=False):
    """Interpolate two-dimensional complex field
    
    Parameters
    ----------
    field: 2d ndarray of shape (M,M)
        The field image
    newsize: int
        Size of the new square array
    info: dict
        Information dictionary that will be updated with the new
        resolution.
    
    Returns
    -------
    field_intp: 2d ndarray of shape (newsize,newsize)
        The interpolated field
    ingo_intp: dict
        The update information dictionary
    
    See Also
    --------
    meta.get_sim_info: The method to obtain the `info` dict
    """
    assert field.shape[0] == field.shape[1], "only square fields allowed" 
    size = field.shape[0]
    if verbose:
        print("...Interpolating complex data from {}px to {}px.".format(
                                            field.shape[0], newsize))
    
    ampl = np.abs(field)
    phas = unwrap.unwrap(np.angle(field))
    x = np.arange(size)
    ampl_i = interpolate.RectBivariateSpline(x,x, ampl, kx=1, ky=1)
    phas_i = interpolate.RectBivariateSpline(x,x, phas, kx=1, ky=1)
    xn = np.linspace(0, size-1, newsize, endpoint=True)
    field_intp = ampl_i(xn,xn) * np.exp(1j*phas_i(xn,xn))

    ##### This is very important:
    ##### If we rescale the image, we need to change the parameters
    ##### accordingly:
    smaller = newsize/size
    # less pixels are needed for a micrometer
    # pixel size increases
    if info is not None:
        info = info.copy()
        info["effective pixel size [um]"] /= smaller
        # there are less pixels -> everything in px decreases
        for key in info.keys():
            if key.endswith("[px]"):
                info[key] *= smaller
        return field_intp, info
    else:
        warnings.warn("Info data not edited, because not in argument!")
        return field_intp



def interpolate_volume(ref, resize):
    """Intperolate a 3D volume
    """
    assert len(ref.shape) == 3, "3D data only"
    (sx, sy, sz) = ref.shape
    assert sx==sy, "Only cubic data allowed"
    assert sx==sz, "Only cubic data allowed"

    # 2D interpolation along first two axes
    twodintp = np.zeros((sx, resize, resize))
    xn = np.linspace(0, ref.shape[0], resize, endpoint=True)
    for i in range(sx):
        x = np.arange(sx)
        pr = interpolate.RectBivariateSpline(x,x,ref[i], kx=1, ky=1)
        twodintp[i] = pr(xn,xn)
    
    med = np.transpose(twodintp, [1,0,2])

    out = np.zeros((resize, resize, resize))
    # 1D interpolation along last axis
    for i in range(resize):
        y = np.arange(sx)
        pr = interpolate.RectBivariateSpline(y,xn,med[i], kx=1, ky=1)
        out[i] = pr(xn,xn)

    return np.transpose(out, [1,0,2])


