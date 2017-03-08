#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division, print_function, unicode_literals

import h5py
import numpy as np
import os
import warnings

from . import meep, common


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
    crop = np.int(np.ceil(crop))
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


def get_field_at_ld(sim_path, ld=0, crop_pml=True):
    """ Obtain cross-section of field from h5 file

    Get the EM field at a distance ld [px] from the center of the
    FDTD domain in direction of propagation. 
    
    The direction of propagation is assumed to be the last axis
    for 2D and 3D data.
    
    Parameters
    ----------
    h5file: str
        Path to the .h5 field file or folder containing it
    ld: int
        The axial position from the center of the image in pixels.
        For even images, the center is the right hand pixel
        of the actual center.
    """
    # check h5file
    if os.path.isdir(sim_path):
        files = os.listdir(sim_path)
        files = [ f for f in files if f.endswith(".h5") ]
        files = [ f for f in files if not f.startswith("eps-") ]
        files.sort()
        h5file=os.path.join(sim_path, files[0])
    else:
        h5file=sim_path
    # check ld
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
            field = (freal[:,pos] + 1j*fimag[:,pos])
        elif len(freal.shape)-ls == 3:
            psh = freal.shape[2]
            pos = np.int(np.floor(psh*.5)) + ld
            if pos+1 > psh:
                warnings.warn("`ld` is probably too large")
            field = (freal[:,:,pos] + 1j*fimag[:,:,pos])
        else:
            msg="Cannot handle array of shape {}.".format(freal.shape)
            raise NotImplementedError(msg)

    field = np.transpose(np.squeeze(field))

    if crop_pml:
        info = get_sim_info(os.path.dirname(h5file))
        crop = info["pmlsize [px]"]
        field = crop_arr(field, crop)

    return field


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


def get_ri_structure(path, crop_pml=True, outd=None, outf=None, savepng=False):
    """Obtain the 3D refractive index from an .h5 file
    
    Parameters
    ----------
    path: str
        Path to the .h5 file or a folder containing it. In the second
        case, we will look for a file similar to "eps-000000.00.h5".
    crop_pml: bool
        Crops the perfectly matching layer from the simulation

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
        riref = np.sqrt(np.array(h5fd.items()[0][1]))
    
    if crop_pml:
        info = get_sim_info(os.path.dirname(path))
        crop = info["pmlsize [px]"]
        riref = crop_arr(riref, crop)

    # Make a cubic array
    rshape = np.shape(riref)
    if len(rshape) == 2:   #2D
        (xs, ys) = rshape
        dpadl = np.ceil((xs-ys)/2)
        dpadr = np.floor((xs-ys)/2)
        riref = np.pad(riref,((0,0),(dpadl,dpadr)), mode="edge").transpose()
    else:   #3D
        (xs, ys, zs) = np.shape(riref)
        dpadl = np.ceil((xs-zs)/2)
        dpadr = np.floor((xs-zs)/2)
        riref = np.pad(riref,((0,0),(0,0),(dpadl,dpadr)), mode="edge").transpose()

    if dpadl != dpadr:
        warnings.warn("Uneven cell size! Reference is shifted by "+
                      "half a pixel. -> Comparison might be faulty.")

    # return RI
    return riref



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
    
    if "cytoplasm_a" in info:
        info["lateral_object_size"] = 2*info["cytoplasm_a"]
    if "cytoplasm_b" in info:
        if "cytoplasm_c" in info:
            info["axial_object_size"] = 2*max(info["cytoplasm_b"],info["cytoplasm_c"])
        else:
            info["axial_object_size"] = 2*info["cytoplasm_b"]
    
    wl = info["wavelength"]
    for kk in list(info.keys()):
        if kk.endswith(("size", "_a", "_b", "_c", "_x", "_y", "_z")):
            info[kk+" [px]"] = info[kk]*wl
            info[kk+" [wavelengths]"] = info[kk]
            info.pop(kk)

    return info


def get_tomo_dirlist(tomo_path):
    """Returns all relevant simulation directories of a tomographic simulation
    
    Parameters
    ----------
    tomo_path: str
        Path to a tomographic simulation series
    
    Returns
    -------
    bg, [ph1, ph2, ...]: list of str
        Paths to the background simulation "bg" and to the
        phantom simulations.
    """
    dirs = os.listdir(tomo_path)
    dirs = [ d for d in dirs if os.path.isdir(os.path.join(tomo_path,d)) ]
    phs = [ d for d in dirs if d.startswith("ph_") ]
    bgs = [ d for d in dirs if d.startswith("bg_") ]
    assert len(bgs)==1, "None or more than one bg simulation found!"
    assert len(phs), "No phantom simulations found!"
    bg = os.path.join(tomo_path, bgs[0])
    phs = [ os.path.join(tomo_path,p) for p in phs ]
    phs.sort()
    return bg, phs


def get_tomo_ri_structure(tomo_path, crop_pml=True):
    """Obtain the 3D refractive index from a tomogaphic simulation series
    
    Same as `get_ri_structure`, except it works with a tomography simulation
    directory and returns the structure with the smallest rotation angle.

    Parameters
    ----------
    tomo_path: str
        Path to a tomographic simulation series
    """
    _bg, phs = get_tomo_dirlist(tomo_path)
    npyfile = os.path.abspath(tomo_path)+"_results/ri_structure.npy"
    common.mkdir_p(os.path.dirname(npyfile))
    if not os.path.exists(npyfile):
        riref = get_ri_structure(phs[0], crop_pml=crop_pml)
        np.save(npyfile, riref)
    else:
        riref = np.load(npyfile)
    return riref


def get_tomo_sinogram_at_ld(tomo_path, ld):
    """Returns the sinogram from a tomographic simulation
    
    Parameters
    ----------
    tomo_path: str
        Path to tomographic soimulation directory
    ld: int
        The axial position from the center of the image in pixels.
        For even images, the center is the right hand pixel
        of the actual center (see `get_field_at_ld`).

    """
    bg, phs = get_tomo_dirlist(tomo_path)
    bgfield = get_field_at_ld(bg)
    fields = []
    for ph in phs:
        fields.append(get_field_at_ld(ph)/bgfield)
    return np.array(fields)


def get_tomo_angles(tomo_path):
    _bg, phs = get_tomo_dirlist(tomo_path)
    angles = []
    for f in phs:
        f = os.path.basename(f)
        angles.append(float(f.split("_")[-1].strip("out-")))
    angles = np.array(angles)
    return angles
