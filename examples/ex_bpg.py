#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""backpropagation methods using ODTbrain and radontea"""
from __future__ import division, print_function, unicode_literals

import argparse
import matplotlib.pylab as plt
import numpy as np
import os
import re
import sys

import nrefocus
import odtbrain as odt
import radontea as rt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))+"/../meep_tomo")

from meep_tomo import extract, common, postproc


def backpropagate_fdtd_data(tomo_path,
                            approx,
                            ld_offset = 1,
                            autofocus = True,
                            interpolate=False,
                            force=False,
                            verbose=0):
    
    assert approx in ["radon", "born", "rytov"]
    
    res_dir = get_results_dir(tomo_path)
    
    # Determine name of resulting npy file
    name = "ri_{}_lmeas{}".format(approx, ld_offset)
    if autofocus:
        name += "_af"
    if interpolate:
        name += "_intp{}".format(interpolate)
    name += "_{}.npy".format(os.path.basename(tomo_path))
    name = os.path.join(res_dir, name)
    
    if os.path.exists(name) and not force:
        if verbose:
            print("...Using existing refractive index: {}".format(name))
        ri = np.load(name)
    else:
        # Get some parameters
        _bg, phs = extract.get_tomo_dirlist(tomo_path)
        info = extract.get_sim_info(phs[0])
        res = info["wavelength [px]"]
        nm = info["medium_ri"]

        sino, angles = get_sinogram(tomo_path, ld_offset, autofocus=autofocus, force=force)
        
        if autofocus:
            ld = 0
        else:
            ld = ld_offset

        ri = backpropagate_sinogram(sinogram=sino,
                                    angles=angles,
                                    approx=approx,
                                    res=res,
                                    nm=nm,
                                    ld=ld)
        # save ri
        np.save(name, ri)
        
    return ri
    

def backpropagate_sinogram(sinogram,
                           angles,
                           approx,
                           res,
                           nm,
                           ld=0,
                           ):

    sshape = len(sinogram.shape)
    assert sshape in [2,3], "sinogram must have dimension 2 or 3"


    print("...distance lD:", ld)
    uSin = sinogram
    assert approx in ["radon", "born", "rytov"]


    if approx == "rytov":
        uSin = odt.sinogram_as_rytov(uSin)
    elif approx == "radon":
        uSin = odt.sinogram_as_radon(uSin)
    
    if approx in ["born", "rytov"]:
        # Perform reconstruction with ODT
        if sshape == 2:
            f = odt.backpropagate_2d(uSin,
                                     angles=angles,
                                     res=res,
                                     nm=nm,
                                     lD=ld
                                     )
        else:
            f = odt.backpropagate_3d(uSin,
                                     angles=angles,
                                     res=res,
                                     nm=nm,
                                     lD=ld
                                     )

        ri = odt.odt_to_ri(f, res, nm)
    else:
        # Perform reconstruction with OPT
        # works in 2d and 3d
        f = rt.backproject(uSin, angles=angles)
        ri = odt.opt_to_ri(f, res, nm)                

    return ri


def get_results_dir(tomo_path):
    res_dir = os.path.abspath(tomo_path)+"_results"
    common.mkdir_p(res_dir)
    return res_dir


def get_sinogram(tomo_path,
                 ld_offset=1,
                 interpolate=False,
                 autofocus=True,
                 force=False):
    """ Retreives the sinogram
    
    Either from an existing sino_lmeasXX_tsYY[_af]_intpZZ.npy or from experimental files.
    
    Parameters
    ----------
    directory : str
        Location to look for sinogram data
    distance_from_phantom : int
        Distance from center of simulation [px]
    interpolate : int or False
        Perform interpolation?
    autofocus : bool
        Perform autofocusing?
    timestep : int
        If there are multiple h5 files (timesteps), take this one.
    """
    res_dir = get_results_dir(tomo_path)
    _bg, phs = extract.get_tomo_dirlist(tomo_path)
    info = extract.get_sim_info(phs[0])

    res = info["wavelength"]
    nmed = info["medium_ri"]

    ld_guess = (ld_offset + info["axial_object_size [wavelengths]"]/2)*res

    # raw sinogram
    rawname = "sinogram_raw_{}.npy".format(ld_guess)
    rawname = os.path.join(res_dir, rawname)
    if os.path.exists(rawname) and not force:
        print("...Loading extracted sinogram")
        sino_raw = np.load(rawname)
    else:
        print("...Extracting sinogram")
        sino_raw = extract.get_tomo_sinogram_at_ld(tomo_path, ld=ld_guess)
        np.save(rawname, sino_raw)
    
    # processed sinogram
    name = "sinogram_lmeas{}".format(ld_guess)
    if autofocus:
        name += "_af"
    if interpolate:
        name += "_intp{}".format(interpolate)
    name += "_{}.npy".format(os.path.basename(tomo_path))
    name = os.path.join(res_dir, name)

    if os.path.exists(name) and not force:
        print("...Using preprocessed sinogram: {}".format(name))
        u = np.load(name)
    else:
        print("...Processing raw sinogram: {}".format(name))
        
        if interpolate:
            assert len(sino_raw.shape) == 3, "interpolation only for 3D data!"
            u = []
            for ii in range(sino_raw.shape[0]):
                uii = postproc.interpolate_field(sino_raw[ii], interpolate)
                u.append(uii)
            u = np.array(u)
        else:
            u = sino_raw
        
        if autofocus:
            print("......Performing autofocusing")
            u, dopt, gradient = nrefocus.autofocus_stack(u, nmed,
                                                         res, ival=(-1.5*ld_guess, 0), 
                                                         same_dist=True, 
                                                         ret_ds=True, ret_grads=True)
            print("Autofocusing distance:", np.average(dopt))
            print(nmed)
            # save gradient
            plt.figure(figsize=(4,4), dpi=600)
            plt.plot(gradient[0][0][0], gradient[0][0][1], color="black")
            plt.plot(gradient[0][1][0], gradient[0][1][1], color="red")
            plt.xlabel("distance from original slice")
            plt.ylabel("average gradient metric")
            plt.tight_layout()
            plt.savefig(os.path.join(res_dir, "Refocus_Gradient.png"))
            plt.close()

        # save sinogram
        np.save(name, u)
    
    
    angles = extract.get_tomo_angles(tomo_path)
    
    return u, angles


def save_cross_sections(directory, ri, wavelength=None, cut=(2,2,2),
                        name=None):
    DIR = directory
    FINALOUT = get_resultsdir(DIR)
    
    if name is None:
        name = os.path.split(DIR)[1]
    
    name = os.path.split(name)[1]
    if not name.endswith(".png"):
        name += ".png"
    
    riref = get_phantom_refractive_index(DIR)
    BGDIR, dirlist = tool.GetDirlistSimulation(DIR)
    info = tool.GetInfoFromFolder(dirlist[-1])
    res = info["Sampling per wavelength [px]"]
    if wavelength is not None:
        px_um = res*(1000/wavelength)
    else:
        px_um = None
    
    ndims = len(ri.shape)

    cut = list(cut)
    
    if ndims == 2:
        cx = cut[0] * res
        cy = cut[1] * res
        tool.arr2im(riref, scale=True).save(os.path.join(FINALOUT,
                                  "diel_{}".format(name)))
        tool.arr2im(ri.real, scale=True).save(os.path.join(FINALOUT,
                                  "recon_{}".format(name)))
        tool.saveprofile_xy(1*ri.real, "profile_"+name, FINALOUT,
                            cut=(cx,cy), ref=riref.real, px_um=px_um)
    else:
        cx = cut[0] = cut[0] * res
        cy = cut[1] = cut[1] * res
        cz = cut[2] = cut[2] * res
        tool.saveslice_xyz(ri.real, name[:-4]+".png", FINALOUT, cut=cut)
        tool.saveslice_xyz(riref, "ri_ref.png", FINALOUT, cut=cut)
        tool.saveprofile_xyz(ri.real, name[:-4]+".png", FINALOUT,
                             ref=riref, cut=cut, px_lambs=1/res)

    np.save(os.path.join(FINALOUT, "ri_reference.npy"), riref)


def sort_folder_list(folders):
    # sort the folders according to the appearance of values in the name
    sortlist = list()
    non_decimal = re.compile(r'[^\d.]+')
    
    for af in folders:
        af = af.split("_")
        af = [ float(non_decimal.sub('0', a)) for a in af ]
        sortlist.append(af)

    numf = len(folders)
    sortlist = np.array(sortlist)
    # iteratively sort the folder list
    while True:
        indices = np.argsort(sortlist, axis=0)
        for i in range(indices.shape[1]):
            indices = np.argsort(sortlist, axis=0)
            if not np.allclose(sortlist[:,i], sortlist[0,i]):
                sortlist = np.array(sortlist[indices[:,i]].tolist())
                folders = np.array(folders)[indices[:,i]].tolist()
                continue
        break
    return folders


if __name__ == "__main__":
    out_dir = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(
            description='Inverse of FDTD tomorgaphy simulation.',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-a', '--approximation', type=str, default="rytov",
                        help='approximation: "born", "rytov, or "radon"')
    parser.add_argument('directory', type=str, default=out_dir,
                        help='directory of FDTD simulation')
    parser.add_argument('-l', '--detector_distance', type=int, default=1,
                        help='take field at distance from phantom'+\
                             ' in wavelengths')
    parser.add_argument('-f', '--autofocus', type=int, default=True,
                        help='perform autofocusing')


    args = parser.parse_args()
    out_dir = os.path.abspath(args.directory)
    
    if not os.path.exists(out_dir):
        print("Could not find {}".format(out_dir))
        print("Please specify directory with an argument.")
        exit()

    ri = backpropagate_fdtd_data(directory=out_dir,
                                 approx=args.approximation,
                                 ld_offset=args.detector_distance,
                                 autofocus=args.autofocus)
    
