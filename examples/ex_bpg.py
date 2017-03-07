#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""backpropagation methods using ODTbrain and radontea"""
from __future__ import division, print_function

import argparse
import numpy as np
import os
import re
import sys
import unwrap
import warnings

import nrefocus
import odtbrain as odt
import radontea as rt

import fdtd_tool as tool


def backpropagate_fdtd_data(directory,
                            approximation,
                            distance_from_phantom = 1,
                            timestep = -1,
                            autofocus = True,
                            interpolate=False,
                            save_png=True,
                            wavelength=None,
                            override=False):
    
    DIR = directory
    APPROXIMATION = approximation
    lD_meas = distance_from_phantom
    TIMESTEP = timestep
    AUTOFOCUS = autofocus
    INTERPOLATE = interpolate
    WAVELENGTH = wavelength
    
    FINALOUT = get_resultsdir(DIR)
    
    assert approximation in ["radon", "born", "rytov"]
    
    name = "ri_{}_lmeas{}_ts{}".format(APPROXIMATION, lD_meas,
                                       TIMESTEP)
    if AUTOFOCUS:
        name += "_af"
    if INTERPOLATE:
        name += "_intp{}".format(INTERPOLATE)
    name += "_{}.npy".format(os.path.split(DIR)[1])
    name = os.path.join(FINALOUT, name)
    
    if os.path.exists(name) and not override:
        print("...Using existing refractive index: {}".format(name))
        ri = np.load(name)
    else:
        # Get some parameters
        BGDIR, dirlist = tool.GetDirlistSimulation(DIR)
        info = tool.GetInfoFromFolder(dirlist[-1])
        res = info["Sampling per wavelength [px]"]
        lD = (lD_meas + info["Axial object size [wavelengths]"]/2)*res
        nm = info["Medium RI"]
        # how many pixels from the border (PML) should we remove from the images?
        pxcut = int(np.ceil(info["PML thickness [wavelengths]"] * res))

        uSin = get_sinogram(DIR, lD_meas, autofocus=AUTOFOCUS, timestep=TIMESTEP,
                            save_png=save_png, override=override)
        
        #uSin /= np.abs(uSin)
        
        if AUTOFOCUS:
            lD = 0

        angles = get_sinogram_angles(DIR)

        #np.savetxt("fdtd_real.txt", uSin[0].real, fmt="%.4f")
        #np.savetxt("fdtd_imag.txt", uSin[0].imag, fmt="%.4e")
        #with open("fdtd_info.txt", "w") as tf:
        #    tf.write("nm = {}\n".format(nm))
        #    tf.write("lD = {}\n".format(lD))
        #    tf.write("res = {}\n".format(res))
        

        #np.savetxt("fdtd_angles.txt", angles)
        #np.savetxt("fdtd_real.txt", uSin.real)
        #np.savetxt("fdtd_imag.txt", uSin.imag)
        #with open("fdtd_info.txt", "w") as tf:
        #    tf.write("nm = {}\n".format(nm))
        #    tf.write("lD = {}\n".format(lD))
        #    tf.write("res = {}\n".format(res))
        #riref = get_phantom_refractive_index(DIR)
        #np.savetxt("fdtd_phantom.txt", riref)

        ri = backpropagate_sinogram(sinogram=uSin,
                                    angles=angles,
                                    approximation=APPROXIMATION,
                                    res=res,
                                    nm=nm,
                                    distance_center_detector=lD)

        # save ri
        np.save(name, ri)
        
        
    if save_png:
        save_cross_sections(DIR, ri, WAVELENGTH, name=name[:-4])
    
    return ri
    

def backpropagate_sinogram(sinogram,
                           angles,
                           approximation,
                           res,
                           nm,
                           distance_center_detector=0,
                           ):

    sshape = len(sinogram.shape)
    assert sshape in [2,3], "sinogram must have dimension 2 or 3"

    APPROXIMATION = approximation
    lD = distance_center_detector
    print("distance lD:", lD)
    uSin = sinogram
    assert approximation in ["radon", "born", "rytov"]


    if APPROXIMATION == "rytov":
        uSin = odt.sinogram_as_rytov(uSin)
    elif APPROXIMATION == "radon":
        uSin = odt.sinogram_as_radon(uSin)
    
    if APPROXIMATION in ["born", "rytov"]:
        # Perform reconstruction with ODT
        if sshape == 2:
            f = odt.backpropagate_2d(uSin,
                                     angles = angles,
                                     res = res,
                                     nm = nm,
                                     lD = lD
                                )
        else:
            f = odt.backpropagate_3d(uSin,
                                     angles = angles,
                                     res = res,
                                     nm = nm,
                                     lD = lD
                                )

        ri = odt.odt_to_ri(f, res, nm)
    else:
        # Perform reconstruction with OPT
        # works in 2d and 3d
        f = rt.backproject(uSin, angles=angles)
        ri = odt.opt_to_ri(f, res, nm)                

    return ri


def get_phantom_refractive_index(directory):
    BGDIR, dirlist = tool.GetDirlistSimulation(directory)
    info = tool.GetInfoFromFolder(dirlist[-1])
    res = info["Sampling per wavelength [px]"]
    pxcut = int(np.ceil(info["PML thickness [wavelengths]"] * res))
    files = os.listdir(dirlist[0])
    for f in files:
        if f.startswith("eps-") and f.endswith(".h5"):
            epsname = f
    riref = tool.GetDielecticStructure(dirlist[0], epsname, cut=pxcut)
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

    return riref



def get_resultsdir(directory):
    FINALOUT = os.path.abspath(directory)+"_results"
    tool.mkdir(FINALOUT)
    return FINALOUT



def get_sinogram(directory,
                 distance_from_phantom,
                 interpolate=False,
                 autofocus=True,
                 timestep=-1,
                 save_png=True,
                 override=False
                 ):
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
    FINALOUT = get_resultsdir(directory)
    INTERPOLATE = interpolate
    AUTOFOCUS = autofocus
    DIR = directory
    lD_meas = distance_from_phantom
    
    res = tool.get_info_resolution(directory)
    pxcut = tool.get_info_pml_px(directory)
    aos = tool.get_info_axial_obj_size(directory)
    nmed = tool.get_info_medium_ri(directory)

    lD = (lD_meas + aos/2)*res


    name = "sinogram_lmeas{}_ts{}".format(lD, timestep)

    if AUTOFOCUS:
        name += "_af"
    if INTERPOLATE:
        name += "_intp{}".format(INTERPOLATE)
    
    name += "_{}.npy".format(os.path.split(DIR)[1])

    name = os.path.join(FINALOUT, name)

    if os.path.exists(name) and not override:
        print("...Using existing sinogram: {}".format(name))
        u = np.load(name)
    else:
        files = os.listdir(DIR)
        print("...Computing sinogram: {}".format(name))
        u = list()
        BGDIR, dirlist = tool.GetDirlistSimulation(directory)


        # Get background image
        bgfiles = tool.GetH5files(BGDIR)
        BE = tool.GetFieldAtLineFromCenter(BGDIR, bgfiles[timestep], lD)
        BEx = tool.CropPML(BE,pxcut)
        
        for d in range(len(dirlist)):
            print(dirlist[d])
            sys.stdout.flush()
            # input directories
            EPSDIR = dirlist[d]
            dirstr = os.path.split(EPSDIR)[1]
            
            # Get all entries in the directory
            files = tool.GetH5files(EPSDIR)
            # Actual fields, line scans

            if len(files) == 0:
                continue
            E = tool.GetFieldAtLineFromCenter(EPSDIR, files[timestep], lD)

            Ex = tool.CropPML(E, pxcut)
            
            if INTERPOLATE:
                Ex = tool.InterpolateFields(Ex, INTERPOLATE)

            Field_nf = Ex/BEx
            u.append(Field_nf)
    
        u = np.array(u)
        
        if AUTOFOCUS:
            print("......Performing autofocusing")
            u, dopt, gradient = nrefocus.autofocus_stack(u, nmed,
                                      res, ival=(-1.5*lD, 0), 
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
            plt.savefig(os.path.join(FINALOUT, "Refocus_Gradient.png"))
            plt.close()

        # save sinogram
        np.save(name, u)
    
    if save_png:
        if len(u.shape)==2:
            tool.arr2im(unwrap.unwrap(np.angle(u)), scale=True).save(
                               os.path.join(FINALOUT, name[:-4]+".png"))
        else:
            # save one slice
            cpos = int(u.shape[2]/2)
            tool.arr2im(unwrap.unwrap(np.angle(u[:,:,cpos])), scale=True).save(
                               os.path.join(FINALOUT, name[:-4]+".png"))
            # save angle series
            sga = os.path.join(FINALOUT, "sino_amp")
            sgp = os.path.join(FINALOUT, "sino_pha")

            tool.saveslice_xyz(np.abs(u), "amplitude", sga)
            tool.saveslice_xyz(np.angle(u), "phase", sgp,
                               unwrap_data=True)
    
    return u


def get_sinogram_angles(directory):
    BGDIR, dirlist = tool.GetDirlistSimulation(directory)
    angles = list()
    for f in dirlist:
        f = os.path.split(f)[1]
        angles.append(float(f.split("_")[-1].strip("out-")))
    angles = np.array(angles)
    return angles


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

    DIR = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(
            description='Inverse of FDTD tomorgaphy simulation.',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-a', '--approximation', type=str,
                        default="rytov",
                        help='approximation: "born", "rytov, or "radon"')
    parser.add_argument('directory',  metavar='DIR', type=str,
                        default=DIR,
                        help='directory of FDTD simulation')
    parser.add_argument('-l', '--detector_distance',  metavar='lD', type=int,
                        default=1,
                        help='take field at distance from phantom'+\
                             ' in wavelengths')
    parser.add_argument('-t', '--timestep', type=int,
                        default=-1,
                        help='take this timestep from simulation')
    parser.add_argument('-f', '--autofocus', type=int,
                        default=True,
                        help='perform autofocusing')
    parser.add_argument('-w', '--wavelength', type=int,
                        default=500,
                        help='wavelength in nm')

    args = parser.parse_args()
    DIR = os.path.abspath(args.directory)
    APPROXIMATION = args.approximation
    lD_meas = args.detector_distance
    TIMESTEP = args.timestep
    AUTOFOCUS = args.autofocus
    WAVELENGTH = args.wavelength
    
    if not os.path.exists(DIR):
        print("Could not find {}".format(DIR))
        print("Please specify directory with an argument.")
        exit()

    FINALOUT = get_resultsdir(DIR)

    ri = backpropagate_fdtd_data(directory=DIR,
                                 approximation=APPROXIMATION,
                                 distance_from_phantom=lD_meas,
                                 timestep=TIMESTEP,
                                 autofocus=AUTOFOCUS,
                                 save_png=True,
                                 wavelength=WAVELENGTH)
    
