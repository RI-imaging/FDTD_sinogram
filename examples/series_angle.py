#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    FDTD_backpropagate_2d.py
    
    Backpropagate data that has been produces with compile_and_run.py.

    - The first argument is the working directory (can also be edited
      in the script)

python angle_series_quality.py /media/data/simulations/phantom_3d/phantom_3d_A0240_R13_T00015000_Nmed1.333_Ncyt1.365_Nnuc1.36_Nleo1.387/


"""
from __future__ import division, print_function, unicode_literals

import matplotlib as mpl
mpl.use('Agg')


import argparse
import gc
from matplotlib import pylab as plt
import numpy as np
import os
import re
from scipy import interpolate
import sys
import unwrap


import ex_bpg
import ex_toolkit


def get_matching_angle_series(directory):
    """
       from a directory
       
           phantom_2d_A0061_R13_T00015000
           
       find all directories, that have a different angle, e.g.
       
           phantom_2d_A0071_R13_T00015000
        
       but not
        
            phantom_2d_A0061_R14_T00015000
    """
    root, td = os.path.split(directory)
    
    # identifiers are items that do not start with A (angle)
    ids = td.split("_")
    ids = [ (not i.startswith("A"))*i  for i in ids ]
    
    # find all folders that match
    folderlist = os.listdir(root)
    newfolderlist = list()
    
    for folder in folderlist:
        if (not folder.endswith("_results")) and os.path.isdir(os.path.join(root,folder)):
            fids = folder.split("_")
            fids = [ (not i.startswith("A"))*i  for i in fids ]
            # There was a problem with successive simulations having
            # different values for e.g. Nnuc: 1.36 vs. 1.359999999999.
            # We fix this with this for loop:
            for i in range(len(fids)):
                wrong = 0
                if fids[i] == ids[i]:
                    pass
                elif alphanum2num(fids[i]) != "":
                    if (np.allclose(alphanum2num(fids[i]),
                                    alphanum2num(ids[i]))):
                        pass
                    else:
                        wrong += 1
                else:
                    wrong += 1
                
            if wrong == 0:
                #print(folder)
                newfolderlist.append(os.path.join(root, folder))
    
    newfolderlist = sort_folder_list(newfolderlist)
    
    return newfolderlist



def divisorGenerator(n):
    for i in xrange(1,int(n/2)+1):
        if n%i == 0: 
            yield i



def get_divisor_sinograms(directory,
                          distance_from_phantom,
                          interpolate=False,
                          autofocus=True,
                          timestep=-1,
                          save_png=True,
                          override=False
                          ):
    """ Given a directory, return divisor sinogram subslices.
    
    Parameters are passed to get_sinogram.
    """
    DIR = directory
    lD_meas = distance_from_phantom
    TIMESTEP = timestep
    AUTOFOCUS = autofocus
    INTERPOLATE = interpolate


    #name = "ri_{}_lmeas{}_ts{}".format(APPROXIMATION, lD_meas,
    #                                   TIMESTEP)
    #if AUTOFOCUS:
    #    name += "_af"
    #if INTERPOLATE:
    #    name += "_intp{}".format(INTERPOLATE)
    #name += "_{}.npy".format(os.path.split(DIR)[1])
    #name = os.path.join(FINALOUT, name)

    uSin = get_sinogram(directory=directory,
                        distance_from_phantom=distance_from_phantom,
                        interpolate=interpolate,
                        autofocus=autofocus,
                        timestep=timestep,
                        save_png=save_png,
                        override=override
                        )

    angles = get_sinogram_angles(f)

    A = uSin.shape[0]
    
    # This list contains number of angles used for reconstructions,
    # and the filenames of the reconstructions.
    #rilist = [[A, name]]

    # define the fileneames of the divisor reconstructions.
    
    # list of sinograms
    sinograms = list()
    sino_angles = list()
    
    
    for d in divisorGenerator(int(A)):
        sinograms.append(uSin[::d])
        sino_angles.append(angles[::d])
#        if A == 240:
#            print("Sinogram {}; divided by {} is {}".format(A, d, A/d))
#    
#    if A == 240:
#        sys.exit()
    
    return sinograms, sino_angles


if __name__ == "__main__":

    DIR = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(
            description='Inverse from FDTD tomoraphy simulation. '+\
                        'Makes use of the divisors.',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
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
    parser.add_argument('-w', '--wavelength', type=int,
                        default=500,
                        help='wavelength in nm')
    parser.add_argument('-m', '--minimum_angle', type=int,
                        default=2,
                        help='Minimum angle to plot (b/c of noise)')

    args = parser.parse_args()
    DIR = os.path.abspath(args.directory)
    lD_meas = args.detector_distance
    TIMESTEP = args.timestep
    AUTOFOCUS = True                #force autofocusing
    WAVELENGTH = args.wavelength
    MIN_ANG = args.minimum_angle
    
    if not os.path.exists(DIR):
        print("Could not find {}".format(DIR))
        print("Please specify directory with an argument.")
        exit()

    FINALOUT = get_resultsdir(DIR)

    folders = get_matching_angle_series(DIR)
    
    sum_square = dict()
    tv_diff = dict()
    
    for APPROX in ["rytov", "born", "radon"]:

        sum_square[APPROX] = list()
        tv_diff[APPROX] = list()
    
        for f in folders:
            finalout_loc = get_resultsdir(f)
            # get divisior sinograms
            uSin_div, angles_div = get_divisor_sinograms(
                                        directory=f,
                                        distance_from_phantom=lD_meas,
                                        timestep=TIMESTEP,
                                        autofocus=AUTOFOCUS,
                                        save_png=False,
                                        override=False)

            res = tool.get_info_resolution(f)
            nm = tool.get_info_medium_ri(f)
            
            riref = get_phantom_refractive_index(f)
            
            for s in range(len(uSin_div)):

                name = "ri_divisor{:04d}_{}_{}.npy".format(
                                         s, APPROX, os.path.split(f)[1])
                name = os.path.join(finalout_loc, name)
                if os.path.exists(name):
                    print("Using existing ri: {}".format(name))
                    # Do not load npy file, we don't need it anymore here.
                    # We only need the ss, tvd (computed separately)
                    #ri = np.load(name)
                else:
                    ri = backpropagate_sinogram(sinogram=uSin_div[s],
                                                angles=angles_div[s],
                                                approximation=APPROX,
                                                res=res,
                                                nm=nm
                                               )
                    np.save(name, ri)
                    del ri
                    
                ss, tvd = compute_metrices(name, APPROX)
            
                A = len(angles_div[s])

                sum_square[APPROX].append([A, ss])
                tv_diff[APPROX].append([A, tvd])
            
            del uSin_div
            del angles_div
            del riref
            gc.collect()
            
            # Now the .npy file containing the sinogram is on the hard
            # disk. We can use it to create divisor-sinograms and add
            # them to the results.


    OUT = os.path.dirname(DIR)
    name = os.path.split(DIR)[1]
    temp = [ (not i.startswith("A"))*i  for i in name.split("_") ]
    name = "angle_series_"+os.path.split(OUT)[1]
    for t in temp:
        name += "_"+t
    name = name.strip("_")+".png"
    
    
    # Create a plot in the parent directory
    fig , (ax1, ax2) = plt.subplots(2, 1)
    approxs = list(sum_square.keys())
    approxs.sort()
    colors = ["blue", "black", "red"]

    savedata = []

    for i in range(len(approxs)):
        appr_data = sum_square[approxs[i]]
        appr_data.sort()
        appr_data = np.array(appr_data)
        maid = np.sum(appr_data[:,0] < MIN_ANG)
        x = appr_data[maid:,0]
        y1 = np.array(appr_data[maid:,1])
        ax1.plot(x,y1, color=colors[i], label=approxs[i].capitalize())
        
        if i == 0:
            # x values
            savedata.append(x)
        savedata.append(y1)
        
        
    for i in range(len(approxs)):
        appr_data = tv_diff[approxs[i]]
        appr_data.sort()
        appr_data = np.array(appr_data)
        maid = np.sum(appr_data[:,0] < MIN_ANG)
        x = appr_data[maid:,0]
        y2 = np.array(appr_data[maid:,1])
        ax2.plot(x,y2, color=colors[i], label=approxs[i].capitalize())
        savedata.append(y2)

    ax1.legend(loc=1)
    ax2.legend(loc=1)
    ax1.set_xlabel("total number of projections")
    ax2.set_xlabel("total number of projections")
    ax1.set_ylabel("nrms error")
    ax2.set_ylabel("ntv error")
    ax1.set_yscale("log")
    #ax2.set_yscale("log")
    ax1.grid()
    ax2.grid()

    x = np.array(x)
    
    dx = (x.max()-x.min())*.03
    ax1.set_xlim(x.min()-dx, x.max()+dx)
    ax2.set_xlim(x.min()-dx, x.max()+dx)
#    ax1.set_ylim(y1.min()*.9, y1.max()*1.1)
#    ax2.set_ylim(y2.min()*.9, y2.max()*1.1)

    plt.tight_layout()

    plt.savefig(os.path.join(OUT, name))
    plt.savefig(os.path.join(os.path.dirname(OUT), name))
    plt.close()
    
    with open(os.path.join(os.path.dirname(OUT),name+".txt"), "w") as outfile:
        outfile.write("# lines: x, ss_born, ss_radon, ss_rytov, tv_born, tv_radon, tv_rytov\n")
        np.savetxt(outfile, savedata)
    
