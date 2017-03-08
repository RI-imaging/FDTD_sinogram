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


from matplotlib import pylab as plt
import numpy as np
from os.path import abspath, dirname, exists, join
import sys

sys.path.insert(0, dirname(abspath(__file__))+"/../meep_tomo")

import ex_toolkit
import ex_bpg

import meep_tomo as mt


def divisor_gen(n):
    for i in xrange(1,int(n/2)+1):
        if n%i == 0: 
            yield i


if __name__ == "__main__":
    print("Starting simulation. This will take a while!")
    # Run the simulation
    here = dirname(abspath(__file__))
    ph_file = here+"/../phantoms_meep/phantom_2d.cpp"
    dir_out = here+"/simulations/series_angle"
    dir_res = here+"/simulations/series_angle_results"
    num_angles = 360
    wavelength = 13.
    medium_ri = 1.333
    mt.meep.run_tomography(phantom_template=ph_file,
                           num_angles=num_angles,
                           dir_out=dir_out,
                           timesteps=15000,
                           wavelength=wavelength,
                           medium_ri=medium_ri,
                           verbose=1)


    print("Obtaining full sinogram!")
    # Get full, autofocused sinogram
    sino, angles = ex_bpg.get_sinogram(dir_out)


    print("Performing reconstructions. This will take a while!")
    mt.common.mkdir_p(dir_res)
    tv_metrices = []
    rms_metrices = []
    # Compute all possible divisors for the number of angles
    for dd in divisor_gen(int(num_angles)):
        # Get subsection of sinogram
        subsin=1*sino[::dd]
        subang=1*angles[::dd]
        # Perform reconstructions
        for approx in ["rytov", "born", "radon"]:
            npyfile = "divisor{}_{}.npy".format(dd, approx) 
            npyfile = join(dir_res, npyfile)
            if not exists(npyfile):
                print("...recostr divisor {} w/approx. {}".format(dd, approx))
                ri = ex_bpg.backpropagate_sinogram(sinogram=subsin,
                                                   angles=subang,
                                                   approx=approx,
                                                   res=wavelength,
                                                   nm=medium_ri,
                                                   ld=0)
                np.save(npyfile, ri)
            else:
                print("...using {}".format(npyfile))
            # Compute metrices
            rms, tv = ex_toolkit.compute_metrices(npyfile, approx=approx)
            tv_metrices.append([approx, dd, tv])
            rms_metrices.append([approx, dd, rms])
            
    # Plot the result
    