#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Error estimation for varying number of sinogram images (Born, Radon, Rytov)

This example reproduces figure 1 and partially figure 3 of [1], showcasing
the usage of the Python wrapper meep_tomo for MEEP FDTD simulations and
the subsequent tomographic reconstruction using ODTbrain (backpropagation),
radontea (backpojection), and nrefocus (numerical focusing).

Notes
-----
This script will create a folder "simulations/series_angle" (~750MB) containing
the MEEP simulation data and a folder "simulations/series_angle_results"
(~150MB) that contains the postprocessing data including sinograms and
3D refractive index reconstructions in the .npy file format.

To be able to test the three reconstruction methods (Born, Radon, Rytov)
for the experimental case, the sinogram data are autofocused. However,
autofocusing does not always find the correct focal position (the center
of the rotation), which introduces errors in the reconstruction. This
script has a parameter `autofocusing` which, when set to `False` will use
the exact focusing distance (best-case-scenario). In this case, the
gradient of the nucleolus is recovered much better than for autofocusing
(series_angle_b_exact_focus.png vs. series_angle_b_autofocusing.png).
The overall message stays the same: The root-mean-square (RMS) and
total variation (TV) errors are lowest for the Rytov approximation
(series_angle_a_exact_focus.png vs. series_angle_a_autofocusing.png). 



[1] Paul Müller, Mirjam Schürmann and Jochen Guck, "ODTbrain: a Python library
for full-view, dense diffraction tomography", Bioinformatics 2015,
DOI: 10.1186/s12859-015-0764-0
"""
from __future__ import division, print_function, unicode_literals

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
            yield n//i


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
    autofocus = True
    mt.meep.run_tomography(phantom_template=ph_file,
                           num_angles=num_angles,
                           dir_out=dir_out,
                           timesteps=15000,
                           wavelength=wavelength,
                           medium_ri=medium_ri,
                           verbose=1)

    print("...Computing angle series")
    sino, angles = ex_bpg.get_sinogram(dir_out, autofocus=autofocus)
    mt.common.mkdir_p(dir_res)
    tv_metrices = []
    rms_metrices = []
    # Compute all possible divisors for the number of angles
    for dd in divisor_gen(int(num_angles)):
        # Get subsection of sinogram
        subsin=1*sino[::num_angles//dd]
        subang=1*angles[::num_angles//dd]
        # Perform reconstructions
        for approx in ["rytov", "born", "radon"]:
            if autofocus:
                ap="_af"
            else:
                ap=""
            npyfile = "divisor{}_{}{}.npy".format(dd, approx, ap) 
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
            rms, tv = ex_toolkit.compute_metrices(npyfile,
                                                  approx=approx)
            tv_metrices.append([approx, dd, tv])
            rms_metrices.append([approx, dd, rms])

    print("...Creating figure 1")
    colors = ['#3925f9', "#3c3c3c", "#d84242"]
    kwargs_ri=dict(vmin=1.333, vmax=1.387, cmap=plt.cm.get_cmap("Spectral_r"))
    fig1 = plt.figure("Figure 1 (Müller et. al, Bioinformatics 2015)", (10, 10))
    grid = [plt.subplot2grid((3,3),(0,0), title="phantom"),
            plt.subplot2grid((3,3),(1,0), title="backpropagation (Born)"),
            plt.subplot2grid((3,3),(1,1), title="backprojection (Radon)"),
            plt.subplot2grid((3,3),(1,2), title="backpropagation (Rytov)"),
            plt.subplot2grid((3,3),(0,1), colspan=2, title="phase sinogram"),
            plt.subplot2grid((3,3),(2,0), colspan=3,
                             title="cross-section through phantom nucleolus"),
           ]
    # phantom
    phantom = mt.extract.get_tomo_ri_structure(tomo_path=dir_out)
    grid[0].imshow(phantom, **kwargs_ri)
    cutcoord=phantom.shape[0]//2+int(2*wavelength)
    for gd in grid[:4]:
        gd.set_xticks([])
        gd.set_yticks([])
    grid[5].plot(phantom[cutcoord], linewidth=2, color="k", label="phantom")
    # sinogram
    grid[4].imshow(np.unwrap(np.angle(sino), axis=0).transpose(),
                   aspect="auto",
                   cmap=plt.get_cmap("coolwarm"))
    grid[4].set_yticks([])
    grid[4].set_ylabel("detector")
    # this is deg by accident, becaus we have 360 projections
    grid[4].set_xlabel("angle [deg]")
    for ii, approx in enumerate(["born", "radon", "rytov"]):
        ri = ex_bpg.backpropagate_fdtd_data(tomo_path=dir_out,
                                            approx=approx,
                                            autofocus=autofocus)
        grid[ii+1].imshow(ri.real, **kwargs_ri)
        grid[5].plot(ri[cutcoord], color=colors[ii], label=approx.capitalize())
    grid[5].legend()
    grid[5].grid()
    grid[5].set_ylabel("refractive index")
    grid[5].set_xlabel("pixel position")
    plt.tight_layout()

    print("...Creating figure 3")
    # Plot the result
    fig3 = plt.figure("Figure 3 (Müller et. al, Bioinformatics 2015)", (10, 7))
    ax1 = plt.subplot(211)
    ax1.set_ylabel("normalized RMS error\n of reconstruction")
    ax2 = plt.subplot(212)
    ax2.set_ylabel("normalized TV error\n of reconstruction")
    for ii, approx in enumerate(["born", "radon", "rytov"]):
        rms = np.array([[it[1], it[2]] for it in rms_metrices if it[0]==approx])
        ax1.plot(rms[:,0], rms[:,1], colors[ii], label=approx.capitalize()+" 2D")
        tv = np.array([[it[1], it[2]] for it in tv_metrices if it[0]==approx])
        ax2.plot(tv[:,0], tv[:,1], colors[ii], label=approx.capitalize()+" 2D")
    for ax in [ax1, ax2]:
        ax.legend()
        ax.set_xlabel("total number of projections for reconstruction")
        ax.grid()
    plt.tight_layout()

    plt.show()