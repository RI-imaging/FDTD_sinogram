#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division, print_function, unicode_literals

import numpy as np

from . import meta, plot


def SetEllipseInArray(array, val, x, y, a, b, phi, sampling, phi_tot):
    """
        Using an input array, set everything inside an elliptical
        region to a specified `val`. The ellipse is described by
        its location `x` and `y`, the major `a` and minor `b` axis,
        and its rotation `phi` (major axis is on x-axis at `phi`=0rad).
        `sampling` describes the sampling units of the region.
        
        If an ellipse is outside of the visible region, then no error
        or warning is raised.
    """
    print("x,y", x,y)
    print("a,b", a,b)
    
    x *= sampling
    y *= sampling
    a *= sampling
    b *= sampling
    
    # TODO: rotation
    (sx,sy) = array.shape
    lx = np.linspace(-sx/2, +sx/2, sx)
    ly = np.linspace(-sy/2, +sy/2, sy)
    (cox, coy) = np.meshgrid(lx, ly)
    
    #loc = np.where((cox-x)**2/a**2 + (coy-y)**2/b**2 <= 1)

    rottx = coy*np.cos(phi_tot) - cox*np.sin(phi_tot)
    rotty = coy*np.sin(phi_tot) + cox*np.cos(phi_tot)

    rotx = (rottx-x)*np.cos(phi) - (rotty-y)*np.sin(phi)
    roty = (rottx-x)*np.sin(phi) + (rotty-y)*np.cos(phi)


    loc = np.where((rotx)**2/a**2 + (roty)**2/b**2 <= 1)


    array[loc] = val
    

    
    return array


def plot_phantom(filename="phantom_2d.cpp"):

    specs = meta.GetPhantomSpecs(filename)
    # initiate domain with medium RI
    samp = specs["SAMPLING"]
    dsize = specs["LATERALSIZE"]
    domain = np.ones((dsize*samp, dsize*samp)) * specs["MEDIUM_RI"]

    # add cytoplasm
    domain = SetEllipseInArray(array=domain,
                               val=specs["CYTOPLASM_RI"],
                               x=0,
                               y=0,
                               a=specs['CYTOPLASM_A'],
                               b=specs['CYTOPLASM_B'],
                               phi=0,
                               sampling=samp,
                               phi_tot=specs["ACQUISITION_PHI"])

    # add nucleus
    domain = SetEllipseInArray(array=domain,
                               val=specs["NUCLEUS_RI"],
                               x=specs["NUCLEUS_X"],
                               y=specs["NUCLEUS_Y"],
                               a=specs['NUCLEUS_A'],
                               b=specs['NUCLEUS_B'],
                               phi=specs["NUCLEUS_PHI"],
                               sampling=samp,
                               phi_tot=specs["ACQUISITION_PHI"])


    # add nucleolus
    domain = SetEllipseInArray(array=domain,
                               val=specs["NUCLEOLUS_RI"],
                               x=specs["NUCLEOLUS_X"],
                               y=specs["NUCLEOLUS_Y"],
                               a=specs['NUCLEOLUS_A'],
                               b=specs['NUCLEOLUS_B'],
                               phi=specs["NUCLEOLUS_PHI"],
                               sampling=samp,
                               phi_tot=specs["ACQUISITION_PHI"])
                               
    # plot the output
    plot.arr2im(domain, cut=False, scale=True, invert=False).save("1.png")
