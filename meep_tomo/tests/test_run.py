#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Test if MEEP runs through
"""
from __future__ import division, print_function

import numpy as np
import os
from os.path import abspath, basename, dirname, join, split, exists
import shutil
import sys
import tempfile
from scipy.io.wavfile import WAVE_FORMAT_EXTENSIBLE

# Add parent directory to beginning of path variable
parent_dir = dirname(dirname(abspath(__file__)))
sys.path.insert(0, parent_dir)

import meep_tomo as mt

phantom_dir = join(dirname(parent_dir), "phantoms_meep")


def stest_run():
    ph="phantom_2d"
    timesteps=10
    wavelength=3.0
    wdir = tempfile.mkdtemp(prefix="meep_test_")
    ph_file = join(phantom_dir, ph+".cpp")

    bgdir = mt.meep.run_projection(phantom_template=ph_file,
                                   onlymedium=True,
                                   dir_out=wdir,
                                   timesteps=timesteps,
                                   wavelength=wavelength)

    assert exists(join(bgdir, "eps-000000.00.h5"))
    assert exists(join(bgdir, "ez-000001.67.h5"))
    assert exists(join(bgdir, "bg_"+ph+"_exec.log"))
    assert mt.meep.simulation_completed(bgdir)
    
    angle=0.1
    phdir = mt.meep.run_projection(phantom_template=ph_file,
                                   angle=angle,
                                   dir_out=wdir,
                                   onlymedium=False,
                                   timesteps=timesteps,
                                   wavelength=wavelength)

    assert exists(join(phdir, "eps-000000.00.h5"))
    assert exists(join(phdir, "ez-000001.67.h5"))
    assert mt.meep.simulation_completed(phdir)

    shutil.rmtree(wdir)


def stest_run_tomography():
    wdir = tempfile.mkdtemp(prefix="meep_test_")
    ph = "phantom_2d"
    ph_file = join(phantom_dir, ph+".cpp")

    mt.meep.run_tomography(phantom_template=ph_file,
                           num_angles=3,
                           dir_out=wdir,
                           timesteps=3,
                           wavelength=3.0,
                           verbose=1)
    
    shutil.rmtree(wdir)


def test_run_tomography_scale():
    wdir = tempfile.mkdtemp(prefix="meep_test_")
    ph = "phantom_2d"
    ph_file = join(phantom_dir, ph+".cpp")

    mt.meep.run_tomography(phantom_template=ph_file,
                           num_angles=3,
                           scale=2,
                           dir_out=wdir,
                           timesteps=3,
                           wavelength=3.0,
                           verbose=1)
    cpfile = "ph_phantom_2d_0.000000000000000-out/ph_phantom_2d_0.000000000000000.cpp"
    cppath = os.path.join(wdir, cpfile)
    with open(cppath, "r") as fd:
        script = fd.read()
    # The default is "#define NUCLEOLUS_Y  2.0"
    assert script.count("#define NUCLEOLUS_Y  4.0")

    shutil.rmtree(wdir)


if __name__ == "__main__":
    # Run all tests
    loc = locals()
    for key in list(loc.keys()):
        if key.startswith("test_") and hasattr(loc[key], "__call__"):
            loc[key]()
    
