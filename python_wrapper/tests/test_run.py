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

# Add parent directory to beginning of path variable
parent_dir = dirname(dirname(abspath(__file__)))
sys.path.insert(0, parent_dir)

import meep_tomo as mt

phantom_dir = join(dirname(parent_dir), "phantoms_meep")


def test_run(ph="phantom_2d"):
    wdir = tempfile.mkdtemp(prefix="meep_test")
    ph = "phantom_2d"
    ph_file = join(phantom_dir, ph+".cpp")

    mt.meep.run_projection(angle=None,
                           P=ph_file,
                           WDIR=wdir,
                           C=1,
                           T=10,
                           R=3)

    empdir = join(wdir, "empty_"+ph+"-out")
    assert exists(join(empdir, "eps-000000.00.h5"))
    assert exists(join(empdir, "ez-000001.67.h5"))
    assert exists(join(empdir, "output.txt"))
    
    assert mt.meep.simulation_completed(join(wdir, "empty_"+ph+".bin"))
    
    mt.meep.run_projection(angle=0.1,
                           P=ph_file,
                           WDIR=wdir,
                           C=1,
                           T=10,
                           R=3)

    epsdir = join(wdir, "eps_"+ph+"_0.1-out")
    assert exists(join(epsdir, "eps-000000.00.h5"))
    assert exists(join(epsdir, "ez-000001.67.h5"))
    assert exists(join(epsdir, "output.txt"))

    assert mt.meep.simulation_completed(join(wdir, "eps_"+ph+"_0.1.bin"))

    shutil.rmtree(wdir)


def test_run_tomography():

    wdir = tempfile.mkdtemp(prefix="meep_test")
    ph = "phantom_2d"
    ph_file = join(phantom_dir, ph+".cpp")

    mt.meep.run_tomography(A=3,
                           P=ph_file,
                           DIR=wdir,
                           C=1,
                           T=3,
                           R=3)



if __name__ == "__main__":
    # Run all tests
    loc = locals()
    for key in list(loc.keys()):
        if key.startswith("test_") and hasattr(loc[key], "__call__"):
            loc[key]()
    
