#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Test if we can compile the phantoms
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


def test_compile(ph="phantom_2d"):
    wdir = tempfile.mkdtemp(prefix="meep_test")
    ph = "phantom_2d"
    ph_file = join(phantom_dir, ph+".cpp")
    
    mt.meep.make_binary(P=ph_file,
                        WDIR=wdir,
                        A=0.1,
                        T=1000,
                        R=12,
                        onlymedium=True)

    assert exists(join(wdir, "compile.txt"))
    # The source file
    assert exists(join(wdir, "empty_"+ph+".cpp"))
    # The binary
    assert exists(join(wdir, "empty_"+ph+".bin"))
    
    mt.meep.make_binary(P=ph_file,
                        WDIR=wdir,
                        A=0.1,
                        T=1000,
                        R=12,
                        onlymedium=False)

    # The source file
    assert exists(join(wdir, "eps_"+ph+"_0.1.cpp"))
    # The binary
    assert exists(join(wdir, "eps_"+ph+"_0.1.bin"))
    
    shutil.rmtree(wdir)


def test_compile_all():
    """Test all phantom files"""
    phs = [p for p in os.listdir(phantom_dir) if p.endswith(".cpp") ]
    for p in phs:
        print("test compiling: ", p)
        test_compile(p)


if __name__ == "__main__":
    # Run all tests
    loc = locals()
    for key in list(loc.keys()):
        if key.startswith("test_") and hasattr(loc[key], "__call__"):
            loc[key]()
    
