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
    ph_file = join(phantom_dir, ph+".cpp")
    bin_path = join(wdir, ph+".bin")
    cpp_path = join(wdir, ph+".cpp")
    log_path = join(wdir, ph+"_compile.log")
    
    mt.meep.make_binary(phantom_template=ph_file,
                        bin_path=bin_path,
                        verbose=1)

    assert exists(log_path)
    # The source file
    assert exists(cpp_path)
    # The binary
    assert exists(bin_path)
    
    shutil.rmtree(wdir, ignore_errors=True)


def test_compile_all():
    """Test all phantom files"""
    phs = [p for p in os.listdir(phantom_dir) if p.endswith(".cpp") ]
    for p in phs:
        print("test compiling: ", p)
        test_compile(p[:-4])


if __name__ == "__main__":
    # Run all tests
    loc = locals()
    for key in list(loc.keys()):
        if key.startswith("test_") and hasattr(loc[key], "__call__"):
            loc[key]()
    
