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


def test_wrongkey(ph="phantom_2d"):
    wdir = tempfile.mkdtemp(prefix="meep_test")
    ph_file = join(phantom_dir, ph+".cpp")
    bin_path = join(wdir, ph)

    try:
        mt.meep.make_binary(phantom_template=ph_file,
                            bin_path=bin_path,
                            verbose=1,
                            hastalavista=42)
    except KeyError:
        pass


if __name__ == "__main__":
    # Run all tests
    loc = locals()
    for key in list(loc.keys()):
        if key.startswith("test_") and hasattr(loc[key], "__call__"):
            loc[key]()
