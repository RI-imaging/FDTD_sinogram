"""Export 3D FDTD simulation as compressed lzma

I use this file to compress ~600MB of data by a factor of 1000 using
lzma compression. The data is downsampled with the parameter `skip`
"""
import os
from os.path import join
import numpy as np
import tempfile

try:
    import lzma
except ImportError:
    from backports import lzma

import tarfile


def compress_folder_lzma(folder, outname="sino"):
    """
    Compress the contents of a folder using lzma compression method.
    """
    outtar = outname+".tar"
    outtarlzma = outname+".tar.lzma"

    # first create one big tarfile
    with tarfile.open(outtar, "w") as t:
        prev = os.path.abspath(os.curdir)
        os.chdir(folder)
        for f in os.listdir("./"):
            t.add(f)
        os.chdir(prev)

    # then compress with lzma
    with lzma.LZMAFile(outtarlzma, "wb", preset=9 | lzma.PRESET_EXTREME) as l:
        with open(outtar, "rb") as t:
            a = t.read()
            l.write(a)

    os.remove(outtar)


def write_license(folder):
    with open(join(folder, "license.txt"), "w") as f:
        f.write("""This work is licensed under a Creative Commons Attribution 4.0 International License.
http://creativecommons.org/licenses/by/4.0/""")


def write_readme(folder, skip=2):
    with open(join(folder, "readme.txt"), "w") as f:
        f.write("""This archive contains finite difference time domain simulation results
created with the meep package. The data is background corrected
and numerically refocused. The projections are equally distributed
between 0 and 2PI. The FDTD data is downsampled by a factor of {}
and the real and imaginary values are rounded to the second decimal
digit. The phantom data is rounded to the third decimal digit.""".format(skip))


def write_info(folder, kw):
    with open(join(folder, "fdtd_info.txt"), "w") as f:
        for k in kw:
            f.write("{} = {}\n".format(k, kw[k]))


def export_sinogram(sinoname, phantomname, outname):
    out = tempfile.mkdtemp()
    print("TEMP:", out)
    #sinoname = "sinogram_lmeas123.5_ts-1_af_phantom_3d_A0180_R13_T00015000_Nmed1.333_Ncyt1.365_Nnuc1.36_Nleo1.387.npy"
    #sinoname = "sinogram_lmeas123.5_ts-1_af_phantom_3d_tilted_A0220_R13_T00015000_Nmed1.333_Ncyt1.365_Nnuc1.36_Nleo1.387_volmax30.0_scale1.0.npy"
    #phantomname = "ri_reference.npy"

    skip = 2  # make sure that angles are dividable by skip

    # SINOGRAM
    sino = np.load(sinoname)
    angles = np.linspace(0, 2*np.pi, sino.shape[0], endpoint=False)
    sino2 = sino[:, ::skip, ::skip]
    angles2 = angles[::skip]

    for ii in range(sino2.shape[0]):
        np.savetxt(join(out, "field_{:03d}_real.txt".format(
            ii)), sino2[ii].real, fmt="%.2f")
        np.savetxt(join(out, "field_{:03d}_imag.txt".format(
            ii)), sino2[ii].imag, fmt="%.2f")

    # REFERENCE
    phantom = np.load(phantomname)
    phantom2 = phantom[::skip, ::skip, ::skip]
    for ii in range(phantom2.shape[0]):
        np.savetxt(join(out, "phantom_{:03d}_real.txt".format(
            ii)), phantom2[ii].real, fmt="%.3f")

    # get parameters
    parm = sinoname.split("_")
    for p in parm:
        if p.startswith("R"):
            res = float(p[1:])/skip
        if p.startswith("Nmed"):
            nmed = float(p[4:])

    write_info(out, {"nm": nmed,
                     "res": res,
                     "lD": 0,
                     #"tilt_yz":0.2,
                     })

    write_license(out)
    write_readme(out, skip=skip)

    #outname="fdtd_3d_sino_A{:d}_R{:.3f}".format(sino.shape[0], res)
    #outname="fdtd_3d_sino_A{:d}_R{:.3f}_tiltyz0.2".format(sino.shape[0], res)

    compress_folder_lzma(out, outname=outname)
