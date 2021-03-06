"""Tools used by the examples """
import numpy as np
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))+"/../meep_tomo")

from meep_tomo import extract, common
import ex_bpg


def compute_metrices(tomo_path, approx, autofocus=False):
    """Compute RMS and TV metrices for a MEEP-simulated ODT reconstruction

    Parameters
    ----------
    tomo_path: str
        Simulation directory or .npy file of a reconstructed simulation
    approx: str
        Approximation to use, one of ["radon", "born", "rytov"]
    autofocus: bool
        If `True`, perform autofocusing. If `False` uses the exact
        focusing (the center of rotation in the simulation).
        This only makes sense if `tomo_path` is not an .npy file.

    Returns
    -------
    rms, tv: floats
        root-mean-square and total variation errors

    Notes
    -----
    A second call with the same arguments will be fast, because the
    result is saved on disk.



    See Also
    --------
    metric_rms, metric_tv: The used metrics
    """
    assert approx in ["radon", "born", "rytov"]

    tomo_path = os.path.abspath(tomo_path)

    if os.path.isdir(tomo_path):
        sim_dir = os.path.abspath(tomo_path)
        res_dir = os.path.abspath(tomo_path)+"_results"
        common.mkdir_p(res_dir)
        metr_file = os.path.join(res_dir, "metrices.txt")
        npy_file = False
    elif tomo_path.endswith(".npy"):
        res_dir = os.path.dirname(os.path.abspath(tomo_path))
        sim_dir = res_dir[:-8]
        msg = "Simulation directory not found! The .npy file should be in a " +\
              "folder named after the simulation with '_results' appended!"
        assert os.path.exists(sim_dir), msg
        metr_file = tomo_path[:-4]+"_metrices.txt"
        npy_file = tomo_path
    else:
        raise ValueError("simulation must be a directory or an .npy file!")

    tv = None
    ss = None

    # Check if the results_file exists and read parameters
    if os.path.exists(metr_file):
        with open(metr_file, "r") as fd:
            lines = fd.readlines()
            for line in lines:
                line = line.strip()
                if line.startswith("TV_"+approx):
                    try:
                        tv = float(line.split()[1])
                    except:
                        pass
                elif line.startswith("SS_"+approx):
                    try:
                        ss = float(line.split()[1])
                    except:
                        pass

    if tv is None or ss is None:
        if npy_file:
            ri = np.load(npy_file)
            assert autofocus == False, "`autofocus` has no effect for .npy files!"
        else:
            # Recompute everything
            ri = ex_bpg.backpropagate_fdtd_data(sim_dir,
                                                approximation=approx,
                                                autofocus=autofocus)

        # reference
        riref = extract.get_tomo_ri_structure(sim_dir)

        ss = metric_rms(ri, riref)
        tv = metric_tv(ri, riref)

        # Save result in resf files
        with open(metr_file, "a") as resfdata:
            lines = "# metrices of ri-riref\n"
            lines += "TV_{} {:.15e}\n".format(approx, tv)
            lines += "SS_{} {:.15e}\n".format(approx, ss)
            resfdata.writelines(lines)

    return ss, tv


def cutout(a):
    """Cut out circle/sphere from 2D/3D square/cubic array"""
    x = np.arange(a.shape[0])
    c = a.shape[0] / 2

    if len(a.shape) == 2:
        x = x.reshape(-1, 1)
        y = x.reshape(1, -1)
        zero = ((x-c)**2 + (y-c)**2) < c**2
    elif len(a.shape) == 3:
        x = x.reshape(-1, 1, 1)
        y = x.reshape(1, -1, 1)
        z = x.reshape(1, -1, 1)
        zero = ((x-c)**2 + (y-c)**2 + (z-c)**2) < c**2
    else:
        raise ValueError("Cutout array must have dimension 2 or 3!")
    a *= zero
    #tool.arr2im(a, scale=True).save("test.png")
    return a


def metric_rms(ri, ref):
    """Root mean square metric (normalized)

    This metric was used and described in
    Müller et. al, "ODTbrain: a Python library for full-view,
    dense diffraction tomography" Bioinformatics 2015
    """
    rms = np.sum(cutout(ri.real-ref.real)**2)
    norm = np.sum(cutout(ref.real-1)**2)
    return np.sqrt(rms/norm)


def metric_tv(ri, ref):
    """Total variation metric (normalized)

    This metric was used and described in
    Müller et. al, "ODTbrain: a Python library for full-view,
    dense diffraction tomography" Bioinformatics 2015
    """
    grad = np.gradient(ri.real-ref)
    result = 0
    for g in grad:
        result += np.sum(cutout(np.abs(g)))
    tv = result / len(grad)
    norm = np.sum(cutout(ref.real-1)**2)
    return np.sqrt(tv/norm)
