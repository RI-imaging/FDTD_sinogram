"""Backpropagation methods used in the examples"""
import matplotlib.pylab as plt
import numpy as np
import os
import sys

import nrefocus
import odtbrain as odt
import radontea as rt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))+"/../meep_tomo")

from meep_tomo import extract, common, postproc


def backpropagate_fdtd_data(tomo_path,
                            approx,
                            ld_offset=1,
                            autofocus=False,
                            interpolate=False,
                            force=False,
                            verbose=0):
    """Reconstruct a tomographic simulation folder

    Parameters
    ----------
    tomo_path: str
        Simulation directory
    approx: str
        Approximation to use, one of ["radon", "born", "rytov"]
    ld_offset: float
        Number of wavelengths behind the phantom border at which to
        take the complex field from the simulation. This makes use
        of the "axial_object_size" info parameter.
    autofocus: bool
        If `True`, perform autofocusing. If `False` uses the exact
        focusing (the center of rotation in the simulation).
    interpolate: False or int
        If not `False`, interpolates the field data for faster
        reconstruction.
    force: bool
        If set to `True`, no cached data are used.
    verbose: int
        Increment to increase verbosity

    Returns
    -------
    ri: ndarray
        The 2D or 3D reconstructed refractive index
    """
    assert approx in ["radon", "born", "rytov"]

    res_dir = get_results_dir(tomo_path)

    # Determine name of resulting npy file
    name = "ri_{}_lmeas{}".format(approx, ld_offset)
    if autofocus:
        name += "_af"
    if interpolate:
        name += "_intp{}".format(interpolate)
    name += "_{}.npy".format(os.path.basename(tomo_path))
    name = os.path.join(res_dir, name)

    if os.path.exists(name) and not force:
        if verbose:
            print("...Using existing refractive index: {}".format(name))
        ri = np.load(name)
    else:
        # Get some parameters
        _bg, phs = extract.get_tomo_dirlist(tomo_path)
        info = extract.get_sim_info(phs[0])
        res = info["wavelength"]
        nm = info["medium_ri"]

        sino, angles = get_sinogram(
            tomo_path, ld_offset, autofocus=autofocus, force=force)

        ri = backpropagate_sinogram(sinogram=sino,
                                    angles=angles,
                                    approx=approx,
                                    res=res,
                                    nm=nm,
                                    ld=0)
        # save ri
        np.save(name, ri)

    return ri


def backpropagate_sinogram(sinogram,
                           angles,
                           approx,
                           res,
                           nm,
                           ld=0,
                           ):
    """Backpropagate a 2D or 3D sinogram

    Parameters
    ----------
    sinogram: complex ndarray
        The scattered field data
    angles: 1d ndarray
        The angles at which the sinogram data were recorded
    approx: str
        Approximation to use, one of ["radon", "born", "rytov"]
    res: float
        Size of vacuum wavelength in pixels
    nm: float
        Refractive index of surrounding medium
    ld: float
        Reconstruction distance. Values !=0 only make sense for the
        Born approximation (which itself is not very usable).
        See the ODTbrain documentation for more information.

    Returns
    -------
    ri: ndarray
        The 2D or 3D reconstructed refractive index
    """
    sshape = len(sinogram.shape)
    assert sshape in [2, 3], "sinogram must have dimension 2 or 3"

    uSin = sinogram
    assert approx in ["radon", "born", "rytov"]

    if approx == "rytov":
        uSin = odt.sinogram_as_rytov(uSin)
    elif approx == "radon":
        uSin = odt.sinogram_as_radon(uSin)

    if approx in ["born", "rytov"]:
        # Perform reconstruction with ODT
        if sshape == 2:
            f = odt.backpropagate_2d(uSin,
                                     angles=angles,
                                     res=res,
                                     nm=nm,
                                     lD=ld
                                     )
        else:
            f = odt.backpropagate_3d(uSin,
                                     angles=angles,
                                     res=res,
                                     nm=nm,
                                     lD=ld
                                     )

        ri = odt.odt_to_ri(f, res, nm)
    else:
        # Perform reconstruction with OPT
        # works in 2d and 3d
        f = rt.backproject(uSin, angles=angles)
        ri = odt.opt_to_ri(f, res, nm)

    return ri


def get_results_dir(tomo_path):
    """Return/Create the results directory"""
    res_dir = os.path.abspath(tomo_path)+"_results"
    common.mkdir_p(res_dir)
    return res_dir


def get_sinogram(tomo_path,
                 ld_offset=1,
                 autofocus=False,
                 interpolate=False,
                 force=False):
    """Obtain the sinogram of a MEEP tomographic simulation

    Parameters
    ----------
    tomo_path: str
        Simulation directory
    ld_offset: float
        Number of wavelengths behind the phantom border at which to
        take the complex field from the simulation. This makes use
        of the "axial_object_size" info parameter.
    autofocus: bool
        If `True`, perform autofocusing. If `False` uses the exact
        focusing (the center of rotation in the simulation).
    interpolate: False or int
        If not `False`, interpolates the field data for faster
        reconstruction.
    force: bool
        If set to `True`, no cached data are used.

    Returns
    -------
    sino: complex ndarray
        The 2D/3D complex field sinogam
    """
    res_dir = get_results_dir(tomo_path)
    _bg, phs = extract.get_tomo_dirlist(tomo_path)
    info = extract.get_sim_info(phs[0])

    res = info["wavelength"]
    nmed = info["medium_ri"]

    ld_guess = (ld_offset + info["axial_object_size [wavelengths]"]/2)*res

    # raw sinogram
    rawname = "sinogram_raw_{}.npy".format(ld_guess)
    rawname = os.path.join(res_dir, rawname)
    if os.path.exists(rawname) and not force:
        print("...Loading extracted sinogram")
        sino_raw = np.load(rawname)
    else:
        print("...Extracting sinogram")
        sino_raw = extract.get_tomo_sinogram_at_ld(tomo_path, ld=ld_guess)
        np.save(rawname, sino_raw)

    # processed sinogram
    name = "sinogram_lmeas{}".format(ld_guess)
    if autofocus:
        name += "_af"
    if interpolate:
        name += "_intp{}".format(interpolate)
    name += "_{}.npy".format(os.path.basename(tomo_path))
    name = os.path.join(res_dir, name)

    if os.path.exists(name) and not force:
        print("...Using preprocessed sinogram: {}".format(name))
        u = np.load(name)
    else:
        print("...Processing raw sinogram: {}".format(name))

        if interpolate:
            assert len(sino_raw.shape) == 3, "interpolation only for 3D data!"
            u = []
            for ii in range(sino_raw.shape[0]):
                uii = postproc.interpolate_field(sino_raw[ii], interpolate)
                u.append(uii)
            u = np.array(u)
        else:
            u = sino_raw

        if autofocus:
            print("......Performing autofocusing")
            u, dopt, gradient = nrefocus.autofocus_stack(u, nmed,
                                                         res, ival=(-1.5 *
                                                                    ld_guess, 0),
                                                         same_dist=True,
                                                         ret_ds=True, ret_grads=True,
                                                         metric="average gradient",
                                                         )
            print("Autofocusing distance:", np.average(dopt))
            # save gradient
            plt.figure(figsize=(4, 4), dpi=600)
            plt.plot(gradient[0][0][0], gradient[0][0][1], color="black")
            plt.plot(gradient[0][1][0], gradient[0][1][1], color="red")
            plt.xlabel("distance from original slice")
            plt.ylabel("average gradient metric")
            plt.tight_layout()
            plt.savefig(os.path.join(res_dir, "Refocus_Gradient.png"))
            plt.close()
        else:
            u = nrefocus.refocus_stack(u,
                                       d=-ld_guess,
                                       nm=nmed,
                                       res=res)

        # save sinogram
        np.save(name, u)

    angles = extract.get_tomo_angles(tomo_path)

    return u, angles
