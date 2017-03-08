#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division, print_function, unicode_literals

import numpy as np
from scipy import interpolate
import unwrap
import warnings



def interpolate_field(field, newsize, info=None, verbose=False):
    """Interpolate two-dimensional complex field
    
    Parameters
    ----------
    field: 2d ndarray of shape (M,M)
        The field image
    newsize: int
        Size of the new square array
    info: dict
        Information dictionary that will be updated with the new
        resolution.
    
    Returns
    -------
    field_intp: 2d ndarray of shape (newsize,newsize)
        The interpolated field
    ingo_intp: dict
        The update information dictionary
    
    See Also
    --------
    meta.get_sim_info: The method to obtain the `info` dict
    """
    assert field.shape[0] == field.shape[1], "only square fields allowed" 
    size = field.shape[0]
    if verbose:
        print("...Interpolating complex data from {}px to {}px.".format(
                                            field.shape[0], newsize))
    
    ampl = np.abs(field)
    phas = unwrap.unwrap(np.angle(field))
    x = np.arange(size)
    ampl_i = interpolate.RectBivariateSpline(x,x, ampl, kx=1, ky=1)
    phas_i = interpolate.RectBivariateSpline(x,x, phas, kx=1, ky=1)
    xn = np.linspace(0, size-1, newsize, endpoint=True)
    field_intp = ampl_i(xn,xn) * np.exp(1j*phas_i(xn,xn))

    ##### This is very important:
    ##### If we rescale the image, we need to change the parameters
    ##### accordingly:
    smaller = newsize/size
    # less pixels are needed for a micrometer
    # pixel size increases
    if info is not None:
        info = info.copy()
        info["effective pixel size [um]"] /= smaller
        # there are less pixels -> everything in px decreases
        for key in info.keys():
            if key.endswith("[px]"):
                info[key] *= smaller
        return field_intp, info
    else:
        warnings.warn("Info data not edited, because not in argument!")
        return field_intp


def interpolate_volume(ref, resize):
    """Intperolate a 3D volume
    """
    assert len(ref.shape) == 3, "3D data only"
    (sx, sy, sz) = ref.shape
    assert sx==sy, "Only cubic data allowed"
    assert sx==sz, "Only cubic data allowed"

    # 2D interpolation along first two axes
    twodintp = np.zeros((sx, resize, resize))
    xn = np.linspace(0, ref.shape[0], resize, endpoint=True)
    for i in range(sx):
        x = np.arange(sx)
        pr = interpolate.RectBivariateSpline(x,x,ref[i], kx=1, ky=1)
        twodintp[i] = pr(xn,xn)
    
    med = np.transpose(twodintp, [1,0,2])

    out = np.zeros((resize, resize, resize))
    # 1D interpolation along last axis
    for i in range(resize):
        y = np.arange(sx)
        pr = interpolate.RectBivariateSpline(y,xn,med[i], kx=1, ky=1)
        out[i] = pr(xn,xn)

    return np.transpose(out, [1,0,2])

