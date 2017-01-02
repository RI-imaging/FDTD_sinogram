#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division, print_function, unicode_literals

import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['text.usetex']=True
matplotlib.rcParams['text.latex.unicode']=True
matplotlib.rcParams['font.family']='serif'
#matplotlib.rcParams['text.latex.preamble']=[r"""\usepackage{amsmath}
#                                            \usepackage[utf8x]{inputenc}
#                                            \usepackage{amssymb}"""] 
from matplotlib import pylab as plt
import numpy as np
import os
from PIL import Image # install python image library (PIL)
from scipy.interpolate import interp1d

from .common import mkdir

def arr2im(arr, cut=False, scale=False, invert=False):
    """ Convert the real part of an array to an image.
        If scale is True, then the image will be resized to fit into
        the interval (0,254.9). Scale overrides cut.
        If cut is True, then the image will not be resized but values
        will be cut according to uint8.
    
    """
    a=arr.real.copy()
    if a.dtype == bool:
        a = np.array(a, dtype=float)
    # Remove Nan -> 0
    a[np.where(np.isnan(a))] = 0

    if len(arr.shape) > 2:
        a = np.squeeze(a)
    
    if scale:
        #im=Image.Image()
        # This also takes care of negative ri's
        a -= np.min(a) 
        a *= 254.9/np.max(a)
    if np.min(a) < 0:
        print("WARNING (image output): Data file contains negative")
        print("        pixel values. I set them to zero.")
        a[np.where(a<0)] = 0
    if np.max(a) > 255.:
        if cut == False:
            print("WARNING (image output): Maximum pixel value exceeds")
            print("        uint8 ({:.2f}). I resized the scale.".format(np.max(a)))
            a *= 255/a.max()
        else:
            print("WARNING (image output): Maximum pixel value exceeds")
            print("        uint8 ({:.2f}). I cut at 255.".format(np.max(a)))
            cuti = np.where(a>255)
            a[cuti]= 255
    if invert:
        a = 255.-a
    try:
        im=Image.fromarray(np.uint8(a))
    except AttributeError:
        c = np.ones(a.shape)
        b = a*c
        im=Image.fromarray(np.uint8(b))
    return im
 
 
 
def saveslice_xyz(a, fname, fdir, resize=None, lim=(None,None),
                  unwrap_data=False, cut=(0,0,0)):
    """
        from a 3D array `a`, save slices at the center from each
        cartesian coordinate in .png image format.
        shape convention: (z,y,x)
    
    cut : tuple of 3 elements
        coordinates in pixels where to cut along an axis

    """
    OUTDIR = fdir
    mkdir(OUTDIR)
    if fname[-4:] != ".png":
        fname += ".png"


    # grayscale rescale
    if lim == (None,None):
        if not a.min() == a.max():
            ri = (a-a.min())/(a.max()-a.min()) * 254.9
        else:
            ri = a
    else:
        # do the rescaling locally in the slice further down
        ri = a
        
    
    if len(ri.shape) != 3:
        print("   Input array must be 3-dimensional! (saveslice_xyz)")
        return
    (liz, liy, lix) = ri.shape
    lix = int(lix/2) + cut[0]
    liy = int(liy/2) + cut[1]
    liz = int(liz/2) + cut[2]

    # We reconstructed a 3D image
    imagex = 1.*ri.real[:,:,lix]
    imagey = 1.*ri.real[:,liy,:]
    imagez = 1.*ri.real[liz,:,:]
    
    for im, lab in zip([imagex, imagey, imagez], ["x", "y", "z"]):
    
        if unwrap_data:
            import unwrap
            im[:] = unwrap.unwrap(im[:])

        if lim != (None,None):
            # translate lim to grayscale
            im[im>lim[1]] = lim[1]
            im[im<lim[0]] = lim[0]
            im[:] = (im-im.min())/(im.max()-im.min()) * 254.9

        ax=arr2im(im, scale=False)

        if resize is not None:
            ax.thumbnail((resize, resize), Image.ANTIALIAS)
        
        ax.save(os.path.join(OUTDIR, lab+"_"+fname))
    
    del ri, a



def saveprofile_xy(a, fname, fdir, cut=(0,0), ref=None,
                   transparent=False, px_um=None):
    """
        from a 3D array `a`, save the pofile along the coordinate axes
        in all three dimensiona at the center from each axis.
        Uses matplotlib and saves as .png.
        
        cut: (x,y) coordinates for line cut from center
    """
    plotfontsize = 14
    OUTDIR = fdir
    if fname[-4:] != ".png":
        fname += ".png"
    ri = 1*a
    if len(ri.shape) != 2:
        print("Input array must be 2-dimensional! (tool.saveslice_xyz)")
        return
    (liy, lix) = ri.shape
    
    cx = int(lix/2) + cut[0]
    cy = int(liy/2) + cut[1]
    
    # We reconstructed a 2D image
    profiles = [a[:,cx], a[cy,:]]
    # Reference graphs
    if ref is not None:
        (riy, rix) = ref.shape
        rcx = int(rix/2) + cut[0]
        rcy = int(riy/2) + cut[1]
        refprofs = [ref[:,rcx], ref[rcy,:]]

        # Resize everything to smallest ref square
        rmin = min(riy, rix)
        for i in range(2):
            lr = len(refprofs[i])
            refprofs[i] = refprofs[i][lr/2-rmin/2:lr/2+rmin/2]
            lp = len(profiles[i])
            profiles[i] = profiles[i][lp/2-rmin/2:lp/2+rmin/2]
        
    # Create matplotlib plot
    #myfig=plt.figure(num=None, figsize=(8,6 ), dpi=300, facecolor='w',
    #                 edgecolor='w')
    (myfig, axes) = plt.subplots(1,2, num=None, figsize=(8,6 ), dpi=300,
                                      facecolor='w', edgecolor='w')
    labels = [r"cut at x={}".format(cut[0]),
              r"cut at y={}".format(cut[1])]
    colors = ["g","b"]
    refstr = ["k-", "k-"]

    ymin = np.min(np.array(profiles))
    ymax = np.max(np.array(profiles))

    y_formatter = matplotlib.ticker.ScalarFormatter(useOffset=False)

    for i in range(len(profiles)):
        size = len(profiles[i])
        x=np.arange(size)
        lsize=size/2
        x = np.linspace(-lsize,+lsize,size,endpoint=False)
        y_orig = profiles[i]

        # interpolate (nearest) the data before matplotlib does it
        xi = np.linspace(x.min(), x.max(), 10000, endpoint=True)
        
        f_origi = interp1d(x, y_orig, kind="nearest")

        if ref is not None:
            xr = np.linspace(-lsize, lsize, len(refprofs[i]))
            axes[i].plot(xr, refprofs[i], refstr[i])

        axes[i].plot(xi, f_origi(xi), colors[i]+"-")#, label=labels[i])

        axes[i].yaxis.set_major_formatter(y_formatter)

        #axes[i].legend(fontsize=plotfontsize)

        #plt.ylabel(r"pixel value", fontsize=plotfontsize)
        
        axes[i].set_xlabel(labels[i]+" [px]", fontsize=plotfontsize)
        axes[i].set_ylabel(r"pixel value", fontsize=plotfontsize)

        #ticks = axes[i].get_yticks()
        #start = np.floor(ticks[0]*100)/100
        #end = np.ceil(ticks[-1]*100)/100
        #leng = round((end-start)*100)
        #leng = min(leng, 42)

        # try to get good numbering:
        ylocator = plt.MaxNLocator(15, prune="both")
        axes[i].yaxis.set_major_locator(ylocator)


        axes[i].set_ylim((ymin,ymax))
        axes[i].set_xlim((xi.min(),xi.max()))

    plt.tight_layout()
    for c in range(10):
        # Let LaTeX try it several times
        try:
            plt.savefig(os.path.join(fdir,fname), transparent=transparent)
        except:
            pass
        else:
            break
    plt.close()




def saveprofile_xyz(a, fname, fdir, cut=(0,0,0), ref=None,
                    resizeref=None, ylabel="refractive index",
                    px_um=None, px_lambs=None, transparent=False,
                    ymin=None, ymax=None):
    """
    from a 3D array `a`, save the pofile along the coordinate axes
    in all three dimensions at the center from each axis.
    Uses matplotlib and saves as .png.
    
    Parameters
    ----------
    a : 3d ndarray
        3D volume with refractive index values
    fname : str
        Filename to use
    fdir : str
        directory
    cut : tuple of 3 elements
        coordinates in pixels where to cut along an axis
    ref : None or array of shape `a.shape`
        Reference refractive index (plotted in black)
    resizeref : None
        Reserved for future use
    ylabel : str
        Label of the y axis
    px_um : float
        Size of one pixel in µm. This is usually a number <1.
        If specified, the x-axis will be labeled in µm.
        Not used of px_lambs is set.
    px_lambs : float
        Size of one pixel in wavelengths.
        If specified, the x-axis will be labeled in wavelengths.
    transparent : bool
        If True, set the background of the plot to transparent.
    ymin, ymax : float
        define the plotting interval of the y axis


    Returns
    -------
    None

    """
    plotfontsize = 14
    OUTDIR = fdir
    if fname[-4:] != ".png":
        fname += ".png"
    ri = 1*a
    if len(ri.shape) != 3:
        print("Input array must be 3-dimensional! (tool.saveslice_xyz)")
        return
        
    (liz, liy, lix) = ri.shape
    cx = lix = int(lix/2) + cut[0]
    cy = liy = int(liy/2) + cut[1]
    cz = liz = int(liz/2) + cut[2]
    # We reconstructed a 3D image
    imagex = 1.*ri.real[liz,liy,:]
    imagey = 1.*ri.real[liz,:,lix]
    imagez = 1.*ri.real[:,liy,lix]
    profiles = [imagex, imagey, imagez]
    # Create matplotlib plot
    #myfig=plt.figure(num=None, figsize=(8,6 ), dpi=300, facecolor='w',
    #                 edgecolor='w')
    #(myfig, axes) = plt.subplots(1,1, num=None, figsize=(8,6 ), dpi=300,
    #                                  facecolor='w', edgecolor='w')
    
    if ref is not None:
        (riz, riy, rix) = ref.shape
        rcx = int(rix/2) + cut[0]
        rcy = int(riy/2) + cut[1]
        rcz = int(riz/2) + cut[2]
        refprofs = [ref[rcz,rcy,:], ref[rcz,:,rcx], ref[:,rcy,rcx]]

        # Resize everything to smallest ref square
        rmin = min(riy, rix, riz)
        
        for i in range(3):
            lr = len(refprofs[i])
            refprofs[i] = refprofs[i][lr/2-rmin/2:lr/2+rmin/2]
            lp = len(profiles[i])
            profiles[i] = profiles[i][lp/2-rmin/2:lp/2+rmin/2]

    # Create matplotlib plot
    #myfig=plt.figure(num=None, figsize=(8,6 ), dpi=300, facecolor='w',
    #                 edgecolor='w')
    (myfig, axes) = plt.subplots(1,3, num=None, figsize=(8,5 ), dpi=300,
                                      facecolor='w', edgecolor='w')

    if px_lambs is not None:
        labels = [r"cut at x={:.2f} [lambda]".format(cut[0]*px_lambs),
                  r"cut at y={:.2f} [lambda]".format(cut[1]*px_lambs),
                  r"cut at z={:.2f} [lambda]".format(cut[2]*px_lambs)]
    else:
        if px_um is None:
            labels = [r"cut at x={:.2f} [px]".format(cut[0]),
                      r"cut at y={:.2f} [px]".format(cut[1]),
                      r"cut at z={:.2f} [px]".format(cut[2])]
        else:
            labels = [r"cut at x={:.2f} [um]".format(cut[0]*px_um),
                      r"cut at y={:.2f} [um]".format(cut[1]*px_um),
                      r"cut at z={:.2f} [um]".format(cut[2]*px_um)]
        
    colors = ["g","b","r"]
    refstr = ["k-", "k-", "k-"]

    if ymin is None:
        ymin = np.min(np.hstack(profiles))*.999
        if ref is not None:
            ymin = min(ymin, np.min(np.hstack(refprofs))*.999)
    if ymax is None:
        ymax = np.max(np.hstack(profiles))*1.001
        if ref is not None:
            ymax = max(ymax, np.max(np.hstack(refprofs))*1.001)
       
    y_formatter = matplotlib.ticker.ScalarFormatter(useOffset=False)


    for i in range(len(profiles)):
        size = len(profiles[i])
        x=np.arange(size)
        lsize=size/2
        x = np.linspace(-lsize,+lsize,size,endpoint=False)
        if px_lambs is not None:
            x *= px_lambs
        elif px_um is not None:
            # px to um conversion
            x *= px_um
        y_orig = profiles[i]

        # interpolate (nearest) the data before matplotlib does it
        xi = np.linspace(x.min(), x.max(), 100000, endpoint=True)

        f_origi = interp1d(x, y_orig, kind="nearest")

        axes[i].set_ylim((ymin, ymax))

        axes[i].yaxis.set_major_formatter(y_formatter)
        axes[i].plot(xi, f_origi(xi), colors[i]+"-")

        if ref is not None:
            ref_origi = interp1d(x, refprofs[i], kind="nearest")
            axes[i].plot(xi, ref_origi(xi), refstr[i])
            #xr = np.linspace(-lsize, lsize, len(refprofs[i]))
            #axes[i].plot(xr, refprofs[i], refstr[i])

        axes[i].set_xlabel(labels[i], fontsize=plotfontsize)
        axes[i].set_ylabel(ylabel, fontsize=plotfontsize)

        # try to get good numbering:
        ylocator = plt.MaxNLocator(15, prune="both")
        axes[i].yaxis.set_major_locator(ylocator)
        

        axes[i].set_xlim((xi.min(),xi.max()))




    plt.tight_layout()
    for c in range(10):
        # Let LaTeX try it several times
        try:
            plt.savefig(os.path.join(fdir,fname), transparent=transparent, dpi=300)
        except:
            pass
        else:
            break
    del ri, ref, a
    plt.close()

