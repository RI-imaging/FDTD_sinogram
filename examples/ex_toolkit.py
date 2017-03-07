#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Tool kit for the examples"""
from __future__ import division, print_function, unicode_literals

import numpy as np
import os
import re
from scipy import interpolate
import sys
import unwrap



def alphanum2num(string):
    """
    removes all non-numeric characters from a string and returns a
    float.
    """
    try:
        ret = float(re.sub("[^0-9]", "", string))
    except:
        print(string)
        ret = ""
    return ret


def compute_metrices(simulation, approx, normalize=True, cropto=None):
    """  
    Given a the directory of a simulation, e.g.
    phantom_3d_A0200_R13_T15000_Nmed1.333_Ncyt1.335_Nnuc1.335_Nleo1.335
    and an approximation type that is accepted by 
    `backpropagate.backpropagate_fdtd_data`, returns the metrices
    [sum_of_squares, total_variation]
    
    A second call with the same arguments will be very fast, because
    the results are saved in the results folder.
    
    If simulation is not a directory, but points to a .npy file, then
    it is assumed that this file points to the refractive index 
    reconstruction of a simulation.
    
    Parameters
    ----------
    simulation : str
        simulation directory
    approx : str
        "radon", "born", "rytov"
    normalize : bool
        Normalize to (avg-ref)
    cropto : int
        Crops the image to this size.
    """
    simulation = os.path.abspath(simulation)
    # define lambda functions
    
    def sumsq(ri, ref):
        return np.sum(cutout(ri.real-riref)**2)
    
    def tvdev(ri, ref):
        grad = np.gradient(ri.real - riref)
        result = 0
        for g in grad:
            result += np.sum(cutout(np.abs(g)))
        return result / len(grad)
        

    ## Norm also uses cutout
    def norm(val, ref, method=None):
        """
        method has no effect
        """
        if method == sumsq:
            # normalized root mean square error nrmse
            return np.sqrt(val/sumsq(1, ref))
        elif method == tvdev:
            #return val/(np.sum(cutout(np.ones_like(ref)))*np.std(ref))
            # normalization only takes into account parts of phantom
            #return val/(np.sum(ref != np.min(ref)))
            #return val/np.sum(np.abs(ref-1))
            return np.sqrt(val/sumsq(1, ref))
        else:
            raise ValueError("wrong use of norm")
    
    if os.path.isdir(simulation):
        simulation_dir = simulation
        resf = os.path.abspath(simulation_dir)+"_results/metrices.txt"
    elif simulation.endswith(".npy"):
            # we have the refractive index in the "_results" dir
            simulation_dir = None
            resf = simulation[:-4]+"_metrices.txt"
    else:
        raise ValueError("simulation must be directory or npy file!")
    
    tv = None
    ss = None
    
    # Check if the results_file exists and read parameters
    if os.path.exists(resf):
        with open(resf, "r") as resfdata:
            lines = resfdata.readlines()
            for line in lines:
                if line.startswith("TV_"+approx):
                    try:
                        tv = float(line[3+len(approx):])
                    except:
                        pass
                elif line.startswith("SS_"+approx):
                    try:
                        ss = float(line[3+len(approx):])
                    except:
                        pass
    
    tv=ss=None
    
    if tv is None or ss is None:
        if simulation_dir is None:
            ## Only perform metric calculations for npy files
            ri = np.load(simulation)
            # simulation directory without _results
            odir = os.path.dirname(simulation)[:-8]
            riref = get_phantom_refractive_index(odir)
        else:
            ## Backpropagate if we need to
            riref = get_phantom_refractive_index(simulation_dir)
            # obtain refractive index            
            ri = backpropagate_fdtd_data(simulation_dir, approximation=approx)
            ## sum of squares of difference

        if cropto is not None:
            # crop the images before computing metrices
            raise ValueError("I decided not to use cropto.")
            old = ri.shape[0]
            crop = int((old-cropto)/2)
            ri = tool.CropPML(ri, crop)
            riref = tool.CropPML(riref, crop)

        ss = sumsq(ri.real, riref)
        tv = tvdev(ri.real, riref)

        ## Save result in resf files
        with open(resf, "a") as resfdata:
            lines = "# metrices of ri-riref\n"
            lines += "TV_{} {:.10e}\n".format(approx, tv)
            lines += "SS_{} {:.10e}\n".format(approx, ss)
            resfdata.writelines(lines)


    if normalize:
        resfnorm = resf[:-4]+"_norm.txt"
        
        tvn = None
        ssn = None
        
        try:
            os.remove(resfnorm)
        except:
            pass
        
        if os.path.exists(resfnorm):
            with open(resfnorm, "r") as resfdata:
                lines = resfdata.readlines()
                for line in lines:
                    if line.startswith("TV_"+approx):
                        try:
                            tvn = float(line[3+len(approx):])
                        except:
                            pass
                    elif line.startswith("SS_"+approx):
                        try:
                            ssn = float(line[3+len(approx):])
                        except:
                            pass

        if tvn is None or ssn is None:
            if simulation_dir is None:
                # simulation directory without _results
                odir = os.path.dirname(simulation)[:-8]
                riref = get_phantom_refractive_index(odir)
            else:
                ## Backpropagate if we need to
                riref = get_phantom_refractive_index(simulation_dir)
            
            ssn = norm(ss, riref, sumsq)
            tvn = norm(tv, riref, tvdev)

            #with open(resfnorm, "a") as resfdata:
            #    lines = "# normalized metrices of ri-riref\n"
            #    lines += "TV_{} {:.10e}\n".format(approx, tvn)
            #    lines += "SS_{} {:.10e}\n".format(approx, ssn)
            #    resfdata.writelines(lines)
        

            
        ss = ssn
        tv = tvn
            
    return ss, tv


def cutout(a):
    """ cut out circle/sphere from 2D/3D square/cubic array
    """
    x = np.arange(a.shape[0])
    c = a.shape[0] / 2
    
    if len(a.shape) == 2:
        x = x.reshape(-1,1)
        y = x.reshape(1,-1)
        zero = ((x-c)**2 + (y-c)**2) < c**2
    elif len(a.shape) == 3:
        x = x.reshape(-1,1,1)
        y = x.reshape(1,-1,1)
        z = x.reshape(1,-1,1)
        zero = ((x-c)**2 + (y-c)**2 + (z-c)**2) < c**2
    else:
        raise ValueError("Cutout array must have dimension 2 or 3!")
    a *= zero
    #tool.arr2im(a, scale=True).save("test.png")
    return a


