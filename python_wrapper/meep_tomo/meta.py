#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division, print_function, unicode_literals

import os
import warnings


def GetInfoFromFolder(DIR, logfile="output.txt"):
    """ Given a directory, get output.txt and do `ÌnfoSimulation`
    """
    DIR = os.path.realpath(DIR)

    lofgile = os.path.join(DIR,logfile)
    
    if os.path.exists(logfile):
        newlogfile = logfile
    elif os.path.exists(os.path.join(DIR,"output.txt")):
        newlogfile = os.path.join(DIR,"output.txt")
    elif os.path.exists(os.path.join(os.path.split(DIR)[0],"output.txt")):
        newlogfile = os.path.join(os.path.split(DIR)[0],"output.txt")
    elif os.path.exists(os.path.realpath(DIR)+"put.txt"):
        newlogfile = os.path.realpath(DIR)+"put.txt"
    else:
        files = os.listdir(DIR)
        for f in files:
            if f.endswith(".txt") and f.count("output") != 0:
                newlogfile = os.path.join(DIR,f)

    if logfile != newlogfile:
        warnings.warn("Using info logfile: {}".format(newlogfile))
    
    return InfoSimulation(newlogfile)


def get_info_resolution(DIR):
    BGDIR, dirlist = GetDirlistSimulation(DIR)
    info = GetInfoFromFolder(dirlist[-1])
    res = info["Sampling per wavelength [px]"]
    return res


def GetDirlistSimulation(DIR):
    """ Get dirnames of simulated background and simulation runs
    
    Parameters
    ----------
    DIR : str
        working directory

    
    Returns
    -------
    [bgdir, [sim1, sim2, etc...]]
    """
    DIR = os.path.realpath(DIR)
    # Go into DIR and find all subdirectories empty and eps
    files = os.listdir(DIR)
    dirstrings = list()
    emptydir = None
    for f in files:
        if f.startswith("eps_") and os.path.isdir(DIR+"/"+f):
            dirstrings.append(f[4:])
        if f.startswith("empty_") and os.path.isdir(DIR+"/"+f):
            emptydir = os.path.realpath(DIR+"/"+f)
    
    dirstrings.sort(key=lambda x: float(x.split("_")[-1].split("-out")[0]))

    dirlist = list()
    for dirend in dirstrings:
        dirlist.append( os.path.realpath(DIR+"/eps_" + dirend) )
    
    assert emptydir is not None, "{} - emptydir not found. simulation incomplete?".format(DIR)
    return [emptydir, dirlist]



def GetPhantomSpecs(DIR, filename):
    """
        Get the #define statements from a cpp file and return a 
        dictionary with them and their values.
    """
    f = open(os.path.join(DIR, filename))
    s = f.readlines()
    specs = dict()
    for line in s:
        line = line.partition("//")[0].strip()
        line = line.partition("#define")[2].strip()
        if len(line) != 0:
            name, n, value = line.partition(" ")
            name = name.strip()
            value = value.strip()
            if not value.isalpha():
                value = float(value)
            specs[name] = value
    return specs





def InfoSimulation(logfile="output.txt", lamb=None):
    """
        Parse all info from an FDTD simulation and make it available
        as a dictionary.
        
        Optionally, set the wavelength of the simulation with `lamb` in
        nanometers. If it is not set, the wavelength of `logfile` will
        be used.
    """
    logfile = os.path.realpath(logfile)
    ld = lamb
    try:
        f = open(logfile, "r")
    except:
        raise ValueError("Logfile {} not found. Try using GetInfoFromFolder function.".format(logfile))
        
    d = f.readlines()
    f.close()
    
    info = dict()
    for line in d:
        if line.count(":") == 1:
            split = line.strip(".").split(":")
            try:
                val = float(split[1].strip())
            except ValueError:
                continue
            else:
                info[split[0].strip()] = val
    
    sp = info["Sampling per wavelength [px]"]
    info["PML thickness [px]"] = info["PML thickness [wavelengths]"]*sp
    
    for key in list(info.keys()):
        if key.endswith("[wavelenghts]"):
            info[key[:-4]+"ths]"] = info[key]

    if ( not info.has_key("Axial object size [wavelengths]") and
         info.has_key("Axial object size 1 [wavelengths]") and
         info.has_key("Axial object size 2 [wavelengths]")      ):
        info["Axial object size [wavelengths]"] = max(
                info["Axial object size 1 [wavelengths]"],
                info["Axial object size 2 [wavelengths]"])

    if ( not info.has_key("Lateral object size [wavelengths]") and
         info.has_key("Lateral object size 1 [wavelengths]") and
         info.has_key("Lateral object size 2 [wavelengths]")      ):
        info["Lateral object size [wavelengths]"] = max(
                info["Lateral object size 1 [wavelengths]"],
                info["Lateral object size 2 [wavelengths]"])
                
    info["Object size [wavelengths]"] = max(
            info["Lateral object size [wavelengths]"],
            info["Axial object size [wavelengths]"])

    return info
