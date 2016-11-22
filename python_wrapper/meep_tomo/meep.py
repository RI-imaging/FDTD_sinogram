#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division, print_function, unicode_literals

import argparse
import multiprocessing as mp
import numpy as np
import os
import time
import warnings

from .common import mkdir


# Default values. DO NOT CHANGE!
_Nmed = 1.333
_Ncyt = 1.365
_Nnuc = 1.360
_Nleo = 1.387



def create_cpp(P, A, T, R, onlymedium=True,
               Nmed=_Nmed, Ncyt=_Ncyt, Nnuc=_Nnuc, Nleo=_Nleo,
               scale=None):
    CPPSCRIPT = os.path.split(P)[1]
    if onlymedium:
        EPS = "empty_{}".format(CPPSCRIPT[:-4], A)
        med = "true"
    else:
        EPS = "eps_{}_{}".format(CPPSCRIPT[:-4], A)
        med = "false"
        
    # read in the original cpp script
    f = open(P, "r")
    d = f.readlines()
    f.close()
    
    replacing = [
                 [ "ACQUISITION_PHI", float(A)],
                 [ "TIME", float(T)],
                 [ "SAMPLING", float(R)],
                 [ "MEDIUM_RI", float(Nmed)],
                 [ "CYTOPLASM_RI", float(Ncyt)],
                 [ "NUCLEUS_RI", float(Nnuc)],
                 [ "NUCLEOLUS_RI", float(Nleo)],
                 [ "ONLYMEDIUM", med]
                ]
    
    for item in replacing:
        string = "#define {} ".format(item[0])
        value = item[1]
        for i in range(len(d)):
            if d[i].count(string) > 0:
                d[i] = "{} {}\n".format(string, value)
    
    if scale is not None:
        # scale all size parameters of the cell with scale
        scaling = [ "_A", "_B", "_C", "_X", "_Y", "_Z" ]
        for item in scaling:
            for i in range(len(d)):
                triplet = d[i].split()
                if (len(triplet) == 3 and
                    triplet[0] == "#define" and
                    triplet[1].endswith(item)   ):
                    newval = float(triplet[2])*scale
                    d[i] = "{}  {}  {}\n".format(
                            triplet[0], triplet[1], newval)
    
    return EPS, d


def compile_cpp(cpp_lines, WDIR, EPS):
    gpplib = " -lmeep_mpi -lhdf5 -lz -lgsl -lharminv -llapack "+\
             "-lcblas -latlas -lfftw3 -lm "
    
    # copy cpp script
    eps = open(os.path.join(WDIR,EPS+".cpp"), "w")
    eps.writelines(cpp_lines)
    eps.close()

    library_paths = "-L/usr/include/hdf5/serial/ "+\
                    "-L/usr/lib/x86_64-linux-gnu/hdf5/openmpi/ "+\
                    "-L/usr/lib/x86_64-linux-gnu/hdf5/serial/ "+\
                    "-L/usr/local/lib/"

    runstring = "g++ {} -malign-double {} -o {}  {} 2>&1 | tee {}".format(
                 library_paths,
                 os.path.join(WDIR,EPS+".cpp"),
                 os.path.join(WDIR,EPS+".bin"),
                 gpplib,
                 os.path.join(WDIR,"compile.txt"))
    binfilename = os.path.join(WDIR,EPS+".bin")

    if not os.path.exists(binfilename):
        os.system(runstring)
        time.sleep(0.1)

    return binfilename


def make_binary(P, WDIR, A, T, R, onlymedium=True,
                Nmed=_Nmed, Ncyt=_Ncyt, Nnuc=_Nnuc, Nleo=_Nleo):
    """
        Copies the original cpp script into subdirectories for eps and
        empty samples and replaces the definitions:
        
            - ONLYMEDIUM
            - ACQUISITION_PHI
            - refractive indices
    
        The result is a directory WDIR that contains several binaries
        
            g++ -malign-double twodphantom.cpp -o twodphantom.bin -lmeep_mpi -lhdf5 -lz -lgsl -lharminv -llapack -lcblas -latlas -lfftw3 -lm

        
        that end with .bin and which can then be executed using

            mpirun -n 7 ./twodphantom.bin 2>&1 | tee output.txt >> /dev/null
            
        Errors and compile messages are displayed on stdout and are
        written to a text file in WDIR.
    """
    EPS, d = create_cpp(P, A, T, R, onlymedium=onlymedium,
                   Nmed=Nmed, Ncyt=Ncyt, Nnuc=Nnuc, Nleo=Nleo)

    return compile_cpp(d, WDIR, EPS)


def run_projection(angle, R, T, C, WDIR, P, remove_unfinished=True,
                   Nmed=_Nmed, Ncyt=_Ncyt, Nnuc=_Nnuc, Nleo=_Nleo):
    """ If angle is None, the background will be computed and the
    refractive indices will be ignored.
    
    """
    # make sure directory exists
    mkdir(WDIR)
    if angle is None:
        print("Compiling backgrund".format(angle))
        binary = make_binary(P, WDIR, 0, T, R, onlymedium=True,
                             Nmed=Nmed, Ncyt=Ncyt, Nnuc=Nnuc, Nleo=Nleo)
    else:
        print("Compiling angle {}".format(angle))
        binary = make_binary(P, WDIR, angle, T, R, onlymedium=False,
                             Nmed=Nmed, Ncyt=Ncyt, Nnuc=Nnuc, Nleo=Nleo)
    os.chdir(WDIR)
    if not simulation_completed(binary, 
                                remove_unfinished=remove_unfinished):
        outputname = binary[:-4]+"_output.txt"
        print("Running {}".format(binary))
        runstring = "mpirun -n {} {} 2>&1 | tee {} ".format(
                    C, binary, outputname)
        os.system(runstring)
        outdir = binary[:-4]+"-out"
        os.rename(os.path.join(WDIR, outputname),
                  os.path.join(outdir, "output.txt"))
    else:
        print("Simulation already completed.")


def run_tomography(A, R, T, C, DIR, P, remove_unfinished=True,
                   Nmed=_Nmed, Ncyt=_Ncyt, Nnuc=_Nnuc, Nleo=_Nleo):
    ## Create binaries
    # Create output directory
    script = os.path.split(P)[1]
    dname = "{}_A{:04d}_R{:02d}_T{:08d}_Nmed{}_Ncyt{}_Nnuc{}_Nleo{}/".\
            format(script[:-4], A, R, T, Nmed, Ncyt, Nnuc, Nleo)
    WDIR = os.path.join(DIR, dname)
    print("Creating directory: {}".format(WDIR))
    mkdir(WDIR)
    # Run background:
    run_projection(None, R, T, C, WDIR, P,
                   Nmed=Nmed, Ncyt=Ncyt, Nnuc=Nnuc, Nleo=Nleo)

    angles = np.linspace(0, 2*np.pi, A, endpoint=False)

    for angle in angles:
        run_projection(angle, R, T, C, WDIR, P, 
                       remove_unfinished=remove_unfinished,
                       Nmed=Nmed, Ncyt=Ncyt, Nnuc=Nnuc, Nleo=Nleo)


def simulation_completed(path, remove_unfinished=False, verbose=0):
    """ Check if a FDTD simulation was completed successfully.
    
    The argument is a binary `.bin` file.
    
    This is usually the case if `path[:-4]+"-out"` exists and there are
    h5 files in it.
    """
    if path.endswith("/"):
        path = os.path.dirname(path)
    
    if ( path.endswith(".bin") or 
           path.endswith(".cpp") or 
           path.endswith("-out") ):
        pass
    else:
        if verbose > 0:
            warnings.warn("Path has unknown ending: {}".format(path))

    folder = path[:-4]+"-out"
    if verbose > 1:
        print("Searching: ", folder)
    
    if os.path.isdir(folder):
        files = os.listdir(folder)
        ok_counter = 0
        for f in files:
            if ( (f.startswith("eps") and f.endswith(".h5")) or
                 ((f.startswith("ez") or f.startswith("ey")) and
                   f.endswith(".h5")) or
                 ((f.startswith("eps") or f.startswith("empty")) and 
                   f.endswith(".cpp")) or
                 ( f == "output.txt" )
                ):
                ok_counter += 1
        if ok_counter >= 4:
            return True
        elif remove_unfinished:
            print("Removing unfinished simulation: {}".format(folder))
            print("Press Ctrl+C to abort. You have 5s.")
            time.sleep(5)
            for f in files:
                print("Deleting {}".format(f))
                os.remove(os.path.join(folder,f))
            os.rmdir(folder)
            print("Deleting {}".format(folder))
    # default
    return False



if __name__ == "__main__":
    ## Parse all arguments
    C = mp.cpu_count()
    WDIR = os.path.dirname(os.path.abspath(__file__))
    P = os.path.join(WDIR, "phantom_2d.cpp")
    DIR = os.path.join(WDIR, "simulations")
    parser = argparse.ArgumentParser(
            description='Finite difference timed domain (FDTD) '+\
                        'tomographic data acquisition.',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-a', '--angles',  metavar='A', type=int,
                        default=100,
                        help='total number of acquisition angles')
    parser.add_argument('-c', '--cpus', metavar='C', type=int,
                        default=C,
                        help='number of CPUs'.format(C))
    parser.add_argument('-d', '--directory', metavar='DIR', type=str,
                        default=DIR, 
                        help='output directory'.format(DIR))
    parser.add_argument('-p', '--phantom_script', metavar='P', type=str,
                        default=P,
                        help='dielectric phantom'.format(P))
    parser.add_argument('-r', '--resolution', metavar='R', type=int,
                        default=13,
                        help='number of pixels per wavelength (in vacuum)')
    parser.add_argument('-t', '--timesteps',  metavar='T', type=int,
                        default=15000,
                        help='number of FDTD time steps to perform')
    parser.add_argument('--medium',  metavar='Nmed', type=float,
                        default=_Nmed,
                        help='Refractive index of medium')
    parser.add_argument('--cytoplasm',  metavar='Ncyt', type=float,
                        default=_Ncyt,
                        help='Refractive index of cytoplasm')
    parser.add_argument('--nucleus',  metavar='Nnuc', type=float,
                        default=_Nnuc,
                        help='Refractive index of nucleus')
    parser.add_argument('--nucleolus',  metavar='Nleo', type=float,
                        default=_Nleo,
                        help='Refractive index of nucleolus')


    args = parser.parse_args()
    A = args.angles
    R = args.resolution
    T = args.timesteps
    C = args.cpus
    DIR = os.path.abspath(args.directory)
    P = os.path.abspath(args.phantom_script)
    Nmed = args.medium
    Ncyt = args.cytoplasm
    Nnuc = args.nucleus
    Nleo = args.nucleolus
    
    run_tomography(A, R, T, C, DIR, P, 
                   Nmed=Nmed, Ncyt=Ncyt, Nnuc=Nnuc, Nleo=Nleo)
