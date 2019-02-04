import codecs
import multiprocessing as mp
import numpy as np
import os
import shutil
import time

from .common import mkdir_p


#: The meep library name to use. On a Ubuntu 17.10 machine and
#: when meep is installed via
#: `apt install libmeep-mpi-default-dev libmeep-mpi-default`, the name
#: will be "meep_mpi-default". To find out the name, run
#: `locate meep | grep /usr/lib`, which would yield something like
#: `/usr/lib/x86_64-linux-gnu/libmeep_mpi-default.so.8`,
#: `/usr/lib/x86_64-linux-gnu/libmeep_mpi-default.so.8.0.0`.
MEEP_LIBRARY_NAME = "meep_mpi-default"


def create_cpp(phantom_template, cpp_path, **pkwargs):
    """Create a MEEP simulation C++ script from a phantom template

    Parameters
    ----------
    phantom_template: str
        Path to a phantom C++ script
    cpp_path: str
        Path to the resulting C++ script
    pkwargs: dict
        The kwargs are used to set the parameters that are
        given with "#define"-statements in `phantom_template`.
        The names can be mixed upper- and lower-case. However, 
        in the C++ template, they should be all-caps.
    """
    # read in the original cpp script
    with open(phantom_template, "r") as fd:
        script = fd.readlines()

    for key in pkwargs:
        string = "#define {} ".format(key.upper())
        # Add lower() to correctly map True/False
        value = "{}".format(pkwargs[key]).lower()
        for ii in range(len(script)):
            if script[ii].startswith(string):
                script[ii] = "{} {}\n".format(string, value)
                break
        else:
            msg = "'#define {}' not found in {}!".format(key.upper(),
                                                         phantom_template)
            raise KeyError(msg)

    with open(cpp_path, "w") as fd:
        fd.writelines(script)


def compile_cpp(cpp_path, verbose=False):
    """Compile a cpp file for usage with meep

    Parameters
    ----------
    cpp_path: str
        Full path to C++ file
    verbose: bool
        Save output of g++ to log file

    Returns
    -------
    bin_path: str
        Full path to the compiled binary

    Notes
    -----
    The resulting binaries can be executed with 

        mpirun `bin_file`
    """
    assert cpp_path.endswith(".cpp")
    bin_path = cpp_path[:-4]+".bin"

    gpplib = " -lhdf5 -lz -lgsl -lharminv -llapack " +\
             "-lcblas -latlas -lfftw3 -lm "
    gpplib += "-l{}".format(MEEP_LIBRARY_NAME)

    library_paths = "-L/usr/include/hdf5/serial/ " +\
                    "-L/usr/lib/x86_64-linux-gnu/hdf5/openmpi/ " +\
                    "-L/usr/lib/x86_64-linux-gnu/hdf5/serial/ " +\
                    "-L/usr/lib/x86_64-linux-gnu/ " +\
                    "-L/usr/local/lib/"

    runstring = "g++ {} -malign-double {} -o {}  {} 2>&1".format(
        library_paths,
        os.path.join(cpp_path),
        os.path.join(bin_path),
        gpplib)

    if verbose:
        log_path = cpp_path[:-4]+"_compile.log"
        runstring += " | tee {}".format(log_path)

    if os.path.exists(bin_path):
        os.remove(bin_path)
    os.system(runstring)

    return bin_path


def get_phantom_kwargs(phantom_cpp):
    """Returns a dict for all "#define" statements of a cpp file"""
    # read in the original cpp script
    with codecs.open(phantom_cpp, "r", encoding="utf-8") as fd:
        script = fd.readlines()

    kwargs = {}
    for ii in range(len(script)):
        if script[ii].startswith("#define"):
            splitted = script[ii].split()
            key = splitted[1].lower()
            val = splitted[2].lower()
            if val == "true":
                val = True
            elif val == "false":
                val = False
            elif val.count("."):
                val = float(val)
            else:
                val = int(val)
            kwargs[key] = val

    return kwargs


def make_binary(phantom_template, bin_path, verbose=False, **pkwargs):
    """Compile a meep binary with custom parameters from a C++ phantom template

    Parameters
    ----------
    phantom_template: str
        Path to a phantom template
    bin_path: str
        Path to the resulting binary. The extension ".bin" is appended
        if non-existent.
    pkwargs: dict
        The kwargs are used to set the parameters that are
        given with "#define"-statements in `phantom_template`.
        The names can be mixed upper- and lower-case. However, 
        in the C++ template, they should be all-caps.
    """
    if not bin_path.endswith(".bin"):
        bin_path += ".bin"

    cpp_path = bin_path[:-4]+".cpp"
    create_cpp(phantom_template=phantom_template,
               cpp_path=cpp_path,
               **pkwargs)
    # this will create `bin_path`
    compile_cpp(cpp_path, verbose=verbose)


def run_projection(phantom_template, dir_out,
                   num_cpus=mp.cpu_count(), remove_unfinished=True,
                   verbose=0, **pkwargs):
    """Run a meep simulation for a phantom

    Paramters
    ---------
    phantom_template: str
        Path to a phantom template
    dir_out: str
        Output directory where cpp, bin, and log files will be created
    num_cpus: bool
        Number of CPUs to use with mpirun
    remove_unfinished: bool
        Remove files from unfinished simulations before starting anew
    verbose: int
        Increases verbosity
    pkwargs: dict
        The kwargs are used to set the parameters that are
        given with "#define"-statements in `phantom_template`.
        The names can be mixed upper- and lower-case. However, 
        in the C++ template, they should be all-caps.
        `pkwargs` must at least contain the "angle" keyword
        if `pkwargs["onlymedium"]` is `False`.
    """
    phtbase = os.path.basename(phantom_template)
    # make sure directory exists
    mkdir_p(dir_out)
    # make lower-case
    npkw = {}
    for kk in pkwargs:
        npkw[kk.lower()] = pkwargs[kk]

    if "onlymedium" in npkw and npkw["onlymedium"] == True:
        bin_base = "bg_{}.bin".format(phtbase[:-4])
        if verbose:
            print("...Running background")
    else:
        assert "angle" in pkwargs, "`pkwargs` must contain the 'angle' key"
        npkw["onlymedium"] = False
        bin_base = "ph_{}_{:.15f}.bin".format(phtbase[:-4], pkwargs["angle"])
        if verbose:
            print("...Running angle {:.3f}".format(npkw["angle"]))

    bin_path = os.path.join(dir_out, bin_base)
    # This output directory is some kind of default in meep
    outdir = bin_path[:-4]+"-out"

    prevdir = os.path.abspath(os.curdir)
    os.chdir(dir_out)
    if not simulation_completed(bin_path,
                                remove_unfinished=remove_unfinished):
        # Create the binary for the simulation
        make_binary(phantom_template=phantom_template,
                    bin_path=bin_path,
                    verbose=verbose,
                    **pkwargs)

        base_bin_path = os.path.basename(bin_base)
        logfile = base_bin_path[:-4]+"_exec.log"
        if verbose:
            print("...Executing {}".format(bin_base))
        runstring = "mpirun -n {} {} 2>&1 | tee {} ".format(
                    num_cpus, base_bin_path, logfile)
        os.system(runstring)
        # move log file
        os.rename(os.path.join(dir_out, logfile),
                  os.path.join(outdir, logfile))
        # move cpp file
        os.rename(bin_path[:-4]+".cpp",
                  os.path.join(outdir, os.path.basename(bin_path[:-4]+".cpp")))
        # move compile log
        complog = bin_path[:-4]+"_compile.log"
        if os.path.exists(complog):
            os.rename(complog,
                      os.path.join(outdir, os.path.basename(complog)))
        # remove binary
        os.remove(bin_path)
    else:
        if verbose:
            print("...Simulation already completed.")
    os.chdir(prevdir)

    return outdir


def run_tomography(phantom_template, num_angles, dir_out, scale=1,
                   scale_vars_end=["_A", "_B", "_C", "_X", "_Y", "_Z"],
                   remove_unfinished=True, verbose=0,
                   **pkwargs):
    """Run a tomographic series of projections

    Parameters
    ----------
    phantom_template: str
        Path to a phantom template
    num_angles: int
        Number of angles to compute over the range [0,2PI)
    dir_out: str
        Path to the simulation output directory
    scale: float
        Scales the phantom size (see `scale_vars_end`)
    scale_vars_end: list of str
        The items of the list define the end of the names
        of the variables that will be scaled, e.g. by default,
        all variables that end with "_A" such as "CYTOPLASM_A"
        are scaled by one.
    remove_unfinished: bool
        Remove files from unfinished simulations before starting anew
    verbose: int
        Increases verbosity
    pkwargs: dict
        The kwargs are used to set the parameters that are
        given with "#define"-statements in `phantom_template`.
        The names can be mixed upper- and lower-case. However, 
        in the C++ template, they should be all-caps.

    See Also
    --------
    run_projection: this method is called for each projection

    Notes
    -----
    The phantom files must have "#define" statements for at least
    "ANGLE" and "ONLYMEDIUM".
    """
    # scaling variable identifiers
    scale_vars_end = list(set(scale_vars_end))
    # convert pkwargs to lower-case and scale keyword arguments
    pkwargs_lower = {}
    for kk in pkwargs:
        pkwargs_lower[kk.lower()] = pkwargs[kk]
    # load default kwargs
    npkw = get_phantom_kwargs(phantom_template)
    # update default kwargs with user-defined kwargs
    npkw.update(pkwargs_lower)
    for kk in npkw:
        if scale != 1:
            for end in scale_vars_end:
                if kk.endswith(end.lower()):
                    npkw[kk] *= scale
    # Run background:
    npkw["onlymedium"] = True
    run_projection(phantom_template=phantom_template,
                   dir_out=dir_out,
                   remove_unfinished=remove_unfinished,
                   verbose=verbose,
                   **npkw)
    # Run other phantom projections
    npkw["onlymedium"] = False
    angles = np.linspace(0, 2*np.pi, num_angles, endpoint=False)

    for angle in angles:
        npkw["angle"] = angle
        run_projection(phantom_template=phantom_template,
                       dir_out=dir_out,
                       remove_unfinished=remove_unfinished,
                       verbose=verbose,
                       **npkw)


def simulation_completed(path, remove_unfinished=False, verbose=0):
    """Check if an FDTD simulation was completed successfully

    This will check whether a `run_projection` completed successfully

    Paramters
    ---------
    path: str
        Path to the simulation  C++, binary or "-out" folder
    remove_unfinished: bool
        Remove unfinished simulation files. This removes the "-out"
        folder which is important for subsequent simulations, because
        meep will name subsequent folders "-out1", etc. which would
        break our analysis pipeline.

    Returns
    -------
    completed: bool
        `True` if simulation was successfully completed.

    Notes
    -----
    Each simulation is run in a separate folder. Meep creates the "-out"
    folder in those folders that contains the simulation result in the h5
    file format. 
    """
    if (path.endswith(".bin") or
        path.endswith(".cpp") or
            path.endswith("-out")):
        out_path = path[:-4]+"-out"
    else:
        raise ValueError("Path must be a C++, binary file or output folder!")

    if os.path.isdir(out_path):
        files = os.listdir(out_path)
        ok_counter = 0
        for f in files:
            if (  # phantom eps structure file
                (f.startswith("eps") and f.endswith(".h5")) or
                # fields results file
                ((f.startswith("ez") or f.startswith("ey")) and
                 f.endswith(".h5")) or
                # simulation cpp script (is copied there afterwards)
                ((f.startswith("ph") or f.startswith("bg")) and
                 f.endswith(".cpp")) or
                # execution log file
                (f.endswith("_exec.log"))
            ):
                ok_counter += 1
        if ok_counter >= 4:
            return True
        elif remove_unfinished:
            print("...Removing unfinished simulation: {}".format(out_path))
            print("...Press Ctrl+C to abort. You have 5s.")
            time.sleep(5)
            shutil.rmtree(out_path, ignore_errors=True)
            print("...Deleted.")
    return False
