meep_tomo
=========
A Python wrapper for tomography using MEEP C++ scripts 

|Build Status| |Coverage Status|


System dependencies
-------------------
On debian-like systems run


   sudo apt install libatlas-base-dev libharminv-dev libgsl-dev libmeep-mpi-default-dev libmeep-mpi-default8
   sudo apt install hdf5-tools h5utils libhdf5-openmpi-dev libhdf5-serial-dev


Python dependencies
-------------------


To install all dependencies, run

    pip install -r requirements.txt


To test the wrapper, run

    py.test


.. |Build Status| image:: http://img.shields.io/travis/RI-imaging/FDTD_sinogram.svg
   :target: https://travis-ci.org/RI-imaging/FDTD_sinogram
.. |Coverage Status| image:: https://img.shields.io/coveralls/RI-imaging/FDTD_sinogram.svg
   :target: https://coveralls.io/r/RI-imaging/FDTD_sinogram
