## Finite-Difference Time-Domain (FDTD) simulations with cell phantoms
This is a collection of [Meep](http://ab-initio.mit.edu/wiki/index.php/Main_Page) scripts and Python wrappers to generate tomographic scattering data of artificial cell phantoms, to reconstruct the refractive index (RI) from these scattering data, and to analyze the quality of the reconstructed RI in dependence of cell phantom properties and used reconstruction algorithm.

- **phantoms_meep**: Meep C++ scripts of 2D and 3D cell phantoms
- **meep_tomo**: Python wrapper for the C++ scripts
- **examples**: Example usages of the Python wrapper that reproduce figures from the paper below


### References
I used these scripts for the 2015 publication [*ODTbrain: a Python library for full-view, dense diffraction tomography* (Müller et al.) **BMC Bioinformatics**](http://dx.doi.org/10.1186/s12859-015-0764-0). Please cite this paper if you are using them in a scientific publication.
