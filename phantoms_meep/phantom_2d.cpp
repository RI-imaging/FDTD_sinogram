// 2D phantom for MEEP simulation
# include <meep.hpp>
# include <cstdlib>
# include <iostream>
# include <iomanip>
# include <fstream>
# include <cmath>
# include <complex>


// Do not set this TIMESTEPS too short. Wee need the system to equilibrate.
// Very good agreement with  "#define TIMESTEPS 1000000"
// This worked well with     "#define TIMESTEPS 100000"
// This worked worse with    "#define TIMESTEPS 10000"
#define TIMESTEPS 15000

// Coordinates
// The cytoplasm is centered at the origin.  All coordinates are
// given in wavelengths (Assuming a wavelength of 500nm, 20 wavelengths
// are 10um). The cytoplasm and nucleus have major axis (A) and minor
// axis (B). The angle of rotation PHI is angle between major axis (A)
// and x-axis in radians.

#define ANGLE .02

// These SIZEs include the PML and are total domain sizes (not half)
// LATERALSIZE can be large to grab all scattered fields
#define LATERALSIZE     30.0
// up to which distance in axial direction do we need the field?
#define AXIALSIZE       20.0

#define MEDIUM_RI       1.333

// A and B are half of the full minor/major axes
#define CYTOPLASM_RI    1.365
#define CYTOPLASM_A     8.5
#define CYTOPLASM_B     7.0

#define NUCLEUS_RI      1.360
#define NUCLEUS_A       4.5
#define NUCLEUS_B       3.5
#define NUCLEUS_PHI     0.5
#define NUCLEUS_X       2.0
#define NUCLEUS_Y       1.0

#define NUCLEOLUS_RI    1.387
#define NUCLEOLUS_A     1.0
#define NUCLEOLUS_B     1.0
#define NUCLEOLUS_PHI   0.0
#define NUCLEOLUS_X     2.0
#define NUCLEOLUS_Y     2.0

// Choosing the resolution
// http://ab-initio.mit.edu/wiki/index.php/Meep_Tutorial
// In general, at least 8 pixels/wavelength in the highest
// dielectric is a good idea.
#define WAVELENGTH 13.       // How many pixels for a wavelength?
// A PML thickness of about half the wavelength is ok.
// http://www.mail-archive.com/meep-discuss@ab-initio.mit.edu/msg00525.html
#define PMLSIZE 0.5 // PML thickness in wavelengths

#define ONLYMEDIUM false

using namespace meep;

double eps(const vec &p)
{
    if (ONLYMEDIUM) {
         return pow(MEDIUM_RI,2.0);
    } else {
        double cox = p.x()-LATERALSIZE/2.0;
        double coy = p.y()-AXIALSIZE/2.0;
        
        double rottx = cox*cos(ANGLE) - coy*sin(ANGLE);
        double rotty = cox*sin(ANGLE) + coy*cos(ANGLE);

        // Cytoplasm
        double crotx = rottx;
        double croty = rotty;
        
        // Nucleus
        double nrotx = (rottx-NUCLEUS_X)*cos(NUCLEUS_PHI) - (rotty-NUCLEUS_Y)*sin(NUCLEUS_PHI);
        double nroty = (rottx-NUCLEUS_X)*sin(NUCLEUS_PHI) + (rotty-NUCLEUS_Y)*cos(NUCLEUS_PHI);

        // Nucleolus
        double nnrotx = (rottx-NUCLEOLUS_X)*cos(NUCLEOLUS_PHI) - (rotty-NUCLEOLUS_Y)*sin(NUCLEOLUS_PHI);
        double nnroty = (rottx-NUCLEOLUS_X)*sin(NUCLEOLUS_PHI) + (rotty-NUCLEOLUS_Y)*cos(NUCLEOLUS_PHI);        
        
        // Nucleolus
        if ( pow(nnrotx,2.0)/pow(NUCLEOLUS_A,2.0) + pow(nnroty,2.0)/pow(NUCLEOLUS_B,2.0) <= 1 ) {
            return pow(NUCLEOLUS_RI,2.0);
        }
        // Nucleus
        else if ( pow(nrotx,2.0)/pow(NUCLEUS_A,2.0) + pow(nroty,2.0)/pow(NUCLEUS_B,2.0) <= 1 ) {
            return pow(NUCLEUS_RI,2.0);
        }
        // Cytoplasm
        else if ( pow(crotx,2.0)/pow(CYTOPLASM_A,2.0) + pow(croty,2.0)/pow(CYTOPLASM_B,2.0) <= 1 ) {
            return pow(CYTOPLASM_RI,2.0);
        }
        // Medium
        else {
            return pow(MEDIUM_RI,2.0);
        }

    }
}


std::complex<double> one(const vec &p) 
{   
    return 1.0;
    //source modulation
    //return exp(-(p.x())*(p.x())/100000);
}


int main(int argc, char **argv) {

    initialize mpi(argc, argv);       // do this even for non-MPI Meep

    ////////////////////////////////////////////////////////////////////
    // CHECK THE eps() FUNCTION WHEN CHANGING STUFF IN THIS BLOCK.
    //
    // determine the size of the copmutational volume
    // 1. The wavelength defines the grid size. One wavelength
    //    is sampled with WAVELENGTH pixels.
    double resolution = WAVELENGTH;
    // The lateral extension sr of the computational grid.
    // make sure sr is even
    double sr = LATERALSIZE;
    double sz = AXIALSIZE;
    // The axial extension is the same as the lateral extension,
    // since we rotate the sample.

    // Wavelength has size of one unit. c=1, so frequency is one as 
    // well.
    double frequency = 1.;
    ////////////////////////////////////////////////////////////////////
    
    if (ONLYMEDIUM){
        // see eps() for more info
        master_printf("...Using empty structure \n");
    }
    else{
        // see eps() for more info
        master_printf("...Using phantom structure \n");
    }

    master_printf("...Lateral object size [wavelengths]: %f \n", 2.0 * CYTOPLASM_A);
    // take the largest number (A) here
    master_printf("...Axial object size [wavelengths]: %f \n", 2.0 * CYTOPLASM_B);
    master_printf("...PML thickness [wavelengths]: %f \n", PMLSIZE);
    master_printf("...Medium RI: %f \n", MEDIUM_RI);
    master_printf("...Sampling per vacuum wavelength [px]: %f \n", WAVELENGTH);
    master_printf("...Radial extension [px]: %f \n", sr * WAVELENGTH);
    master_printf("...Axial extension [px]: %f \n", sz * WAVELENGTH);
    
    int clock0=clock();

    master_printf("...Initializing grid volume \n");
    grid_volume v = vol2d(sr,sz,resolution); 

    master_printf("...Initializing structure \n");
    structure s(v, eps, pml(PMLSIZE)); // structure
    s.set_epsilon(eps,true); //switch eps averaging on or off

    master_printf("...Initializing fields \n");
    fields f(&s);

    const char *dirname = make_output_directory(__FILE__);
    f.set_output_directory(dirname);
    
    master_printf("...Saving dielectric structure \n");
    f.output_hdf5(Dielectric, v.surroundings(),0,false,true,0);

    master_printf("...Adding light source \n");
    // Wavelength is one unit. Since c=1, frequency is also 1.
    continuous_src_time src(1.);
    // Light source is a plane at z = 2*PMLSIZE
    // to not let the source sit in the PML
    volume src_plane(vec(0.0,2*PMLSIZE),vec(sr,2*PMLSIZE));
    
    f.add_volume_source(Ez,src,src_plane,one,1.0);
    master_printf("...Starting simulation \n");


    for (int i=0; i<TIMESTEPS;i++) {
        f.step();
        if ( i == TIMESTEPS - 1){
                f.output_hdf5(Ez,v.surroundings(),0,true,false,0);
        }
    }

return 0;
}


