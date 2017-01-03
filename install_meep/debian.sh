#!/bin/bash
## Stop if errors occur
set -e

## --- Settings ---------------------------------------------------------------

## (Optional) Install utilities
echo "...Installing utilities"
sudo apt-get install -y python-matplotlib mayavi2 h5utils

## --- Build dependencies -----------------------------------------------------
## (list obtained from https://launchpad.net/ubuntu/quantal/+source/meep)
echo "...Installing build dependencies"
sudo apt-get install -y autotools-dev chrpath debhelper gfortran \
    libatlas-base-dev guile-2.0-dev libfftw3-dev libgsl0-dev \
    libharminv-dev libhdf5-serial-dev liblapack-dev pkg-config zlib1g-dev

## Unfortunately we may not install libctl-dev directly, as MEEP 1.2.1 needs version >=3.2
echo "...Installing libctl"
wget http://ab-initio.mit.edu/libctl/libctl-3.2.1.tar.gz
tar xzf libctl* && cd libctl-3.2.1/
./configure LIBS=-lm  &&  make  &&  sudo make install
cd ..

## --- MEEP -------------------------------------------------------------------
echo "...Installing meep dependencies 1"
## Skip this line if no multiprocessing used:
meep_opt="--with-mpi"; sudo apt-get -y install openmpi-bin libopenmpi-dev
## Install really everything that has to do with hdf5
echo "...Installing meep dependencies 2"
sudo aptitude -y install hdf5-tools h5utils libhdf5-openmpi-dev libhdf5-serial-dev
# You might also want to try these:
# sudo apt-get -y install libhdf5-openmpi-10 libhdf5-10 hdf5-helpers 

echo "...Setting meep flags"
export CFLAGS=" -fPIC"; export CXXFLAGS=" -fPIC"; export FFLAGS="-fPIC" 
export CPPFLAGS="-I/usr/local/include -I/usr/lib/x86_64-linux-gnu/hdf5/openmpi/ -I/usr/lib/x86_64-linux-gnu/hdf5/serial/ -I/usr/include/ -I/usr/include/hdf5/openmpi/ -I/usr/include/hdf5/serial/"
export LDFLAGS="-L/usr/local/include -L/usr/lib/x86_64-linux-gnu/hdf5/openmpi/ -L/usr/lib/x86_64-linux-gnu/hdf5/serial/ -L/usr/include/ -L/usr/include/hdf5/openmpi/ -L/usr/include/hdf5/serial/"
export LD_RUN_PATH="/usr/local/lib"

echo "...Installing meep"
# Does not work on ubuntu 16.04
#wget http://jdj.mit.edu/~stevenj/meep-1.2.1.tar.gz
#tar xzf meep-1.2.1.tar.gz  &&  cd meep-1.2.1/

wget http://ab-initio.mit.edu/meep/meep-1.3.tar.gz
tar xzf meep-1.3.tar.gz  &&  cd meep-1.3/

./configure $meep_opt --enable-shared --prefix=/usr/local  &&  make  &&  sudo make install
cd ..

echo "...Updating library cache"
# Update library cache if you get the error "cannot open shared object file: No such file or directory":
sudo ldconfig

