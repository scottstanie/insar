#!/bin/bash
set -e

# sudo apt-get -y update
# sudo apt-get -y upgrade
# sudo apt-get install -y aptitude
# sudo aptitude install -y make cmake-curses-gui build-essential git libtool automake libbz2-dev python-dev libjpeg-dev libboost1.55-dev
#
TOPDIR="$HOME/otb"
mkdir -p $TOPDIR
cd $TOPDIR
mkdir -p installdir build
export OTB_SRC=$TOPDIR/OTB
export OTB_BUILD=$TOPDIR/build
export OTB_INSTALL=$TOPDIR/installdir
git clone https://github.com/orfeotoolbox/OTB.git OTB

cd build
cmake $OTB_SRC/SuperBuild -DCMAKE_INSTALL_PREFIX=$TOPDIR/installdir -DOTB_USE_OPENCV=OFF -DOTB_USE_QT4=OFF
# cmake $TOPDIR/OTB/SuperBuild -DCMAKE_INSTALL_PREFIX=$TOPDIR/installdir

make -j4

cmake \
 -DCMAKE_CXX_COMPILER:FILEPATH=/usr/bin/g++-4.5
 -DCMAKE_C_COMPILER:FILEPATH=/usr/bin/gcc-4.5
 -DCMAKE_INSTALL_PREFIX:PATH=$OTB_INSTALL \
 -DGDAL_INCLUDE_DIR:PATH=$GDAL_INSTALL/include \
 -DBUILD_APPLICATIONS:BOOL=ON \
 -DWRAP_PYTHON:BOOL=ON \
 -DBUILD_TESTING:BOOL=OFF \
 -DBUILD_EXAMPLES:BOOL=OFF \
 -DOTB_USE_VISU_GUI:BOOL=OFF \
 -DOTB_USE_CURL:BOOL=ON \
 -DOTB_USE_PQXX:BOOL=OFF \
 -DOTB_USE_PATENTED:BOOL=OFF \
 -DOTB_USE_EXTERNAL_BOOST:BOOL=ON \
 -DOTB_USE_EXTERNAL_EXPAT:BOOL=ON \
 -DOTB_USE_EXTERNAL_FLTK:BOOL=ON \
 -DOTB_USE_MAPNIK:BOOL=OFF \
 $OTB_SRC
