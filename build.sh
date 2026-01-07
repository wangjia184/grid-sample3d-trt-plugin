#!/bin/bash

rm -rf ./build
mkdir -p build
cd build
cmake .. -DTensorRT_ROOT=/usr/local/tensorrt -DCMAKE_BUILD_TYPE=Release
make
cp -f libgrid_sample_3d_plugin.so ../../bin/