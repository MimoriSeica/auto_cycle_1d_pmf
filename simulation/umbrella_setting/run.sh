#!/bin/bash

sander -O \
 -p ../init/alat.prmtop \
 -i run_%d.in \
 -c ../init/run.rst \
 -o run.out \
 -r run.rst \
 -x run.nc

