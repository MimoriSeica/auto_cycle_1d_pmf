#!/bin/bash

sander -O \
 -p ../0_init/alat.prmtop \
 -i run.in \
 -c ../0_init/alat.crd \
 -o run.out \
 -r run.rst \
 -x run.nc

