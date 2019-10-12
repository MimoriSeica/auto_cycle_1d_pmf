#!/bin/bash

sander -O \
 -p alat.prmtop \
 -i run.in \
 -c alat.crd \
 -o run.out \
 -r run.rst \
 -x run.nc
