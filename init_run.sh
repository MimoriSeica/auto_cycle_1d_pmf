#!/bin/bash

sander -O \
 -p simulation/init/alat.prmtop \
 -i simulation/init/run.in \
 -c simulation/init/alat.crd \
 -o simulation/init/run.out \
 -r simulation/init/run.rst \
 -x simulation/init/run.nc

