#!/bin/bash

sander -O \
 -p simulation/init/alat.prmtop \
 -i simulation/production/run.in \
 -c simulation/umbrella_setting/run.rst \
 -o simulation/production/run.out \
 -r simulation/production/run.rst \
 -x simulation/production/run.nc

