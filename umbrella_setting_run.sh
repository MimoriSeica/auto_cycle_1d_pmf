#!/bin/bash

sander -O \
 -p simulation/init/alat.prmtop \
 -i simulation/umbrella_setting/run.in \
 -c simulation/init/run.rst \
 -o simulation/umbrella_setting/run.out \
 -r simulation/umbrella_setting/run.rst \
 -x simulation/umbrella_setting/run.nc

