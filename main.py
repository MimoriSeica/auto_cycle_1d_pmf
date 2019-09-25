import os
import json
import random

# グローバル変数
SETTING_FILE_NAME = "setting_file.txt"
BIN_SIZE = 181

def read_setting_file():
    with open(SETTING_FILE_NAME,'r') as f:
        return json.load(f)

def write_setting_file(setting_data):
    with open(SETTING_FILE_NAME,'w') as f:
        json.dump(setting_data, f, indent = 4)

def make_init_setting_file():
    init_data = {}
    init_data["outputFiles"] = {}
    for i in range(BIN_SIZE):
        init_data["outputFiles"]["{}".format(i)] = []

    init_data["nextSearch"] = 90
    init_data["tryCount"] = 0

    return init_data

def initialize():
    write_setting_file(make_init_setting_file())

def simulation_init():
    os.system("echo simulation_init")
    #os.system("bash simulation/init/run.sh")

class simulation_umbrella_setting:
    def make_input(self, setting_data):
        filename = "simulation/umbrella_setting/run.in"
        print("writing %s..." % (filename))
        with open(filename, 'w') as f:
            ig = random.randint(0,1000000)
            f.write("equilibration with restraint\n")
            f.write(" &cntrl\n")
            f.write("   ig=%d, \n" % (ig))
            f.write("   irest=1, ntx=5,\n")
            f.write("   igb=8, gbsa=1,\n")
            f.write("   cut=9999.0, rgbmax=9998.0,\n")
            f.write("   ntc=2, ntf=1, tol=0.000001,\n")
            f.write("   ntt=3, gamma_ln=2.0, temp0=300.0,\n")
            f.write("   ntb=0, nscm=10000,\n")
            f.write("   ioutfm=1,\n")
            f.write("   nstlim=500000, dt=0.002,\n")
            f.write("   ntpr=50000, ntwx=50000, ntwv=0, ntwr=500000,\n")
            f.write("   nmropt=1,\n")
            f.write(" /\n")
            f.write(" &wt\n")
            f.write("  type='END',\n")
            f.write(" /\n")
            f.write("DISANG=run.disang\n")
            f.write("\n")

    def make_sh(self, setting_data):
        filename = "simulation/umbrella_setting/run.sh"
        print("writing %s..." % (filename))
        with open(filename, 'w') as f:
            f.write("#!/bin/bash\n")
            f.write("\n")
            f.write("sander -O \\\n")
            f.write(" -p ../init/alat.prmtop \\\n")
            f.write(" -i run_%d.in \\\n")
            f.write(" -c ../init/run.rst \\\n")
            f.write(" -o run.out \\\n")
            f.write(" -r run.rst \\\n")
            f.write(" -x run.nc\n")
            f.write("\n")
        os.chmod(filename, 0o755)

    def make_disang(self, setting_data):
        filename = "simulation/umbrella_setting/run.disang"
        value = setting_data["nextSearch"]
        print("writing %s..." % (filename))
        with open(filename, 'w') as f:
            f.write("harmonic restraint changing spring constant\n")
            f.write(" &rst\n")
            f.write("   iat=9,15,17,19,\n")
            f.write("   r0=%f, r0a=%f, k0=0.01, k0a=200.0,\n" % (value, value))
            f.write("   ifvari=1, nstep1=0, nstep2=250000,\n")
            f.write(" /\n")
            f.write(" &rst\n")
            f.write("   iat=9,15,17,19,\n")
            f.write("   r0=%f, r0a=%f, k0=200.0, k0a=200.0,\n" % (value, value))
            f.write("   ifvari=1, nstep1=250001, nstep2=500000,\n")
            f.write(" /\n")
            f.write("\n")

    def __init__(self, setting_data):
        self.make_disang(setting_data)
        self.make_input(setting_data)
        self.make_sh(setting_data)
        os.system("echo simulation_umbrella_setting")
        #os.system("bash simulation/umbrella_setting/run.sh")

class simulation_production:
    def make_input(self, setting_data):
        filename = "simulation/production/run.in"
        value = setting_data["tryCount"]
        print("writing %s..." % (filename))
        with open(filename, 'w') as f:
            ig = random.randint(0,1000000)
            f.write("production with restraint\n")
            f.write(" &cntrl\n")
            f.write("   ig=%d, \n" % (ig))
            f.write("   irest=1, ntx=5,\n")
            f.write("   igb=8, gbsa=1,\n")
            f.write("   cut=9999.0, rgbmax=9998.0,\n")
            f.write("   ntc=2, ntf=1, tol=0.000001,\n")
            f.write("   ntt=3, gamma_ln=2.0, temp0=300.0,\n")
            f.write("   ntb=0, nscm=10000,\n")
            f.write("   ioutfm=1,\n")
            f.write("   nstlim=5000000, dt=0.002,\n")
            f.write("   ntpr=5000, ntwx=5000, ntwv=0, ntwr=500000,\n")
            f.write("   nmropt=1,\n")
            f.write(" /\n")
            f.write(" &wt\n")
            f.write("  type='DUMPFREQ', istep1=5000,\n")
            f.write(" /\n")
            f.write(" &wt\n")
            f.write("  type='END',\n")
            f.write(" /\n")
            f.write("DISANG=run.disang\n")
            f.write("DUMPAVE=../data/run_%d.dat\n" % (value))
            f.write("\n")

    def make_sh(self, setting_data):
        filename = "simulation/production/run.sh"
        print("writing %s..." % (filename))
        with open(filename, 'w') as f:
            f.write("#!/bin/bash\n")
            f.write("\n")
            #f.write("NPROC=2\n")
            f.write("sander -O \\\n")
            f.write(" -p ../init/alat.prmtop \\\n")
            f.write(" -i run.in \\\n")
            f.write(" -c ../umbrella_setting/run.rst \\\n")
            f.write(" -o run.out \\\n")
            f.write(" -r run.rst \\\n")
            f.write(" -x run.nc\n")
            f.write("\n")
        os.chmod(filename, 0o755)

    def make_disang(self, setting_data):
        filename = "simulation/production/run.disang"
        value = setting_data["nextSearch"]
        print("writing %s..." % (filename))
        with open(filename, 'w') as f:
            f.write("harmonic restraint fixed spring constant\n")
            f.write(" &rst\n")
            f.write("   iat=9,15,17,19,\n")
            f.write("   r0=%f, k0=200.0,\n" % (value))
            f.write(" /\n")
            f.write("\n")

    def __init__(self, setting_data):
        self.make_disang(setting_data)
        self.make_input(setting_data)
        self.make_sh(setting_data)
        os.system("echo simulation_production")
        #os.system("bash simulation/umbrella_setting/run.sh")

def simulation(setting_data):
    simulation_init()
    simulation_umbrella_setting(setting_data)
    simulation_production(setting_data)

def analyze(setting_data):
    pass

def main():
    initialize()

    playCount = 1
    for _ in range(playCount):
        setting_data = read_setting_file()
        simulation(setting_data)
        analyze(setting_data)
        write_setting_file(setting_data)

if __name__ == "__main__":
    main()
