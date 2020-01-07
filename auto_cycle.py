import os
import json
import random
import numpy as np
import math
import torch
import gpytorch
import sys
from statistics import mean, median,variance,stdev
from matplotlib import pyplot as plt
from scipy.stats import gaussian_kde

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
    init_data["free_energy"] = {}
    init_data["variance"] = {}

    for i in range(BIN_SIZE):
        init_data["outputFiles"]["{}".format(i)] = []

    init_data["nextSearch"] = 90
    init_data["tryCount"] = 0

    return init_data

def initialize():
    write_setting_file(make_init_setting_file())

class simulation_init:

    def make_sh(self):
        filename = "init_run.sh"
        print("writing %s..." % (filename))
        with open(filename, 'w') as f:
            f.write("#!/bin/bash\n")
            f.write("\n")
            f.write("sander -O \\\n")
            f.write(" -p simulation/init/alat.prmtop \\\n")
            f.write(" -i simulation/init/run.in \\\n")
            f.write(" -c simulation/init/alat.crd \\\n")
            f.write(" -o simulation/init/run.out \\\n")
            f.write(" -r simulation/init/run.rst \\\n")
            f.write(" -x simulation/init/run.nc\n")
            f.write("\n")
        os.chmod(filename, 0o755)

    def __init__(self):
        os.system("echo simulation_init")
        self.make_sh()
        os.system("bash init_run.sh")

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
            f.write("DISANG=simulation/umbrella_setting/run.disang\n")
            f.write("\n")

    def make_sh(self, setting_data):
        filename = "umbrella_setting_run.sh"
        print("writing %s..." % (filename))
        with open(filename, 'w') as f:
            f.write("#!/bin/bash\n")
            f.write("\n")
            f.write("sander -O \\\n")
            f.write(" -p simulation/init/alat.prmtop \\\n")
            f.write(" -i simulation/umbrella_setting/run.in \\\n")
            f.write(" -c simulation/init/run.rst \\\n")
            f.write(" -o simulation/umbrella_setting/run.out \\\n")
            f.write(" -r simulation/umbrella_setting/run.rst \\\n")
            f.write(" -x simulation/umbrella_setting/run.nc\n")
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
        os.system("bash umbrella_setting_run.sh")

class simulation_production:
    def make_input(self, setting_data, outputFilePath):
        filename = "simulation/production/run.in"
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
            f.write("   nstlim=10000000, dt=0.002,\n")
            f.write("   ntpr=5000, ntwx=5000, ntwv=0, ntwr=500000,\n")
            f.write("   nmropt=1,\n")
            f.write(" /\n")
            f.write(" &wt\n")
            f.write("  type='DUMPFREQ', istep1=5000,\n")
            f.write(" /\n")
            f.write(" &wt\n")
            f.write("  type='END',\n")
            f.write(" /\n")
            f.write("DISANG=simulation/production/run.disang\n")
            f.write("DUMPAVE={}\n".format(outputFilePath))
            f.write("\n")

    def make_sh(self, setting_data):
        filename = "production_run.sh"
        print("writing %s..." % (filename))
        with open(filename, 'w') as f:
            f.write("#!/bin/bash\n")
            f.write("\n")
            #f.write("NPROC=2\n")
            f.write("sander -O \\\n")
            f.write(" -p simulation/init/alat.prmtop \\\n")
            f.write(" -i simulation/production/run.in \\\n")
            f.write(" -c simulation/umbrella_setting/run.rst \\\n")
            f.write(" -o simulation/production/run.out \\\n")
            f.write(" -r simulation/production/run.rst \\\n")
            f.write(" -x simulation/production/run.nc\n")
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
        outputFilePath = "simulation/data/run_{}.dat".format(setting_data["tryCount"])
        setting_data["outputFiles"]["{}".format(setting_data["nextSearch"])].append(outputFilePath)

        self.make_disang(setting_data)
        self.make_input(setting_data, outputFilePath)
        self.make_sh(setting_data)
        os.system("echo simulation_production")
        os.system("bash production_run.sh")

def simulation(setting_data):
    simulation_umbrella_setting(setting_data)
    simulation_production(setting_data)

class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

class analyze():
    def gauss(self, x, mean):
        _x = np.linalg.norm(np.array(x) - np.array(mean))
        return math.exp(-((_x ** 2) / (2 * (self.SIGMA_POW_2))))

    def biased_energy(self, x, center):
        dx = np.linalg.norm(x - center)
        dx = dx - (math.trunc(dx / 360.0)) * 360.0
        return (self.SPRING_CONSTANT / self.KBT) * (dx ** 2)

    def cal_delta_PMF(self, x_from, x_to, center, kde):
        tmp_PMF_to =  self.KBT * math.log(kde(x_to)) + self.biased_energy(x_to, center)
        tmp_PMF_from =  self.KBT * math.log(kde(x_from)) + self.biased_energy(x_from, center)
        return - (tmp_PMF_to - tmp_PMF_from)

    def train(self, training_times, model, likelihood, optimizer, mll, train_x, train_y):
        for i in range(training_times):
            optimizer.zero_grad()
            output = model(train_x)
            loss = -mll(output, train_y)
            loss.backward()
            print('Iter %d/%d - Loss: %.3f' % (i + 1,
                                               training_times,
                                               loss.item()))
            optimizer.step()

    def __init__(self, setting_data):

        self.TEMPARETURE = 300.
        self.KB_KCALPERMOL = 0.0019872041
        self.KBT = self.KB_KCALPERMOL * self.TEMPARETURE
        self.SPRING_CONSTANT = 200.0 * ((math.pi/180.0) ** 2)
        self.SIGMA = 5
        self.SIGMA_POW_2 = self.SIGMA ** 2
        self.DATA_TRAIN_SPLIT = 5

        pred_x = []
        for i in range(BIN_SIZE-1):
            for j in range(self.DATA_TRAIN_SPLIT):
                pred_x.append([i+j*(1.0/self.DATA_TRAIN_SPLIT)])

        train_x = []
        train_y = []
        for angle in range(BIN_SIZE):
            for filePath in setting_data["outputFiles"]["{}".format(angle)]:
                with open(filePath) as file:
                    rowData = np.array([str.strip().split() for str in file.readlines()], dtype = 'float')[:, 1]
                    now_kde = gaussian_kde(rowData.T)
                    data_stdev = stdev(rowData)

                    for x in pred_x:
                        if x[0] < angle - (data_stdev/2) or angle + (data_stdev/2) - (1.0/self.DATA_TRAIN_SPLIT) < x[0]:
                            continue

                        now_x = [x[0]]
                        now_y = self.cal_delta_PMF(x[0], x[0] + (1.0/self.DATA_TRAIN_SPLIT), angle, now_kde)
                        train_x.append(now_x)
                        train_y.append(now_y)

        train_x = torch.Tensor(np.array(train_x))
        train_y = torch.Tensor(np.array(train_y))
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        model = ExactGPModel(train_x, train_y, likelihood)

        model.train()
        likelihood.train()

        optimizer = torch.optim.Adam([
            {'params': model.parameters()},  # Includes GaussianLikelihood parameters
        ], lr=0.1)

        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
        self.train(50, model, likelihood, optimizer, mll, train_x, train_y)

        model.eval()
        likelihood.eval()

        plot_x = range(BIN_SIZE)
        freeEnegy_y = [0 for i in range(BIN_SIZE)]
        freeEnegy_diff_y = [0 for i in range(BIN_SIZE)]
        variance_y = [0 for i in range(BIN_SIZE)]
        variance_y_cou = [0 for i in range(BIN_SIZE)]
        variance_sum_y = [0 for i in range(BIN_SIZE)]

        pred_y = likelihood(model(torch.Tensor(pred_x)))
        grad = pred_y.mean
        lower, upper = pred_y.confidence_region()

        for i in range(len(plot_x) - 1):
            nowEnegy = 0
            nowVariance = 0

            for j in range(self.DATA_TRAIN_SPLIT):
                id = i * self.DATA_TRAIN_SPLIT + j
                diff_variance = (upper[id] - lower[id]) / 2
                nowEnegy += grad[id]
                nowVariance += diff_variance ** 2

                if (j / self.DATA_TRAIN_SPLIT) < 0.5:
                    variance_y[i] += diff_variance ** 2
                    variance_y_cou[i] += 1
                else:
                    variance_y[i+1] += diff_variance ** 2
                    variance_y_cou[i+1] += 1

            freeEnegy_y[i+1] = freeEnegy_y[i] + nowEnegy.item()
            freeEnegy_diff_y[i+1] = nowEnegy.item()
            variance_sum_y[i+1] = variance_sum_y[i] + nowVariance.item()

        for i in range(len(variance_sum_y)):
            variance_sum_y[i] = math.sqrt(variance_sum_y[i])
            variance_y[i] /= max(1, variance_y_cou[i])
            variance_y[i] = variance_y[i].item()

        variance_upper_y = (np.array(freeEnegy_y) + np.array(variance_sum_y)).tolist()
        variance_lower_y = (np.array(freeEnegy_y) - np.array(variance_sum_y)).tolist()

        f, ax = plt.subplots(1, 1, figsize=(8, 6))
        ax.plot(plot_x, freeEnegy_y, 'b')
        ax.set_ylim([np.array(variance_lower_y).min() - 1, np.array(variance_upper_y).max() + 1])
        ax.fill_between(plot_x, variance_lower_y, variance_upper_y, alpha=0.5)
        ax.legend(['Observed Data', 'Mean', 'Confidence'])
        print("freeEnegy_y is {}".format(freeEnegy_y))
        plt.savefig('simulation/output/freeEnegy_{}.png'.format(setting_data["tryCount"]))

        f, ax = plt.subplots(1, 1, figsize=(8, 6))
        ax.plot(plot_x, freeEnegy_diff_y, 'b')
        ax.set_ylim([np.array(freeEnegy_diff_y).min() - 0.5, np.array(freeEnegy_diff_y).max() + 0.5])
        print("freeEnegy_diff_y is {}".format(freeEnegy_diff_y))
        plt.savefig('simulation/output/diff_{}.png'.format(setting_data["tryCount"]))

        f, ax = plt.subplots(1, 1, figsize=(8, 6))
        ax.plot(plot_x, variance_y, 'b')
        print("variance_y is {}".format(variance_y))
        plt.savefig('simulation/output/variance_{}.png'.format(setting_data["tryCount"]))

        max_variance = 0
        max_id = -1
        for i in range(181):
            if variance_y[i] > max_variance:
                max_id = i
                max_variance = variance_y[i]

        setting_data["nextSearch"] = max_id
        setting_data["free_energy"][setting_data["tryCount"]] = freeEnegy_y
        setting_data["variance"][setting_data["tryCount"]] = variance_y
        print("nextSearch is {}".format(max_id))


def main():
    initialize()
    #これは最初の一回で良い
    simulation_init()

    playCount = 200
    for _ in range(playCount):
        setting_data = read_setting_file()
        print("{} th try".format(setting_data["tryCount"]))
        simulation(setting_data)
        analyze(setting_data)
        setting_data["tryCount"] += 1
        write_setting_file(setting_data)

if __name__ == "__main__":
    main()
