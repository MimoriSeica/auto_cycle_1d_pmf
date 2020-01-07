import os
import json
import random
import numpy as np
import math
import torch
import gpytorch
import sys
import statistics
from matplotlib import pyplot as plt
from scipy.stats import gaussian_kde
from sklearn.linear_model import LinearRegression

# グローバル変数
SETTING_FILE_NAME = "setting_file.txt"
BIN_SIZE = 181

def read_setting_file():
    with open(SETTING_FILE_NAME,'r') as f:
        return json.load(f)

class linear_analyze():
    def gauss(self, x, mean):
        _x = np.linalg.norm(np.array(x) - np.array(mean))
        return math.exp(-((_x ** 2) / (2 * (self.SIGMA_POW_2))))

    def biased_energy(self, x, center):
        dx = np.linalg.norm(x - center)
        dx = dx - (math.trunc(dx / 360.0)) * 360.0
        return (self.SPRING_CONSTANT / self.KBT) * (dx ** 2)

    def cal_Umbrella_Centers_Gaussian(self, x):
        ret_array = [self.gauss(x, angle) for angle in self.umbrella_centers_r]
        return np.array(ret_array)

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
        self.DATA_TRAIN_SPLIT = 3
        self.umbrella_centers_r = range(181)

        train_x = []
        train_y = []
        for angle in range(BIN_SIZE):
            for filePath in setting_data["outputFiles"]["{}".format(angle)]:
                with open(filePath) as file:
                    rowData = np.array([str.strip().split() for str in file.readlines()], dtype = 'float')[:, 1]
                    now_kde = gaussian_kde(rowData.T)

                    plot_x = np.linspace(angle - 5, angle + 5, 200)
                    plot_y = now_kde(plot_x)

                    for x in rowData:
                        now_x = self.cal_Umbrella_Centers_Gaussian(x) - self.cal_Umbrella_Centers_Gaussian(angle)
                        now_y = self.cal_delta_PMF(angle, x, angle, now_kde)

                        train_x.append(now_x)
                        train_y.append(now_y)

        test_x = torch.linspace(0, 180, 181)
        test_xx = []
        for i in test_x:
            test_xx.append(self.cal_Umbrella_Centers_Gaussian(i))

        model = LinearRegression()

        model.fit(train_x, train_y)
        tmp_test_y = self.KBT * model.predict(test_xx)
        tmp_test_y = tmp_test_y - tmp_test_y[0]
        self.y = tmp_test_y

    def get_y(self):
        return self.y

class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

class gauss_analyze():
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
        self.DATA_TRAIN_SPLIT = 3

        train_x = []
        train_y = []
        for angle in range(BIN_SIZE):
            for filePath in setting_data["outputFiles"]["{}".format(angle)]:
                with open(filePath) as file:
                    rowData = np.array([str.strip().split() for str in file.readlines()], dtype = 'float')[:, 1]
                    now_kde = gaussian_kde(rowData.T)

                    tmp_x = np.linspace(rowData.min(), rowData.max(), 70)
                    for i in range(len(tmp_x) - 1):
                        now_x = [tmp_x[i], tmp_x[i+1]]
                        now_y = self.cal_delta_PMF(tmp_x[i], tmp_x[i+1], angle, now_kde)

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
        self.train(40, model, likelihood, optimizer, mll, train_x, train_y)

        model.eval()
        likelihood.eval()

        plot_x = range(BIN_SIZE)
        freeEnegy_y = [0 for i in range(BIN_SIZE)]
        freeEnegy_diff_y = [0 for i in range(BIN_SIZE)]
        variance_y = [0 for i in range(BIN_SIZE)]
        variance_y_cou = [0 for i in range(BIN_SIZE)]
        variance_sum_y = [0 for i in range(BIN_SIZE)]

        pred_x = []
        for i in range(BIN_SIZE-1):
            for j in range(self.DATA_TRAIN_SPLIT):
                pred_x.append([i+j*(1.0/self.DATA_TRAIN_SPLIT), i+(j+1)*(1.0/self.DATA_TRAIN_SPLIT)])

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

        self.y = freeEnegy_y

    def get_y(self):
        return self.y

def wham_analyze():
    with open("wham_from_cycle_data.txt") as file:
        return np.array([str.strip().split() for str in file.readlines()], dtype = 'float')[:, 1]

def main():
    setting_data = read_setting_file()
    linear_y = linear_analyze(setting_data)
    gauss_y = gauss_analyze(setting_data)
    wham_y = wham_analyze()

    plot_x = range(BIN_SIZE)
    f, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.plot(plot_x, linear_y.get_y(), color="red")
    ax.plot(plot_x, gauss_y.get_y(), color="orange")
    ax.plot(plot_x, wham_y)
    plt.show()


if __name__ == "__main__":
    main()
