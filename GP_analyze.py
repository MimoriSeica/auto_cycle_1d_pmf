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

SETTING_FILE_NAME = "setting_file.txt"
DATA_TRAIN_SPLIT = 5
BIN_SIZE = 181
TEMPARETURE = 300.
KB_KCALPERMOL = 0.0019872041
KBT = KB_KCALPERMOL * TEMPARETURE
SPRING_CONSTANT = 200.0 * ((math.pi/180.0) ** 2)

class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

def biased_energy(x, center):
    dx = np.linalg.norm(x - center)
    dx = dx - (dx // 360.0) * 360.0
    return SPRING_CONSTANT * (dx ** 2)

def cal_delta_PMF(x_from, x_to, center, kde):
    tmp_PMF_to =  KBT * math.log(kde(x_to)) + biased_energy(x_to, center)
    tmp_PMF_from =  KBT * math.log(kde(x_from)) + biased_energy(x_from, center)
    return - (tmp_PMF_to - tmp_PMF_from)

def train(training_times, model, likelihood, optimizer, mll, train_x, train_y):
    cou = []
    losses = []
    for i in range(training_times):
        optimizer.zero_grad()
        output = model(train_x)
        loss = -mll(output, train_y)
        loss.backward()
        print('Iter %d/%d - Loss: %.5f' % (i + 1,
                                           training_times,
                                           loss.item()))
        cou.append(i)
        losses.append(loss.item())
        optimizer.step()

def read_setting_file():
    with open(SETTING_FILE_NAME,'r') as f:
        return json.load(f)

def add_train_data(from_x, train_x, train_y, plot_dict, angle, now_kde):
    now_x = [from_x]
    now_y = cal_delta_PMF(from_x, from_x + (1.0/DATA_TRAIN_SPLIT), angle, now_kde)
    train_x.append(now_x)
    train_y.append(now_y)
    if plot_dict != None:
        plot_dict[from_x].append(now_y)

def wham_analyze():
    with open("wham_from_cycle_data.txt") as file:
        return np.array([str.strip().split() for str in file.readlines()], dtype = 'float')[:, 1]

class analyze_class():
    def gauss(self, x, mean):
        _x = np.linalg.norm(np.array(x) - np.array(mean))
        return math.exp(-((_x ** 2) / (2 * (self.SIGMA_POW_2))))

    def biased_energy(self, x, center):
        dx = np.linalg.norm(x - center)
        dx = dx - (dx // 360.0) * 360.0
        return self.SPRING_CONSTANT * (dx ** 2)

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
        for now_dict in setting_data["outputFiles"]:
            filePath = now_dict["filename"]
            angle = float(now_dict["angle"])
            with open(filePath) as file:
                rowData = np.array([str.strip().split() for str in file.readlines()], dtype = 'float')[:, 1]
                now_kde = gaussian_kde(rowData.T)
                data_stdev = stdev(rowData)
                data_mean = mean(rowData)

                for x in pred_x:
                    if x[0] < data_mean - data_stdev or data_mean + data_stdev - (1.0/self.DATA_TRAIN_SPLIT) < x[0]:
                        continue

                    now_x = [x[0]]
                    now_y = self.cal_delta_PMF(x[0], x[0] + (1.0/self.DATA_TRAIN_SPLIT), angle, now_kde)
                    train_x.append(now_x)
                    train_y.append(now_y)

        f, ax = plt.subplots(1, 1, figsize=(8, 6))
        ax.scatter(train_x, train_y, c='blue', alpha=0.3)
        plt.show()

        train_x = torch.Tensor(np.array(train_x))
        train_y = torch.Tensor(np.array(train_y))
        likelihood = gpytorch.likelihoods.GaussianLikelihood(
        noise_constraint=gpytorch.constraints.Positive()
        )
        model = ExactGPModel(train_x, train_y, likelihood)

        model.train()
        likelihood.train()

        optimizer = torch.optim.Adam([
            {'params': model.parameters()},  # Includes GaussianLikelihood parameters
        ], lr=0.1)

        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
        self.train(110, model, likelihood, optimizer, mll, train_x, train_y)

        model.eval()
        likelihood.eval()

        plot_x = range(BIN_SIZE)
        max_variance = 0
        max_id = -1
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

                if max_variance < diff_variance:
                    max_variance = diff_variance
                    max_id = id / self.DATA_TRAIN_SPLIT

            freeEnegy_y[i+1] = freeEnegy_y[i] + nowEnegy.item()
            freeEnegy_diff_y[i+1] = nowEnegy.item()
            variance_sum_y[i+1] = variance_sum_y[i] + nowVariance.item()

        for i in range(len(variance_sum_y)):
            variance_sum_y[i] = math.sqrt(variance_sum_y[i])
            variance_y[i] /= max(1, variance_y_cou[i])
            variance_y[i] = variance_y[i].item()

        variance_upper_y = (np.array(freeEnegy_y) + np.array(variance_sum_y)).tolist()
        variance_lower_y = (np.array(freeEnegy_y) - np.array(variance_sum_y)).tolist()

        print(variance_y)


def analyze():
    pred_x = []
    for i in range(BIN_SIZE-1):
        for j in range(DATA_TRAIN_SPLIT):
            pred_x.append([i+j*(1.0/DATA_TRAIN_SPLIT)])

    plot_dict = {}
    for x in pred_x:
        plot_dict[x[0]] = []

    train_x = []
    train_y = []
    setting_data = read_setting_file()
    for now_dict in setting_data["outputFiles"]:
        filePath = now_dict["filename"]
        angle = float(now_dict["angle"])
        with open(filePath) as file:
            rowData = np.array([str.strip().split() for str in file.readlines()], dtype = 'float')[:, 1]
            now_kde = gaussian_kde(rowData.T)
            data_stdev = stdev(rowData)
            data_mean = mean(rowData)

            for x in pred_x:
                if not (x[0] < data_mean - (data_stdev) or data_mean + (data_stdev) - (1.0/DATA_TRAIN_SPLIT) < x[0]):
                    add_train_data(x[0], train_x, train_y, plot_dict, angle, now_kde)

    f, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.scatter(train_x, train_y, c='blue', alpha=0.3)
    plt.show()

    train_x = torch.Tensor(np.array(train_x))
    train_y = torch.Tensor(np.array(train_y))
    likelihood = gpytorch.likelihoods.GaussianLikelihood(
    noise_constraint=gpytorch.constraints.Positive()
    )
    model = ExactGPModel(train_x, train_y, likelihood)

    model.train()
    likelihood.train()

    optimizer = torch.optim.Adam([
        {'params': model.parameters()},  # Includes GaussianLikelihood parameters
    ], lr=0.1)

    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    train(110, model, likelihood, optimizer, mll, train_x, train_y)

    model.eval()
    likelihood.eval()

    plot_x = range(BIN_SIZE)
    freeEnegy_y = [0 for i in range(BIN_SIZE)]
    variance_y = [0 for i in range(BIN_SIZE)]
    variance_y_cou = [0 for i in range(BIN_SIZE)]
    variance_sum_y = [0 for i in range(BIN_SIZE)]

    pred_y = likelihood(model(torch.Tensor(pred_x)))
    grad = pred_y.mean
    lower, upper = pred_y.confidence_region()

    for i in range(len(plot_x) - 1):
        nowEnegy = 0
        nowVariance = 0

        for j in range(DATA_TRAIN_SPLIT):
            id = i * DATA_TRAIN_SPLIT + j
            diff_variance = (upper[id] - lower[id]) / 2
            nowEnegy += grad[id]
            nowVariance += diff_variance ** 2

            if (j / DATA_TRAIN_SPLIT) < 0.5:
                variance_y[i] += diff_variance ** 2
                variance_y_cou[i] += 1
            else:
                variance_y[i+1] += diff_variance ** 2
                variance_y_cou[i+1] += 1

        freeEnegy_y[i+1] = freeEnegy_y[i] + nowEnegy.item()
        variance_sum_y[i+1] = variance_sum_y[i] + nowVariance.item()

    for i in range(len(variance_sum_y)):
        variance_sum_y[i] = math.sqrt(variance_sum_y[i])
        variance_y[i] /= max(1, variance_y_cou[i])
        variance_y[i] =  variance_y[i].item()

    variance_upper_y = (np.array(freeEnegy_y) + np.array(variance_sum_y))
    variance_lower_y = (np.array(freeEnegy_y) - np.array(variance_sum_y))

    wham_y = wham_analyze()
    f, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.plot(plot_x, wham_y, )
    ax.plot(plot_x, freeEnegy_y, color="red")
    ax.legend(['WHAM','Gaussian process'], fontsize=16)
    plt.xlabel("angle[degree]", fontsize=18)
    plt.ylabel("potential of mean force[kcal / mol]", fontsize=18)
    plt.show()

    f, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.plot(plot_x, variance_y)
    plt.xlabel("angle[degree]", fontsize=18)
    plt.ylabel("variance", fontsize=18)
    plt.show()
    print(variance_y)


if __name__ == "__main__":
    # analyze()
    setting_data = read_setting_file()
    analyze_class(setting_data)
