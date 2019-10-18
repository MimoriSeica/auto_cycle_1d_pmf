import os
import json
import random
import numpy as np
import math
import torch
import gpytorch
import sys
import re
import statistics
from matplotlib import pyplot as plt
from scipy.stats import gaussian_kde
from sklearn.linear_model import LinearRegression

# グローバル変数
SETTING_FILE_NAME = "setting_file.txt"
BIN_SIZE = 181

def read_json_file(path):
    with open(path,'r') as f:
        return json.load(f)

def write_json_file(path, dict_data):
    with open(path,'w') as f:
        json.dump(dict_data, f, indent = 4)

def get_wham_data():
    with open("wham_pmf.txt") as file:
        return np.array([str.strip().split() for str in file.readlines()], dtype = 'float')[:, 1]

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

    def cal_Umbrella_Centers_Gaussian(self, x):
        ret_array = [self.gauss(x, angle) for angle in self.umbrella_centers_r]
        return np.array(ret_array)

    def __init__(self, setting_data, limit, wham_data, row_data_dict):

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
                num = int(re.search(r'\d+', filePath).group())
                if num > limit:
                    continue

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
        freeEnegy_y = self.KBT * model.predict(test_xx)
        freeEnegy_y = freeEnegy_y - freeEnegy_y[0]

        row_data_dict["free_enegy"][limit] = freeEnegy_y.tolist()

def main():
    playCount = 100
    wham_data = get_wham_data()
    row_data_dict = {}
    row_data_dict["free_enegy"] = {}
    for count in range(playCount):
        setting_data = read_json_file(SETTING_FILE_NAME)
        print("{} th try".format(count))
        analyze(setting_data, count, wham_data, row_data_dict)

    write_json_file("row_data_liner.json", row_data_dict)

if __name__ == "__main__":
    main()
