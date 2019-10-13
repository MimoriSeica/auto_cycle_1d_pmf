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

class analyze():
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

        f, ax = plt.subplots(1, 1, figsize=(8, 6))
        ax.plot(test_x.numpy(), tmp_test_y)
        ax.set_ylim([-10, 15])
        plt.show()


def main():
    setting_data = read_setting_file()
    analyze(setting_data)

if __name__ == "__main__":
    main()
