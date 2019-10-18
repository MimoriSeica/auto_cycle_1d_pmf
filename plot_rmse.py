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

# グローバル変数
BIN_SIZE = 181

def read_json_file(path):
    with open(path,'r') as f:
        return json.load(f)

def get_wham_data():
    with open("wham_from_cycle_data.txt") as file:
        return np.array([str.strip().split() for str in file.readlines()], dtype = 'float')[:, 1]

def main():
    row_datas_gp = read_json_file("row_data.json")
    wham_data = get_wham_data()
    rmse_list_gp = []

    for count in range(100):
        free_enegy = row_datas_gp["free_enegy"]["{}".format(count)]
        sum = 0;

        for i in range(BIN_SIZE):
            sum += (free_enegy[i] - wham_data[i]) ** 2

        rmse_list_gp.append(sum / BIN_SIZE)

    row_datas_liner = read_json_file("row_data_liner.json")
    rmse_list_liner = []

    for count in range(100):
        free_enegy = row_datas_liner["free_enegy"]["{}".format(count)]
        sum = 0;

        for i in range(BIN_SIZE):
            sum += (free_enegy[i] - wham_data[i]) ** 2

        rmse_list_liner.append(sum / BIN_SIZE)

    f, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.plot(range(len(rmse_list_gp)), rmse_list_gp, 'b')
    ax.plot(range(len(rmse_list_liner)), rmse_list_liner, color = "red")
    ax.plot(range(len(rmse_list_gp)), [0.1909531092466414 for i in range(len(rmse_list_gp))], color = "orange")
    ax.plot(range(len(rmse_list_gp)), [2.6104119312491014 for i in range(len(rmse_list_gp))])
    ax.set_ylim([0, 10])
    ax.set_xlabel("cycle count")
    ax.set_ylabel("RMSE")
    plt.show()

if __name__ == "__main__":
    main()
