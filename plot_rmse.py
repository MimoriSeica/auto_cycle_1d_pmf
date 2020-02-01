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
SETTING_FILE_NAME = "setting_file.txt"
WHAM_DATA_FILE = "wham_from_cycle_data.txt"
BIN_SIZE = 181

def read_setting_file():
    with open(SETTING_FILE_NAME,'r') as file:
        return json.load(file)

def load_wham_data():
    with open(WHAM_DATA_FILE,'r') as file:
        return np.array([str.strip().split() for str in file.readlines()], dtype = 'float')[:, 1]

"""
サイクルのRMSEの遷移をplotする
"""
def main():
    free_energys = read_setting_file()['free_energy']
    wham_data = load_wham_data()
    plot_x = []
    plot_y = []

    for id, free_energy in free_energys.items():
        rmse = 0
        for i in range(BIN_SIZE):
            rmse += (wham_data[i] - free_energy[i]) ** 2

        rmse /= BIN_SIZE
        plot_x.append(id)
        plot_y.append(rmse)
        print(id, rmse)

    f, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.plot(plot_x[15:], plot_y[15:])
    plt.show()


if __name__ == "__main__":
    main()
