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
BIN_SIZE = 181

def read_setting_file():
    with open(SETTING_FILE_NAME,'r') as file:
        return json.load(file)


"""
サイクルのvarianceの遷移をplotする
"""
def main():
    variances = read_setting_file()['variance']
    plot_x = []
    plot_y = []

    for id, variance in variances.items():
        ave = 0
        for i in range(BIN_SIZE):
            ave += variance[i] ** 2

        ave /= BIN_SIZE
        plot_x.append(int(id)+1)
        plot_y.append(ave)
        print(id, ave)
        if int(id) == 90:
            break


    f, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.plot(plot_x[20:], plot_y[20:])
    ax.tick_params(labelsize=12)
    ax.ticklabel_format(style='sci',axis='y',scilimits=(0,0))
    plt.xlabel("number of sampling", fontsize=18)
    plt.ylabel("mean of doubled variance", fontsize=18)
    plt.show()


if __name__ == "__main__":
    main()
