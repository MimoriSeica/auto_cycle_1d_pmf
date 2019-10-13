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

def main():
    rmse_list = read_json_file("rmse_list.json")["rmse_list"]

    f, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.plot(range(len(rmse_list)), rmse_list, 'b')
    ax.plot(range(len(rmse_list)), [1.210058043189025 for i in range(len(rmse_list))], 'b', color = "red")
    ax.set_ylim([0, 10])
    ax.set_xlabel("cycle count")
    ax.set_ylabel("RMSE")
    plt.show()

if __name__ == "__main__":
    main()
