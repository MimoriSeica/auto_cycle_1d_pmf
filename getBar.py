import os
import json
import random
import numpy as np
import math
import sys
import statistics
import re
from matplotlib import pyplot as plt

# グローバル変数
SETTING_FILE_NAME = "setting_file.txt"
BIN_SIZE = 181

def read_setting_file():
    with open(SETTING_FILE_NAME,'r') as f:
        return json.load(f)

"""
サイクルがどこを選択したかをplotするもの。
"""
def main():
    outputFiles = read_setting_file()['outputFiles']
    hist_list = []

    for files in outputFiles:
        hist_list.append(float(files["angle"]))

    f, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.hist(hist_list, bins=180)
    plt.show()

if __name__ == "__main__":
    main()
