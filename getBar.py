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
    setting_data = read_setting_file()['outputFiles']
    hist_x = [i for i in range(BIN_SIZE)]
    hist_y = [0 for i in range(BIN_SIZE)]

    for i in range(BIN_SIZE):
        strList = setting_data["{}".format(i)]
        for str in strList:
            num = int(re.search(r'\d+', str).group())
            if num <= 60:
                hist_y[i] += 1

    print(hist_y)
    f, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.bar(hist_x, hist_y)
    #ax.set_ylim([0, 3])
    ax.set_xlabel("angle")
    ax.set_ylabel("search number")
    plt.show()

if __name__ == "__main__":
    main()
