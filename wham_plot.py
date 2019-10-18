from matplotlib import pyplot as plt
import numpy as np

with open("wham_pmf.txt") as file:
    thisArray = np.array([str.strip().split() for str in file.readlines()], dtype = 'float')

    f, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.plot(thisArray[:, 0], thisArray[:, 1])
    ax.set_ylim([-10, 15])
    ax.set_xlabel("angle")
    ax.set_ylabel("search number")
    plt.show()
