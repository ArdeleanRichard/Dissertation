import numpy as np
import matplotlib.pyplot as plt
import time


import SBM
import SBM_functions as fs
import scatter
import datasets as ds

def main():
    data, y = ds.getGenData()

    pn = 25
    start = time.time()
    labels = SBM.multiThreaded(data, pn, ccThreshold=5)
    end = time.time()
    print('SBM: ' + str(end - start))

    scatter.griddedPlotFunction('SBM' + str(len(data)), data, labels, pn, marker='o')
    # plt.show()
    plt.savefig('./figures/SBM')

    unique, counts = np.unique(labels, return_counts=True)
    print(dict(zip(unique, counts)))

    # plt.show()

main()

