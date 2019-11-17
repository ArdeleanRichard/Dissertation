import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn import metrics

import SBM
import SBM_functions as fs
import scatter
import datasets as ds

def main():
    #data = ds.getTINSData()
    data, y = ds.getGenData()
    # data, y = ds.getDatasetSimulation(simNr=79)

    pn = 25
    start = time.time()
    labels = SBM.multiThreaded(data, pn, ccThreshold=5, version=2)
    end = time.time()
    print('SBM: ' + str(end - start))

    scatter.griddedPlotFunction('SBM' + str(len(data)), data, labels, pn, marker='o')
    # plt.show()
    plt.savefig('./figures/SBMv2_gen')

    unique, counts = np.unique(labels, return_counts=True)
    print(dict(zip(unique, counts)))

    # plt.show()

main()

