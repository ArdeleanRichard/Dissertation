import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn import metrics

import SBM
import SBM_functions as fs
import scatter_plot
import datasets as ds

def main():
    #data = ds.getTINSData()
    #data, y = ds.getGenData()
    data, y = ds.getDatasetSimulationPCA2D(simNr=79, align_to_peak=0)

    pn = 25
    start = time.time()
    labels = SBM.multiThreaded(data, pn, ccThreshold=5, version=2)
    end = time.time()
    print('SBM: ' + str(end - start))

    scatter_plot.griddedPlotFunction('SBM' + str(len(data)), data, pn, labels, marker='o')
    # plt.show()
    plt.savefig('./figures/SBMv2_sim79')

    scatter_plot.griddedPlotFunction('SBM' + str(len(data)), data, pn, y, marker='o')
    # plt.show()
    plt.savefig('./figures/ground_truth')

    unique, counts = np.unique(labels, return_counts=True)
    print(dict(zip(unique, counts)))

    # plt.show()

main()
