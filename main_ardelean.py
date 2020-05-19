import time

import matplotlib.pyplot as plt
import numpy as np

import SBM
import datasets as ds
import scatter_plot


def main():
    # data = ds.getTINSData()
    # data, y = ds.getGenData()
    data, y = ds.get_dataset_simulation_features(simNr=3, align_to_peak=True)

    print(data)

    pn = 25
    start = time.time()
    labels = SBM.parallel(data, pn, ccThreshold=5, version=2)
    end = time.time()
    print('SBM: ' + str(end - start))

    scatter_plot.plot_grid('SBM' + str(len(data)), data, pn, labels, marker='o')
    plt.savefig('./figures/SBMv2_sim79')

    scatter_plot.plot('GT' + str(len(data)), data, y, marker='o')
    plt.savefig('./figures/ground_truth')

    unique, counts = np.unique(labels, return_counts=True)
    print(dict(zip(unique, counts)))

    # plt.show()


main()
