import time

import matplotlib.pyplot as plt
import numpy as np

from utils.sbm import SBM
from utils.dataset_parsing import datasets as ds
from utils import scatter_plot
from autoencoder.model_train import create_autoencoder_model, verify_output, get_codes


def main(program):
    # data = ds.getTINSData()
    # data, y = ds.getGenData()

    if program == "sbm":
        # data = ds.getTINSData()
        # data, y = ds.getGenData()
        data, y = ds.get_dataset_simulation_pca_2d(simNr=79, align_to_peak=True)

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

    elif program == "autoencoder":
        simulation_number = 79
        spikes, labels = ds.get_dataset_simulation(simNr=simulation_number)
        print(spikes.shape)

        encoder, decoder = create_autoencoder_model(spikes)
        verify_output(spikes, encoder, decoder)
        autoencoder_features = get_codes(spikes, encoder)

        scatter_plot.plot('GT' + str(len(autoencoder_features)), autoencoder_features, labels, marker='o')
        plt.savefig(f'./figures/autoencoder/ground_truth_sim{simulation_number}')

main("autoencoder")
