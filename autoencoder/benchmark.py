import time

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score

from utils.dataset_parsing.datasets import stack_simulations_array
from utils.sbm import SBM
from utils.dataset_parsing import datasets as ds
from utils import scatter_plot
from utils.constants import autoencoder_layer_sizes, autoencoder_code_size
from autoencoder.model_auxiliaries import verify_output, get_codes, verify_random_outputs, create_plot_folder
from autoencoder.autoencoder import AutoencoderModel


def validate_model(autoencoder_layers, pt=False):
    range_min = 1
    range_max = 96

    autoencoder = AutoencoderModel(encoder_layer_sizes=autoencoder_layers[:-1],
                                   decoder_layer_sizes=autoencoder_layers[:-1],
                                   code_size=autoencoder_layers[-1])

    encoder, autoenc = autoencoder.return_encoder()

    if pt == True:
        autoencoder.load_weights(f'./autoencoder/weights/autoencoder_allsim_e100_d80_nonoise_c{autoencoder_layers[-1]}_pt')
    else:
        autoencoder.load_weights(f'./autoencoder/weights/autoencoder_allsim_e100_d80_nonoise_c{autoencoder_layers[-1]}')

    results = []
    for simulation_number in range(range_min, range_max):
        if simulation_number == 25 or simulation_number == 44:
            continue
        spikes, labels = ds.get_dataset_simulation(simNr=simulation_number)

        spikes = spikes[labels != 0]
        labels = labels[labels != 0]

        autoencoder_features = get_codes(spikes, encoder)

        pca_2d = PCA(n_components=2)
        autoencoder_features = pca_2d.fit_transform(autoencoder_features)

        pn = 25
        clustering_labels = SBM.parallel(autoencoder_features, pn, ccThreshold=5, version=2)

        results.append([adjusted_rand_score(labels, clustering_labels),
                        adjusted_mutual_info_score(labels, clustering_labels)])

    return results
