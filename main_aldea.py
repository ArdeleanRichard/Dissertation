import csv
import pickle
import struct

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PyEMD import EMD
from scipy import fft
from scipy.fftpack import hilbert
from sklearn.decomposition import PCA

from licenta.roxi import benchmark_data as bd
from utils import constants as cs, scatter_plot
from utils.dataset_parsing import datasets as ds
import libraries.SimpSOM as sps
import libraries.som as som2
from pipeline import pipeline
from feature_extraction import shape_features, feature_extraction as fe
from utils.feature_extraction_methods import feature_extraction_method


def run_sim(sim_nr):
    bd.accuracy_all_algorithms_on_simulation(simulation_nr=sim_nr,
                                             feature_extract_method=None,
                                             # dim_reduction_method='pca2d',
                                             # dim_reduction_method='derivatives2d',
                                             plot=True,
                                             pe_labeled_data=True,
                                             pe_unlabeled_data=False,
                                             pe_extra=False,
                                             nr_features=10,
                                             weighted=False
                                             # save_folder='kohonen',
                                             # som_dim=[20, 20],
                                             # som_epochs=1000,
                                             # title='sim' + str(sim_nr),
                                             # extra_plot=True,
                                             )


if __name__ == "__main__":
    run_sim(4)
