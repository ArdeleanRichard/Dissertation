import csv
import sys
import warnings

import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
from sklearn import metrics
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans

from utils.benchmark.benchmark_data import write_benchmark_labeled_data, print_benchmark_labeled_data, \
    benchmark_algorithm_labeled_data, benchmark_algorithm_unlabeled_data, print_benchmark_unlabeled_data, \
    write_benchmark_unlabeled_data, print_benchmark_extra, benchmark_algorithm_extra
from utils.sbm import SBM
from utils import constants as cs, scatter_plot
from utils.dataset_parsing import datasets as ds
from feature_extraction import feature_extraction as fe

warnings.simplefilter(action='ignore', category=FutureWarning)
sys.setrecursionlimit(100000)


def distribution_filter_features(X, number_of_features):
    return scatter_plot.make_distributions(X, number_of_features)


def accuracy_all_algorithms_on_simulation(simulation_nr, feature_extract_method, dim_reduction_method=None, plot=False,
                                          pe_labeled_data=True, pe_unlabeled_data=True, pe_extra=False,
                                          save_folder="", nr_features=None, weighted=False, distribution_filtering=False, **kwargs):
    # get original data
    X, y = ds.get_dataset_simulation(simulation_nr)

    title_suffix = str(simulation_nr)

    # reduce the feature space
    if feature_extract_method is not None:
        X = fe.apply_feature_extraction_method(X, feature_extract_method, dim_reduction_method, **kwargs)
        title_suffix = title_suffix + "_" + feature_extract_method

    if nr_features is not None:
        nr_features = min(nr_features, X.shape[1])
        X, peaks = distribution_filter_features(X, nr_features)
        if weighted is True:
            X = apply_weights(X, peaks)
            title_suffix = title_suffix + 'features_weighted'

    # apply algorithm(s) and save clustering labels
    labels = [[], [], []]
    for a in range(0, 1):
        labels[a] = apply_algorithm(X, y, a)

    # apply dimensionality reduction for visualization
    if X.shape[1] > 2:
        X = fe.reduce_dimensionality(X, 'pca2d')

    # display ground truth
    if X.shape[1] == 2:
        scatter_plot.plot("Ground truth for Sim" + title_suffix, X, y, marker='o')
        if save_folder != "":
            plt.savefig('figures/' + save_folder + '/' + "sim" + title_suffix + "_0ground" + '.png')
        plt.show()
    elif X.shape[1] == 3:
        fig = px.scatter_3d(X, x=X[:, 0], y=X[:, 1], z=X[:, 2], color=y)
        fig.update_layout(title="Ground truth for Sim" + title_suffix)
        fig.show()

    # plot algorithms labels
    if plot:
        if X.shape[1] == 2:
            for a in range(0, 1):
                scatter_plot.plot(cs.algorithms[a] + " on Sim" + title_suffix, X, labels[a],
                                  marker='o')
                if save_folder != "":
                    plt.savefig('figures/' + save_folder + '/' + "sim" + title_suffix + "_" + cs.algorithms[a] + '.png')
                plt.show()
        elif X.shape[1] == 3:
            for a in range(0, 1):
                fig = px.scatter_3d(X, x=X[:, 0], y=X[:, 1], z=X[:, 2], color=labels[a].astype(str))
                fig.update_layout(title=cs.algorithms[a] + " for Sim" + title_suffix)
                fig.show()

    # performance evaluation
    if pe_labeled_data:
        # print("\nPerformance evaluation - labeled data - " + feature_extract_method)
        pe_labeled_data_results = [[], [], []]
        for a in range(0, 1):
            pe_labeled_data_results[a] = benchmark_algorithm_labeled_data(y, labels[a])
            print_benchmark_labeled_data(simulation_nr, a, pe_labeled_data_results[a])
            write_benchmark_labeled_data(simulation_nr, feature_extract_method, pe_labeled_data_results)

    if pe_unlabeled_data:
        print("\nPerformance evaluation - unlabeled data - " + feature_extract_method)
        pe_unlabeled_data_results = [[], [], []]
        pe_ground_results = benchmark_algorithm_unlabeled_data(X, y)
        for a in range(0, 1):
            pe_unlabeled_data_results[a] = benchmark_algorithm_unlabeled_data(X, labels[a])
            print_benchmark_unlabeled_data(simulation_nr, a, pe_unlabeled_data_results[a], pe_ground_results)
            write_benchmark_unlabeled_data(simulation_nr, feature_extract_method, pe_unlabeled_data_results,
                                           pe_ground_results)
    if pe_extra:
        print("\nPerformance evaluation - extra - " + feature_extract_method)
        pe_extra_results = [[], [], []]
        for a in range(0, 1):
            pe_extra_results[a] = benchmark_algorithm_extra(y, labels[a])
            print_benchmark_extra(simulation_nr, a, pe_extra_results[a])


def apply_algorithm(X, y, alg_number):
    """
        Evaluate the performance of the clustering by using ARI, AMI and Fowlkes_Mallows.
        Specific to labeled data.
        :param y: the ground truth labels
        :param X: the coordinates of the data
        :param alg_number: select the algorithm (see algorithms list in constants.py)
        :param weights: weights of the algorithm
        :returns np.array: the labels as clustered by the algorithm
    """
    if alg_number == 0:
        kmeans = KMeans(n_clusters=np.amax(y) + 1).fit(X)
        labels = kmeans.labels_
    else:
        if alg_number == 1:
            min_samples = np.log(len(X))
            db = DBSCAN(eps=0.02, min_samples=min_samples).fit(X)
            labels = db.labels_
        else:
            labels = SBM.parallel(X, pn=25, version=2)

    return labels


def apply_weights(X, peaks):
    reference = peaks[0]
    for i in range(len(peaks)):
        ratio = peaks[i]/reference
        X[:, i] = X[:, i] * (ratio ** 0.5)
    return X
