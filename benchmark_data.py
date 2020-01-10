import csv
import sys
import warnings
import numpy as np
from sklearn import metrics
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import plotly.express as px

import SBM
import constants as cs
import datasets as ds
import scatter_plot

warnings.simplefilter(action='ignore', category=FutureWarning)
sys.setrecursionlimit(100000)


def apply_algorithm(X, y, alg_number):
    """
        Evaluate the performance of the clustering by using ARI, AMI and Fowlkes_Mallows.
        Specific to labeled data.
        :param y: the ground truth labels
        :param X: the coordinates of the data
        :param alg_number: select the algorithm (see algorithms list in constants.py)
        :returns np.array: the labels as clustered by the algorithm
    """
    if alg_number == 0:
        kmeans = KMeans(n_clusters=np.amax(y) + 1).fit(X)
        labels = kmeans.labels_
    else:
        if alg_number == 1:
            min_samples = np.log(len(X))
            db = DBSCAN(eps=0.1, min_samples=min_samples).fit(X)
            labels = db.labels_
        else:
            labels = SBM.parallel(X, pn=25, version=2)

    return labels


def benchmark_algorithm_labeled_data(y, labels):
    """
        Evaluate the performance of the clustering by using ARI, AMI and Fowlkes_Mallows.
        Specific to labeled data.
        :param labels: the result of the algorithm
        :param x: the coordinates of the data
        :returns np.array: list - the result of the performance evaluation
    """
    all_ari = metrics.adjusted_rand_score(y, labels)
    all_ami = metrics.adjusted_mutual_info_score(y, labels)
    all_fmi = metrics.fowlkes_mallows_score(y, labels)

    adj = labels > 0
    y_nn = y[adj]
    labels_nn = labels[adj]
    nnp_ari = metrics.adjusted_rand_score(y_nn, labels_nn)
    nnp_ami = metrics.adjusted_mutual_info_score(y_nn, labels_nn)

    return np.array([all_ari, all_ami, nnp_ari, nnp_ami, all_fmi])


def print_benchmark_labeled_data(sim_nr, algorithm_number, pe_results):
    """
        Print the results of benchmarking for labeled data in the console
        :param algorithm_number: integer - number of the algorithm (0 = K-Means, 1=DBSCAN, 2=SBM)
        :returns None
    """
    print(
        "Sim" + str(sim_nr) + " - " + cs.algorithms[algorithm_number] + " - " + 'ARI-ALL: {: .3f}'.format(
            pe_results[0]))
    print(
        "Sim" + str(sim_nr) + " - " + cs.algorithms[algorithm_number] + " - " + 'AMI-ALL: {: .3f}'.format(
            pe_results[1]))
    print(
        "Sim" + str(sim_nr) + " - " + cs.algorithms[algorithm_number] + " - " + 'ARI-NNP: {: .3f}'.format(
            pe_results[3]))
    print(
        "Sim" + str(sim_nr) + " - " + cs.algorithms[algorithm_number] + " - " + 'AMI-NNP: {: .3f}'.format(
            pe_results[4]))
    print(
        "Sim" + str(sim_nr) + " - " + cs.algorithms[algorithm_number] + " - " + 'FMI: {: .3f}'.format(
            pe_results[2]))


def write_benchmark_labeled_data(simulation_number, feature_extr_method, pe_values):
    """
        Write to csv file (Sim_X_labeled_nameOfExtractionMethod) the results of benchmarking for labeled data
        :returns None
    """
    formatted_kmeans = ["%.3f" % number for number in pe_values[0]]
    formatted_kmeans.insert(0, "K-means")
    formatted_dbscan = ["%.3f" % number for number in pe_values[1]]
    formatted_dbscan.insert(0, "DBSCAN")
    formatted_sbm = ["%.3f" % number for number in pe_values[2]]
    formatted_sbm.insert(0, "S.B.M.")

    header_labeled_data = ['Algor', 'ARI-a', 'AMI-a', 'ARI-n', 'AMI-n', 'FMI-a']
    row_list = [header_labeled_data, formatted_kmeans, formatted_dbscan, formatted_sbm]
    with open('./results/Sim_%s_labeled_%s.csv' % (simulation_number, feature_extr_method), 'w', newline='') as file:
        writer = csv.writer(file, delimiter='\t')
        writer.writerows(row_list)


def benchmark_algorithm_unlabeled_data(x, labels):
    """
        Evaluate the performance of the clustering by using Silhouette, Calinski_Harabasz and Davies_Bouldin.
        Specific to unlabeled data.
        :param labels: the result of the algorithm
        :param x: the coordinates of the data
        :returns np.array: list - the result of the performance evaluation
    """
    silhouette_algorithm = metrics.silhouette_score(x, labels)
    cbi_algorithm = metrics.calinski_harabasz_score(x, labels)
    dbi_algorithm = metrics.davies_bouldin_score(x, labels)

    return np.array([silhouette_algorithm, dbi_algorithm, cbi_algorithm])


def print_benchmark_unlabeled_data(sim_nr, algorithm_number, pe_results, ground):
    """
        Print the results of benchmarking for unlabeled data in the console
        :param algorithm_number: integer - number of the algorithm (0 = K-Means, 1=DBSCAN, 2=SBM)
        :returns None
    """
    print(
        "Sim" + str(sim_nr) + " - " + cs.algorithms[algorithm_number] + " - " + 'Sil: {: .3f}'.format(
            pe_results[0]) + " vs " + '{: .3f}'.format(ground[0]))
    print(
        "Sim" + str(sim_nr) + " - " + cs.algorithms[algorithm_number] + " - " + 'DBI: {: .3f}'.format(
            pe_results[1]) + " vs " + '{: .3f}'.format(ground[1]))
    print(
        "Sim" + str(sim_nr) + " - " + cs.algorithms[algorithm_number] + " - " + 'CHI: {: .3f}'.format(
            pe_results[2]) + " vs " + '{: .3f}'.format(ground[2]))


def write_benchmark_unlabeled_data(simulation_number, feature_extr_method, pe_values, ground):
    """
        Write to csv file (Sim_X_unlabeled_nameOfExtractionMethod) the results of benchmarking for unlabeled data
        :returns None
    """
    formatted_kmeans = ["%.3f" % number for number in pe_values[0]]
    formatted_kmeans.insert(0, "K-means")
    formatted_dbscan = ["%.3f" % number for number in pe_values[1]]
    formatted_dbscan.insert(0, "DBSCAN")
    formatted_sbm = ["%.3f" % number for number in pe_values[2]]
    formatted_sbm.insert(0, "S.B.M.")
    formatted_ground = ["%.3f" % number for number in ground]
    formatted_ground.insert(0, "Ground")

    # Silhouette, Davies-Bouldin, Calinski-Harabasz
    # g - ground truth, a - algorithm
    header_labeled_data = ['Algor', 'Sil-a', 'DBI-a', 'CHI-a']
    row_list = [header_labeled_data, formatted_kmeans, formatted_dbscan, formatted_sbm, formatted_ground]
    with open('./results/Sim_%s_unlabeled_%s.csv' % (simulation_number, feature_extr_method), 'w', newline='') as file:
        writer = csv.writer(file, delimiter='\t')
        writer.writerows(row_list)


def benchmark_algorithm_extra(labels, y):
    """
        Evaluate the performance of the clustering by using Homogenity, Completeness and V-score.
        Specific to labeled data.
        :param labels: the result of the algorithm
        :param y: the ground truth
        :returns np.array: list - the result of the performance evaluation
    """
    # ALL settings points
    all_homogeneity = metrics.homogeneity_score(y, labels)
    all_completeness = metrics.completeness_score(y, labels)
    all_v_score = metrics.v_measure_score(y, labels)

    # NNP settings
    adj = labels > 0
    y_nn = y[adj]
    labels_nn = labels[adj]

    nnp_homogeneity = metrics.homogeneity_score(y_nn, labels_nn)
    nnp_completeness = metrics.completeness_score(y_nn, labels_nn)
    nnp_v_score = metrics.v_measure_score(y_nn, labels_nn)

    return np.array([all_homogeneity, all_completeness, all_v_score, nnp_homogeneity, nnp_completeness, nnp_v_score])


def print_benchmark_extra(sim_nr, algorithm_number, pe_results):
    """
        Print the results of extra benchmarking for labeled data in the console
        :param algorithm_number: integer - number of the algorithm (0 = K-Means, 1=DBSCAN, 2=SBM)
        :returns None
    """
    print('\nALL SETTING')
    print("Sim" + str(sim_nr) + " - " + cs.algorithms[algorithm_number] + " - " + 'Hom: {: .3f}'.format(pe_results[0]))
    print("Sim" + str(sim_nr) + " - " + cs.algorithms[algorithm_number] + " - " + 'Com: {: .3f}'.format(pe_results[1]))
    print("Sim" + str(sim_nr) + " - " + cs.algorithms[algorithm_number] + " - " + 'V-s: {: .3f}'.format(pe_results[2]))
    print('NNP SETTING')
    print("Sim" + str(sim_nr) + " - " + cs.algorithms[algorithm_number] + " - " + 'Hom: {: .3f}'.format(pe_results[3]))
    print("Sim" + str(sim_nr) + " - " + cs.algorithms[algorithm_number] + " - " + 'Com: {: .3f}'.format(pe_results[4]))
    print("Sim" + str(sim_nr) + " - " + cs.algorithms[algorithm_number] + " - " + 'V-s: {: .3f}'.format(pe_results[5]))


def accuracy_all_algorithms_on_simulation(simulation_nr, feature_extract_method, plot=False,
                                          pe_labeled_data=True, pe_unlabeled_data=True, pe_extra=False):
    # get data
    X, y = ds.apply_feature_extraction_method(simulation_nr, feature_extract_method)
    if cs.feature_space_dimensions[feature_extract_method] == 2:
        scatter_plot.plot("Ground truth for Sim_" + str(simulation_nr), X, y, marker='o')
        plt.show()
    else:
        fig = px.scatter_3d(X, x=X[:, 0], y=X[:, 1], z=X[:, 2], color=y)
        fig.update_layout(title="Ground truth for Sim_" + str(simulation_nr))
        fig.show()

    # apply algorithm(s) and save clustering labels
    labels = [[], [], []]
    for a in range(0, 3):
        labels[a] = apply_algorithm(X, y, a)

    # plot algorithms labels
    if plot:
        if cs.feature_space_dimensions[feature_extract_method] == 2:
            for a in range(0, 3):
                scatter_plot.plot(cs.algorithms[a] + " on Sim_" + str(simulation_nr), X, labels[a], marker='o')
                plt.show()
        else:
            for a in range(0, 3):
                fig = px.scatter_3d(X, x=X[:, 0], y=X[:, 1], z=X[:, 2], color=labels[a])
                fig.update_layout(title=cs.algorithms[a] + " for Sim_" + str(simulation_nr))
                fig.show()

    # performance evaluation
    if pe_labeled_data:
        print("\nPerformance evaluation - labeled data - " + cs.feature_extraction_methods[feature_extract_method])
        pe_labeled_data_results = [[], [], []]
        for a in range(0, 3):
            pe_labeled_data_results[a] = benchmark_algorithm_labeled_data(y, labels[a])
            print_benchmark_labeled_data(simulation_nr, a, pe_labeled_data_results[a])
            write_benchmark_labeled_data(simulation_nr, cs.feature_extraction_methods[feature_extract_method],
                                         pe_labeled_data_results)

    if pe_unlabeled_data:
        print("\nPerformance evaluation - unlabeled data - " + cs.feature_extraction_methods[feature_extract_method])
        pe_unlabeled_data_results = [[], [], []]
        pe_ground_results = benchmark_algorithm_unlabeled_data(X, y)
        for a in range(0, 3):
            pe_unlabeled_data_results[a] = benchmark_algorithm_unlabeled_data(X, labels[a])
            print_benchmark_unlabeled_data(simulation_nr, a, pe_unlabeled_data_results[a], pe_ground_results)
            write_benchmark_unlabeled_data(simulation_nr, cs.feature_extraction_methods[feature_extract_method],
                                           pe_unlabeled_data_results, pe_ground_results)
    if pe_extra:
        print("\nPerformance evaluation - extra - " + cs.feature_extraction_methods[feature_extract_method])
        pe_extra_results = [[], [], []]
        for a in range(0, 3):
            pe_extra_results[a] = benchmark_algorithm_extra(y, labels[a])
            print_benchmark_extra(simulation_nr, a, pe_extra_results[a])
