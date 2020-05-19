import csv
import functools
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
from PyEMD import EMD, Visualisation
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances_chunked
from sklearn.metrics.cluster.unsupervised import check_number_of_labels
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import check_X_y

import benchmark_data as bd
import constants as cs
import datasets as ds
import libraries.SimpSOM as sps
import libraries.som as som2
import scatter_plot
import spike_features


def get_mutual_info(simulation_nr):
    spikes, labels = ds.get_dataset_simulation(simulation_nr)
    # features = spike_features.get_features(spikes)
    features = spike_features.get_shape_phase_distribution_features(spikes)
    pca_2d = PCA(n_components=2)
    features = pca_2d.fit_transform(features)
    dims = ['fd_max', 'spike_max', 'fd_min']
    loading_scores = pd.Series(pca_2d.components_[0], index=dims)
    sorted_loading_scores = loading_scores.abs().sort_values(ascending=False)
    print(sorted_loading_scores)

    # sbm_labels = SBM.parallel(features, pn=25, version=2)
    # res = dict(zip(["fd_max", "fd_min"],
    #                mutual_info_classif(features, sbm_labels, discrete_features="auto")
    #                ))
    # print(res)
    # print(mutual_info_classif(features, sbm_labels, discrete_features="auto"))


def generate_som(simulation_nr, dim, start_learn_rate, epochs):
    spikes, labels = ds.get_dataset_simulation(simulation_nr)

    filename = 'models/kohonen' + str(simulation_nr) + '_' + str(dim) + 'x' + str(dim) + '_' + str(epochs) + 'e.sav'

    try:
        k_map = pickle.load(open(filename, 'rb'))
    except FileNotFoundError:
        k_map = sps.somNet(dim, dim, spikes, PBC=True, PCI=True)
        k_map.train(startLearnRate=start_learn_rate, epochs=epochs)
        pickle.dump(k_map, open(filename, 'wb'))

    return spikes, labels, k_map


def som_features(simulation_nr, dim, start_learn_rate, epochs):
    spikes, labels, k_map = generate_som(simulation_nr, dim, start_learn_rate, epochs)
    return spikes, labels, k_map, np.array(k_map.project(spikes, show=True, printout=True))


def som_metrics(simulation_nr, dim, start_learn_rate, epochs, show=True):
    spikes, labels, k_map, features = som_features(simulation_nr, dim, start_learn_rate, epochs)

    alg_labels = [[], [], []]
    for alg in range(0, 3):
        alg_labels[alg] = bd.apply_algorithm(features, labels, alg)

    pe_labeled_data_results = [[], [], []]
    for alg in range(0, 3):
        pe_labeled_data_results[alg] = bd.benchmark_algorithm_labeled_data(labels, alg_labels[alg])
        bd.print_benchmark_labeled_data(simulation_nr, alg, pe_labeled_data_results[alg])
        bd.write_benchmark_labeled_data(simulation_nr, 'kohonen', pe_labeled_data_results)

    if show:
        filename_png = 'figures/kohonen' + str(simulation_nr) + '_' + str(dim) + 'x' + str(dim) + '_' + str(epochs) + \
                       'e.png'
        k_map.diff_graph(show=True, printout=True, filename=filename_png)

        scatter_plot.plot("Ground truth for Sim_" + str(simulation_nr), features, labels, marker='o')
        plt.show()
        for a in range(0, 3):
            scatter_plot.plot(cs.algorithms[a] + " on Sim_" + str(simulation_nr), features, alg_labels[a], marker='o')
            plt.show()


def som_err_graph(simulation_nr, dim, start_learn_rate, epochs):
    if __name__ == '__main__':
        spikes, labels = ds.get_dataset_simulation(simulation_nr)

        som = som2.SOM(dim, dim, alpha_start=start_learn_rate)  # initialize the SOM
        som.fit(spikes, epochs, save_e=True,
                interval=100)  # fit the SOM for 10000 epochs, save the error every 100 steps
        filename = 'figures/k_err' + str(simulation_nr) + '_' + str(dim) + 'x' + str(dim) + '_' + str(
            epochs) + 'e.png'
        som.plot_error_history(filename=filename)  # plot the training error history
        return som


# som_metrics(simulation_nr=22, dim=35, start_learn_rate=0.1, epochs=6500)
# som_err_graph(simulation_nr=2, dim=40, start_learn_rate=0.1, epochs=10000)

# print("Mutual Info alg", a, " ", mutual_info_classif(X, labels[a], discrete_features="auto"))
# get_mutual_info(21)

def hilbert_emd(simulation_nr, visualisation=False, plot_spikes=False):
    spikes, labels = ds.get_dataset_simulation(simulation_nr)
    emd = EMD()
    time = np.arange(79)

    for num, spike in enumerate(spikes):
        emd(spike)
        IMFs, res = emd.get_imfs_and_residue()

        if visualisation:
            vis = Visualisation()
            vis.plot_imfs(imfs=IMFs, residue=res, t=time, include_residue=True)
            # vis.plot_instant_freq(IMFs, imfs=IMFs)
            vis.show()

        if plot_spikes:
            fig = plt.figure()
            ax0 = fig.add_subplot()
            ax0.set_title("Sim" + str(simulation_nr) + "_Spike" + str(num))
            ax0.plot(time, spike, label="spike")
            imf_signal = spike - res
            ax0.plot(time, imf_signal, label="no_res")
            # ax0.plot(time, res, label=res)
            # ax0.plot(time, IMFs[0], label="IMF[0]")
            # ax0.plot(time, IMFs[1], label="IMF[1]")
            # ax0.plot(time, IMFs[2], label="IMF[2]")
            ax0.legend()
            # plt.savefig('figures/spikes/sim22/sim' + str(simulation_nr) + "_spike" + str(num) + '.png')
            plt.show()


def accuracy_alex(spikes_arg, labels_arg, plot=False,
                  pe_labeled_data=True, pe_unlabeled_data=True, pe_extra=False,
                  save_folder="", title=""):
    title_suffix = title
    X, y = ds.get_hilbert_features(0, spikes_arg=spikes_arg, labels_arg=labels_arg)
    # feature_reduction='derivativesPCA2D')

    if X.shape[1] == 2:
        scatter_plot.plot("Ground truth for Sim_" + title_suffix, X, y, marker='o')
        if save_folder != "":
            plt.savefig('figures/' + save_folder + '/' + "sim" + title_suffix + "_0ground" + '.png')
        plt.show()
    elif X.shape[1] == 3:
        fig = px.scatter_3d(X, x=X[:, 0], y=X[:, 1], z=X[:, 2], color=y)
        fig.update_layout(title="Ground truth for Sim_" + title_suffix)
        fig.show()

    # apply algorithm(s) and save clustering labels
    labels = [[], [], []]
    for a in range(0, 3):
        labels[a] = bd.apply_algorithm(X, y, a)

    # plot algorithms labels
    if plot:
        if X.shape[1] == 2:
            for a in range(0, 3):
                scatter_plot.plot(cs.algorithms[a] + " on Sim_" + title_suffix, X, labels[a],
                                  marker='o')
                if save_folder != "":
                    plt.savefig('figures/' + save_folder + '/' + "sim" + title_suffix + "_" + cs.algorithms[a] + '.png')
                plt.show()
        elif X.shape[1] == 3:
            for a in range(0, 3):
                fig = px.scatter_3d(X, x=X[:, 0], y=X[:, 1], z=X[:, 2], color=labels[a])
                fig.update_layout(title=cs.algorithms[a] + " for Sim_" + title_suffix)
                fig.show()

    # performance evaluation
    if pe_labeled_data:
        print("\nPerformance evaluation - labeled data")
        pe_labeled_data_results = [[], [], []]
        for a in range(0, 3):
            pe_labeled_data_results[a] = bd.benchmark_algorithm_labeled_data(y, labels[a])
            bd.print_benchmark_labeled_data(0, a, pe_labeled_data_results[a])

    if pe_unlabeled_data:
        print("\nPerformance evaluation - unlabeled data")
        pe_unlabeled_data_results = [[], [], []]
        pe_ground_results = bd.benchmark_algorithm_unlabeled_data(X, y)
        for a in range(0, 3):
            pe_unlabeled_data_results[a] = bd.benchmark_algorithm_unlabeled_data(X, labels[a])
            bd.print_benchmark_unlabeled_data(0, a, pe_unlabeled_data_results[a], pe_ground_results)
    if pe_extra:
        print("\nPerformance evaluation - extra")
        pe_extra_results = [[], [], []]
        for a in range(0, 3):
            pe_extra_results[a] = bd.benchmark_algorithm_extra(y, labels[a])
            bd.print_benchmark_extra(0, a, pe_extra_results[a])


def generate_dataset_from_simulations2(simulations, simulation_labels, save=False, pca=False):
    spikes = []
    labels = []
    index = 1
    for sim_index in np.arange(len(simulations)):
        s, l = ds.get_dataset_simulation(simulations[sim_index], 79, True, False)
        for spike_index in np.arange(len(s)):
            for wanted_label in np.arange(len(simulation_labels[sim_index])):
                if simulation_labels[sim_index][wanted_label] == l[spike_index]:
                    spikes.append(s[spike_index])
                    labels.append(index + wanted_label)
        index = index + len(simulation_labels[sim_index])

    # spikes = np.array(spikes)
    # labels = np.array(labels)

    # ds.get_dataset_simulation_hilbert(0, spikes_arg=spikes, labels_arg=labels)

    if save:
        np.savetxt("spikes.csv", spikes, delimiter=",")
        np.savetxt("labels.csv", labels, delimiter=",")

    if pca:
        spikes_for_pca = np.array(spikes)
        pca_2d = PCA(n_components=2)
        spikes_for_pca = pca_2d.fit_transform(spikes_for_pca)
        scatter_plot.plot("Ground truth for Sim_" + "new sim", spikes_for_pca, labels, marker='o')
        plt.show()

    return spikes, labels


# bd.accuracy_all_algorithms_on_multiple_simulations(1, 20, feature_extract_method=3)
# hilbert_emd(2)

# spikes_, labels_ = generate_dataset_from_simulations2([1, 2, 6, 12, 24, 28, 2, 15, 17],
#                                                       [[10], [7], [6], [15], [2], [8], [13], [8], [2]], pca=True)
# spikes_, labels_ = generate_dataset_from_simulations2([2],[[2,4,5,6,8,9,10,12,14,15,16,18,19]], pca=True)
# accuracy_alex(spikes_, labels_, plot=False, pe_unlabeled_data=False)

# Diana: rosu, albastru, verde
# Alex: negru, galben, aqua
# Andreea: ticlam, mov, orange


def write_cluster_info(sim_nr_left, sim_nr_right):
    results = []
    for sim_nr in range(sim_nr_left, sim_nr_right + 1):
        if sim_nr == 25 or sim_nr == 44:
            continue
        print("Processing sim", sim_nr)
        spikes, labels = ds.get_dataset_simulation(sim_nr)
        for i in range(1 + max(labels)):
            cluster_spikes, cluster_labels = generate_dataset_from_simulations2([sim_nr], [[i]])
            cluster_features = {"sim_nr": sim_nr, "spike_nr": i}
            cluster_features.update(spike_features.describe_cluster(cluster_spikes))
            results.append(cluster_features)
    with open('./results/Sim_%s_%s_features.csv' % (sim_nr_left, sim_nr_right), 'w', newline='') as file:
        writer = csv.DictWriter(file, results[0].keys())
        writer.writeheader()
        writer.writerows(results)


# write_cluster_info(1, 40)
# def test_silhouette_sample(spikes, labels):
#     sil_coeffs = metrics.silhouette_samples(spikes, labels, metric='manhattan')
#     means = []
#     for label in range(max(labels) + 1):
#         means.append(sil_coeffs[labels == label].mean())
#     for i in np.arange(len(means)):
#         print(means[i])


def test_silhouette(sim_nr=0):
    # spikes_arg, labels_arg = generate_dataset_from_simulations2([1, 2, 6, 12, 24, 28, 2, 15, 17],
    #                                    [[10], [7], [6], [15], [2], [8], [13], [8], [2]])
    # spikes, labels = ds.get_dataset_simulation_hilbert(0, spikes_arg=spikes_arg, labels_arg=labels_arg,
    #                                          feature_reduction='derivativesPCA2D')
    # scatter_plot.plot("Test sil", spikes, labels)
    # plt.show()
    # test_silhouette_sample(spikes, labels)
    spikes, labels = ds.get_hilbert_features(sim_nr, feature_reduction='derivativesPCA2D')
    test_silhouette_sample(spikes, labels)


def silhouette_samples2(X, labels, metric='euclidean', **kwds):
    X, labels = check_X_y(X, labels, accept_sparse=['csc', 'csr'])
    le = LabelEncoder()
    labels = le.fit_transform(labels)
    n_samples = len(labels)
    label_freqs = np.bincount(labels)
    check_number_of_labels(len(le.classes_), n_samples)

    kwds['metric'] = metric
    reduce_func = functools.partial(silhouette_reduce2,
                                    labels=labels, label_freqs=label_freqs)
    results = zip(*pairwise_distances_chunked(X, reduce_func=reduce_func,
                                              **kwds))
    intra_clust_dists, inter_clust_dists = results
    intra_clust_dists = np.concatenate(intra_clust_dists)
    inter_clust_dists = np.concatenate(inter_clust_dists)

    denom = (label_freqs - 1).take(labels, mode='clip')
    with np.errstate(divide="ignore", invalid="ignore"):
        intra_clust_dists /= denom

    # sil_samples = inter_clust_dists - intra_clust_dists
    sil_samples = inter_clust_dists
    # with np.errstate(divide="ignore", invalid="ignore"):
    #     sil_samples /= np.maximum(intra_clust_dists, inter_clust_dists)
    # nan values are for clusters of size 1, and should be 0
    return np.nan_to_num(sil_samples)


def silhouette_reduce2(D_chunk, start, labels, label_freqs):
    """Accumulate silhouette statistics for vertical chunk of X

    Parameters
    ----------
    D_chunk : shape (n_chunk_samples, n_samples)
        precomputed distances for a chunk
    start : int
        first index in chunk
    labels : array, shape (n_samples,)
        corresponding cluster labels, encoded as {0, ..., n_clusters-1}
    label_freqs : array
        distribution of cluster labels in ``labels``
    """
    # accumulate distances from each sample to each cluster
    clust_dists = np.zeros((len(D_chunk), len(label_freqs)),
                           dtype=D_chunk.dtype)
    for i in range(len(D_chunk)):
        clust_dists[i] += np.bincount(labels, weights=D_chunk[i],
                                      minlength=len(label_freqs))

    # intra_index selects intra-cluster distances within clust_dists
    intra_index = (np.arange(len(D_chunk)), labels[start:start + len(D_chunk)])
    # intra_clust_dists are averaged over cluster size outside this function
    intra_clust_dists = clust_dists[intra_index]
    # of the remaining distances we normalise and extract the minimum
    clust_dists[intra_index] = np.inf
    clust_dists /= label_freqs
    inter_clust_dists = clust_dists.min(axis=1)
    return intra_clust_dists, inter_clust_dists


def test_silhouette_sample(spikes, labels, metric='euclidean'):
    sil_coeffs = silhouette_samples2(spikes, labels, metric=metric)
    print(metric)
    means = []
    for label in range(max(labels) + 1):
        means.append(sil_coeffs[labels == label].mean())
    for i in np.arange(len(means)):
        print(means[i])


def test_silhouette_on_pca(sim_nr):
    amplitude_pca, labels = ds.get_hilbert_features(sim_nr)
    distances = ['euclidean', 'manhattan', 'mahalanobis', 'sqeuclidean', 'chebyshev', 'minkowski',
                 'braycurtis', 'canberra', 'correlation']
    for i in range(len(distances)):
        test_silhouette_sample(amplitude_pca, labels, metric=distances[i])
        print()


# for i in range(11, 20):
#     print("SIM", i)
#     test_silhouette(i)
#     print()
# test_silhouette()


for i in range(3, 4):
    if i == 24 or i == 25 or i == 44:
        continue
    print("SIM", i)
    test_silhouette_on_pca(i)