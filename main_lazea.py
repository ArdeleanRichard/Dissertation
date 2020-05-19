import functools

import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances_chunked
from sklearn.metrics.cluster.unsupervised import check_number_of_labels
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_X_y

import datasets as ds
import scatter_plot
import superlets as slt


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


def plot_spikes_from_cluster(simnr, spikes, labels, label):
    i = 0
    found = 0
    for spike in spikes:
        if labels[i] == label:
            found = found + 1
            plt.plot(np.arange(len(spikes[i])), spikes[i])
        i = i + 1
        if found == 20:
            break
    plt.title("Cluster_" + str(label) + "_Sim" + str(simnr))
    plt.show()


def purity_score(y_true, y_pred):
    # compute contingency matrix (also called confusion matrix)
    contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
    # return purity
    return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix)


def generate_dataset_from_simulations2(simulations, simulation_labels, save=False):
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

    spikes = np.array(spikes)
    labels = np.array(labels)
    if save:
        np.savetxt("spikes.csv", spikes, delimiter=",")
        np.savetxt("labels.csv", labels, delimiter=",")

    return spikes, labels


def generate_dataset_from_simulations(simulations, simulation_labels, save=False):
    all_spikes = []
    all_labels = []
    for sim_index in np.arange(len(simulations)):
        s, l = ds.get_dataset_simulation(simulations[sim_index], 79, True, False)
        all_spikes.append(np.array(s))
        all_labels.append(np.array(l))
    spikes = []
    labels = []

    for sim_index in np.arange(len(all_spikes)):
        for spike_index in np.arange(len(all_spikes[sim_index])):
            # now all_spikes[sim_index][spike_index] will be the spike
            # check if the spike label is found in the wanted labels
            for wanted_label in np.arange(len(simulation_labels[sim_index])):
                if simulation_labels[sim_index][wanted_label] == all_labels[sim_index][spike_index]:
                    spikes.append(all_spikes[sim_index][spike_index])
                    labels.append(simulation_labels[sim_index][wanted_label])

    spikes = np.array(spikes)
    labels = np.array(labels)
    if save:
        np.savetxt("spikes.csv", spikes, delimiter=",")
        np.savetxt("labels.csv", labels, delimiter=",")

    return spikes, labels


def remove_separated_clusters_old(spikes, labels):
    labels_to_delete = []
    for i in np.arange(max(labels) + 1):
        label_to_keep = i
        labels_new = []
        for j in np.arange(len(labels)):
            if labels[j] == label_to_keep:
                labels_new.append(labels[j])
            else:
                labels_new.append(max(labels) + 2)
        silhouette_algorithm = metrics.silhouette_score(spikes, labels_new)
        # print("Label" + str(i) + " " + str(silhouette_algorithm))
        if silhouette_algorithm >= 0.3:
            labels_to_delete.append(label_to_keep)
            print("DELETED CLUSTER " + str(label_to_keep))
    new_spikes = []
    new_labels = []
    for i in np.arange(len(labels)):
        if labels[i] not in labels_to_delete:
            new_spikes.append(spikes[i])
            new_labels.append(labels[i])
    return np.array(new_spikes), np.array(new_labels)


def remove_separated_clusters(spikes, labels, metric, threshold):
    labels_to_delete = []
    sil_coeffs = silhouette_samples2(spikes, labels, metric=metric)
    means = []
    for label in range(max(labels) + 1):
        means.append(sil_coeffs[labels == label].mean())
    for i in np.arange(len(means)):
        print(means[i])
        if means[i] >= threshold:
            labels_to_delete.append(i)
            # print("DELETED CLUSTER " + str(i))
    new_spikes = []
    new_labels = []
    for i in np.arange(len(labels)):
        if labels[i] not in labels_to_delete:
            new_spikes.append(spikes[i])
            new_labels.append(labels[i])
    return np.array(new_spikes), np.array(new_labels)


def main():
    spikes, labels = generate_dataset_from_simulations2([1, 2, 6, 12, 24, 28, 2, 15, 17],
                                                        [[10], [7], [6], [15], [2], [8], [13], [8], [2]], False)
    result_spikes1 = slt.slt(spikes, 5, 1.8)
    scaler = StandardScaler()
    result_spikes1 = scaler.fit_transform(result_spikes1)
    pca_2d = PCA(n_components=2)
    result_spikes = pca_2d.fit_transform(result_spikes1)
    scatter_plot.plot("Ground truth for Sim_generated", result_spikes, labels, marker='o')
    plt.show()


main()
