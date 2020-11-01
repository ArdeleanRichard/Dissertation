from scipy import signal
from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from collections import Counter
import pandas as pd

import datasets as ds
import scatter_plot
import derivatives as deriv
import superlets as slt


def generate_dataset_from_simulations(simulations, simulation_labels, save=False):
    spikes = []
    labels = []
    index = 0
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


def remove_separated_clusters(features, spikes, labels, metrics_list, threshold_list):
    labels_to_delete = []
    for i in np.arange(len(metrics_list)):
        print("Applying metric " + metrics_list[i] + " with threshold " + str(threshold_list[i]))
        sil_coeffs = metrics.silhouette_samples(features, labels, metric=metrics_list[i])
        means = []
        for label in np.arange(max(labels) + 1):
            if label not in labels:
                means.append(-1)
            else:
                means.append(sil_coeffs[labels == label].mean())
        for j in np.arange(len(means)):
            if means[j] != -1:
                print(means[j])
                if means[j] >= threshold_list[i]:
                    labels_to_delete.append(j)
                    print("Deleted cluster " + str(j))
    new_spikes = []
    new_labels = []
    new_features = []
    for i in np.arange(len(labels)):
        if labels[i] not in labels_to_delete:
            new_spikes.append(spikes[i])
            new_labels.append(labels[i])
            new_features.append(features[i])

    return np.array(new_spikes), np.array(new_features), np.array(new_labels)


def pipeline(simnr, spikes, labels, methods):
    stop = False
    features = []
    new_labels = []
    new_spikes = []
    j = 0
    while len(labels) > 0 and stop == False:
        changed = False
        for i in np.arange(len(methods)):
            if methods[i] == 'hil':
                print("Pipeline step " + str(j) + " applying hil")

                spikes_hilbert = ds.hilbert(spikes)
                envelope = np.abs(spikes_hilbert)
                features = ds.reduce_dimensionality(envelope, 'derivativesPCA2D')
                new_spikes, new_features, new_labels = remove_separated_clusters(features, spikes, labels,
                                                                                 ['mahalanobis'],
                                                                                 [0.65])
                if len(Counter(new_labels).keys()) >= 2:
                    new_spikes, new_features, new_labels = remove_separated_clusters(new_features, new_spikes,
                                                                                     new_labels,
                                                                                     ['canberra'],
                                                                                     [0.65])
            if methods[i] == 'stft':
                print("Pipeline step " + str(j) + " applying stft")
                sampled_frequencies, time_segments, Zxx = signal.stft(spikes, window='blackman', fs=1,
                                                                      nperseg=45)
                amplitude = np.abs(Zxx)
                amplitude = np.apply_along_axis(deriv.compute_fdmethod_1spike, 2, amplitude)
                amplitude = amplitude.reshape(*amplitude.shape[:1], -1)
                pca_2d = PCA(n_components=2)
                features = pca_2d.fit_transform(amplitude)
                new_spikes, new_features, new_labels = remove_separated_clusters(features, spikes, labels,
                                                                                 ['mahalanobis'],
                                                                                 [0.52])
            if methods[i] == 'slt':
                print("Pipeline step " + str(j) + " applying slt")
                result_spikes1 = slt.slt(spikes, 5, 1.8)
                scaler = StandardScaler()
                result_spikes1 = scaler.fit_transform(result_spikes1)
                pca_2d = PCA(n_components=2)
                features = pca_2d.fit_transform(result_spikes1)
                new_spikes, new_features, new_labels = remove_separated_clusters(features, spikes, labels,
                                                                                 ['euclidean'],
                                                                                 [0.70])
            scatter_plot.plot("GT sim%d step %d using %s" % (simnr, j, methods[i]), X=features, labels=labels,
                              marker='o')
            plt.savefig("./figures/pipeline/sim%d_step_%d_using_%s" % (simnr, j, methods[i]))
            plt.show()
            if len(labels) != len(new_labels):
                changed = True
            labels = new_labels
            spikes = new_spikes
            j = j + 1
            if len(Counter(labels).keys()) < 2:
                changed = False
                break
        stop = not changed


def wrapper_pipeline():
    spikes, labels = generate_dataset_from_simulations([22], [[0, 1, 2, 3, 4, 5, 6]], save=False)
    pipeline(22, spikes, labels, ['stft', 'hil', 'slt'])


def plot_threshold_all():
    distances = ['euclidean', 'manhattan', 'mahalanobis', 'seuclidean', 'sqeuclidean', 'chebyshev', 'minkowski',
                 'braycurtis', 'canberra', 'correlation']
    for i in range(len(distances)):
        metric = distances[i]
        data = pd.read_csv('./results/d3_STFT_%s.csv' % metric, delimiter='\n')
        # print(data)
        data = data.to_numpy()
        # print(data)
        nr_clusters = np.array(0)
        for i in range(0, 101):
            th = float(i / 100)
            print(th)
            eliminate = data[data >= th]
            print(eliminate.size)
            nr_clusters = np.insert(nr_clusters, i, eliminate.size)
        print(nr_clusters)
        nr_clusters = nr_clusters / 30
        print(nr_clusters.size)

        plt.plot(np.arange(102), nr_clusters, label="%s" % metric)
    # plt.title("Silhouette th with %s metric on STFTd sim 1-30" % metric)
    plt.title("Silhouette th on STFTd 3d sim 1-30")
    plt.xlabel('threshold * 100')
    plt.ylabel('average number of clusters')
    plt.legend(loc="upper right")
    plt.savefig('./figures/pipeline/d3_STFT_th_all')
    plt.show()


def plot_threshold(metric):
    data = pd.read_csv('./results/d3_STFT_%s.csv' % metric, delimiter='\n')
    data = data.to_numpy()
    nr_clusters = np.array(0)
    for i in range(0, 101):
        th = float(i / 100)
        print(th)
        eliminate = data[data >= th]
        print(eliminate.size)
        nr_clusters = np.insert(nr_clusters, i, eliminate.size)
    print(nr_clusters)
    nr_clusters = nr_clusters / 30
    print(nr_clusters.size)

    plt.plot(np.arange(102), nr_clusters, label="%s" % metric)
    plt.title("Silhouette %s th on STFTd 3d sim 1-30" % metric)
    plt.xlabel('threshold * 100')
    plt.ylabel('average number of clusters')
    plt.savefig('./figures/pipeline/d3_STFT_th_%s' % metric)
    plt.show()


def individual_th_plots_wrapper():
    distances = ['euclidean', 'manhattan', 'mahalanobis', 'seuclidean', 'sqeuclidean', 'chebyshev', 'minkowski',
                 'braycurtis', 'canberra', 'correlation']
    for i in range(len(distances)):
        plot_threshold(distances[i])


plot_threshold_all()
individual_th_plots_wrapper()
# wrapper_pipeline()
