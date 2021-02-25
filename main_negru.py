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


def get_mutual_info(simulation_nr):
    spikes, labels = ds.get_dataset_simulation(simulation_nr)
    # features = spike_features.get_features(spikes)
    features = shape_features.get_shape_phase_distribution_features(spikes)
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
    features = np.array(k_map.project(spikes, show=True, printout=True))
    return spikes, labels, k_map, features


def som_metrics(simulation_nr, dim, start_learn_rate, epochs, show=True):
    spikes, labels, k_map, features = som_features(simulation_nr, dim, start_learn_rate, epochs)
    print("ala")
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


def get_features_shape_phase_distribution(spikes):
    pca_2d = PCA(n_components=2)

    features = shape_features.get_shape_phase_distribution_features(spikes)
    features = pca_2d.fit_transform(features)
    print("Variance Ratio = ", np.sum(pca_2d.explained_variance_ratio_))

    return features


def write_cluster_info(sim_nr_left, sim_nr_right):
    results = []
    for sim_nr in range(sim_nr_left, sim_nr_right + 1):
        if sim_nr == 25 or sim_nr == 44:
            continue
        print("Processing sim", sim_nr)
        spikes, labels = ds.get_dataset_simulation(sim_nr)
        for i in range(1 + max(labels)):
            cluster_spikes, cluster_labels = pipeline.generate_dataset_from_simulations2([sim_nr], [[i]])
            cluster_features = {"sim_nr": sim_nr, "spike_nr": i}
            cluster_features.update(shape_features.describe_cluster(cluster_spikes))
            results.append(cluster_features)
    with open('./results/Sim_%s_%s_features.csv' % (sim_nr_left, sim_nr_right), 'w', newline='') as file:
        writer = csv.DictWriter(file, results[0].keys())
        writer.writeheader()
        writer.writerows(results)


# def test_silhouette_sample(spikes, labels):
#     sil_coeffs = metrics.silhouette_samples(spikes, labels, metric='manhattan')
#     means = []
#     for label in range(max(labels) + 1):
#         means.append(sil_coeffs[labels == label].mean())
#     for i in np.arange(len(means)):
#         print(means[i])


def get_dataset_simulation_emd_quartiles(sim_nr, spike_length=79, align_to_peak=True, normalize_spike=False,
                                         spikes_arg=None, labels_arg=None):
    if sim_nr == 0:
        spikes = np.array(spikes_arg)
        labels = np.array(labels_arg)
    else:
        spikes, labels = ds.get_dataset_simulation(sim_nr, spike_length, align_to_peak, normalize_spike)
    emd = EMD()

    features = np.zeros((spikes.shape[0], 12))
    for i, spike in enumerate(spikes):
        emd(spike)
        IMFs, res = emd.get_imfs_and_residue()
        hb = np.abs(hilbert(spike))
        ffts = fft.fft(IMFs)
        freq = fft.fftfreq(len(ffts[0]))
        t = np.arange(79)

        # Only a name
        IMFs = np.abs(ffts)
        # f = np.array(deriv.compute_fdmethod(IMFs))

        f1 = np.array([np.percentile(IMFs[0], 25), np.percentile(IMFs[0], 50), np.percentile(IMFs[0], 75)])
        f2 = np.array([np.percentile(IMFs[1], 25), np.percentile(IMFs[1], 50), np.percentile(IMFs[1], 75)])
        f3 = np.array([0, 0, 0])
        f4 = np.array([0, 0, 0])
        if IMFs.shape[0] >= 3:
            f3 = np.array([np.percentile(IMFs[2], 25), np.percentile(IMFs[2], 50), np.percentile(IMFs[2], 75)])
        if IMFs.shape[0] >= 4:
            f4 = np.array([np.percentile(IMFs[3], 25), np.percentile(IMFs[3], 50), np.percentile(IMFs[3], 75)])

        # print(np.concatenate((np.array([f1, f2, f3, f4]))))
        features[i] = np.concatenate((np.array([f1, f2, f3, f4])))
        # print(freq)
        # plt.plot(freq, fft1.real, freq, fft1.imag)
        # plt.show()
        # plt.clf()
        # plt.plot(freq, fft2.real, freq, fft2.imag)
        # plt.show()

    features = fe.reduce_dimensionality(features, method='PCA2D')
    return features, labels


spikes_per_channel = np.array([11837, 2509, 1443, 2491, 18190, 9396, 876, 9484, 9947, 10558, 2095, 3046, 898,
                               1284, 5580, 6409, 9274, 13625, 419, 193, 3220, 2128, 281, 219, 4111, 1108, 5045, 6476,
                               973, 908,
                               787, 10734])


def read_timestamps():
    with open('./datasets/real_data/Waveforms/M017_S001_SRCS3L_25,50,100_0004_5stdv.spiket', 'rb') as file:
        timestamps = []
        read_val = file.read(4)
        timestamps.append(struct.unpack('i', read_val)[0])

        while read_val:
            read_val = file.read(4)
            try:
                timestamps.append(struct.unpack('i', read_val)[0])
            except struct.error:
                break
        timestamps = np.array(timestamps)

        return np.array(timestamps)


def read_waveforms(filename):
    with open(filename, 'rb') as file:
        waveforms = []
        read_val = file.read(4)
        waveforms.append(struct.unpack('f', read_val)[0])

        while read_val:
            read_val = file.read(4)
            try:
                waveforms.append(struct.unpack('f', read_val)[0])
            except struct.error:
                break

        return np.array(waveforms)


def extract_spikes(timestamps, waveform, channel):
    left_limit = np.sum(spikes_per_channel[:channel])
    right_limit = left_limit + spikes_per_channel[channel]
    timestamps = timestamps[left_limit:right_limit]
    # waveform = waveform[channel * 36297600: (channel + 1) * 36297600]

    print(waveform.shape)

    spikes = np.zeros((spikes_per_channel[channel], 58))
    print(spikes.shape)
    for index in range(len(timestamps)):
        # print('index', index)
        # print(timestamps[index])
        # print(timestamps[index] + 58)
        # print(waveform.shape)
        # print()
        spikes[index] = waveform[timestamps[index]: timestamps[index] + 58]
    print(spikes.shape)
    # print(spikes[-2].shape)

    peak_ind = np.argmin(spikes, axis=1)
    # avg_peak = np.floor(np.mean(peak_ind))
    timestamps = timestamps - (19 - peak_ind)
    timestamps = timestamps.astype(int)

    spikes = []
    for i in range(len(timestamps)):
        spikes.append(waveform[timestamps[i]:timestamps[i] + 58])
        plt.plot(np.arange(58), -spikes[i])
    plt.show()


def extract_spikes3(waveform, channel):
    left_limit = np.sum(spikes_per_channel[:channel])
    right_limit = left_limit + spikes_per_channel[channel]

    spikes = []
    for i in range(0, len(waveform), 58):
        spikes.append(waveform[i:i + 58])
    print(len(spikes))
    for i in range(0, len(spikes), 1000):
        plt.plot(np.arange(58), -spikes[i])
    plt.show()

    return spikes[left_limit:right_limit]


units_per_channel = [
    [],
    [1629, 474, 5951, 255],
    [],
    [],
    [686],
    [15231, 1386, 678],
    [1269, 1192, 3362, 2263, 192],
    [79],
    [684, 2053, 3125],
    [4313, 160, 123, 2582, 211],
    [1303, 6933, 1298],
    [],
    [285],
    [],
    [],
    [2658, 1489, 461],
    [1742, 150, 277],
    [5845, 542],
    [8762, 886, 699],
    [],
    [],
    [],
    [252],
    [],
    [],
    [1480, 745, 203],
    [],
    [2397, 512],
    [658, 1328, 138],
    [],
    [],
    [],
    [5899, 239],
]


def sum_until_channel(channel):
    ch_sum = 0
    for i in units_per_channel[:channel]:
        ch_sum += np.sum(np.array(i)).astype(int)

    return ch_sum


def get_spike_units(waveform, channel, plot_spikes=False):
    spikes = []
    new_spikes = np.zeros((np.sum(units_per_channel[channel]), 58))

    for i in range(0, len(waveform), 58):
        spikes.append(waveform[i: i + 58])

    left_limit_spikes = sum_until_channel(channel)
    right_limit_spikes = left_limit_spikes + np.sum(np.array(units_per_channel[channel]))
    print(left_limit_spikes, right_limit_spikes)
    spikes = spikes[left_limit_spikes: right_limit_spikes]

    if plot_spikes:
        for i in range(0, len(spikes), 1000):
            plt.plot(np.arange(58), -spikes[i])
        plt.show()

    labels = np.array([])
    for i, units in enumerate(units_per_channel[channel]):
        labels = np.append(labels, np.repeat(i, units))

    for i, units in enumerate(units_per_channel[channel]):
        left_lim = sum_until_channel(channel)
        right_lim = left_lim + units

        spike_index = 0
        for j in range(len(spikes)):
            # print(j, left_lim, right_lim)
            new_spikes[spike_index] = spikes[j]
            spike_index += 1
        # scaler = StandardScaler()
        # spikes = scaler.fit_transform(spikes)
    return new_spikes, labels.astype(int)


def real_dataset(channel, feature_extraction_method, dim_reduction_method):
    # waveforms_ = read_waveforms('./datasets/real_data/Waveforms/M017_S001_SRCS3L_25,50,100_0004_5stdv.spikew')
    waveforms_ = read_waveforms('./datasets/real_data/Units/M017_0004_5stdv.ssduw')
    spikes, labels = get_spike_units(waveforms_, channel=channel)

    spikes2d = fe.apply_feature_extraction_method(spikes, feature_extraction_method, dim_reduction_method)
    scatter_plot.plot_clusters(spikes2d, labels,
                               'channel' + str(channel) + '_' + feature_extraction_method + '_' + dim_reduction_method,
                               save_folder='real_data')
    plt.show()


# for i in range(1, 32):
#     try:
#         real_dataset(i, 'shape', 'pca2d')
#     except TypeError:
#         print('Error at ', i)
# write_cluster_info(1, 79)

def run_sim(sim_nr):
    bd.accuracy_all_algorithms_on_simulation(simulation_nr=sim_nr,
                                             feature_extract_method=None,
                                             # dim_reduction_method='',
                                             # dim_reduction_method='pca2d',
                                             plot=True,
                                             pe_labeled_data=True,
                                             pe_unlabeled_data=False,
                                             pe_extra=False,
                                             weighted=True,
                                             nr_features=45
                                             # save_folder='kohonen',

                                             # som_dim=[20, 20],
                                             # som_epochs=1000,
                                             # title='sim' + str(sim_nr),
                                             # extra_plot=True,
                                             )
    bd.accuracy_all_algorithms_on_simulation(simulation_nr=sim_nr,
                                             feature_extract_method=None,
                                             # dim_reduction_method='',
                                             # dim_reduction_method='pca2d',
                                             plot=True,
                                             pe_labeled_data=True,
                                             pe_unlabeled_data=False,
                                             pe_extra=False,
                                             weighted=False,
                                             nr_features=45
                                             # save_folder='kohonen',

                                             # som_dim=[20, 20],
                                             # som_epochs=1000,
                                             # title='sim' + str(sim_nr),
                                             # extra_plot=True,
                                             )


def run_pipeline():
    spikes, labels = ds.get_dataset_simulation(64)
    pipeline.pipeline(spikes, labels, [
        ['hilbert', 'derivatives_pca2d', 'mahalanobis', 0.65],
        ['stft_d', 'PCA2D', 'mahalanobis', 0.65],
        # ['superlets', 'PCA2D', 'euclidean', 0.65],
    ])


# run_sim(64)
run_sim(5)
# run_pipeline()

# bd.accuracy_all_algorithms_on_multiple_simulations(1, 3, feature_extract_method='hilbert',
#                                                    reduce_dimensionality_method='derivatives_pca2d')
