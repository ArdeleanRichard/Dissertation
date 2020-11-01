import csv
import struct

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
import plotly.express as px

from utils.benchmark import benchmark_data as bd
from utils import constants as cs, scatter_plot as sp
from utils.datasets import datasets as ds
from utils.constants import LABEL_COLOR_MAP
from feature_extraction import derivatives
from feature_extraction import feature_extraction as fe


def gui():
    bd.accuracy_all_algorithms_on_simulation(simulation_nr=4,
                                             feature_extract_method='stft',
                                             dim_reduction_method='pca2d',
                                             plot=True,
                                             pe_labeled_data=True,
                                             pe_unlabeled_data=True,
                                             pe_extra=False,
                                             save_folder='demo',
                                             )


def plot_all_ground_truths():
    pca_2d = PCA(n_components=2)
    for sim_nr in range(95, 96):
        spikes, labels = ds.get_dataset_simulation(sim_nr, spike_length=79, align_to_peak=2, normalize_spike=False)
        clusters = max(labels)
        signal_pca = pca_2d.fit_transform(spikes)
        sp.plot(title="GT with PCA Sim_%d (%d clusters)" % (sim_nr, clusters), X=signal_pca, labels=labels, marker='o')
        plt.savefig('./figures/sim_%d_c%d' % (sim_nr, clusters))
        # plt.show()
        # print(max(labels))


def spikes_per_cluster(sim_nr):
    sim_nr = sim_nr
    spikes, labels = ds.get_dataset_simulation(sim_nr, spike_length=79, align_to_peak=2, normalize_spike=False)
    print(spikes.shape)

    pca2d = PCA(n_components=2)

    for i in range(np.amax(labels) + 1):
        spikes_by_color = spikes[labels == i]
        print(len(spikes_by_color))
        sp.plot_spikes(spikes_by_color, "Sim_%d_Cluster_%d" % (sim_nr, i))
        cluster_pca = pca2d.fit_transform(spikes_by_color)
        # sp.plot(title="GT with PCA Sim_%d" % sim_nr, X=cluster_pca, marker='o')
        plt.scatter(cluster_pca[:, 0], cluster_pca[:, 1], c=LABEL_COLOR_MAP[i], marker='o', edgecolors='k')
        plt.title("Cluster %d Sim_%d" % (i, sim_nr))
        plt.savefig('figures/spikes_on_cluster/Sim_%d_Cluster_%d_color' % (sim_nr, i))
        plt.show()
        # print(cluster_pca)


def all_spikes():
    for sim_nr in range(77, 96):
        if sim_nr != 25 and sim_nr != 44:
            spikes_per_cluster(sim_nr)


def csf_db():
    pca_2d = PCA(n_components=2)
    alg_labels = [[], [], []]
    pe_labeled_data_results = [[], [], []]

    header_labeled_data = ['Simulation', 'Clusters', 'Algorithm', 'Index', 'Value']
    # with open('./results/PCA_2d_DBD.csv', 'w', newline='') as file:
    #     writer = csv.writer(file, delimiter=',')
    #     writer.writerows(header_labeled_data)

    for sim_nr in range(95, 96):
        spikes, labels = ds.get_dataset_simulation(sim_nr, spike_length=79, align_to_peak=2, normalize_spike=False)
        signal_pca = pca_2d.fit_transform(spikes)

        for alg_nr in range(0, 3):
            alg_labels[alg_nr] = bd.apply_algorithm(signal_pca, labels, alg_nr)
            pe_labeled_data_results[alg_nr] = bd.benchmark_algorithm_labeled_data(labels, alg_labels[alg_nr])

        formatted_kmeans = ["%.3f" % number for number in pe_labeled_data_results[0]]
        formatted_dbscan = ["%.3f" % number for number in pe_labeled_data_results[1]]
        formatted_sbm = ["%.3f" % number for number in pe_labeled_data_results[2]]
        row1 = [sim_nr, max(labels), 'K-means', "ari_all", formatted_kmeans[0]]
        row2 = [sim_nr, max(labels), 'K-means', "ami_all", formatted_kmeans[1]]
        row3 = [sim_nr, max(labels), 'K-means', "ari_nnp", formatted_kmeans[2]]
        row4 = [sim_nr, max(labels), 'K-means', "ami_nnp", formatted_kmeans[3]]
        row5 = [sim_nr, max(labels), 'SBM', "ari_all", formatted_sbm[0]]
        row6 = [sim_nr, max(labels), 'SBM', "ami_all", formatted_sbm[1]]
        row7 = [sim_nr, max(labels), 'SBM', "ari_nnp", formatted_sbm[2]]
        row8 = [sim_nr, max(labels), 'SBM', "ami_nnp", formatted_sbm[3]]
        row9 = [sim_nr, max(labels), 'DBSCAN', "ari_all", formatted_dbscan[0]]
        row10 = [sim_nr, max(labels), 'DBSCAN', "ami_all", formatted_dbscan[1]]
        row11 = [sim_nr, max(labels), 'DBSCAN', "ari_nnp", formatted_dbscan[2]]
        row12 = [sim_nr, max(labels), 'DBSCAN', "ami_nnp", formatted_dbscan[3]]
        row_list = [row1, row2, row3, row4, row5, row6, row7, row8, row9, row10, row11, row12]
        with open('./results/PCA_2d_DBD.csv', 'a+', newline='') as file:
            writer = csv.writer(file, delimiter=',')
            writer.writerows(row_list)


def save_all_pca2d():
    pca_2d = PCA(n_components=2)

    for alg_nr in range(2, 3):
        average = [0, 0, 0, 0, 0]
        # average = [0, 0, 0, 0, 0, 0]
        simulation_counter = 0
        for sim_nr in range(1, 96):
            if sim_nr != 25 and sim_nr != 27 and sim_nr != 44:
                spikes, labels = ds.get_dataset_simulation(sim_nr, spike_length=79, align_to_peak=2,
                                                           normalize_spike=False)
                signal_pca = pca_2d.fit_transform(spikes)
                # signal_pca = derivatives.compute_fdmethod(spikes)
                alg_labels = bd.apply_algorithm(signal_pca, labels, alg_nr)
                results = bd.benchmark_algorithm_labeled_data(labels, alg_labels)
                # results = bd.benchmark_algorithm_extra(alg_labels, labels)
                simulation_counter += 1
                average += results
                formatted = ["%.3f" % number for number in results]
                row = [sim_nr, formatted[0], formatted[1], formatted[2], formatted[3], formatted[4]]
                # row = [sim_nr, formatted[0], formatted[1], formatted[2], formatted[3], formatted[4], formatted[5]]
                with open('./results/all_%s_pca3d.csv' % cs.algorithms[alg_nr], 'a+', newline='') as file:
                    writer = csv.writer(file, delimiter=',')
                    writer.writerow(row)
        average = average / simulation_counter
        with open('./results/all_%s_pca3d.csv' % cs.algorithms[alg_nr], 'a+', newline='') as file:
            writer = csv.writer(file, delimiter=',')
            writer.writerow(average)


def sim_details():
    for sim_nr in range(45, 96):
        spikes, labels = ds.get_dataset_simulation(sim_nr, spike_length=79, align_to_peak=2, normalize_spike=False)
        row = [sim_nr, max(labels), spikes.shape[0]]
        with open('./results/simulaton_details.csv', 'a+', newline='') as file:
            writer = csv.writer(file, delimiter=',')
            writer.writerow(row)


def plot_single_spike():
    sim_nr = 15
    spikes, labels = ds.get_dataset_simulation(sim_nr, spike_length=79, align_to_peak=2, normalize_spike=False)
    spike = spikes[0]
    print(spike)
    plt.plot(np.arange(79), spike)
    plt.show()
    for i in range(1, 79):
        print('f(%d,%d)=1' % (i, spike[i]))


spikes_per_channel = np.array([0, 11837, 2509, 1443, 2491, 18190, 9396, 876, 9484, 9947, 10558, 2095, 3046, 898,
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
    # timestamps = timestamps[1:51]
    # waveform = waveform[channel * 36297600: (channel + 1) * 36297600]

    print(waveform.shape)

    spikes = np.zeros((spikes_per_channel[channel], 58))
    # spikes = np.zeros((50, 58))
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


def extract_spikes2(timestamps, waveform, channel):
    left_limit = np.sum(spikes_per_channel[:channel])
    right_limit = left_limit + spikes_per_channel[channel]
    waveform_channel = waveform[left_limit:right_limit]
    print(left_limit)
    print(right_limit)

    spikes = []
    spike_index = 0
    for i in range(int(left_limit), int(right_limit), 58):
        print(len(waveform[i:i + 58]))
        spikes.append(waveform_channel[i:i + 58])
        print(spike_index)
        spike_index += 1
    # plt.show()

    spikes = []
    for i in range(0, spikes_per_channel[channel]):
        spikes.append(waveform_channel[i * 58:(i + 1) * 58])

    for i in range(0, 10):
        plt.plot(np.arange(58), -spikes[i])
    plt.show()


def extract_spikes3(waveform):
    spikes = []
    for i in range(0, len(waveform), 58):
        spikes.append(waveform[i:i + 58])
    print(len(spikes))
    for i in range(0, len(spikes), 1000):
        plt.plot(np.arange(58), -spikes[i])
    plt.show()

    pca2d = PCA(n_components=2)
    # X = pca2d.fit_transform(spikes[0:int(np.floor(spikes_per_channel[2]/58))])
    X = pca2d.fit_transform(spikes[0:11837])
    plt.scatter(X[:, 0], X[:, 1], marker='o', edgecolors='k')
    plt.show()

    pca3d = PCA(n_components=3)
    X = pca3d.fit_transform(spikes[0:11837])
    fig = px.scatter_3d(X, x=X[:, 0], y=X[:, 1], z=X[:, 2])
    fig.update_layout(title="Ground truth for channel 1")
    fig.show()

    der_spikes = derivatives.compute_fdmethod(spikes[0:11837])
    plt.scatter(der_spikes[:, 0], der_spikes[:, 1], marker='o', edgecolors='k')
    plt.show()


units_per_channel_5 = [
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

units_per_channel = [
    [],
    [1773, 483, 2282],
    [2149, 280],
    [],
    [993, 2828],
    [32565, 200],
    [1061, 1362, 135, 1102],
    [],
    [2085, 3056, 692],
    [145, 349, 220],
    [1564],
    [9537],
    [14264],
    [4561],
    [6926],
    [1859, 439, 1359],
    [309, 1877],
    [1379, 242],
    [2739],
    [],
    [],
    [],
    [],
    [],
    [],
    [1149],
    [201, 244],
    [],
    [109, 209],
    [413],
    [377],
    [421],
    [276, 19014],
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
    return new_spikes, labels.astype(int)


def fe_per_channels():
    # waveforms_ = read_waveforms('./datasets/real_data/M017_004_5stdv/Units/M017_0004_5stdv.ssduw')
    # channel_list = [1, 4, 5, 6, 7, 8, 9, 10, 12, 15, 16, 17, 18, 22, 25, 27, 28, 32]
    waveforms_ = read_waveforms('./datasets/real_data/M017_004_3stdv/Units/M017_S001_SRCS3L_25,50,100_0004.ssduw')
    channel_list = [1, 2, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 25, 26, 28, 29, 30, 31, 32]
    for ch in range(5, 6):
        spikes, labels = get_spike_units(waveforms_, channel=channel_list[ch])
        spikes2d = fe.apply_feature_extraction_method(spikes, 'pca3d')
        sp.plot_clusters(spikes2d, labels, 'pca3d channel_%d' % channel_list[ch], 'real_data')
        plt.show()


# fe_per_channels()

gui()
# plot_all_ground_truths()
# spikes_per_cluster(2)
# all_spikes()
# csf_db()
# sim_details()
# plot_single_spike()
# save_all_pca2d()
