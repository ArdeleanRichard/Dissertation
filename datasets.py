import matplotlib.pyplot as plt
import numpy as np
import numpy.fft as fft
import pandas as pd
from PyEMD import EMD
from peakdetect import peakdetect
from scipy.io import loadmat
from scipy.signal import hilbert
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import derivatives as deriv
import discretewlt as dwt
import scatter_plot
import shape_features
import superlets as slt
import wavelets as wlt

dataName = ["S1", "S2", "U", "UO", "Simulation"]
files = ["s1_labeled.csv", "s2_labeled.csv", "unbalance.csv"]


def spike_extract(signal, spike_start, spike_length):
    """
    Extract the spikes from the signal knowing where the spikes start and their length
    :param signal: matrix - height for each point of the spikes
    :param spike_start: vector - each entry represents the first point of a spike
    :param spike_length: integer - constant, 79

    :returns spikes: matrix - each row contains 79 points of one spike
    """
    spikes = np.zeros([len(spike_start), spike_length])

    for i in range(len(spike_start)):
        spikes[i, :] = signal[spike_start[i]: spike_start[i] + spike_length]

    return spikes


def spike_preprocess(signal, spike_start, spike_length, align_to_peak, normalize_spikes, spike_label):
    spikes = spike_extract(signal, spike_start, spike_length)

    # align to max
    if align_to_peak:
        # peak_ind is a vector that contains the index (0->78 / 79 points for each spike) of the maximum of each spike
        peak_ind = np.argmax(spikes, axis=1)
        # avg_peak is the avg of all the peaks
        avg_peak = np.floor(np.mean(peak_ind))
        # spike_start is reinitialized so that the spikes are aligned
        spike_start = spike_start - (avg_peak - peak_ind)
        spike_start = spike_start.astype(int)
        # the spikes are re-extracted using the new spike_start
        spikes = spike_extract(signal, spike_start, spike_length)

    # normalize spikes using Z-score: (value - mean)/ standard deviation
    if normalize_spikes:
        normalized_spikes = [(spike - np.mean(spike)) / np.std(spike) for spike in spikes]
        return normalized_spikes
    return spikes


def get_dataset_simulation_features(simNr, spike_length=79, align_to_peak=True, normalize_spike=False):
    """
    Load the dataset with 2 chosen features (amplitude and distance between min peaks)
    :param simNr: integer - the number of the wanted simulation
    :param spike_length: integer - length of spikes in number of samples
    :param align_to_peak: integer - aligns each spike to it's maximum value
    :param normalize_spike: boolean - applies z-scoring normalization to each spike

    :returns spikes_features: matrix - the 2-dimensional points resulted
    :returns labels: vector - the vector of labels for each point
    """
    spikes, labels = get_dataset_simulation(simNr, spike_length, align_to_peak, normalize_spike)
    spikes_features = np.empty((len(spikes), 2))

    for i in range(len(spikes)):
        # print(i)
        max_peaks, min_peaks = peakdetect(spikes[i], range(spike_length), lookahead=1)
        # print(max_peaks)
        # print(min_peaks)
        max_peaks = np.array(max_peaks)

        amplitude_information = np.argmax(max_peaks[:, 1])
        amplitude_position = max_peaks[amplitude_information][0]
        spike_amplitude = max_peaks[amplitude_information][1]

        spike_distance = 0

        if amplitude_position < min_peaks[0][0]:
            spike_distance = min_peaks[0][0] - 0
        else:
            for j in range(0, len(min_peaks)):
                if j + 1 >= len(min_peaks):
                    spike_distance = 79 - min_peaks[j][0]
                    # plt.figure()
                    # plt.plot(spikes[i])
                    # plt.savefig(f"./figures/FirstSpike{i}")
                    break
                else:
                    if min_peaks[j][0] < amplitude_position < min_peaks[j + 1][0]:
                        spike_distance = min_peaks[j + 1][0] - min_peaks[j][0]
                        break

        spikes_features[i] = [spike_amplitude, spike_distance]

        # if spike_amplitude < 0.5:
        #     plt.figure()
        #     plt.plot(spikes[i])
        #     plt.savefig(f"./figures/Noise{i},{spike_distance}")

    return spikes_features, labels


def get_dataset_simulation_pca_2d(simNr, spike_length=79, align_to_peak=True, normalize_spike=False):
    """
    Load the dataset after PCA on 2 dimensions
    :param simNr: integer - the number of the wanted simulation
    :param spike_length: integer - length of spikes in number of samples
    :param align_to_peak: integer - aligns each spike to it's maximum value
    :param normalize_spike: boolean - applies z-scoring normalization to each spike

    :returns spikes_pca_2d: matrix - the 2-dimensional points resulted
    :returns labels: vector - the vector of labels for each point
    """
    spikes, labels = get_dataset_simulation(simNr, spike_length, align_to_peak, normalize_spike)
    # apply pca
    pca_2d = PCA(n_components=2)
    spikes_pca_2d = pca_2d.fit_transform(spikes)
    # getDatasetSimulationPlots(spikes, spikes_pca_2d, spikes_pca_3d, labels)

    # np.save('79_ground_truth', label)
    # np.save('79_x', spikes_reduced[:, 0])
    # np.save('79_y', spikes_reduced[:, 1])

    return spikes_pca_2d, labels


def get_dataset_simulation_wavelets(simNr, spike_length=79, align_to_peak=True, normalize_spike=False):
    """
    Load the dataset after wavelets on 2 dimensions
    :param simNr: integer - the number of the wanted simulation
    :param spike_length: integer - length of spikes in number of samples
    :param align_to_peak: integer - aligns each spike to it's maximum value
    :param normalize_spike: boolean - applies z-scoring normalization to each spike

    :returns result_spikes: matrix - the 2-dimensional points resulted
    :returns labels: vector - the vector of labels for each point
    """
    spikes, labels = get_dataset_simulation(simNr, spike_length, align_to_peak, normalize_spike)

    result_spikes1 = wlt.fd_wavelets(spikes)
    scaler = StandardScaler()
    result_spikes1 = scaler.fit_transform(result_spikes1)
    pca_2d = PCA(n_components=2)
    result_spikes = pca_2d.fit_transform(result_spikes1)
    return result_spikes, labels


def get_dataset_simulation_wavelets_3d(simNr, spike_length=79, align_to_peak=True, normalize_spike=False):
    """
    Load the dataset after derivatives on 2 dimensions
    :param simNr: integer - the number of the wanted simulation
    :param spike_length: integer - length of spikes in number of samples
    :param align_to_peak: integer - aligns each spike to it's maximum value
    :param normalize_spike: boolean - applies z-scoring normalization to each spike

    :returns result_spikes: matrix - the 2-dimensional points resulted
    :returns labels: vector - the vector of labels for each point
    """
    spikes, labels = get_dataset_simulation(simNr, spike_length, align_to_peak, normalize_spike)

    result_spikes1 = wlt.fd_wavelets(spikes)
    scaler = StandardScaler()
    result_spikes1 = scaler.fit_transform(result_spikes1)
    pca_3d = PCA(n_components=3)
    result_spikes = pca_3d.fit_transform(result_spikes1)
    return result_spikes, labels


def get_dataset_simulation_superlets_2d(simNr, spike_length=79, align_to_peak=True, normalize_spike=False):
    """
    Load the dataset after derivatives on 2 dimensions
    :param simNr: integer - the number of the wanted simulation
    :param spike_length: integer - length of spikes in number of samples
    :param align_to_peak: integer - aligns each spike to it's maximum value
    :param normalize_spike: boolean - applies z-scoring normalization to each spike

    :returns result_spikes: matrix - the 2-dimensional points resulted
    :returns labels: vector - the vector of labels for each point
    """
    spikes, labels = get_dataset_simulation(simNr, spike_length, align_to_peak, normalize_spike)

    result_spikes1 = slt.slt(spikes, 2, 1.1)
    # result_spikes1 = slt.slt2(spikes, 5, 1.5)
    scaler = StandardScaler()
    result_spikes1 = scaler.fit_transform(result_spikes1)
    pca_2d = PCA(n_components=2)
    result_spikes = pca_2d.fit_transform(result_spikes1)
    return result_spikes, labels


def get_dataset_simulation_superlets_3d(simNr, spike_length=79, align_to_peak=True, normalize_spike=False):
    """
    Load the dataset after derivatives on 2 dimensions
    :param simNr: integer - the number of the wanted simulation
    :param spike_length: integer - length of spikes in number of samples
    :param align_to_peak: integer - aligns each spike to it's maximum value
    :param normalize_spike: boolean - applies z-scoring normalization to each spike

    :returns result_spikes: matrix - the 2-dimensional points resulted
    :returns labels: vector - the vector of labels for each point
    """
    spikes, labels = get_dataset_simulation(simNr, spike_length, align_to_peak, normalize_spike)

    result_spikes1 = slt.slt(spikes, 2, 1.1)
    scaler = StandardScaler()
    result_spikes1 = scaler.fit_transform(result_spikes1)
    pca_3d = PCA(n_components=3)
    result_spikes = pca_3d.fit_transform(result_spikes1)
    return result_spikes, labels


def get_dataset_simulation_derivatives(simNr, spike_length=79, align_to_peak=True, normalize_spike=False):
    """
    Load the dataset after derivatives on 2 dimensions
    :param simNr: integer - the number of the wanted simulation
    :param spike_length: integer - length of spikes in number of samples
    :param align_to_peak: integer - aligns each spike to it's maximum value
    :param normalize_spike: boolean - applies z-scoring normalization to each spike

    :returns result_spikes: matrix - the 2-dimensional points resulted
    :returns labels: vector - the vector of labels for each point
    """
    spikes, labels = get_dataset_simulation(simNr, spike_length, align_to_peak, normalize_spike)

    result_spikes1 = deriv.compute_fdmethod(spikes)
    scaler = StandardScaler()
    result_spikes1 = scaler.fit_transform(result_spikes1)
    pca_2d = PCA(n_components=2)
    result_spikes = pca_2d.fit_transform(result_spikes1)
    # result_spikes = result_spikes1

    return result_spikes, labels


def get_dataset_simulation_dwt2d(simNr, spike_length=79, align_to_peak=True, normalize_spike=False):
    """
    Load the dataset after dwt on 2 dimensions
    :param simNr: integer - the number of the wanted simulation
    :param spike_length: integer - length of spikes in number of samples
    :param align_to_peak: integer - aligns each spike to it's maximum value
    :param normalize_spike: boolean - applies z-scoring normalization to each spike

    :returns result_spikes: matrix - the 2-dimensional points resulted
    :returns labels: vector - the vector of labels for each point
    """
    spikes, labels = get_dataset_simulation(simNr, spike_length, align_to_peak, normalize_spike)

    result_spikes1 = dwt.dwt_fd_method(spikes)
    scaler = StandardScaler()
    result_spikes1 = scaler.fit_transform(result_spikes1)
    pca_2d = PCA(n_components=2)
    result_spikes = pca_2d.fit_transform(result_spikes1)
    return result_spikes


# spike extraction options
# original sampling rate 96KHz, with each waveform at 316 points(dimensions/features)
# downsampled to 24KHz, (regula-3-simpla) => 79 points (de aici vine 79 de mai jos)
def get_dataset_simulation_pca_3d(simNr, spike_length=79, align_to_peak=True, normalize_spike=False):
    """
    Load the dataset after PCA on 3 dimensions
    :param simNr: integer - the number of the wanted simulation
    :param spike_length: integer - length of spikes in number of samples
    :param align_to_peak: integer - aligns each spike to it's maximum value
    :param normalize_spike: boolean - applies z-scoring normalization to each spike

    :returns spikes_pca_3d: matrix - the 3-dimensional points resulted
    :returns labels: vector - the vector of labels for each point
    """
    spikes, labels = get_dataset_simulation(simNr, spike_length, align_to_peak, normalize_spike)
    # apply pca
    pca_3d = PCA(n_components=3)
    spikes_pca_3d = pca_3d.fit_transform(spikes)

    return spikes_pca_3d, labels


def get_dataset_simulation_derivatives_3d(simNr, spike_length=79, align_to_peak=True, normalize_spike=False):
    """
    Load the dataset after PCA on 3 dimensions
    :param simNr: integer - the number of the wanted simulation
    :param spike_length: integer - length of spikes in number of samples
    :param align_to_peak: integer - aligns each spike to it's maximum value
    :param normalize_spike: boolean - applies z-scoring normalization to each spike

    :returns spikes_pca_3d: matrix - the 3-dimensional points resulted
    :returns labels: vector - the vector of labels for each point
    """
    spikes, labels = get_dataset_simulation(simNr, spike_length, align_to_peak, normalize_spike)

    result_spikes = deriv.compute_fdmethod3d(spikes)

    return result_spikes, labels


def get_dataset_simulation(simNr, spike_length=79, align_to_peak=True, normalize_spike=False):
    """
    Load the dataset
    :param spike_length: integer - length of spikes in number of samples
    :param align_to_peak: integer - aligns each spike to it's maximum value
    :param normalize_spike: boolean - applies z-scoring normalization to each spike
    :param simNr: integer - the number of the wanted simulation

    :returns spikes: matrix - the 79-dimensional points resulted
    :returns labels: vector - the vector of labels for each point
    """
    simulation_dictionary = loadmat('./datasets/simulation_' + str(simNr) + '.mat')
    ground_truth_dictionary = loadmat('./datasets/ground_truth.mat')

    labels = ground_truth_dictionary['spike_classes'][0][simNr - 1][0, :]
    start = ground_truth_dictionary['spike_first_sample'][0][simNr - 1][0, :]
    data = simulation_dictionary['data'][0, :]

    # each spike will contain the first 79 points from the data after it has started
    spikes = spike_preprocess(data, start, spike_length, align_to_peak, normalize_spike, labels)

    return spikes, labels


def get_shape_phase_distribution_features(sim_nr, spike_length=79, align_to_peak=True, normalize_spike=False,
                                          spikes_arg=None, labels_arg=None):
    if sim_nr == 0:
        spikes = np.array(spikes_arg)
        labels = np.array(labels_arg)
    else:
        spikes, labels = get_dataset_simulation(sim_nr, spike_length, align_to_peak, normalize_spike)
    pca_2d = PCA(n_components=2)

    features = shape_features.get_shape_phase_distribution_features(spikes)
    features = pca_2d.fit_transform(features)
    print("Variance Ratio = ", np.sum(pca_2d.explained_variance_ratio_))

    return features, labels


def get_hilbert_features(sim_nr, spike_length=79, feature_reduction='derivativesPCA2D', align_to_peak=True,
                         normalize_spike=False, plot=False, spikes_arg=None, labels_arg=None):
    if sim_nr == 0:
        spikes = np.array(spikes_arg)
        labels = np.array(labels_arg)
    else:
        spikes, labels = get_dataset_simulation(sim_nr, spike_length, align_to_peak, normalize_spike)

    spikes_hilbert = hilbert(spikes)

    envelope = np.abs(spikes_hilbert)

    features = reduce_dimensionality(envelope, feature_reduction)
    return features, labels


def get_emd_signal_no_residuum_features(simNr, spike_length=79, align_to_peak=True, normalize_spike=False):
    spikes, labels = get_dataset_simulation(simNr, spike_length, align_to_peak, normalize_spike)
    emd = EMD()

    # Signal without residuum
    features = np.zeros((spikes.shape[0], spikes.shape[1]))
    for i, spike in enumerate(spikes):
        emd(spike)
        IMFs, res = emd.get_imfs_and_residue()
        features[i] = np.sum(IMFs, axis=0)

    features = reduce_dimensionality(features, method='derivativesPCA2D')
    return features, labels


def get_emd_imf_derivatives_features(sim_nr, spike_length=79, align_to_peak=True, normalize_spike=False,
                                     spikes_arg=None, labels_arg=None):
    if sim_nr == 0:
        spikes = np.array(spikes_arg)
        labels = np.array(labels_arg)
    else:
        spikes, labels = get_dataset_simulation(sim_nr, spike_length, align_to_peak, normalize_spike)
    emd = EMD()

    features = np.zeros((spikes.shape[0], 8))
    for i, spike in enumerate(spikes):
        emd(spike)
        IMFs, res = emd.get_imfs_and_residue()

        f = np.array(deriv.compute_fdmethod(IMFs))

        if IMFs.shape[0] >= 4:
            features[i] = np.concatenate((f[0], f[1], f[2], f[3]))
        elif IMFs.shape[0] >= 3:
            features[i] = np.concatenate((f[0], f[1], f[2], [0, 0]))
        else:
            features[i] = np.concatenate((f[0], f[1], [0, 0], [0, 0]))

    features = reduce_dimensionality(features, method='PCA2D')
    return features, labels


def reduce_dimensionality(n_features, method='PCA2D'):
    if method == 'PCA2D':
        pca_2d = PCA(n_components=2)
        features = pca_2d.fit_transform(n_features)
    elif method == 'PCA3D':
        pca_3d = PCA(n_components=3)
        features = pca_3d.fit_transform(n_features)
    elif method == 'derivatives':
        features = deriv.compute_fdmethod(n_features)
    elif method == 'derivativesPCA2D':
        features = deriv.compute_fdmethod(n_features)
        pca_2d = PCA(n_components=2)
        features = pca_2d.fit_transform(features)
    else:
        features = []
    return features


# FOR SIMULATION 79
# dataset.mat in key 'data' == simulation_97.mat in key 'data'
# dataset.mat in key 'ground_truth' == ground_truth.mat in key 'spike_classes'[78]
# dataset.mat in key 'start_spikes' == ground_truth.mat in key 'spike_first_sample'[78]
# dataset.mat in key 'spike_wf' == ground_truth.mat in key 'su_waveforms'[78] (higher precision in GT)
def getDatasetSim79():
    """
    Load the dataset Simulation79
    :param None

    :returns spikes_pca_2d: matrix - the points that have been taken through 2D PCA
    :returns labels: vector - the vector of labels for simulation79
    """
    dictionary = loadmat('./datasets/dataset.mat')

    # dataset file is a dictionary (the data has been extracted from ground_truth.mat and simulation_79.mat), containing following keys:
    # ground_truth (shape = 14536): the labels of the points
    # start_spikes (shape = 14536): the start timestamp of each spike
    # data (shape = 14400000): the raw spikes with all their points
    # spike_wf (shape = (20, 316)): contains the form of each spike (20 spikes, each with 316 dimensions/features) NOT USED YET

    labels = dictionary['ground_truth'][0, :]
    start = dictionary['start_spikes'][0, :]
    data = dictionary['data'][0, :]

    # spike extraction options
    # original sampling rate 96KHz, with each waveform at 316 points(dimensions/features)
    # downsampled to 24KHz, (regula-3-simpla) => 79 points (de aici vine 79 de mai jos)
    spike_length = 79  # length of spikes in number of samples
    align_to_peak = False  # aligns each spike to it's maximum value
    normalize_spike = False  # applies z-scoring normalization to each spike

    # each spike will contain the first 79 points from the data after it has started
    spikes = spike_preprocess(data, start, spike_length, align_to_peak, normalize_spike, labels)

    # apply pca
    pca_2d = PCA(n_components=2)
    pca_3d = PCA(n_components=3)
    spikes_pca_2d = pca_2d.fit_transform(spikes)
    spikes_pca_3d = pca_3d.fit_transform(spikes)

    return spikes_pca_2d, labels


def getDatasetSim97Plots(spikes, spike_pca_2d, spikes_pca_3d, labels):
    # plot some spikes
    ind = np.random.randint(0, len(labels), [20])
    plt.plot(np.transpose(spikes[ind, :]))
    plt.show()

    # plot all spikes from one unit
    unit = 15
    ind = np.squeeze(np.argwhere(labels == unit))
    plt.plot(np.transpose(spikes[ind, :]))
    plt.title('Unit {}'.format(unit))
    plt.show()

    plotSimulation_PCA2D_grid(spike_pca_2d, labels)

    plotSimulation_PCA3D(spikes_pca_3d, labels)


def plotSimulation_PCA2D(spike_pca_2d, labels):
    # plot scatter of pca
    plt.scatter(spike_pca_2d[:, 0], spike_pca_2d[:, 1], c=labels, marker='x', cmap='brg')
    plt.show()


def plotSimulation_PCA2D_grid(spike_pca_2d, labels):
    # plot scatter of pca
    scatter_plot.plot_grid('Sim79Gridded', spike_pca_2d, labels + 1, 25, marker='x')
    plt.show()


def plotSimulation_PCA3D(spikes_pca_3d, labels):
    # plot scatter of pca in 3d
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(spikes_pca_3d[:, 0], spikes_pca_3d[:, 1], spikes_pca_3d[:, 2], c=labels, marker='x', cmap='brg')
    plt.show()


def getTINSDataChance():
    # Importing the dataset
    data = pd.read_csv('./datasets/data.csv', skiprows=0)
    f1 = data['F1'].values
    f2 = data['F2'].values
    f3 = data['F3'].values

    # print("Shape:")
    # print(data.shape)
    # print("\n")

    c1 = np.array([f1]).T
    c2 = np.array([f2]).T
    c3 = np.array([f3]).T

    X = np.hstack((c1, c2, c3))
    chanceKeep = 1
    keep = np.random.choice(2, len(X), p=[1 - chanceKeep, chanceKeep])
    keep = keep == 1
    X = X[keep]
    return X


def getTINSData():
    # Importing the dataset
    data = pd.read_csv('./datasets/data.csv', skiprows=0)
    f1 = data['F1'].values
    f2 = data['F2'].values
    f3 = data['F3'].values

    # print("Shape:")
    # print(data.shape)
    # print("\n")

    c1 = np.array([f1]).T
    c2 = np.array([f2]).T
    c3 = np.array([f3]).T

    X = np.hstack((c1, c2, c3))
    return X


def getGenData(plotFig=False):
    np.random.seed(0)
    avgPoints = 250
    C1 = [-2, 0] + .8 * np.random.randn(avgPoints * 2, 2)

    C4 = [-2, 3] + .3 * np.random.randn(avgPoints // 5, 2)
    L4 = np.full(len(C4), 1).reshape((len(C4), 1))

    C3 = [1, -2] + .2 * np.random.randn(avgPoints * 5, 2)
    C5 = [3, -2] + 1.0 * np.random.randn(avgPoints * 4, 2)
    L5 = np.full(len(C5), 2)

    C2 = [4, -1] + .1 * np.random.randn(avgPoints, 2)
    # L2 = np.full(len(C2), 1).reshape((len(C2), 1))

    C6 = [5, 6] + 1.0 * np.random.randn(avgPoints * 5, 2)
    L6 = np.full(len(C6), 3).reshape((len(C6), 1))

    if plotFig:
        plt.plot(C1[:, 0], C1[:, 1], 'b.', alpha=0.3)
        plt.plot(C2[:, 0], C2[:, 1], 'r.', alpha=0.3)
        plt.plot(C3[:, 0], C3[:, 1], 'g.', alpha=0.3)
        plt.plot(C4[:, 0], C4[:, 1], 'c.', alpha=0.3)
        plt.plot(C5[:, 0], C5[:, 1], 'm.', alpha=0.3)
        plt.plot(C6[:, 0], C6[:, 1], 'y.', alpha=0.3)
        plt.figure()
        plt.plot(C1[:, 0], C1[:, 1], 'b.', alpha=0.3)
        plt.plot(C2[:, 0], C2[:, 1], 'b.', alpha=0.3)
        plt.plot(C3[:, 0], C3[:, 1], 'b.', alpha=0.3)
        plt.plot(C4[:, 0], C4[:, 1], 'b.', alpha=0.3)
        plt.plot(C5[:, 0], C5[:, 1], 'b.', alpha=0.3)
        plt.plot(C6[:, 0], C6[:, 1], 'b.', alpha=0.3)

    plt.show()
    X = np.vstack((C1, C2, C3, C4, C5, C6))

    c1Labels = np.full(len(C1), 1)
    c2Labels = np.full(len(C2), 2)
    c3Labels = np.full(len(C3), 3)
    c4Labels = np.full(len(C4), 4)
    c5Labels = np.full(len(C5), 5)
    c6Labels = np.full(len(C6), 6)

    y = np.hstack((c1Labels, c2Labels, c3Labels, c4Labels, c5Labels, c6Labels))
    return X, y


def getDatasetS1():
    X = np.genfromtxt("./datasets/s1_labeled.csv", delimiter=",")
    X, y = X[:, [0, 1]], X[:, 2]
    return X, y


def getDatasetS2():
    X = np.genfromtxt("./datasets/s2_labeled.csv", delimiter=",")
    X, y = X[:, [0, 1]], X[:, 2]
    return X, y


def getDatasetU():
    X = np.genfromtxt("./datasets/unbalance.csv", delimiter=",")
    X, y = X[:, [0, 1]], X[:, 2]
    return X, y


def apply_feature_extraction_method(sim_nr, method_nr):
    if method_nr == 0:
        X, y = get_dataset_simulation_pca_2d(sim_nr, align_to_peak=True)
    elif method_nr == 1:
        X, y = get_dataset_simulation_pca_3d(sim_nr, align_to_peak=True)
    elif method_nr == 2:
        X, y = get_dataset_simulation_derivatives(sim_nr, align_to_peak=True)
    elif method_nr == 3:
        X, y = get_dataset_simulation_superlets_2d(sim_nr, align_to_peak=True)
    elif method_nr == 4:
        X, y = get_dataset_simulation_superlets_3d(sim_nr, align_to_peak=True)
    elif method_nr == 5:
        X, y = get_dataset_simulation_wavelets(sim_nr, align_to_peak=True)
    elif method_nr == 6:
        X, y = get_dataset_simulation_wavelets_3d(sim_nr, align_to_peak=True)
    elif method_nr == 7:
        X, y = get_dataset_simulation_dwt2d(sim_nr, align_to_peak=True)
    elif method_nr == 8:
        X, y = get_hilbert_features(sim_nr, align_to_peak=True)
    elif method_nr == 8:
        # X, y = EMD, TODO
        X, y = get_dataset_simulation_pca_2d(sim_nr, align_to_peak=True)
    else:
        X, y = get_dataset_simulation_pca_2d(sim_nr, align_to_peak=True)
    return X, y


# datasetNumber = 1 => S1
# datasetNumber = 2 => S2
# datasetNumber = 3 => U
# datasetNumber = 4 => UO
# datasetNumber = 5 => Sim97
def load_particular_dataset(datasetNumber):
    """
    Benchmarks K-Means, DBSCAN and SBM on one of 5 selected datasets
    :param datasetNumber: integer - the number that represents one of the datasets (0-4)
    :param plot: boolean - optional, whether the plot function should be called or not (for ease of use)

    :returns None
    """
    print("DATASET: " + dataName[datasetNumber])
    datasetName = dataName[datasetNumber]
    simulation_number = 10
    if datasetNumber < 3:
        X = np.genfromtxt("./datasets/" + files[datasetNumber], delimiter=",")
        X, y = X[:, [0, 1]], X[:, 2]
    elif datasetNumber == 3:
        X, y = getGenData()
    else:
        # X, y = ds.getDatasetSim79()
        X, y = get_dataset_simulation_pca_2d(simNr=10)

    # S2 has label problems
    if datasetNumber == 1:
        for k in range(len(X)):
            y[k] = y[k] - 1
