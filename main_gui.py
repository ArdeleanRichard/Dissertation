import benchmark_data as bd

import datasets as ds
import matplotlib.pyplot as plt
from scipy.fftpack import fft, fft2, fftshift
from scipy.signal import stft
import scatter_plot as sp
import numpy as np
from sklearn.decomposition import PCA
import constants as cs
import plotly.express as px
from constants import LABEL_COLOR_MAP


def gui():
    bd.accuracy_all_algorithms_on_simulation(simulation_nr=4,
                                             feature_extract_method=0,
                                             plot=False,
                                             pe_labeled_data=True,
                                             pe_unlabeled_data=False,
                                             pe_extra=False)


def fourier_feature_extract():
    sim_nr = 79
    spikes, labels = ds.get_dataset_simulation(sim_nr, spike_length=79, align_to_peak=2, normalize_spike=False)
    # sp.plot("Ground truth" + str(sim_nr), spikes, labels, marker='o')
    # print(labels)
    # sp.plot_spikes(spikes)
    # print(spikes[0])

    fft_signal = fft(spikes)

    X = [x.real for x in fft_signal[:, 0:40]]
    # print(np.array(X).shape)
    # sp.plot_spikes(X, "Real")
    Y = [x.imag for x in fft_signal[:, 0:40]]
    # sp.plot_spikes(Y, "Imaginary")

    amplitude = np.sqrt(np.add(np.multiply(X, X), np.multiply(Y, Y)))
    phase = np.arctan2(Y, X)
    power = np.power(amplitude, 2)
    # sp.plot_spikes(amplitude, "Amplitude")
    # sp.plot_spikes(phase, "Phase")
    # sp.plot_spikes(power, "Power")

    label_color = [cs.LABEL_COLOR_MAP[l] for l in labels]
    # for i in range(0, len(X)-1):
    #     plt.scatter(X[i], amplitude[i], color=label_color[i])
    # # plt.scatter(X[:], Y[:], c=label_color, marker='o', edgecolors='k')
    # plt.xlabel('Real axis')
    # plt.ylabel('Amplitude axis')
    # plt.title('Sim 79 with FFT real-amplitude values')
    # plt.show()

    # time = np.arange(40)
    # label_1 = np.array([5, 11, 22, 23, 25])
    # for i in label_1:
    #     plt.plot(time, Y[i])
    #     plt.title("Imaginary FFT for %d" % i)
    #     plt.savefig("sim_15_imag_s%d.png" % i)
    #     plt.show()

    pca_2d = PCA(n_components=2)

    x_y_concat = np.concatenate((X, Y), axis=1)
    print(x_y_concat.shape)
    # sp.plot_spikes(x_y_concat, "Concat")
    x_y_concat_pca = pca_2d.fit_transform(x_y_concat)
    # sp.plot(title="GT on Fourier concat on Sim_" + str(sim_nr), X=x_y_concat_pca, labels=labels, marker='o')
    # plt.show()

    real_signal_pca = pca_2d.fit_transform(X)
    # print("Variance ratio for real")
    # print(pca_2d.explained_variance_ratio_)
    # print(real_signal_pca)
    # print(real_signal_pca.shape)

    img_signal_pca = pca_2d.fit_transform(Y)
    # print("Variance ratio for imag")
    # print(pca_2d.explained_variance_ratio_)
    sp.plot(title="GT on Fourier real coeff on Sim_" + str(sim_nr), X=real_signal_pca, labels=labels, marker='o')
    plt.show()
    sp.plot(title="GT on Fourier img coeff on Sim_" + str(sim_nr), X=img_signal_pca, labels=labels, marker='o')
    plt.show()

    amplitude_signal_pca = pca_2d.fit_transform(amplitude)
    phase_signal_pca = pca_2d.fit_transform(phase)
    sp.plot(title="GT on Fourier amplitude coeff on Sim_" + str(sim_nr), X=amplitude_signal_pca, labels=labels,
            marker='o')
    plt.show()
    sp.plot(title="GT on Fourier phase coeff on Sim_" + str(sim_nr), X=phase_signal_pca, labels=labels, marker='o')
    plt.show()

    power_signal_pca = pca_2d.fit_transform(power)
    sp.plot(title="GT on Fourier power coeff on Sim_" + str(sim_nr), X=power_signal_pca, labels=labels,
            marker='o')
    plt.show()

    # SBM
    # real_sbm_labels = bd.apply_algorithm(real_signal_pca, labels, 2)
    # sp.plot(title="SBM on Fourier real coeff on Sim_" + str(sim_nr), X=real_signal_pca, labels=real_sbm_labels,
    #         marker='o')
    # plt.show()
    # img_sbm_labels = bd.apply_algorithm(img_signal_pca, labels, 2)
    # sp.plot(title="SBM on Fourier img coeff on Sim_" + str(sim_nr), X=img_signal_pca, labels=img_sbm_labels, marker='o')
    # plt.show()
    # amplitude_sbm_labels = bd.apply_algorithm(amplitude_signal_pca, labels, 2)
    # phase_sbm_labels = bd.apply_algorithm(phase_signal_pca, labels, 2)
    # power_sbm_labels = bd.apply_algorithm(power_signal_pca, labels, 2)
    #
    # real_results = bd.benchmark_algorithm_labeled_data(real_sbm_labels, labels)
    # print("Real")
    # bd.print_benchmark_labeled_data(sim_nr, 2, real_results)
    #
    # img_results = bd.benchmark_algorithm_labeled_data(img_sbm_labels, labels)
    # print("Imaginary")
    # bd.print_benchmark_labeled_data(sim_nr, 2, img_results)
    #
    # amplitude_results = bd.benchmark_algorithm_labeled_data(amplitude_sbm_labels, labels)
    # print("Amplitude")
    # bd.print_benchmark_labeled_data(sim_nr, 2, amplitude_results)
    #
    # phase_results = bd.benchmark_algorithm_labeled_data(phase_sbm_labels, labels)
    # print("Phase")
    # bd.print_benchmark_labeled_data(sim_nr, 2, phase_results)
    #
    # power_results = bd.benchmark_algorithm_labeled_data(power_sbm_labels, labels)
    # print("Power")
    # bd.print_benchmark_labeled_data(sim_nr, 2, power_results)
    #
    # # K-means
    # real_kmeans_labels = bd.apply_algorithm(real_signal_pca, labels, 0)
    # sp.plot(title="K-Means on Fourier real coeff on Sim_" + str(sim_nr), X=real_signal_pca, labels=real_kmeans_labels,
    #         marker='o')
    # plt.show()
    # img_kmeans_labels = bd.apply_algorithm(img_signal_pca, labels, 0)
    # sp.plot(title="K-Means on Fourier img coeff on Sim_" + str(sim_nr), X=img_signal_pca, labels=img_kmeans_labels,
    #         marker='o')
    # plt.show()
    #
    # real_results = bd.benchmark_algorithm_labeled_data(real_kmeans_labels, labels)
    # img_results = bd.benchmark_algorithm_labeled_data(img_kmeans_labels, labels)
    # print("Real")
    # bd.print_benchmark_labeled_data(sim_nr, 0, real_results)
    # print("Imaginary")
    # bd.print_benchmark_labeled_data(sim_nr, 0, img_results)
    #
    # # DBSCAN
    # real_dbscan_labels = bd.apply_algorithm(real_signal_pca, labels, 1)
    # sp.plot(title="DBSCAN on Fourier real coeff on Sim_" + str(sim_nr), X=real_signal_pca, labels=real_dbscan_labels,
    #         marker='o')
    # plt.show()
    # img_dbscan_labels = bd.apply_algorithm(img_signal_pca, labels, 1)
    # sp.plot(title="DBSCAN on Fourier img coeff on Sim_" + str(sim_nr), X=img_signal_pca, labels=img_dbscan_labels,
    #         marker='o')
    # plt.show()
    #
    # real_results = bd.benchmark_algorithm_labeled_data(real_dbscan_labels, labels)
    # img_results = bd.benchmark_algorithm_labeled_data(img_dbscan_labels, labels)
    # print("Real")
    # bd.print_benchmark_labeled_data(sim_nr, 1, real_results)
    # print("Imaginary")
    # bd.print_benchmark_labeled_data(sim_nr, 1, img_results)


def fourier_feature_extract_3d():
    sim_nr = 79
    spikes, labels = ds.get_dataset_simulation(sim_nr, spike_length=79, align_to_peak=2, normalize_spike=False)
    # sp.plot("Ground truth" + str(sim_nr), signals, labels, marker='o')
    # sp.plot_spikes(spikes)

    fft_signal = fft(spikes)

    X = [x.real for x in fft_signal[:, 0:40]]
    print(np.array(X).shape)
    # sp.plot_spikes(X, "Real")
    Y = [x.imag for x in fft_signal[:, 0:40]]
    amplitude = np.sqrt(np.add(np.multiply(X, X), np.multiply(Y, Y)))
    phase = np.arctan(np.divide(Y, X))

    pca_3d = PCA(n_components=3)

    x_y_concat = np.concatenate((X, Y), axis=1)
    print(x_y_concat.shape)
    x_y_concat_pca = pca_3d.fit_transform(x_y_concat)
    # fig = px.scatter_3d(x_y_concat_pca, x=x_y_concat_pca[:, 0], y=x_y_concat_pca[:, 1], z=x_y_concat_pca[:, 2],
    #                     color=labels)
    # fig.update_layout(title="Concat for Sim_" + str(sim_nr))
    # fig.show()

    # real_signal_pca = pca_3d.fit_transform(X)
    #
    # img_signal_pca = pca_2d.fit_transform(Y)
    # # print("Variance ratio for imag")
    # # print(pca_2d.explained_variance_ratio_)
    # sp.plot(title="GT on Fourier real coeff on Sim_" + str(sim_nr), X=real_signal_pca, labels=labels, marker='o')
    # plt.show()
    # sp.plot(title="GT on Fourier img coeff on Sim_" + str(sim_nr), X=img_signal_pca, labels=labels, marker='o')
    # plt.show()
    #
    amplitude_signal_pca = pca_3d.fit_transform(amplitude)
    # phase_signal_pca = pca_2d.fit_transform(phase)
    fig = px.scatter_3d(amplitude_signal_pca, x=amplitude_signal_pca[:, 0], y=amplitude_signal_pca[:, 1],
                        z=amplitude_signal_pca[:, 2], color=labels)
    fig.update_layout(title="Concat for Sim_" + str(sim_nr))
    fig.show()
    # sp.plot(title="GT on Fourier phase coeff on Sim_" + str(sim_nr), X=phase_signal_pca, labels=labels, marker='o')
    # plt.show()


def short_time_fourier_feature_extraction():
    sim_nr = 67
    spikes, labels = ds.get_dataset_simulation(sim_nr, spike_length=79, align_to_peak=2, normalize_spike=False)
    stft_result = stft(spikes, window='boxcar', nperseg=79)
    sampled_frequencies = stft_result[0]
    time_segments = stft_result[1]
    stft_signal = stft_result[2]
    print(time_segments)

    X = [x.real for x in stft_signal]
    X = np.array(X)
    Y = [x.imag for x in stft_signal]
    Y = np.array(Y)

    amplitude = np.sqrt(np.add(np.multiply(X, X), np.multiply(Y, Y)))
    phase = np.arctan2(Y, X)
    power = np.power(amplitude, 2)

    pca_2d = PCA(n_components=2)

    sp.plot(title="GT on Sim_" + str(sim_nr), X=pca_2d.fit_transform(spikes), labels=labels, marker='o')
    plt.show()

    for i in range(0, 3):
        level_stft = phase[:, :, i]
        signal_pca = pca_2d.fit_transform(level_stft)
        sp.plot(title="GT on STFT_%d boxcar phase coeff on Sim_" % i + str(sim_nr), X=signal_pca, labels=labels,
                marker='o')
        plt.show()

    # signal_pca = pca_2d.fit_transform(Y[:, :, 0])
    # # signal_pca = pca_2d.fit_transform(spikes)
    # sbm_labels = bd.apply_algorithm(signal_pca, labels, 2)
    # sp.plot(title="SBM on STFT img_1 bartlett on Sim_" + str(sim_nr), X=signal_pca, labels=sbm_labels, marker='o')
    # plt.show()
    #
    # sbm_results = bd.benchmark_algorithm_labeled_data(sbm_labels, labels)
    # print("Img_1")
    # bd.print_benchmark_labeled_data(sim_nr, 2, sbm_results)


import scipy.signal as signal


def plot_stft():
    sim_nr = 84
    spikes, labels = ds.get_dataset_simulation(sim_nr, spike_length=79, align_to_peak=2, normalize_spike=False)
    f, t, Zxx = signal.stft(spikes, window='bartlett', nperseg=25)
    print(f)
    print(t)
    print(Zxx.shape)
    print(Zxx)
    # sample_rate, samples = wav.read(filename)
    # f, t, Zxx = signal.stft(samples, fs=sample_rate)

    # for i in range(1, 10):
    #     im = plt.pcolormesh(t, f, np.abs(Zxx)[i], cmap="jet")
    #     # print((np.abs(Zxx)[i]).shape)
    #     plt.colorbar(im)
    #     plt.show()

    for i in range(np.amax(labels) + 1):
        spikes_by_color = Zxx[labels == i]
        print(spikes_by_color.shape)
        for j in range(5):
            im = plt.pcolormesh(t, f, np.abs(spikes_by_color)[j], cmap="jet")
            plt.colorbar(im)
            plt.title("Cluster %d spike %d " % (i, j))
            # plt.savefig('./figures/bartlett_40/ft_cluster_%d_spike_%d' % (i, j))
            plt.show()


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


def spikes_per_cluster():
    sim_nr = 67
    spikes, labels = ds.get_dataset_simulation(sim_nr, spike_length=79, align_to_peak=2, normalize_spike=False)
    print(spikes.shape)

    pca2d = PCA(n_components=2)

    for i in range(np.amax(labels) + 1):
        spikes_by_color = spikes[labels == i]
        print(len(spikes_by_color))
        sp.plot_spikes(spikes_by_color, "Cluster %d Sim_%d" % (i, sim_nr))
        cluster_pca = pca2d.fit_transform(spikes_by_color)
        # sp.plot(title="GT with PCA Sim_%d" % sim_nr, X=cluster_pca, marker='o')
        plt.scatter(cluster_pca[:, 0], cluster_pca[:, 1], c=LABEL_COLOR_MAP[i], marker='o', edgecolors='k')
        plt.title("Cluster %d Sim_%d" % (i, sim_nr))
        plt.show()
        # print(cluster_pca)


# gui()
# fourier_feature_extract()
# short_time_fourier_feature_extraction()
# plot_all_ground_truths()
# spikes_per_cluster()
plot_stft()
