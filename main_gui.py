import benchmark_data as bd

import datasets as ds
import matplotlib.pyplot as plt
from scipy.fftpack import fft
import scatter_plot as sp
import numpy as np
from sklearn.decomposition import PCA
import constants as cs
import plotly.express as px
from constants import LABEL_COLOR_MAP
import csv


def gui():
    bd.accuracy_all_algorithms_on_simulation(simulation_nr=79,
                                             feature_extract_method=0,
                                             plot=True,
                                             pe_labeled_data=True,
                                             pe_unlabeled_data=True,
                                             pe_extra=False)


def fourier_understanding():
    sim_nr = 79
    spikes, labels = ds.get_dataset_simulation(sim_nr, spike_length=79, align_to_peak=2, normalize_spike=False)
    # sp.plot("Ground truth" + str(sim_nr), spikes, labels, marker='o')
    # print(labels)
    # sp.plot_spikes(spikes)
    # print(spikes[0])

    # fft_signal = fft(spikes)
    # print("A[0]:")
    # print(fft_signal[0:10, 0])

    for i in range(np.amax(labels) + 1):
        spikes_by_color = spikes[labels == i]
        fft_signal = fft(spikes_by_color)
        print("A[0] for cluster %d:" % i)
        print(fft_signal[0:10, 0])
        # sp.plot_spikes(spikes_by_color, "Cluster %d Sim_%d" % (i, sim_nr))


def fourier_feature_extract():
    sim_nr = 79
    spikes, labels = ds.get_dataset_simulation(sim_nr, spike_length=79, align_to_peak=2, normalize_spike=False)
    # sp.plot("Ground truth" + str(sim_nr), spikes, labels, marker='o')
    # print(labels)
    # sp.plot_spikes(spikes)
    # print(spikes[0])

    fft_signal = fft(spikes)
    # print("A[0]:")
    # print(fft_signal[0][0])

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
    sim_nr = 15
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
        plt.savefig('figures/spikes_on_cluster/Sim_%d_Cluster_%d' % (sim_nr, i))
        plt.show()
        # print(cluster_pca)


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


# gui()
# fourier_feature_extract()
# plot_all_ground_truths()
# spikes_per_cluster()
# fourier_understanding()
# csf_db()
# sim_details()
plot_single_spike()
