import benchmark_data as bd

import datasets as ds
import matplotlib.pyplot as plt
from scipy.fftpack import fft
import scatter_plot as sp
import numpy as np
from sklearn.decomposition import PCA
import constants as cs
import plotly.express as px
import csv
import derivatives


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
    sim_nr = 8
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
    img_signal_pca = pca_2d.fit_transform(Y)
    sp.plot(title="GT on Fourier real coeff on Sim_" + str(sim_nr), X=real_signal_pca, labels=labels, marker='o')
    plt.show()
    sp.plot(title="GT on Fourier img coeff on Sim_" + str(sim_nr), X=img_signal_pca, labels=labels, marker='o')
    plt.show()

    amplitude_signal_pca = pca_2d.fit_transform(amplitude)
    phase_signal_pca = pca_2d.fit_transform(phase)
    title = "GT Fourier amplitude on Sim_" + str(sim_nr)
    sp.plot(title=title, X=amplitude_signal_pca, labels=labels,
            marker='o')
    plt.savefig('figures/stft_plots/%s' % title)
    plt.show()
    sp.plot(title="GT on Fourier phase coeff on Sim_" + str(sim_nr), X=phase_signal_pca, labels=labels, marker='o')
    plt.show()

    power_signal_pca = pca_2d.fit_transform(power)
    sp.plot(title="GT on Fourier power coeff on Sim_" + str(sim_nr), X=power_signal_pca, labels=labels,
            marker='o')
    plt.show()

    # SBM
    real_sbm_labels = bd.apply_algorithm(real_signal_pca, labels, 2)
    sp.plot(title="SBM on Fourier real coeff on Sim_" + str(sim_nr), X=real_signal_pca, labels=real_sbm_labels,
            marker='o')
    plt.show()
    img_sbm_labels = bd.apply_algorithm(img_signal_pca, labels, 2)
    sp.plot(title="SBM on Fourier img coeff on Sim_" + str(sim_nr), X=img_signal_pca, labels=img_sbm_labels, marker='o')
    plt.show()
    amplitude_sbm_labels = bd.apply_algorithm(amplitude_signal_pca, labels, 2)
    phase_sbm_labels = bd.apply_algorithm(phase_signal_pca, labels, 2)
    power_sbm_labels = bd.apply_algorithm(power_signal_pca, labels, 2)

    real_results = bd.benchmark_algorithm_labeled_data(real_sbm_labels, labels)
    real_f = ["%.3f" % number for number in real_results]
    print("Real")
    bd.print_benchmark_labeled_data(sim_nr, 2, real_results)

    img_results = bd.benchmark_algorithm_labeled_data(img_sbm_labels, labels)
    print("Imaginary")
    bd.print_benchmark_labeled_data(sim_nr, 2, img_results)

    amplitude_results = bd.benchmark_algorithm_labeled_data(amplitude_sbm_labels, labels)
    print("Amplitude")
    bd.print_benchmark_labeled_data(sim_nr, 2, amplitude_results)

    phase_results = bd.benchmark_algorithm_labeled_data(phase_sbm_labels, labels)
    print("Phase")
    bd.print_benchmark_labeled_data(sim_nr, 2, phase_results)

    power_results = bd.benchmark_algorithm_labeled_data(power_sbm_labels, labels)
    print("Power")
    bd.print_benchmark_labeled_data(sim_nr, 2, power_results)
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


def fourier_all_sim():
    real_average = [0, 0, 0, 0, 0]
    img_average = [0, 0, 0, 0, 0]
    amplitude_average = [0, 0, 0, 0, 0]
    phase_average = [0, 0, 0, 0, 0]
    power_average = [0, 0, 0, 0, 0]
    simulations_counter = 0
    for sim_nr in range(1, 96):
        if sim_nr != 25 and sim_nr != 27 and sim_nr != 44:
            simulations_counter += 1

            spikes, labels = ds.get_dataset_simulation(sim_nr, spike_length=79, align_to_peak=2, normalize_spike=False)
            fft_signal = fft(spikes)

            X = [x.real for x in fft_signal[:, 0:40]]
            Y = [x.imag for x in fft_signal[:, 0:40]]
            amplitude = np.sqrt(np.add(np.multiply(X, X), np.multiply(Y, Y)))
            phase = np.arctan2(Y, X)
            power = np.power(amplitude, 2)

            pca_2d = PCA(n_components=2)

            # real_signal_pca = pca_2d.fit_transform(X)
            # img_signal_pca = pca_2d.fit_transform(Y)
            # amplitude_signal_pca = pca_2d.fit_transform(amplitude)
            # phase_signal_pca = pca_2d.fit_transform(phase)
            # power_signal_pca = pca_2d.fit_transform(power)
            real_signal_pca = derivatives.compute_fdmethod(X)
            img_signal_pca = derivatives.compute_fdmethod(Y)
            amplitude_signal_pca = derivatives.compute_fdmethod(amplitude)
            phase_signal_pca = derivatives.compute_fdmethod(phase)
            power_signal_pca = derivatives.compute_fdmethod(power)

            # SBM
            real_sbm_labels = bd.apply_algorithm(real_signal_pca, labels, 2)
            img_sbm_labels = bd.apply_algorithm(img_signal_pca, labels, 2)
            amplitude_sbm_labels = bd.apply_algorithm(amplitude_signal_pca, labels, 2)
            phase_sbm_labels = bd.apply_algorithm(phase_signal_pca, labels, 2)
            power_sbm_labels = bd.apply_algorithm(power_signal_pca, labels, 2)

            # results
            real_results = bd.benchmark_algorithm_labeled_data(real_sbm_labels, labels)
            real_f = ["%.3f" % number for number in real_results]
            real_row = [sim_nr, real_f[0], real_f[1], real_f[2], real_f[3], real_f[4]]
            with open('./results/fourier_real_deriv.csv', 'a+', newline='') as file:
                writer = csv.writer(file, delimiter=',')
                writer.writerow(real_row)

            img_results = bd.benchmark_algorithm_labeled_data(img_sbm_labels, labels)
            img_f = ["%.3f" % number for number in img_results]
            img_row = [sim_nr, img_f[0], img_f[1], img_f[2], img_f[3], img_f[4]]
            with open('./results/fourier_img_deriv.csv', 'a+', newline='') as file:
                writer = csv.writer(file, delimiter=',')
                writer.writerow(img_row)

            amplitude_results = bd.benchmark_algorithm_labeled_data(amplitude_sbm_labels, labels)
            ampl_f = ["%.3f" % number for number in amplitude_results]
            ampl_row = [sim_nr, ampl_f[0], ampl_f[1], ampl_f[2], ampl_f[3], ampl_f[4]]
            with open('./results/fourier_amplitude_deriv.csv', 'a+', newline='') as file:
                writer = csv.writer(file, delimiter=',')
                writer.writerow(ampl_row)

            phase_results = bd.benchmark_algorithm_labeled_data(phase_sbm_labels, labels)
            phase_f = ["%.3f" % number for number in phase_results]
            phase_row = [sim_nr, phase_f[0], phase_f[1], phase_f[2], phase_f[3], phase_f[4]]
            with open('./results/fourier_phase_deriv.csv', 'a+', newline='') as file:
                writer = csv.writer(file, delimiter=',')
                writer.writerow(phase_row)

            power_results = bd.benchmark_algorithm_labeled_data(power_sbm_labels, labels)
            power_f = ["%.3f" % number for number in power_results]
            power_row = [sim_nr, power_f[0], power_f[1], power_f[2], power_f[3], power_f[4]]
            with open('./results/fourier_power_deriv.csv', 'a+', newline='') as file:
                writer = csv.writer(file, delimiter=',')
                writer.writerow(power_row)

            # average
            real_average += real_results
            img_average += img_results
            amplitude_average += amplitude_results
            phase_average += phase_results
            power_average += power_results

    real_average = real_average / simulations_counter
    with open('./results/fourier_real_deriv.csv', 'a+', newline='') as file:
        writer = csv.writer(file, delimiter=',')
        writer.writerow(real_average)
    img_average = img_average / simulations_counter
    with open('./results/fourier_img_deriv.csv', 'a+', newline='') as file:
        writer = csv.writer(file, delimiter=',')
        writer.writerow(img_average)
    amplitude_average = amplitude_average / simulations_counter
    with open('./results/fourier_amplitude_deriv.csv', 'a+', newline='') as file:
        writer = csv.writer(file, delimiter=',')
        writer.writerow(amplitude_average)
    phase_average = phase_average / simulations_counter
    with open('./results/fourier_phase_deriv.csv', 'a+', newline='') as file:
        writer = csv.writer(file, delimiter=',')
        writer.writerow(phase_average)
    power_average = power_average / simulations_counter
    with open('./results/fourier_power_deriv.csv', 'a+', newline='') as file:
        writer = csv.writer(file, delimiter=',')
        writer.writerow(power_average)

# fourier_understanding()
# fourier_feature_extract()
# fourier_all_sim()
