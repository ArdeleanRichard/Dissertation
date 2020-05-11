import benchmark_data as bd

import datasets as ds
import matplotlib.pyplot as plt
import scatter_plot as sp
import numpy as np
from sklearn.decomposition import PCA
import constants as cs
import plotly.express as px
import scipy.signal as signal
import derivatives as deriv
import csv
import stft_dpss


def short_time_fourier_feature_extraction():
    sim_nr = 1
    spikes, labels = ds.get_dataset_simulation(sim_nr, spike_length=79, align_to_peak=2, normalize_spike=False)
    w = 'hamming'
    fs = 1
    nperseg = 40
    data_nr = 7
    # sampled_frequencies, time_segments, Zxx = signal.stft(spikes, window=w, nperseg=nperseg, noverlap=5)
    sampled_frequencies, time_segments, Zxx = signal.stft(spikes, window=w, fs=fs, nperseg=nperseg)
    print(len(time_segments))
    print(len(sampled_frequencies))

    stft_signal = Zxx.reshape(*Zxx.shape[:1], -1)
    print(stft_signal.shape)

    X = [x.real for x in stft_signal]
    X = np.array(X)
    Y = [x.imag for x in stft_signal]
    Y = np.array(Y)
    # amplitude2 = np.sqrt(np.add(np.multiply(X, X), np.multiply(Y, Y)))
    amplitude = np.abs(stft_signal)
    phase = np.arctan2(Y, X)
    power = np.power(amplitude, 2)

    pca_2d = PCA(n_components=2)

    real_pca = pca_2d.fit_transform(X)
    imaginary_pca = pca_2d.fit_transform(Y)
    amplitude_pca = pca_2d.fit_transform(amplitude)
    phase_pca = pca_2d.fit_transform(phase)
    # phase_pca = pca_2d.fit_transform(spikes)
    power_pca = pca_2d.fit_transform(power)

    # real_pca = deriv.compute_fdmethod(X)
    # imaginary_pca = deriv.compute_fdmethod(Y)
    # amplitude_pca = deriv.compute_fdmethod(amplitude)
    # phase_pca = deriv.compute_fdmethod(phase)
    # power_pca = deriv.compute_fdmethod(power)

    # for spectral_feature in range(3, 8):
    # apply algorithm
    # for data_nr in range(3, 8):
    data = real_pca
    if data_nr == 3:
        data = real_pca
    if data_nr == 4:
        data = imaginary_pca
    if data_nr == 5:
        data = amplitude_pca
    if data_nr == 6:
        data = phase_pca
    if data_nr == 7:
        data = power_pca
    sp.plot(title="GT on STFT %s %s Sim_%d" % (w, cs.feature_extraction_methods[data_nr], sim_nr), X=data,
            labels=labels, marker='o')
    plt.savefig('./figures/stft_plots/STFT_%s_%s_Sim_%d' % (w, cs.feature_extraction_methods[data_nr], sim_nr))
    plt.show()

    alg_labels = [[], [], []]
    for alg_nr in range(0, 3):
        alg_labels[alg_nr] = bd.apply_algorithm(data, labels, alg_nr)
        sp.plot(title="%s on STFT %s %s coeff on Sim_%d" % (
            cs.algorithms[alg_nr], w, cs.feature_extraction_methods[data_nr], sim_nr), X=data,
                labels=alg_labels[alg_nr], marker='o')
        plt.savefig('./figures/stft_plots/%s_STFT_%s_%s_Sim_%d' % (
            cs.algorithms[alg_nr], w, cs.feature_extraction_methods[data_nr], sim_nr))
        plt.show()

    pe_labeled_data_results = [[], [], []]
    for a in range(0, 3):
        pe_labeled_data_results[a] = bd.benchmark_algorithm_labeled_data(labels, alg_labels[a])
        bd.print_benchmark_labeled_data(sim_nr, a, pe_labeled_data_results[a])
        bd.write_benchmark_labeled_data(sim_nr, cs.feature_extraction_methods[data_nr], pe_labeled_data_results)


def stft_on_time_segments():
    sim_nr = 1
    spikes, labels = ds.get_dataset_simulation(sim_nr, spike_length=79, align_to_peak=2, normalize_spike=False)
    w = 'hamming'
    nperseg = 40
    # data_nr between 3 and 7
    data_nr = 3

    # sampled_frequencies, time_segments, stft_signal = signal.stft(spikes, window=w, nperseg=42, noverlap=5)
    sampled_frequencies, time_segments, stft_signal = signal.stft(spikes, fs=1, window=w, nperseg=nperseg)
    print(len(time_segments))
    print(time_segments)
    print(len(sampled_frequencies))

    X = [x.real for x in stft_signal]
    X = np.array(X)
    Y = [x.imag for x in stft_signal]
    Y = np.array(Y)

    amplitude = np.abs(stft_signal)
    phase = np.arctan2(Y, X)
    power = np.power(amplitude, 2)

    pca_2d = PCA(n_components=2)
    #
    # real_pca = pca_2d.fit_transform(X)
    # imaginary_pca = pca_2d.fit_transform(Y)
    # amplitude_pca = pca_2d.fit_transform(amplitude)
    # phase_pca = pca_2d.fit_transform(phase)
    # power_pca = pca_2d.fit_transform(power)

    data = X
    if data_nr == 4:
        data = Y
    if data_nr == 5:
        data = amplitude
    if data_nr == 6:
        data = phase
    if data_nr == 7:
        data = power

    # this section is if we do not concatenate the aarays
    sp.plot(title="GT on Sim_" + str(sim_nr), X=pca_2d.fit_transform(spikes), labels=labels, marker='o')
    plt.show()

    # plt.plot(np.arange(79), spikes[20])
    # plt.show()
    # plt.plot(np.arange(79), amplitude[20])
    for i in range(0, len(time_segments)):
        level_stft = data[:, :, i]
        # plt.plot(np.arange(23), level_stft[20])
        signal_pca = pca_2d.fit_transform(level_stft)
        # signal_pca = deriv.compute_fdmethod(level_stft)
        sp.plot(title="GT on STFT_%d %s %d %s Sim_%d" % (i, w, nperseg, cs.feature_extraction_methods[data_nr],
                                                         sim_nr), X=signal_pca, labels=labels, marker='o')
        plt.savefig('./figures/stft_plots/sim%d_%s_stft_%d_%s' % (sim_nr, cs.feature_extraction_methods[data_nr], i, w))
        plt.show()
    plt.show()


def stft_level_apply_sbm():
    sim_nr = 4
    spikes, labels = ds.get_dataset_simulation(sim_nr, spike_length=79, align_to_peak=2, normalize_spike=False)
    w = 'blackman'
    # data_nr between 3 and 7
    fs = 1
    nperseg = 52

    # sampled_frequencies, time_segments, stft_signal = signal.stft(spikes, window=w, nperseg=nperseg, noverlap=5)
    sampled_frequencies, time_segments, stft_signal = signal.stft(spikes, window=w, fs=fs, nperseg=nperseg)
    print(len(time_segments))
    print(time_segments)
    print(len(sampled_frequencies))

    X = [x.real for x in stft_signal]
    X = np.array(X)
    Y = [x.imag for x in stft_signal]
    Y = np.array(Y)

    amplitude = np.abs(stft_signal)
    phase = np.arctan2(Y, X)
    power = np.power(amplitude, 2)

    pca_2d = PCA(n_components=2)

    # change data here!!
    data_nr = 5
    signal_pca = pca_2d.fit_transform(amplitude[:, :, 1])
    print(signal_pca.shape)
    sbm_labels = bd.apply_algorithm(signal_pca, labels, 2)
    sp.plot(title="SBM on STFT %s %s on Sim_%d" % (cs.feature_extraction_methods[data_nr], w, sim_nr),
            X=signal_pca, labels=sbm_labels, marker='o')
    plt.show()
    #
    sbm_results = bd.benchmark_algorithm_labeled_data(sbm_labels, labels)
    print(sbm_results)

    # alg_labels = [[], [], []]
    # for alg_nr in range(0, 3):
    #     alg_labels[alg_nr] = bd.apply_algorithm(signal_pca, labels, alg_nr)
    #
    # pe_labeled_data_results = [[], [], []]
    # for a in range(0, 3):
    #     pe_labeled_data_results[a] = bd.benchmark_algorithm_labeled_data(labels, alg_labels[a])
    #     bd.print_benchmark_labeled_data(sim_nr, a, pe_labeled_data_results[a])
    #     bd.write_benchmark_labeled_data(sim_nr, cs.feature_extraction_methods[data_nr], pe_labeled_data_results)


def plot_stft_spike():
    sim_nr = 15
    fs = 1
    # nperseg = 40
    nperseg = 52
    w = 'blackman'
    spikes, labels = ds.get_dataset_simulation(sim_nr, spike_length=79, align_to_peak=2, normalize_spike=False)
    f, t, Zxx = signal.stft(spikes, window=w, nperseg=nperseg, fs=fs)
    stft_signal = Zxx.reshape(*Zxx.shape[:1], -1)
    print(f)
    print(t)
    print(Zxx.shape)
    print(Zxx)

    spike_nr = 155
    plt.plot(stft_signal)  # this is interesting
    plt.plot(stft_signal[spike_nr])
    plt.show()
    plt.plot(np.arange(79), spikes[spike_nr])
    plt.show()
    plt.plot(np.abs(Zxx)[spike_nr])
    plt.show()
    # plt.pcolormesh(t, f, np.abs(Zxx)[5], cmap='jet')
    # plt.show()
    s = plt.imshow(np.abs(Zxx)[spike_nr], cmap='jet', aspect='auto', origin='lower')
    plt.colorbar(s)
    plt.xlabel('Time')
    plt.ylabel('Frequency')
    plt.title("Amplitude %s fs_%.2f nperseg_%d" % (w, fs, nperseg))
    plt.savefig('./figures/stft_plots/amplitude_fs_%d_nperseg_%d' % (fs, nperseg))
    plt.show()

    # plot spikes in the same cluster
    # for i in range(np.amax(labels) + 1):
    #     spikes_by_color = Zxx[labels == i]
    #     print(spikes_by_color.shape)
    #     for j in range(5):
    #         # im = plt.pcolormesh(t, f, np.abs(spikes_by_color)[j], cmap="jet")
    #         # plt.colorbar(im)
    #         # plt.title("Cluster %d spike %d " % (i, j))
    #         # plt.savefig('./figures/stft_plots/ft_cluster_%d_spike_%d' % (i, j))
    #         # plt.show()
    #         s = plt.imshow(np.abs(spikes_by_color[j]), cmap='jet', aspect='auto', origin='lower')
    #         plt.colorbar(s)
    #         plt.xlabel('Time')
    #         plt.ylabel('Frequency')
    #         plt.title("Amplitude %s fs_%.2f nperseg_%d" % (w, fs, nperseg))
    #         plt.savefig('./figures/stft_plots/c_%d_s_%d_amplitude_fs_%d_nperseg_%d' % (i, j, fs, nperseg))
    #         plt.show()


def stft_find_nperseg():
    sim_nr = 94
    spikes, labels = ds.get_dataset_simulation(sim_nr, spike_length=79, align_to_peak=2, normalize_spike=False)
    w_list = ['bartlett', 'blackman', 'blackmanharris', 'bohman', 'flattop', 'hamming', 'hann', 'parzen']
    dimensions = [3, 4, 5, 7]
    # dimensions = [4]
    w = 'bartlett'
    fs = 1
    # nperseg = 43
    data_nr = 7
    for data_nr in dimensions:
        for w in w_list:
            for nperseg in range(30, 50):
                sampled_frequencies, time_segments, Zxx = signal.stft(spikes, window=w, fs=fs, nperseg=nperseg)

                stft_signal = Zxx.reshape(*Zxx.shape[:1], -1)
                print(stft_signal.shape)

                X = [x.real for x in stft_signal]
                X = np.array(X)
                Y = [x.imag for x in stft_signal]
                Y = np.array(Y)
                # amplitude2 = np.sqrt(np.add(np.multiply(X, X), np.multiply(Y, Y)))
                amplitude = np.abs(stft_signal)
                phase = np.arctan2(Y, X)
                power = np.power(amplitude, 2)

                pca_2d = PCA(n_components=2)

                real_pca = pca_2d.fit_transform(X)
                imaginary_pca = pca_2d.fit_transform(Y)
                amplitude_pca = pca_2d.fit_transform(amplitude)
                phase_pca = pca_2d.fit_transform(phase)
                power_pca = pca_2d.fit_transform(power)

                # apply algorithm
                # for data_nr in range(3, 8):
                data = real_pca
                if data_nr == 3:
                    data = real_pca
                if data_nr == 4:
                    data = imaginary_pca
                if data_nr == 5:
                    data = amplitude_pca
                if data_nr == 6:
                    data = phase_pca
                if data_nr == 7:
                    data = power_pca
                # sp.plot(title="GT on STFT %s %s nperseg=%d on Sim_%d" % (
                #     w, cs.feature_extraction_methods[data_nr], nperseg, sim_nr), X=data, labels=labels, marker='o')
                # plt.savefig('./figures/stft_plots/STFT_%s_%s_%d_Sim_%d' % (
                #     cs.feature_extraction_methods[data_nr], w, nperseg, sim_nr))
                # plt.show()

                alg_labels = [[], [], []]
                for alg_nr in range(2, 3):
                    alg_labels[alg_nr] = bd.apply_algorithm(data, labels, alg_nr)
                    # sp.plot(title="%s on STFT %s %s coeff on Sim_%d" % (
                    #     cs.algorithms[alg_nr], w, cs.feature_extraction_methods[data_nr], sim_nr), X=data,
                    #         labels=alg_labels[alg_nr], marker='o')
                    # plt.savefig('./figures/stft_plots/%s_STFT_%s_%s_Sim_%d' % (
                    #     cs.algorithms[alg_nr], w, cs.feature_extraction_methods[data_nr], sim_nr))
                    # plt.show()

                pe_labeled_data_results = [[], [], []]
                for a in range(2, 3):
                    pe_labeled_data_results[a] = bd.benchmark_algorithm_labeled_data(labels, alg_labels[a])
                    if (pe_labeled_data_results[2] > [0.5, 0.5, 0.5, 0.5, 0]).all():
                        formatted_sbm = ["%.3f" % number for number in pe_labeled_data_results[2]]
                        row = [cs.feature_extraction_methods[data_nr], w, nperseg, stft_signal.shape[1],
                               formatted_sbm[0],
                               formatted_sbm[1], formatted_sbm[2], formatted_sbm[3]]
                        with open('./results/stft_sim_%s.csv' % sim_nr, 'a+',
                                  newline='') as file:
                            writer = csv.writer(file, delimiter=',')
                            writer.writerow(row)


def derivatives_with_stft(sim_nr, nperseg):
    """
        This function extracts the simulation data, apply stft, apply derivatives, concateates results and apply pca2d.
        Works only with the amplitude for now.
    :return: the SBM results for labeled data
    """
    # sim_nr = 1
    spikes, labels = ds.get_dataset_simulation(sim_nr, spike_length=79, align_to_peak=2, normalize_spike=False)

    w = 'blackman'
    fs = 2
    # nperseg = 48

    # sampled_frequencies, time_segments, Zxx = signal.stft(spikes, window=w, nperseg=nperseg, noverlap=5)
    sampled_frequencies, time_segments, Zxx = signal.stft(spikes, window=w, fs=fs, nperseg=nperseg)
    print(len(time_segments))
    print(len(sampled_frequencies))

    # stft_signal = Zxx.reshape(*Zxx.shape[:1], -1)
    # print(stft_signal.shape)

    amplitude = np.abs(Zxx)
    # print(amplitude.shape)
    # # on axis 1 - terrible results, on axis 2 are good results
    amplitude_deriv = np.apply_along_axis(deriv.compute_fdmethod_1spike, 2, amplitude)
    print(amplitude_deriv.shape)
    amplitude_concat = amplitude_deriv.reshape(*amplitude_deriv.shape[:1], -1)
    # print(amplitude_concat.shape)
    pca_2d = PCA(n_components=2)
    amplitude_pca = pca_2d.fit_transform(amplitude_concat)
    title = "Sim_%d GT %s %s %d" % (sim_nr, cs.feature_extraction_methods[5], w, nperseg)
    sp.plot(title=title, X=amplitude_pca, labels=labels, marker='o')
    plt.savefig('figures/stft_plots/%s' % title)
    plt.show()

    sbm_labels = bd.apply_algorithm(amplitude_pca, labels, 2)
    title_sbm = "SBM on STFT %s %s %d on Sim_%d" % (cs.feature_extraction_methods[5], w, nperseg, sim_nr);
    # sp.plot(title=title_sbm, X=amplitude_pca, labels=sbm_labels, marker='o')
    # plt.savefig('figures/stft_plots/%s' % title_sbm)
    # plt.show()

    sbm_results = bd.benchmark_algorithm_labeled_data(sbm_labels, labels)
    # bd.print_benchmark_labeled_data(sim_nr, 2, sbm_results)
    # sbm_results2 = bd.benchmark_algorithm_extra(sbm_labels, labels)
    # bd.print_benchmark_extra(sim_nr, 2, sbm_results2)
    # gt = bd.benchmark_algorithm_unlabeled_data(amplitude_pca, labels)

    return sbm_results


def loop_deriv_stft():
    """
        Writes the result of the stft with derivatives into csv files.
        Computes the average.
    :return:
    """
    for nperseg in range(52, 53):
        average = [0, 0, 0, 0, 0]
        # average = [0, 0, 0, ]
        simulation_counter = 0
        # nperseg = 49
        for i in range(1, 96):
            if i != 25 and i != 27 and i != 44:
                simulation_counter += 1
                results = derivatives_with_stft(sim_nr=i, nperseg=nperseg)
        #         average += results
        #         formatted = ["%.3f" % number for number in results]
        #         row = [i, formatted[0], formatted[1], formatted[2], formatted[3], formatted[4]]
        #         # row = [i, formatted[0], formatted[1], formatted[2]]
        #         with open('./results/stft_deriv_ampl_black-h_%d.csv' % nperseg, 'a+', newline='') as file:
        #             writer = csv.writer(file, delimiter=',')
        #             writer.writerow(row)
        # average = average / simulation_counter
        # with open('./results/stft_deriv_ampl_black-h_%d.csv' % nperseg, 'a+', newline='') as file:
        #     writer = csv.writer(file, delimiter=',')
        #     writer.writerow(average)


def generate_dataset_from_simulations2(simulations, simulation_labels, save=False, pca=False, stft=False, stftd=False,
                                       dpss=True, trial='0'):
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

    if pca:
        spikes_for_pca = spikes
        pca_2d = PCA(n_components=2)
        spikes_for_pca = pca_2d.fit_transform(spikes_for_pca)
        sp.plot("PCA for new sim trial_%s" % trial, spikes_for_pca, labels, marker='o')
        plt.savefig('figures/stft_plots/%s_pca' % trial)
        plt.show()

    if stft:
        spikes_for_stft = spikes
        sampled_frequencies, time_segments, Zxx = signal.stft(spikes_for_stft, window='blackman', fs=1, nperseg=52)
        amplitude = np.abs(Zxx)
        amplitude_concat = amplitude.reshape(*amplitude.shape[:1], -1)
        pca_2d = PCA(n_components=2)
        amplitude_pca = pca_2d.fit_transform(amplitude_concat)
        sp.plot("STFT GT for new sim trial_%s" % trial, amplitude_pca, labels, marker='o')
        plt.savefig('figures/stft_plots/%s_stft' % trial)
        plt.show()
    if stftd:
        spikes_for_stft = spikes
        sampled_frequencies, time_segments, Zxx = signal.stft(spikes_for_stft, window='blackman', fs=1, nperseg=45)
        amplitude = np.abs(Zxx)
        amplitude = np.apply_along_axis(deriv.compute_fdmethod_1spike, 2, amplitude)
        amplitude = amplitude.reshape(*amplitude.shape[:1], -1)
        pca_2d = PCA(n_components=2)
        amplitude_pca = pca_2d.fit_transform(amplitude)
        sp.plot("STFT D GT for new sim trial_%s" % trial, amplitude_pca, labels, marker='o')
        plt.savefig('figures/stft_plots/%s_stftd' % trial)
        plt.show()
    if dpss:
        spikes_for_stft = spikes
        w = stft_dpss.generate_stft_windows()
        w_nr = 2
        sampled_frequencies, time_segments, Zxx = signal.stft(spikes_for_stft, window=w[w_nr], fs=1, nperseg=79)
        amplitude = np.abs(Zxx)
        amplitude_concat = amplitude.reshape(*amplitude.shape[:1], -1)
        pca_2d = PCA(n_components=2)
        amplitude_pca = pca_2d.fit_transform(amplitude_concat)
        sp.plot("STFT dpss GT for new sim trial_%s" % trial, amplitude_pca, labels, marker='o')
        plt.savefig('figures/stft_plots/%s_stft_dpss_%d' % (trial, w_nr))
        plt.show()


# generate_dataset_from_simulations2([2, 15, 32], [[13], [8], [1]], trial=19)
generate_dataset_from_simulations2([1, 3, 5, 16], [[10, 13], [4], [3, 6, 15], [5, 7]], trial='24')
# stft_find_nperseg()
# stft_on_time_segments()
# stft_level_apply_sbm()
# plot_stft_spike()
# short_time_fourier_feature_extraction()
# derivatives_with_stft(10, 45)
# loop_deriv_stft()
