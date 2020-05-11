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


def generate_stft_windows():
    M = 79  # 512
    NW = 2.5  # 2.5
    win, eigvals = signal.windows.dpss(M, NW, 4, return_ratios=True)
    fig, ax = plt.subplots(1)
    ax.plot(win.T, linewidth=1.)
    ax.set(xlim=[0, M - 1], ylim=[-0.2, 0.2], xlabel='Samples',
           title='DPSS, M=%d, NW=%0.1f' % (M, NW))
    ax.legend(['win[%d] (%0.4f)' % (ii, ratio)
               for ii, ratio in enumerate(eigvals)])
    fig.tight_layout()
    plt.show()
    return win


def stft_with_dpss_windows():
    win = generate_stft_windows()
    w_nr = 0
    sim_nr = 40
    spikes, labels = ds.get_dataset_simulation(sim_nr, spike_length=79, align_to_peak=2, normalize_spike=False)
    sampled_frequencies, time_segments, Zxx = signal.stft(spikes, window=win[w_nr], fs=1, nperseg=79)
    amplitude = np.abs(Zxx)
    amplitude_concat = amplitude.reshape(*amplitude.shape[:1], -1)
    pca_2d = PCA(n_components=2)
    amplitude_pca = pca_2d.fit_transform(amplitude_concat)
    sp.plot("STFT with dpss w%d" % w_nr, amplitude_pca, labels, marker='o')
    plt.savefig('figures/stft_plots/dpss_%d_sim_%d' % (w_nr, sim_nr))
    plt.show()


def save_all_dpss():
    pca_2d = PCA(n_components=2)
    win = generate_stft_windows()
    w_nr = 3
    for alg_nr in range(2, 3):
        average = [0, 0, 0, 0, 0]
        # average = [0, 0, 0, 0, 0, 0]
        simulation_counter = 0
        for sim_nr in range(1, 96):
            if sim_nr != 25 and sim_nr != 27 and sim_nr != 44:
                spikes, labels = ds.get_dataset_simulation(sim_nr, spike_length=79, align_to_peak=2,
                                                           normalize_spike=False)
                sampled_frequencies, time_segments, Zxx = signal.stft(spikes, window=win[w_nr], fs=1, nperseg=79)
                amplitude = np.abs(Zxx)
                amplitude = amplitude.reshape(*amplitude.shape[:1], -1)
                signal_pca = pca_2d.fit_transform(amplitude)
                sp.plot("Sim %d STFT with dpss w%d" % (sim_nr, w_nr), signal_pca, labels, marker='o')
                plt.savefig('figures/dpss/sim_%d_dpss_%d' % (sim_nr, w_nr))
                plt.show()
                alg_labels = bd.apply_algorithm(signal_pca, labels, alg_nr)
                results = bd.benchmark_algorithm_labeled_data(labels, alg_labels)
                # results = bd.benchmark_algorithm_extra(alg_labels, labels)
                simulation_counter += 1
                average += results
                formatted = ["%.3f" % number for number in results]
                row = [sim_nr, formatted[0], formatted[1], formatted[2], formatted[3], formatted[4]]
                # row = [sim_nr, formatted[0], formatted[1], formatted[2], formatted[3], formatted[4], formatted[5]]
                with open('./results/all_%s_dpss_%d.csv' % (cs.algorithms[alg_nr], w_nr), 'a+', newline='') as file:
                    writer = csv.writer(file, delimiter=',')
                    writer.writerow(row)
        average = average / simulation_counter
        with open('./results/all_%s_dpss_%d.csv' % (cs.algorithms[alg_nr], w_nr), 'a+', newline='') as file:
            writer = csv.writer(file, delimiter=',')
            writer.writerow(average)


# stft_dpss()
# stft_with_dpss_windows()
# save_all_dpss()
