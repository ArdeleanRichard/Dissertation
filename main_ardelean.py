import math
import os
import time

import matplotlib.pyplot as plt
import numpy as np
from sklearn import preprocessing, metrics
from sklearn.decomposition import PCA

from autoencoder import fft_input
from autoencoder import lstm_input
from autoencoder.benchmark import validate_model
from autoencoder.lstm_autoencoder import LSTMAutoencoderModel
from autoencoder.lstm_input import lstm_create_code_numpy, lstm_get_codes
from utils.dataset_parsing.datasets import stack_simulations_array
from utils.sbm import SBM, SBM_graph
from utils.dataset_parsing import datasets as ds
from utils import scatter_plot
from utils.constants import autoencoder_layer_sizes, autoencoder_code_size, lstm_layer_sizes, lstm_code_size
from autoencoder.model_auxiliaries import verify_output, get_codes, verify_random_outputs
from autoencoder.autoencoder import AutoencoderModel
import networkx as nx


def main(program, sub=""):
    # data = ds.getTINSData()
    # data, y = ds.getGenData()
    if program == "noise_isolation":
        for simulation_number in range(1, 96):
            if simulation_number == 25 or simulation_number == 44:
                continue
            spike_features, labels = ds.get_dataset_simulation_features(simulation_number)
            scatter_plot.plot('GT' + str(len(spike_features)), spike_features, labels, marker='o')
            plt.savefig(f'./figures/noise/gt_model_sim{simulation_number}')
    if program == "sbm":
        # data = ds.getTINSData()
        # data, y = ds.getGenData()
        data, y = ds.get_dataset_simulation_pca_2d(simNr=79, align_to_peak=True)

        print(data)

        pn = 25
        start = time.time()
        labels = SBM.parallel(data, pn, ccThreshold=5, version=2)
        end = time.time()
        print('SBM: ' + str(end - start))

        scatter_plot.plot_grid('SBM' + str(len(data)), data, pn, labels, marker='o')
        plt.savefig('./figures/SBMv2_sim79')

        scatter_plot.plot('GT' + str(len(data)), data, y, marker='o')
        plt.savefig('./figures/ground_truth')

        unique, counts = np.unique(labels, return_counts=True)
        print(dict(zip(unique, counts)))

        # plt.show()

    elif program == "autoencoder":
        simulation_number = 4
        spikes, labels = ds.get_dataset_simulation(simNr=simulation_number)
        print(spikes.shape)

        stack_simulations_array([1,2,3])

        autoencoder = AutoencoderModel(input_size=len(spikes[0]),
                                       encoder_layer_sizes=autoencoder_layer_sizes,
                                       decoder_layer_sizes=autoencoder_layer_sizes,
                                       code_size=20)
        # encoder, decoder = model.train(spikes, epochs=1000)
        # model.save_weights('./autoencoder/autoencoderCon_weights_1000')
        autoencoder.load_weights('./autoencoder/weights/autoencoderCon_weights_1000')
        encoder, decoder = autoencoder.return_encoder()

        verify_output(spikes, encoder, decoder)
        autoencoder_features = get_codes(spikes, encoder)

        scatter_plot.plot('GT' + str(len(autoencoder_features)), autoencoder_features, labels, marker='o')
        plt.savefig(f'./figures/autoencoder/gt_model_sim{simulation_number}')

        pn = 25
        labels = SBM.parallel(autoencoder_features, pn, ccThreshold=5, version=2)

        scatter_plot.plot_grid('SBM' + str(len(autoencoder_features)), autoencoder_features, pn, labels, marker='o')
        plt.savefig(f'./figures/autoencoder/gt_model_sim{simulation_number}_sbm')

    elif program == "autoencoder_sim_array":
        simulation_array = [4, 8, 79]

        spikes, labels = ds.stack_simulations_array(simulation_array)

        autoencoder = AutoencoderModel(input_size=len(spikes[0]),
                                       encoder_layer_sizes=autoencoder_layer_sizes,
                                       decoder_layer_sizes=autoencoder_layer_sizes,
                                       code_size=20)

        encoder, autoenc = autoencoder.return_encoder()

        if sub == "train":
            autoencoder.train(spikes, epochs=500)
            autoencoder.save_weights('./autoencoder/weights/autoencoder_array4-8-79_500')

            verify_output(spikes, encoder, autoenc)
            verify_random_outputs(spikes, encoder, autoenc, 10)
        elif sub == "test":

            autoencoder.load_weights('./autoencoder/weights/autoencoder_array4-8-79_500')

            for simulation_number in simulation_array:
                spikes, labels = ds.get_dataset_simulation(simNr=simulation_number)

                autoencoder_features = get_codes(spikes, encoder)

                scatter_plot.plot('GT' + str(len(autoencoder_features)), autoencoder_features, labels, marker='o')
                plt.savefig(f'./figures/autoencoder/gt_model_sim{simulation_number}')

                pn = 25
                labels = SBM.parallel(autoencoder_features, pn, ccThreshold=5, version=2)

                scatter_plot.plot_grid('SBM' + str(len(autoencoder_features)), autoencoder_features, pn, labels,
                                       marker='o')
                plt.savefig(f'./figures/autoencoder/gt_model_sim{simulation_number}_sbm')
        else:
            pass

    elif program == "autoencoder_sim_range":
        range_min = 1
        range_max = 96
        epochs = 100
        folder = ""
        inner_folder = ""
        # create_plot_folder()

        spikes, labels = ds.stack_simulations_range(range_min, range_max, True, True)

        autoencoder = AutoencoderModel(input_size=len(spikes[0]),
                                       encoder_layer_sizes=autoencoder_layer_sizes,
                                       decoder_layer_sizes=autoencoder_layer_sizes,
                                       code_size=autoencoder_code_size)

        encoder, autoenc = autoencoder.return_encoder()

        if sub == "train":
            autoencoder.train(spikes, epochs=epochs)
            autoencoder.save_weights(f'./autoencoder/weights/autoencoder_allsim_e100_d80_nonoise_c{autoencoder_code_size}')

            verify_output(spikes, encoder, autoenc, path=f"./figures/autoencoder_c{autoencoder_code_size}/spike_verif")
            verify_random_outputs(spikes, encoder, autoenc, 10, path=f"./figures/autoencoder_c{autoencoder_code_size}/spike_verif")
        elif sub == "test":
            autoencoder.load_weights(f'./autoencoder/weights/autoencoder_allsim_e100_d80_nonoise_c{autoencoder_code_size}')

            for simulation_number in range(range_min, range_max):
                if simulation_number == 25 or simulation_number == 44:
                    continue
                spikes, labels = ds.get_dataset_simulation(simNr=simulation_number)

                spikes = spikes[labels != 0]
                labels = labels[labels != 0]

                autoencoder_features = get_codes(spikes, encoder)

                pca_2d = PCA(n_components=2)
                autoencoder_features = pca_2d.fit_transform(autoencoder_features)

                scatter_plot.plot('GT' + str(len(autoencoder_features)), autoencoder_features, labels, marker='o')
                plt.savefig(f'./figures/autoencoder_c{autoencoder_code_size}/gt_model_sim{simulation_number}')

                pn = 25
                labels = SBM.parallel(autoencoder_features, pn, ccThreshold=5, version=2)

                scatter_plot.plot_grid('SBM' + str(len(autoencoder_features)), autoencoder_features, pn, labels,
                                       marker='o')
                plt.savefig(f'./figures/autoencoder_c{autoencoder_code_size}/gt_model_sim{simulation_number}_sbm')
        elif sub == "pre":
            autoencoder_layer_sizes.append(autoencoder_code_size)
            layer_weights = autoencoder.pre_train(spikes, autoencoder_layer_sizes, epochs=100)
            autoencoder.set_weights(layer_weights)

            autoencoder.train(spikes, epochs=epochs)
            autoencoder.save_weights(f'./autoencoder/autoencoder_allsim_e100_d80_nonoise_c{autoencoder_code_size}_pt')
            autoencoder.load_weights(f'./autoencoder/autoencoder_allsim_e100_d80_nonoise_c{autoencoder_code_size}_pt')

            verify_output(spikes, encoder, autoenc, path=f"./figures/autoencoder_c{autoencoder_code_size}_pt/spike_verif")
            verify_random_outputs(spikes, encoder, autoenc, 10, path=f"./figures/autoencoder_c{autoencoder_code_size}_pt/spike_verif")

            for simulation_number in range(range_min, range_max):
                if simulation_number == 25 or simulation_number == 44:
                    continue
                spikes, labels = ds.get_dataset_simulation(simNr=simulation_number)

                spikes = spikes[labels != 0]
                labels = labels[labels != 0]

                autoencoder_features = get_codes(spikes, encoder)

                pca_2d = PCA(n_components=2)
                autoencoder_features = pca_2d.fit_transform(autoencoder_features)

                scatter_plot.plot('GT' + str(len(autoencoder_features)), autoencoder_features, labels, marker='o')
                plt.savefig(f'./figures/autoencoder_c{autoencoder_code_size}_pt/gt_model_sim{simulation_number}')

                pn = 25
                labels = SBM.parallel(autoencoder_features, pn, ccThreshold=5, version=2)

                scatter_plot.plot_grid('SBM' + str(len(autoencoder_features)), autoencoder_features, pn, labels,
                                       marker='o')
                plt.savefig(f'./figures/autoencoder_c{autoencoder_code_size}_pt/gt_model_sim{simulation_number}_sbm')

        else:
            pass

    elif program == "benchmark":
        autoencoder_layers = [
            [70, 60, 50, 40, 30, 20, 10],
            [70, 60, 50, 40, 30, 20],
            [70, 60, 50, 40, 30],
            [70, 60, 50, 40],
            [70, 60, 50]
        ]

        results = []
        for layers in autoencoder_layers:
            code_results = validate_model(layers, pt=True)
            code_results = np.array(code_results)
            results.append(code_results)

        results = np.stack(results, axis=1)

        results = np.array(results)
        codes_ari = []
        codes_ami = []

        for simulation_result in results:
            #np.argmax(simulation_result[:, 0]) - index of best code for ARI in that simulation
            #np.argmax(simulation_result[:, 1]) - index of best code for AMI in that simulation
            codes_ari.append(((np.argmax(simulation_result[:, 0]) + 1) * 10))
            codes_ami.append(((np.argmax(simulation_result[:, 1]) + 1) * 10))

        codes, counts = np.unique(codes_ari, return_counts=True)
        codes_n_counts_ari = dict(zip(codes, counts))
        codes, counts = np.unique(codes_ami, return_counts=True)
        codes_n_counts_ami = dict(zip(codes, counts))


        for i in range(0, len(autoencoder_layers)):
            ari_ami_values_for_all_sim = results[:, i]
            mean_values = np.mean(ari_ami_values_for_all_sim, axis=0)
            print(f"CODE {(i+1)*10} -> ARI - {mean_values[0]}")
            print(f"CODE {(i+1)*10} -> AMI - {mean_values[1]}")


        print(codes_n_counts_ari)
        print(codes_n_counts_ami)

        # np.savetxt("./autoencoder/test.csv", results, delimiter=",", fmt='%.2f')

    elif program == "pipeline_test":
        range_min = 1
        range_max = 96
        autoencoder_layers = [70, 60, 50, 40, 30, 20]

        autoencoder = AutoencoderModel(input_size=79,
                                       encoder_layer_sizes=autoencoder_layers[:-1],
                                       decoder_layer_sizes=autoencoder_layers[:-1],
                                       code_size=autoencoder_layers[-1])

        encoder, autoenc = autoencoder.return_encoder()

        autoencoder.load_weights(f'./autoencoder/weights/autoencoder_allsim_e100_d80_nonoise_c20')

        for simulation_number in range(range_min, range_max):
            if simulation_number == 25 or simulation_number == 44:
                continue
            spikes, labels = ds.get_dataset_simulation(simNr=simulation_number)

            spikes = spikes[labels != 0]
            labels = labels[labels != 0]

            autoencoder_features = get_codes(spikes, encoder)

            pca_2d = PCA(n_components=2)
            autoencoder_features = pca_2d.fit_transform(autoencoder_features)

            sil_coeffs = metrics.silhouette_samples(autoencoder_features, labels, metric='mahalanobis')
            means = []
            for label in np.arange(max(labels) + 1):
                if label not in labels:
                    means.append(-1)
                else:
                    means.append(sil_coeffs[labels == label].mean())
            for label in np.arange(max(labels) + 1):
                if means[label] > 0.7:
                    print(f"SIM{simulation_number} separates {label}")

    elif program == "lstm":
        range_min = 1
        range_max = 2
        epochs = 100
        timesteps = 20

        spike_verif_path = f'./figures/lstm_c{lstm_code_size}/spike_verif/'
        plot_path = f'./figures/lstm_c{lstm_code_size}/'
        weights_path = f'./autoencoder/weights/lstm_allsim_e100_d80_nonoise_c{lstm_code_size}'
        if not os.path.exists(spike_verif_path):
            os.makedirs(spike_verif_path)
        if not os.path.exists(plot_path):
            os.makedirs(plot_path)

        # spikes, labels = ds.stack_simulations_range(range_min, range_max, True, True)
        # spikes = np.reshape(spikes, (spikes.shape[0], spikes.shape[1], 1))

        spikes, labels = ds.stack_simulations_range(range_min, range_max, True, True)
        # spikes, labels = lstm_input.temporalize_spikes(spikes, labels, timesteps)
        spikes = lstm_input.temporalize_spikes(spikes, timesteps)

        autoencoder = LSTMAutoencoderModel(input_size=spikes.shape,
                                       lstm_layer_sizes=lstm_layer_sizes,
                                       code_size=lstm_code_size)

        encoder, autoenc = autoencoder.return_encoder()

        if sub == "train":
            autoencoder.train(spikes, epochs=epochs)
            autoencoder.save_weights(weights_path)

            # verify_output(spikes, encoder, autoenc, path=spike_verif_path)
            # verify_random_outputs(spikes, encoder, autoenc, 10, path=spike_verif_path)
        elif sub == "test":
            autoencoder.load_weights(weights_path)

            for simulation_number in range(range_min+1, range_max+9):
                if simulation_number == 25 or simulation_number == 44:
                    continue
                spikes, labels = ds.get_dataset_simulation(simNr=simulation_number)
                # spikes, labels = lstm_input.temporalize_data(spikes, labels, timesteps)
                spikes = lstm_input.temporalize_spikes(spikes, timesteps)

                autoencoder_features = lstm_get_codes(spikes, encoder, timesteps)

                pca_2d = PCA(n_components=2)
                autoencoder_features = pca_2d.fit_transform(autoencoder_features)

                scatter_plot.plot('GT' + str(len(autoencoder_features)), autoencoder_features, labels, marker='o')
                plt.savefig(plot_path + f'gt_model_sim{simulation_number}')

                pn = 25
                labels = SBM.parallel(autoencoder_features, pn, ccThreshold=5, version=2)

                scatter_plot.plot_grid('SBM' + str(len(autoencoder_features)), autoencoder_features, pn, labels,
                                       marker='o')
                plt.savefig(plot_path + f'gt_model_sim{simulation_number}_sbm')

    elif program == "split_lstm":
        range_min = 1
        range_max = 2
        epochs = 100
        timesteps = 20
        overlap = 10

        spike_verif_path = f'./figures/lstm_c{lstm_code_size}_TS{timesteps}_OL{overlap}/spike_verif/'
        plot_path = f'./figures/lstm_c{lstm_code_size}_TS{timesteps}_OL{overlap}/'
        weights_path = f'./autoencoder/weights/lstm_allsim_e100_d80_nonoise_c{lstm_code_size}_TS{timesteps}_OL{overlap}'
        if not os.path.exists(spike_verif_path):
            os.makedirs(spike_verif_path)
        if not os.path.exists(plot_path):
            os.makedirs(plot_path)


        train_spikes, train_labels, test_spikes, test_labels = ds.stack_simulations_split_train_test(range_min, range_max, False, True)
        train_spikes = lstm_input.temporalize_spikes(train_spikes, timesteps, overlap)

        autoencoder = LSTMAutoencoderModel(input_size=train_spikes.shape,
                                           lstm_layer_sizes=lstm_layer_sizes,
                                           code_size=lstm_code_size)

        encoder, autoenc = autoencoder.return_encoder()

        if sub == "train":
            # autoencoder.train(train_spikes, epochs=epochs)
            # autoencoder.save_weights(weights_path)
            autoencoder.load_weights(weights_path)

            lstm_input.lstm_verify_output(train_spikes, timesteps, encoder, autoenc, path=spike_verif_path)
            lstm_input.lstm_verify_random_outputs(train_spikes, timesteps, encoder, autoenc, 10, path=spike_verif_path)
        elif sub == "test":
            autoencoder.load_weights(weights_path)

            sim_list_index = range_min
            for simulation_number in range(range_min, range_max):
                if simulation_number == 25 or simulation_number == 44:
                    continue

                spikes = test_spikes[sim_list_index-1]
                labels = test_labels[sim_list_index-1]

                spikes = lstm_input.temporalize_spikes(spikes, timesteps, overlap)

                autoencoder_features = lstm_get_codes(spikes, encoder, timesteps)

                pca_2d = PCA(n_components=2)
                autoencoder_features = pca_2d.fit_transform(autoencoder_features)

                scatter_plot.plot(f'GT{len(autoencoder_features)}/{len(train_spikes)+len(spikes)}', autoencoder_features, labels, marker='o')
                plt.savefig(plot_path + f'gt_model_sim{simulation_number}')

                pn = 25
                try:
                    labels = SBM.parallel(autoencoder_features, pn, ccThreshold=5, version=2)

                    scatter_plot.plot_grid('SBM' + str(len(autoencoder_features)), autoencoder_features, pn, labels,
                                           marker='o')
                    plt.savefig(plot_path + f'gt_model_sim{simulation_number}_sbm')
                except KeyError:
                    pass

                sim_list_index += 1

    elif program == "lstm_pca_check":
        plot_path = f'./figures/lstm_pca_check/'
        if not os.path.exists(plot_path):
            os.makedirs(plot_path)
        for simulation_number in range(1, 96):
            if simulation_number == 25 or simulation_number == 44:
                continue
            spikes, labels = ds.get_dataset_simulation(simNr=simulation_number, align_to_peak=True)

            check_spikes = spikes[:, 20:40]
            print(len(check_spikes[0]))

            pca_2d = PCA(n_components=2)
            new_features = pca_2d.fit_transform(check_spikes)

            scatter_plot.plot(f'GT{len(new_features)}', new_features,
                              labels, marker='o')
            plt.savefig(plot_path + f'check_sim{simulation_number}')



    elif program == "sbm_graph":
        simulation_number = 4
        data, y = ds.get_dataset_simulation(simNr=simulation_number)

        # dims = 2
        for dims in range(2, 3):
            print(dims)
            pca_2d = PCA(n_components=dims)

            pca_2d = PCA(n_components=dims)
            spikes = pca_2d.fit_transform(data)

            # scatter_plot.plot('GT-' + str(len(spikes)), spikes, labels, marker='o')
            # plt.savefig('./figures/sim4_gt')

            pn = 5
            # start = time.time()
            # labels = SBM.best(spikes, pn)
            # print(f"SBMog: {time.time() - start}")
            # scatter_plot.plot_grid('SBM-' + str(len(spikes)), spikes, pn, labels, marker='o')
            # plt.savefig('./figures/sim4_sbm')

            start = time.time()
            labels = SBM_graph.SBM(spikes, pn)
            print(f"SBMv2: {time.time() - start}")

            # sum_time = 0
            # nr = 20
            # for test in range(nr):
            #     start = time.time()
            #     labels = SBM_graph.SBM(spikes, pn)
            #     sum_time += time.time() - start
            # print(f"SBMv2: {sum_time/nr}")

            pca_2d = PCA(n_components=2)
            spikes = pca_2d.fit_transform(data)

            scatter_plot.plot('GT-' + str(len(spikes)), spikes, y, marker='o')
            plt.savefig('./figures/sim4_gt')
            scatter_plot.plot_grid('SBMv2-' + str(len(spikes)), spikes, pn, labels, marker='o')
            plt.savefig(f'./figures/sim4_sbmv2_dim{dims}')

            print()

def get_type(on_type, fft_real, fft_imag):
    if on_type == "real":
        spikes = fft_real
    elif on_type == "imag":
        spikes = fft_imag
    elif on_type == "magnitude":
        spikes = np.sqrt(fft_real * fft_real + fft_imag * fft_imag)
    elif on_type == "power":
        spikes = fft_real * fft_real + fft_imag * fft_imag
    elif on_type == "phase":
        spikes = np.arctan2(fft_imag, fft_real)
    elif on_type == "concatened":
        power = fft_real * fft_real + fft_imag * fft_imag
        phase = np.arctan2(fft_imag, fft_real)
        spikes = np.concatenate(power, phase)

    return spikes

def main_fft(program, case, alignment, on_type):
    range_min = 1
    range_max = 96
    epochs = 100
    spike_verif_path = f'./figures/fft/c{autoencoder_code_size}/{on_type}/{"wA" if alignment else "woA"}/{case}/spike_verif'
    plot_path = f'./figures/fft/c{autoencoder_code_size}/{on_type}/{"wA" if alignment else "woA"}/{case}/'
    weights_path = f'./autoencoder/weights/autoencoder_allsim_e100_d80_nonoise_c{autoencoder_code_size}_fft-{on_type}-{case}_{"wA" if alignment else "woA"}'

    if not os.path.exists(spike_verif_path):
        os.makedirs(spike_verif_path)
    if not os.path.exists(plot_path):
        os.makedirs(plot_path)

    fft_real, fft_imag = fft_input.apply_fft_on_range(case, alignment, range_min, range_max)

    fft_real = np.array(fft_real)
    fft_imag = np.array(fft_imag)

    spikes = get_type(on_type, fft_real, fft_imag)

    spikes = np.array(spikes)

    autoencoder = AutoencoderModel(input_size=len(spikes[0]),
                                   encoder_layer_sizes=autoencoder_layer_sizes,
                                   decoder_layer_sizes=autoencoder_layer_sizes,
                                   code_size=autoencoder_code_size)

    encoder, autoenc = autoencoder.return_encoder()

    if program == "train":
        autoencoder.train(spikes, epochs=epochs)
        autoencoder.save_weights(weights_path)

        verify_output(spikes, encoder, autoenc, path=spike_verif_path)
        verify_random_outputs(spikes, encoder, autoenc, 10, path=spike_verif_path)

    if program == "test":
        autoencoder.save_weights(weights_path)

        for simulation_number in range(range_min, range_max):
            if simulation_number == 25 or simulation_number == 44:
                continue

            fft_real, fft_imag, labels = fft_input.apply_fft_on_sim(sim_nr=simulation_number, case=case,
                                                                    alignment=alignment)
            fft_real = np.array(fft_real)
            fft_imag = np.array(fft_imag)

            spikes = get_type(on_type, fft_real, fft_imag)

            spikes = spikes[labels != 0]
            labels = labels[labels != 0]

            autoencoder_features = get_codes(spikes, encoder)

            pca_2d = PCA(n_components=2)
            autoencoder_features = pca_2d.fit_transform(autoencoder_features)

            scatter_plot.plot('GT' + str(len(autoencoder_features)), autoencoder_features, labels, marker='o')
            plt.savefig(plot_path + f'gt_model_sim{simulation_number}')

            pn = 25
            labels = SBM.parallel(autoencoder_features, pn, ccThreshold=5, version=2)

            scatter_plot.plot_grid('SBM' + str(len(autoencoder_features)), autoencoder_features, pn, labels,
                                   marker='o')
            plt.savefig(plot_path + f'gt_model_sim{simulation_number}_sbm')



def main_fft_windowed(program, alignment, on_type):
    range_min = 1
    range_max = 96
    epochs = 100
    spike_verif_path = f'./figures/fft_windowed/c{autoencoder_code_size}/{on_type}/align{alignment}/spike_verif'
    plot_path = f'./figures/fft_windowed/c{autoencoder_code_size}/{on_type}/align{alignment}/'
    weights_path = f'./autoencoder/weights/autoencoder_allsim_e100_d80_nonoise_c{autoencoder_code_size}_fft_windowed-{on_type}_align{alignment}'

    if not os.path.exists(spike_verif_path):
        os.makedirs(spike_verif_path)
    if not os.path.exists(plot_path):
        os.makedirs(plot_path)

    fft_real, fft_imag = fft_input.apply_fft_windowed_on_range(alignment, range_min, range_max)

    fft_real = np.array(fft_real)
    fft_imag = np.array(fft_imag)

    spikes = get_type(on_type, fft_real, fft_imag)
    spikes = np.array(spikes)

    autoencoder = AutoencoderModel(input_size=len(spikes[0]),
                                   encoder_layer_sizes=autoencoder_layer_sizes,
                                   decoder_layer_sizes=autoencoder_layer_sizes,
                                   code_size=autoencoder_code_size)

    encoder, autoenc = autoencoder.return_encoder()

    if program == "train":
        autoencoder.train(spikes, epochs=epochs)
        autoencoder.save_weights(weights_path)

        verify_output(spikes, encoder, autoenc, path=spike_verif_path)
        verify_random_outputs(spikes, encoder, autoenc, 10, path=spike_verif_path)

    if program == "test":
        autoencoder.save_weights(weights_path)

        for simulation_number in range(range_min, range_max):
            if simulation_number == 25 or simulation_number == 44 or simulation_number== 78:
                continue

            fft_real, fft_imag, labels = fft_input.apply_fft_windowed_on_sim(sim_nr=simulation_number,
                                                                    alignment=alignment)
            fft_real = np.array(fft_real)
            fft_imag = np.array(fft_imag)

            spikes = get_type(on_type, fft_real, fft_imag)
            spikes = np.array(spikes)

            spikes = spikes[labels != 0]
            labels = labels[labels != 0]

            autoencoder_features = get_codes(spikes, encoder)

            pca_2d = PCA(n_components=2)
            autoencoder_features = pca_2d.fit_transform(autoencoder_features)

            scatter_plot.plot('GT' + str(len(autoencoder_features)), autoencoder_features, labels, marker='o')
            plt.savefig(plot_path + f'gt_model_sim{simulation_number}')

            pn = 25
            labels = SBM.parallel(autoencoder_features, pn, ccThreshold=5, version=2)
            try:
                scatter_plot.plot_grid('SBM' + str(len(autoencoder_features)), autoencoder_features, pn, labels,
                                       marker='o')
                plt.savefig(plot_path + f'gt_model_sim{simulation_number}_sbm')
            except KeyError:
                pass

def test_fft_windowed():
    def verify_output(spikes, windowed_spikes, i=0, path=""):
        plt.plot(np.arange(len(spikes[i])), spikes[i])
        plt.plot(np.arange(len(windowed_spikes[i])), windowed_spikes[i])
        plt.xlabel('Time')
        plt.ylabel('Magnitude')
        plt.title(f"Verify spike {i}")
        plt.savefig(f'{path}/spike{i}')
        plt.show()

    def verify_random_outputs(spikes, windowed_spikes, verifications=0, path=""):
        random_list = np.random.choice(range(len(spikes)), verifications, replace=False)

        for random_index in random_list:
            verify_output(spikes, windowed_spikes, random_index, path)

    for alignment in [True, False, 2, 3]:
        plot_path = f'./figures/fft/blackman_test/align{alignment}'

        if not os.path.exists(plot_path):
            os.makedirs(plot_path)

        spikes, labels = ds.get_dataset_simulation(simNr=1, align_to_peak=alignment)

        windowed_spikes = fft_input.apply_blackman_window(spikes)

        # verify_random_outputs(spikes, windowed_spikes, 10, plot_path)
        verify_output(spikes, windowed_spikes, 685, path=plot_path)
        verify_output(spikes, windowed_spikes, 2171, path=plot_path)
        verify_output(spikes, windowed_spikes, 2592, path=plot_path)

# main("autoencoder_sim_array", sub="train")
# main("autoencoder_sim_array", sub="test")
# main("autoencoder_sim_range", sub="train")
# main("autoencoder_sim_range", sub="test")
# main("autoencoder_sim_range", sub="pre")
# main("benchmark", sub="")
# main("pipeline_test", sub="")
# main("sbm_graph", sub="")

# case, alignment, on_type
# for alignment in [True, False]:
#     for case in ["original", "padded", "rolled", "reduced"]:
#         for on_type in ["real", "imag", "magnitude"]:
#             main_fft("train", case, alignment, on_type)
#             main_fft("test", case, alignment, on_type)

# main("lstm", sub="train")
# main("lstm", sub="test")
# main("split_lstm", sub="train")
# main("split_lstm", sub="test")
# main("lstm_pca_check", sub="")

# test_fft_windowed()

# case, alignment, on_type
# for alignment in [True, False, 2, 3]:
#     for on_type in ["real", "imag", "magnitude", "power", "phase", "concatenated"]:
#         main_fft_windowed("train", alignment, on_type)
#         main_fft_windowed("test", alignment, on_type)

for alignment in [3]:
    main_fft_windowed("train", alignment, "power")
    main_fft_windowed("test", alignment, "power")
