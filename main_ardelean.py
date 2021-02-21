import math
import time

import matplotlib.pyplot as plt
import numpy as np
from sklearn import preprocessing, metrics
from sklearn.decomposition import PCA

from autoencoder.benchmark import validate_model
from utils.dataset_parsing.datasets import stack_simulations_array
from utils.sbm import SBM, SBM_graph
from utils.dataset_parsing import datasets as ds
from utils import scatter_plot
from utils.constants import autoencoder_layer_sizes, autoencoder_code_size
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


        autoencoder = AutoencoderModel(encoder_layer_sizes=autoencoder_layer_sizes,
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

        autoencoder = AutoencoderModel(encoder_layer_sizes=autoencoder_layer_sizes,
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

        autoencoder = AutoencoderModel(encoder_layer_sizes=autoencoder_layer_sizes,
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

        autoencoder = AutoencoderModel(encoder_layer_sizes=autoencoder_layers[:-1],
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



# main("autoencoder_sim_array", sub="train")
# main("autoencoder_sim_array", sub="test")
# main("autoencoder_sim_range", sub="train")
# main("autoencoder_sim_range", sub="test")
# main("autoencoder_sim_range", sub="pre")
# main("benchmark", sub="")
# main("pipeline_test", sub="")
main("sbm_graph", sub="")
