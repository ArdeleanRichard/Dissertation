import benchmark_data as bd
import datasets as ds
import scatter_plot
import matplotlib.pyplot as plt
import derivatives as deriv
import wavelets as wt
import numpy as np
from sklearn.decomposition import PCA
import plotly.express as px


def main():
    sim_nr = 20
    X, y = ds.apply_feature_extraction_method(sim_nr, 2)
    scatter_plot.plot("Ground truth for Sim_" + str(sim_nr), X, y, marker='o')
    plt.show()

    # for sim_nr in np.arange(11, 21):
    #     X, y = ds.apply_feature_extraction_method(sim_nr, 2)
    #     scatter_plot.plot("Ground truth for Sim_" + str(sim_nr), X, y, marker='o')
    #     plt.savefig("Ground_Truth_Sim_%d"%sim_nr)

    # bd.accuracy_all_algorithms_on_simulation(simulation_nr=20,
    #                                          feature_extract_method=2,
    #                                          plot=True,
    #                                          pe_labeled_data=True,
    #                                          pe_unlabeled_data=False,
    #                                          pe_extra=True)

    # avg = bd.accuracy_all_algorithms_on_multiple_simulations(11, 20, feature_extract_method=2)
    # print(avg)

    # spikes, y = ds.get_dataset_simulation(16, 79, True, True)
    #
    # result_spikes1 = wt.returncoeffs(spikes)
    # # apply pca
    # pca_3d = PCA(n_components=3)
    # X = pca_3d.fit_transform(result_spikes1)
    # fig = px.scatter_3d(X, x=X[:, 0], y=X[:, 1], z=X[:, 2], color=y)
    # fig.update_layout(title="Ground truth for Sim_" + str(15))
    # fig.show()

    # wt.test()
    # wt.logarithmiccwt(spikes, 0, log=True)
    # wt.cwt(spikes, 8)
    # wt.cwt(spikes, 10)
    # wt.cwt(spikes, 20)
    # wt.cwt(spikes, 42)
    # wt.cwt(spikes, 56)
    # for j in np.arange(0, max(labels)):
    #     label = j
    #     flag = 0
    #     print("######################################")
    #     print("Label %d:" % j)
    #     for i in np.arange(0, len(labels)):
    #         if labels[i] == label:
    #             flag = flag + 1
    #             print(i)
    #         if flag == 5:
    #             break


main()
