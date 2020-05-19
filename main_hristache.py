from sklearn.feature_selection import mutual_info_classif

import SBM
import datasets as ds
import spike_features


def hristache(simulation_nr):
    spikes, labels = ds.get_dataset_simulation(simulation_nr)
    features = spike_features.get_features(spikes)
    sbm_labels = SBM.parallel(features, pn=25, version=2)
    # res = dict(zip(["fd_max", "fd_min"],
    #                mutual_info_classif(features, sbm_labels, discrete_features="auto")
    #                ))
    # print(res)
    print(mutual_info_classif(features, sbm_labels, discrete_features="auto"))


hristache(21)
