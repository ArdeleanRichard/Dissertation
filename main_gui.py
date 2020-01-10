import benchmark_data as bd


def gui():
    bd.accuracy_all_algorithms_on_simulation(simulation_nr=1,
                                             feature_extract_method=0,
                                             plot=True,
                                             pe_labeled_data=True,
                                             pe_unlabeled_data=True,
                                             pe_extra=True)


gui()
