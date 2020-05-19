import benchmark_data as bd


def gui(sim_nr=1):
    bd.accuracy_all_algorithms_on_simulation(simulation_nr=sim_nr,
                                             feature_extract_method=3,
                                             plot=True,
                                             pe_labeled_data=True,
                                             pe_unlabeled_data=False,
                                             pe_extra=False,
                                             # save_folder='EMD',
                                             # title='IMF_derivatives_PCA2D',
                                             # save_folder='hilbert',
                                             title='hb_env_derivPCA2D',
                                             )


gui(74)
# for i in range(20, 30):
#     if i == 24 or i == 25 or i == 44:
#         continue
#     gui(i)

