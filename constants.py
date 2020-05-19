feature_extraction_methods = ["pca_2d", "pca_3d", "derivatives_2d", "hilbert"]
feature_space_dimensions = [2, 3, 2, 2]
algorithms = ["K-Means", "DBSCAN", "SBM"]
perf_eval_labeled_data_results = ["Adjusted_Rand_Index", "Adjusted_Mutual_Info", "Fowlkes_Msllows"]
perf_eval_extra_labeled_data_results = ["Homogenity", "Completeness", "V-score"]
perf_eval_unlabeled_data_results = ["Silhouette", "Calinski_Harabasz", "Davies_Bouldin"]

# constants for particular datasets
kmeansValues = [15, 15, 8, 6, 20]
epsValues = [27000, 45000, 18000, 0.5, 0.1]
pn = 25

LABEL_COLOR_MAP = {-1: 'gray',
                   0: 'white',
                   1: 'red',
                   2: 'blue',
                   3: 'green',
                   4: 'black',
                   5: 'yellow',
                   6: 'cyan',
                   7: 'magenta',
                   8: 'tab:purple',
                   9: 'tab:orange',
                   10: 'tab:brown',
                   11: 'tab:pink',
                   12: 'lime',
                   13: 'orchid',
                   14: 'khaki',
                   15: 'lightgreen',
                   16: 'orangered',
                   17: 'salmon',
                   18: 'silver',
                   19: 'yellowgreen',
                   20: 'royalblue',
                   21: 'beige',
                   22: 'crimson',
                   23: 'indigo',
                   24: 'darkblue',
                   25: 'gold',
                   26: 'ivory',
                   27: 'lavender',
                   28: 'lightblue',
                   29: 'olive',
                   30: 'sienna',
                   31: 'salmon',
                   32: 'teal',
                   33: 'turquoise',
                   34: 'wheat',
                   }
