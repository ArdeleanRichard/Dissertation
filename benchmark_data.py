import csv
import sys

import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans

sys.setrecursionlimit(100000)

import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

import SBM
import datasets as ds
import scatter_plot

algName = ["K-MEANS", "DBSCAN", "SBM"]
files = ["s1_labeled.csv", "s2_labeled.csv", "unbalance.csv"]
kmeansValues = [15, 15, 8, 6, 20]
epsValues = [27000, 45000, 18000, 0.5, 0.1]
pn = 25

dataName = ["S1", "S2", "U", "UO", "Simulation"]
algorithmNames = ["K-MEANS", "K-MEANS", "K-MEANS", "K-MEANS", "DBSCAN", "SBM", ]
settings = ["ARI", "AMI", "ARI", "AMI", "NNP", "NNP"]
table = [algName]


# datasetNumber = 0 => S1
# datasetNumber = 1 => S2
# datasetNumber = 2 => U
# datasetNumber = 3 => UO
# datasetNumber = 4 => Sim97
def benchmark_dataset(datasetNumber, plot=False):
    """
    Benchmarks K-Means, DBSCAN and SBM on one of 5 selected datasets
    :param datasetNumber: integer - the number that represents one of the datasets (0-4)
    :param plot: boolean - optional, whether the plot function should be called or not (for ease of use)
    :returns None
    """
    print("DATASET: " + dataName[datasetNumber])
    datasetName = dataName[datasetNumber]
    simulation_number = 10
    if datasetNumber < 3:
        X = np.genfromtxt("./datasets/" + files[datasetNumber], delimiter=",")
        X, y = X[:, [0, 1]], X[:, 2]
    elif datasetNumber == 3:
        X, y = ds.getGenData()
    else:
        X, y = ds.getDatasetSim79()

    # S2 has label problems
    if datasetNumber == 1:
        for k in range(len(X)):
            y[k] = y[k] - 1

    kmeans = KMeans(n_clusters=kmeansValues[datasetNumber]).fit(X)
    labels = kmeans.labels_
    scatter_plot.plotFunction("K-MEANS on" + datasetName, X, labels, plot, marker='o')
    plt.show()
    calculateAccuracy(datasetName, 0, labels, y, print=True)


    if datasetNumber == 1:
        min_samples = np.log(len(X)) * 10
    else:
        min_samples = np.log(len(X))
    db = DBSCAN(eps=epsValues[datasetNumber], min_samples=min_samples).fit(X)
    labels = db.labels_
    scatter_plot.plotFunction("DBSCAN on" + datasetName, X, labels, plot, marker='o')
    plt.show()
    calculateAccuracy(datasetName, 1, labels, y, print=True)


    labels = SBM.multiThreaded(X, pn=25, version=2)
    scatter_plot.plotFunction("SBM on" + datasetName, X, labels, plot, marker='o')
    plt.show()
    calculateAccuracy(datasetName, 2, labels, y, print=True)


    # results = []
    # results.append(metrics.adjusted_rand_score(y, labels))
    # results.append(metrics.adjusted_mutual_info_score(labels, y))
    #
    # # start of the NO-NOISE-POINTS (NNP) setting
    # # we calculate only the accuracy of points that have been clustered(labeled as non-noise)
    # adj = labels > 0
    # yNN = y[adj]
    # labelsNN = labels[adj]
    #
    # results.append(metrics.adjusted_rand_score(yNN, labelsNN))
    # results.append(metrics.adjusted_mutual_info_score(labelsNN, yNN))
    #
    # print(results)


def printAccuracy(datasetName, algorithmNumber, allARI, allAMI, nnpARI, nnpAMI):
    """
    Print the accuracies of the algorithm on the dataset
    :param datasetName: string - the name of the dataset for ease of view
    :param algorithmNumber: integer - number of the algorithm (0 = K-Means, 1=DBSCAN, 2=SBM)
    :param allARI: float - the accuracy of the selected algorithm on the ALL setting by the Adjusted Rand Index
    :param allAMI: float - the accuracy of the selected algorithm on the ALL setting by the Adjusted Mutual Information
    :param nnpARI: float - the accuracy of the selected algorithm on the NNP setting by the Adjusted Rand Index
    :param nnpAMI: float - the accuracy of the selected algorithm on the NNP setting by the Adjusted Mutual Information
    :returns None
    """
    print('ALL SETTING')
    print(datasetName + " - " + algName[algorithmNumber] + " - " + "ARI:" + str(allARI))
    print(datasetName + " - " + algName[algorithmNumber] + " - " + "AMI:" + str(allAMI))

    print('NNP SETTING')
    print(datasetName + " - " + algName[algorithmNumber] + " - " + "ARI:" + str(nnpARI))
    print(datasetName + " - " + algName[algorithmNumber] + " - " + "AMI:" + str(nnpAMI))


def calculateAccuracy(datasetName, algorithmNumber, labels, y, print=False):
    """
    Calculate the accuracies of the algorithm on the dataset
    :param datasetName: string - the name of the dataset for ease of view
    :param algorithmNumber: integer - number of the algorithm (0 = K-Means, 1=DBSCAN, 2=SBM)
    :param labels: vector - the labels that have resulted from the algorithm
    :param y: vector - the ground_truth labels
    :param print: boolean - whether to print the accuracies
    :return np.array: list - the 4 accuracies of the algorithm of the selected dataset
    """
    allARI = metrics.adjusted_rand_score(y, labels)
    allAMI = metrics.adjusted_mutual_info_score(labels, y)

    # start of the NO-NOISE-POINTS (NNP) setting
    # we calculate only the accuracy of points that have been clustered(labeled as non-noise)

    adj = labels > 0
    yNN = y[adj]
    labelsNN = labels[adj]

    nnpARI = metrics.adjusted_rand_score(yNN, labelsNN)
    nnpAMI = metrics.adjusted_mutual_info_score(labelsNN, yNN)

    if print == True:
        printAccuracy(datasetName, algorithmNumber, allARI, allAMI, nnpARI, nnpAMI)

    return np.array([allARI, allAMI, nnpARI, nnpAMI])


def getSimulationAverageAccuracy():
    """
    Iterate through all 95 simulation calculate the accuracy for each and then make an average
    :param None
    :return None
    """
    # f = open("filename", "a") - append mode
    # f.write to write to it
    averageKMeans = np.array([0, 0, 0, 0])
    averageDBSCAN = np.array([0, 0, 0, 0])
    averageSBMv2 = np.array([0, 0, 0, 0])
    averageSBMv1 = np.array([0, 0, 0, 0])
    header = "Dataset Number, KMEANS-ALL-ARI, KMEANS-ALL-AMI, KMEANS-NNP-ARI, KMEANS-NNP-AMI, DBSCAN-ALL-ARI, DBSCAN-ALL-AMI, DBSCAN-NNP-ARI, DBSCAN-NNP-AMI, SBM-V2-ALL-ARI, SBM-V2-ALL-AMI, SBM-V2-NNP-ARI, SBM-V2-NNP-AMI, SBM-V1-ALL-ARI, SBM-V1-ALL-AMI, SBM-V1-NNP-ARI, SBM-V1-NNP-AMI"
    allAccuracies = np.empty((17,))
    for i in range(1, 96):
        print(i)
        if i == 24 or i == 25 or i == 44:
            continue
        X, y = ds.getDatasetSimulationPCA2D(simNr=i)

        kmeans = KMeans(n_clusters=np.amax(y)).fit(X)
        labels = kmeans.labels_
        accuracy_kmeans = calculateAccuracy('', 0, labels, y)
        averageKMeans = np.add(averageKMeans, accuracy_kmeans)

        min_samples = np.log(len(X))
        db = DBSCAN(eps=1, min_samples=min_samples).fit(X)
        labels = db.labels_
        accuracy_dbscan = calculateAccuracy('', 1, labels, y)
        averageDBSCAN = np.add(averageDBSCAN, accuracy_dbscan)

        labels = SBM.multiThreaded(X, pn=30, version=2)
        accuracy_sbmv2 = calculateAccuracy('', 2, labels, y)
        averageSBMv2 = np.add(averageSBMv2, accuracy_sbmv2)

        labels = SBM.multiThreaded(X, pn=30, version=1)
        accuracy_sbmv1 = calculateAccuracy('', 2, labels, y)
        averageSBMv1 = np.add(averageSBMv1, accuracy_sbmv1)

        allAccuracies = np.vstack((allAccuracies, np.insert(
            np.append(accuracy_kmeans, np.append(accuracy_dbscan, np.append(accuracy_sbmv2, accuracy_sbmv1))) * 100, 0,
            i)))
        # print(allAccuracies)
    np.savetxt("PCA3D_accuracy.csv", allAccuracies, delimiter=',', header=header, fmt="%10.2f")
    print("Average KMeans: {}".format(np.array(averageKMeans) / 92))
    print("Average DBSCAN: {}".format(np.array(averageDBSCAN) / 92))
    print("Average SBMv2: {}".format(np.array(averageSBMv2) / 92))
    print("Average SBMv1: {}".format(np.array(averageSBMv1) / 92))


# getSimulationAverageAccuracy()
# benchmark_dataset(4, plot=True)


def calculate_pca_accuracy(labels, x, y, labeled_data=True):
    """
    Calculate the accuracies of the algorithm on one simulation from the dataset
    :param labels: vector - the labels that have resulted from the algorithm
    :param x: vector - the values for each spike point
    :param y: vector - the ground_truth labels
    :param labeled_data: boolean - whether to use the cluster evaluation for labeled or unlabeled data
    :return np.array: list - the accuracies for the given point "x", ground truth "y" and cluster points "labels"
    """

    if labeled_data:
        # the metrics for all points
        all_ari = metrics.adjusted_rand_score(y, labels)
        all_ami = metrics.adjusted_mutual_info_score(y, labels)
        all_fmi = metrics.fowlkes_mallows_score(y, labels)

        # start of the NO-NOISE-POINTS (NNP) setting
        # we calculate only the accuracy of points that have been clustered(labeled as non-noise)

        adj = labels > 0
        y_nn = y[adj]
        labels_nn = labels[adj]

        nnp_ari = metrics.adjusted_rand_score(y_nn, labels_nn)
        nnp_ami = metrics.adjusted_mutual_info_score(y_nn, labels_nn)
        nnp_fmi = metrics.fowlkes_mallows_score(y, labels)

        return np.array([all_ari, all_ami, all_fmi,
                         nnp_ari, nnp_ami, nnp_fmi])
    else:
        silhouette_ground_truth = metrics.silhouette_score(x, y)
        cbi_ground_truth = metrics.calinski_harabasz_score(x, y)
        dbi_ground_truth = metrics.davies_bouldin_score(x, y)
        silhouette_algorithm = metrics.silhouette_score(x, labels)
        cbi_algorithm = metrics.calinski_harabasz_score(x, labels)
        dbi_algorithm = metrics.davies_bouldin_score(x, labels)

        return np.array([silhouette_ground_truth, cbi_ground_truth, dbi_ground_truth,
                         silhouette_algorithm, cbi_algorithm, dbi_algorithm])


def print_accuracy(dataset_number, algorithm_number, accuracy_values, labeled_data=True):
    """
    Print the accuracy results of the algorithm on one simulation from the data set
    :param dataset_number: integer - the number of the simulation (between 1 and 95)
    :param algorithm_number: integer - number of the algorithm (0 = K-Means, 1=DBSCAN, 2=SBM)
    :param accuracy_values: np.array - the values that have resulted from the accuracy calculation
    :param labeled_data: boolean - whether to use the cluster evaluation for labeled or unlabeled data
    :return None
    """

    if labeled_data:
        print('ALL SETTING')
        print(
            "Sim" + str(dataset_number) + " - " + algName[algorithm_number] + " - " + 'ARI: {: .3f}'.format(
                accuracy_values[0]))
        print(
            "Sim" + str(dataset_number) + " - " + algName[algorithm_number] + " - " + 'AMI: {: .3f}'.format(
                accuracy_values[1]))
        print(
            "Sim" + str(dataset_number) + " - " + algName[algorithm_number] + " - " + 'FMI: {: .3f}'.format(
                accuracy_values[2]))

        print('NNP SETTING')
        print(
            "Sim" + str(dataset_number) + " - " + algName[algorithm_number] + " - " + 'ARI: {: .3f}'.format(
                accuracy_values[3]))
        print(
            "Sim" + str(dataset_number) + " - " + algName[algorithm_number] + " - " + 'AMI: {: .3f}'.format(
                accuracy_values[4]))
        print(
            "Sim" + str(dataset_number) + " - " + algName[algorithm_number] + " - " + 'FMI: {: .3f}'.format(
                accuracy_values[5]))
    else:
        print('GROUND TRUTH')
        print(
            "Sim" + str(dataset_number) + " - " + algName[algorithm_number] + " - " + 'Sil: {: .3f}'.format(
                accuracy_values[0]))
        print(
            "Sim" + str(dataset_number) + " - " + algName[algorithm_number] + " - " + 'CHI: {: .3f}'.format(
                accuracy_values[1]))
        print(
            "Sim" + str(dataset_number) + " - " + algName[algorithm_number] + " - " + 'DBI: {: .3f}'.format(
                accuracy_values[2]))

        print('ALGORITHM')
        print(
            "Sim" + str(dataset_number) + " - " + algName[algorithm_number] + " - " + 'Sil: {: .3f}'.format(
                accuracy_values[3]))
        print(
            "Sim" + str(dataset_number) + " - " + algName[algorithm_number] + " - " + 'CHI: {: .3f}'.format(
                accuracy_values[4]))
        print(
            "Sim" + str(dataset_number) + " - " + algName[algorithm_number] + " - " + 'DBI: {: .3f}'.format(
                accuracy_values[5]))


def calculate_v_score(labels, y):
    """
    Calculate the accuracies of the algorithm on one simulation from the dataset (only for supervised clustering)
    :param labels: vector - the labels that have resulted from the algorithm
    :param y: vector - the ground_truth labels
    :return np.array: list - the accuracy result considering the ground truth "y" and cluster points "labels"
    """
    # ALL settings points
    all_homogeneity = metrics.homogeneity_score(y, labels)
    all_completeness = metrics.completeness_score(y, labels)
    all_v_score = metrics.v_measure_score(y, labels)

    # NNP settings
    adj = labels > 0
    y_nn = y[adj]
    labels_nn = labels[adj]

    nnp_homogeneity = metrics.homogeneity_score(y_nn, labels_nn)
    nnp_completeness = metrics.completeness_score(y_nn, labels_nn)
    nnp_v_score = metrics.v_measure_score(y_nn, labels_nn)

    return np.array([all_homogeneity, all_completeness, all_v_score, nnp_homogeneity, nnp_completeness, nnp_v_score])


def print_v_score(dataset_number, algorithm_number, accuracy_values):
    """
    Print the accuracy results computed with v-score of one algorithm on one simulation from the data set
    :param dataset_number: integer - the number of the simulation (between 1 and 95)
    :param algorithm_number: integer - number of the algorithm (0 = K-Means, 1=DBSCAN, 2=SBM)
    :param accuracy_values: np.array - the values that have resulted from the accuracy calculation by v-score indices
    :return None
    """

    print('ALL SETTING')
    print("Sim" + str(dataset_number) + " - " + algName[algorithm_number] + " - " + "Hom: " + str(accuracy_values[0]))
    print("Sim" + str(dataset_number) + " - " + algName[algorithm_number] + " - " + "Com: " + str(accuracy_values[1]))
    print("Sim" + str(dataset_number) + " - " + algName[algorithm_number] + " - " + "V-s: " + str(accuracy_values[2]))
    print('NNP SETTING')
    print("Sim" + str(dataset_number) + " - " + algName[algorithm_number] + " - " + "Hom: " + str(accuracy_values[3]))
    print("Sim" + str(dataset_number) + " - " + algName[algorithm_number] + " - " + "Com: " + str(accuracy_values[4]))
    print("Sim" + str(dataset_number) + " - " + algName[algorithm_number] + " - " + "V-s: " + str(accuracy_values[5]))


def write_supervised_accuracy_to_file(simulation_number, accuracy_values_kmeans, accuracy_values_dbscan,
                                      accuracy_values_sbm):
    """
    Write to a csv file the accuracy measurement of supervised clustering
    :param simulation_number: integer - in the interval [1, 95]
    :param accuracy_values_kmeans: np.array of floats - the results after performing K-means
    :param accuracy_values_dbscan: np.array of floats - the results after performing DBSCAN
    :param accuracy_values_sbm: np.array of floats - the results after performing SBM
    :return None
    """
    # formatted_kmeans = np.array(tuple(round(x, precision) for x in accuracy_values_kmeans))
    formatted_kmeans = ["%.3f" % number for number in accuracy_values_kmeans]
    formatted_kmeans.insert(0, "K-means")
    formatted_dbscan = ["%.3f" % number for number in accuracy_values_dbscan]
    formatted_dbscan.insert(0, "DBSCAN")
    formatted_sbm = ["%.3f" % number for number in accuracy_values_sbm]
    formatted_sbm.insert(0, "S.B.M.")

    header_labeled_data = ['Algor', 'ARI-a', 'AMI-a', 'FMI-a', 'ARI-n', 'AMI-n', 'FMI-n']
    row_list = [header_labeled_data, formatted_kmeans, formatted_dbscan, formatted_sbm]
    with open('./results/Supervised_accuracy_%s.csv' % simulation_number, 'w', newline='') as file:
        writer = csv.writer(file, delimiter='\t')
        writer.writerows(row_list)


def write_unsupervised_accuracy_to_file(simulation_number, accuracy_values_kmeans, accuracy_values_dbscan,
                                        accuracy_values_sbm):
    """
    Write to a csv file the accuracy measurement of unsupervised clustering
    :param simulation_number: integer - in the interval [1, 95]
    :param accuracy_values_kmeans: np.array of floats - the results after performing K-means
    :param accuracy_values_dbscan: np.array of floats - the results after performing DBSCAN
    :param accuracy_values_sbm: np.array of floats - the results after performing SBM
    :return None
    """
    formatted_kmeans = ["%.3f" % number for number in accuracy_values_kmeans]
    formatted_kmeans.insert(0, "K-means")
    formatted_dbscan = ["%.3f" % number for number in accuracy_values_dbscan]
    formatted_dbscan.insert(0, "DBSCAN")
    formatted_sbm = ["%.3f" % number for number in accuracy_values_sbm]
    formatted_sbm.insert(0, "S.B.M.")

    # Silhouette, Calinski-Harabasz, Davies-Bouldin
    # g - ground truth, a - algorithm
    header_labeled_data = ['Algor', 'Sil-g', 'CBI-g', 'DBI-g', 'Sil-a', 'CBI-a', 'DBI-a']
    row_list = [header_labeled_data, formatted_kmeans, formatted_dbscan, formatted_sbm]
    with open('./results/Unsupervised_accuracy_%s.csv' % simulation_number, 'w', newline='') as file:
        writer = csv.writer(file, delimiter='\t')
        writer.writerows(row_list)

def benchmark_simulation(datasetNumber, plot=False, labeled_data=True):
    """
    Benchmarks K-Means, DBSCAN and SBM on one of 5 selected datasets
    :param datasetNumber: integer - the number that represents one of the datasets (0-4)
    :param plot: boolean - optional, whether the plot function should be called or not (for ease of use)
    :param labeled_data: boolean - optional, whether the clustering is done on data with labeled ground truth or not
    :returns None
    """
    print("DATASET: " + dataName[datasetNumber])
    datasetName = dataName[datasetNumber]
    simulation_number = 10
    if datasetNumber < 3:
        X = np.genfromtxt("./datasets/" + files[datasetNumber], delimiter=",")
        X, y = X[:, [0, 1]], X[:, 2]
    elif datasetNumber == 3:
        X, y = ds.getGenData()
    else:
        X, y = ds.getDatasetSimulationPCA2D(simulation_number)

    # S2 has label problems
    if datasetNumber == 1:
        for k in range(len(X)):
            y[k] = y[k] - 1

    kmeans = KMeans(n_clusters=kmeansValues[datasetNumber]).fit(X)
    labels = kmeans.labels_
    scatter_plot.plotFunction("K-MEANS on" + datasetName, X, labels, plot, marker='o')

    accuracy_result_kmeans = calculate_pca_accuracy(labels, X, y, labeled_data=labeled_data)
    print_accuracy(simulation_number, 0, accuracy_result_kmeans, labeled_data=labeled_data)

    if datasetNumber == 1:
        min_samples = np.log(len(X)) * 10
    else:
        min_samples = np.log(len(X))
    db = DBSCAN(eps=epsValues[datasetNumber], min_samples=min_samples).fit(X)
    labels = db.labels_
    scatter_plot.plotFunction("DBSCAN on" + datasetName, X, labels, plot, marker='o')

    accuracy_result_dbscan = calculate_pca_accuracy(labels, X, y, labeled_data=labeled_data)
    print_accuracy(simulation_number, 1, accuracy_result_dbscan, labeled_data=labeled_data)

    labels = SBM.multiThreaded(X, pn=25, version=2)
    scatter_plot.plotFunction("SBM on" + datasetName, X, labels, plot, marker='o')

    accuracy_result_sbm = calculate_pca_accuracy(labels, X, y, labeled_data=labeled_data)
    print_accuracy(simulation_number, 2, accuracy_result_sbm, labeled_data=labeled_data)

    if labeled_data:
        write_supervised_accuracy_to_file(simulation_number, accuracy_result_kmeans, accuracy_result_dbscan,
                                          accuracy_result_sbm)
    else:
        write_unsupervised_accuracy_to_file(simulation_number, accuracy_result_kmeans, accuracy_result_dbscan,
                                            accuracy_result_sbm)


benchmark_simulation(4, plot=True, labeled_data=True)
