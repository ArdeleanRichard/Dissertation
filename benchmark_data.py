import sys

import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
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

dataName = ["S1", "S2", "U", "UO", "Sim97"]
algorithmNames = ["K-MEANS", "K-MEANS", "K-MEANS", "K-MEANS", "DBSCAN", "SBM", ]
settings = ["ARI", "AMI", "ARI", "AMI", "NNP", "NNP"]
table = [algName]


# datasetNumber = 1 => S1
# datasetNumber = 2 => S2
# datasetNumber = 3 => U
# datasetNumber = 4 => UO
# datasetNumber = 5 => Sim97
def benchmark_dataset(datasetNumber, plot=False):
    """
    Benchmarks K-Means, DBSCAN and SBM on one of 5 selected datasets
    :param datasetNumber: integer - the number that represents one of the datasets (0-4)
    :param plot: boolean - optional, whether the plot function should be called or not (for ease of use)

    :returns None
    """
    print("DATASET: " + dataName[datasetNumber])
    datasetName = dataName[datasetNumber]
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
    scatter_plot.plotFunction("K-MEANS on" + datasetName, X, labels, plot, marker='X')
    plt.show()
    calculateAccuracy(datasetName, 0, labels, y, print=True)

    if datasetNumber == 1:
        min_samples = np.log(len(X)) * 10
    else:
        min_samples = np.log(len(X))
    db = DBSCAN(eps=epsValues[datasetNumber], min_samples=min_samples).fit(X)
    labels = db.labels_
    scatter_plot.plotFunction("DBSCAN on" + datasetName, X, labels, plot, marker='X')
    plt.show()
    calculateAccuracy(datasetName, 1, labels, y, print=True)

    labels = SBM.multiThreaded(X, pn=25, version=2)
    scatter_plot.griddedPlotFunction("SBM on" + datasetName, X, labels, pn, plot, marker='X')
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

def getSimulationAccuracy(simNr, plot=False):
    resultKMeans = np.array([0, 0, 0, 0])
    resultDBSCAN = np.array([0, 0, 0, 0])
    resultSBMv2 = np.array([0, 0, 0, 0])
    resultSBMv1 = np.array([0, 0, 0, 0])
    header = "Dataset Number, KMEANS-ALL-ARI, KMEANS-ALL-AMI, KMEANS-NNP-ARI, KMEANS-NNP-AMI, DBSCAN-ALL-ARI, DBSCAN-ALL-AMI, DBSCAN-NNP-ARI, DBSCAN-NNP-AMI, SBM-V2-ALL-ARI, SBM-V2-ALL-AMI, SBM-V2-NNP-ARI, SBM-V2-NNP-AMI, SBM-V1-ALL-ARI, SBM-V1-ALL-AMI, SBM-V1-NNP-ARI, SBM-V1-NNP-AMI"
    allAccuracies = np.empty((17,))
    if simNr == 24 or simNr == 25 or simNr == 44:
        print("This simulation has anomalies.")
        return
    X, y = ds.getDatasetSimulationPCA2D(simNr=simNr, align_to_peak=2)

    kmeans = KMeans(n_clusters=np.amax(y)).fit(X)
    kmeans_labels = kmeans.labels_
    accuracy_kmeans = calculateAccuracy('', 0, kmeans_labels, y)
    resultKMeans = np.add(resultKMeans, accuracy_kmeans)

    min_samples = np.log(len(X))
    db = DBSCAN(eps=0.1, min_samples=min_samples).fit(X)
    dbscan_labels = db.labels_
    accuracy_dbscan = calculateAccuracy('', 1, dbscan_labels, y)
    resultDBSCAN = np.add(resultDBSCAN, accuracy_dbscan)

    sbmv2_labels = SBM.multiThreaded(X, pn=30, version=2)
    accuracy_sbmv2 = calculateAccuracy('', 2, sbmv2_labels, y)
    resultSBMv2 = np.add(resultSBMv2, accuracy_sbmv2)

    sbmv1_labels = SBM.multiThreaded(X, pn=30, version=1)
    accuracy_sbmv1 = calculateAccuracy('', 2, sbmv1_labels, y)
    resultSBMv1 = np.add(resultSBMv1, accuracy_sbmv1)

    allAccuracies = np.vstack((allAccuracies, np.insert(
        np.append(accuracy_kmeans, np.append(accuracy_dbscan, np.append(accuracy_sbmv2, accuracy_sbmv1))) * 100, 0,
        simNr)))
    # print(allAccuracies)

    np.savetxt("sim" + str(simNr) + "_PCA2D_accuracy.csv", allAccuracies, delimiter=',', header=header, fmt="%10.2f")
    print("KMeans: {}".format(np.array(resultKMeans)))
    print("DBSCAN: {}".format(np.array(resultDBSCAN)))
    print("SBMv2: {}".format(np.array(resultSBMv2)))
    print("SBMv1: {}".format(np.array(resultSBMv1)))

    plot_names = ["ground", "kmeans", "dbscan", "sbmv2", "sbmv1"]
    plot_data = [y, kmeans_labels, dbscan_labels, sbmv2_labels, sbmv1_labels]

    if plot:
        for index in range(len(plot_names)):
            if len(X[0]) == 3:
                fig = px.scatter_3d(X, x=X[:, 0], y=X[:, 1], z=X[:, 2],
                                    color=plot_data[index])
                fig.show()
            else:
                scatter_plot.plotFunction(plot_names[index] + " on sim" + str(simNr), X, plot_data[index], plot,
                                          marker='o')
                # plt.savefig('./figures/sim' + str(simNr) + '_' + plot_names[index] + "_fsde6")
                plt.show()


# for i in range(23, 30):
#     getSimulationAccuracy(i, True)
getSimulationAccuracy(26, True)
