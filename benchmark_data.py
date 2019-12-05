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
import scatter

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
    scatter.plotFunction("K-MEANS on" + datasetName, X, labels, plot, marker='X')
    plt.show()
    calculateAccuracy(datasetName, 0, labels, y, print=True)

    if datasetNumber == 1:
        min_samples = np.log(len(X)) * 10
    else:
        min_samples = np.log(len(X))
    db = DBSCAN(eps=epsValues[datasetNumber], min_samples=min_samples).fit(X)
    labels = db.labels_
    scatter.plotFunction("DBSCAN on" + datasetName, X, labels, plot, marker='X')
    plt.show()
    calculateAccuracy(datasetName, 1, labels, y, print=True)

    labels = SBM.multiThreaded(X, pn=25, version=2)
    scatter.griddedPlotFunction("SBM on" + datasetName, X, labels, pn, plot, marker='X')
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
    print('ALL SETTING')
    print(datasetName + " - " + algName[algorithmNumber] + " - " + "ARI:" + str(allARI))
    print(datasetName + " - " + algName[algorithmNumber] + " - " + "AMI:" + str(allAMI))

    print('NNP SETTING')
    print(datasetName + " - " + algName[algorithmNumber] + " - " + "ARI:" + str(nnpARI))
    print(datasetName + " - " + algName[algorithmNumber] + " - " + "AMI:" + str(nnpAMI))


def calculateAccuracy(datasetName, algorithmNumber, labels, y, print=False):
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
    X, y = ds.getDatasetSimulationPCA2D(simNr=simNr)

    kmeans = KMeans(n_clusters=np.amax(y)).fit(X)
    kmeans_labels = kmeans.labels_
    accuracy_kmeans = calculateAccuracy('', 0, kmeans_labels, y)
    resultKMeans = np.add(resultKMeans, accuracy_kmeans)

    min_samples = np.log(len(X))
    db = DBSCAN(eps=0.5, min_samples=min_samples).fit(X)
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

    if plot:
        scatter.plotFunction("Ground truth on sim" + str(simNr), X, y, plot, marker='o')
        plt.show()
        scatter.plotFunction("K-MEANS on sim" + str(simNr), X, kmeans_labels, plot, marker='o')
        plt.show()
        scatter.plotFunction("DBSCAN on sim" + str(simNr), X, dbscan_labels, plot, marker='o')
        plt.show()
        scatter.plotFunction("SBMv2 on sim" + str(simNr), X, sbmv2_labels, plot, marker='o')
        plt.show()
        scatter.plotFunction("SBMv1 on sim" + str(simNr), X, sbmv1_labels, plot, marker='o')
        plt.show()


# getSimulationAccuracy(23, plot=True)
