import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
import sys
sys.setrecursionlimit(100000)

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import SBM
import SBM_functions as fs
import scatter
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
    print("DATASET: "+dataName[datasetNumber])
    datasetName = dataName[datasetNumber]
    if datasetNumber < 3:
        X = np.genfromtxt("./datasets/"+files[datasetNumber], delimiter=",")
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
    scatter.plotFunction("K-MEANS on"+datasetName, X, labels, plot, marker='X')
    plt.show()
    calculateAccuracy(datasetName, 0, labels, y, print=True)


    if datasetNumber == 1:
        min_samples = np.log(len(X)) * 10
    else:
        min_samples = np.log(len(X))
    db = DBSCAN(eps=epsValues[datasetNumber], min_samples=min_samples).fit(X)
    labels = db.labels_
    scatter.plotFunction("DBSCAN on"+datasetName, X, labels, plot, marker='X')
    plt.show()
    calculateAccuracy(datasetName, 1, labels, y, print=True)

    labels = SBM.multiThreaded(X, pn=25, version=2)
    scatter.griddedPlotFunction("SBM on"+datasetName, X, labels, pn,  plot, marker='X')
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

def calculateAccuracy(datasetName, algorithmNumber, labels, y, print = False):

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
    averageKMeans = np.array([0,0,0,0])
    averageDBSCAN = np.array([0,0,0,0])
    averageSBM = np.array([0,0,0,0])
    for i in range(1, 96):
        X, y = ds.getDatasetSimulation(simNr=1)

        kmeans = KMeans(n_clusters=np.amax(y)).fit(X)
        labels = kmeans.labels_
        averageKMeans = np.add(averageKMeans, calculateAccuracy('', 0, labels, y))

        min_samples = np.log(len(X))
        db = DBSCAN(eps=0.1, min_samples=min_samples).fit(X)
        labels = db.labels_
        averageDBSCAN = np.add(averageDBSCAN, calculateAccuracy('', 1, labels, y))

        labels = SBM.multiThreaded(X, pn=30)
        averageSBM = np.add(averageSBM, calculateAccuracy('', 2, labels, y))

    print("Average KMeans: {}".format(np.array(averageKMeans)/95))
    print("Average DBSCAN: {}".format(np.array(averageDBSCAN)/95))
    print("Average SBM: {}".format(np.array(averageSBM)/95))


getSimulationAverageAccuracy()
#benchmark_dataset(4, plot=True)