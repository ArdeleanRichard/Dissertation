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


algName = ["K-MEANS", "DBSCAN", "SBM"]
files = ["s1_labeled.csv", "s2_labeled.csv", "unbalance.csv"]
kmeansValues = [15, 15, 8, 6, 20]
epsValues = [27000, 45000, 18000, 0.5, 1]
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

def benchmark_dataset(datasetNumber):
    if datasetNumber < 3:
        X = np.genfromtxt("./datasets/"+files[datasetNumber], delimiter=",")
        X, y = X[:, [0, 1]], X[:, 2]
    elif i == 4:
        X, y = ds.getGenData()
    else:
        X, y = ds.getTestDataset79()

    # S2 has label problems
    if datasetNumber == 1:
        for k in range(len(X)):
            y[k] = y[k] - 1

    kmeans = KMeans(n_clusters=kmeansValues[datasetNumber]).fit(X)
    labels = kmeans.labels_
    calculateAccuracy(datasetNumber, 0, labels, y)


    if datasetNumber == 1:
        min_samples = np.log(len(X)) * 10
    else:
        min_samples = np.log(len(X))
    db = DBSCAN(eps=epsValues[datasetNumber], min_samples=min_samples).fit(X)
    labels = db.labels_
    calculateAccuracy(datasetNumber, 1, labels, y)


    labels = SBM.multiThreaded(X, pn)
    calculateAccuracy(datasetNumber, 2, labels, y)



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

def calculateAccuracy(datasetNumber, algorithmNumber, labels, y):
    print('ALL SETTING')
    print(dataName[datasetNumber] + " - " + algName[algorithmNumber] + " - " + "ARI:" + str(metrics.adjusted_rand_score(y, labels)))
    print(dataName[datasetNumber] + " - " + algName[algorithmNumber] + " - " + "AMI:" + str(metrics.adjusted_mutual_info_score(labels, y)))

    # start of the NO-NOISE-POINTS (NNP) setting
    # we calculate only the accuracy of points that have been clustered(labeled as non-noise)
    print('NNP SETTING')

    adj = labels > 0
    yNN = y[adj]
    labelsNN = labels[adj]

    print(dataName[datasetNumber] + " - " + algName[algorithmNumber] + " - " + "ARI:" + str(metrics.adjusted_rand_score(yNN, labelsNN)))
    print(dataName[datasetNumber] + " - " + algName[algorithmNumber] + " - " + "AMI:" + str(metrics.adjusted_mutual_info_score(labelsNN, yNN)))

benchmark_dataset(0)