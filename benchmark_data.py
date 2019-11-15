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
kmeansValues = [15, 15, 8, 6]
epsValues = [27000, 45000, 18000, 0.5]
pn = 25

dataName = ["S1", "S2", "U", "UO", "S97"]
algorithmNames = ["K-MEANS", "K-MEANS", "K-MEANS", "K-MEANS", "DBSCAN", "SBM", ]
settings = ["ARI", "AMI", "ARI", "AMI", "NNP", "NNP"]
table = [algName]
for i in range(0, 5):
    if i < 3:
        X = np.genfromtxt("./datasets/"+files[i], delimiter=",")
        X, y = X[:, [0, 1]], X[:, 2]
    elif i == 4:
        X, y = ds.getGenData()
    else:
        X, y = ds.getTestDataset79()

    if i == 1:
        for k in range(len(X)):
            y[k] = y[k] - 1

    for j in range(0, 3):
        if j == 0:
            kmeans = KMeans(n_clusters=kmeansValues[i]).fit(X)
            labels = kmeans.labels_
        elif j == 1:
            if i == 1:
                min_samples = np.log(len(X)) * 10
            else:
                min_samples = np.log(len(X))
            db = DBSCAN(eps=epsValues[i], min_samples=min_samples).fit(X)
            labels = db.labels_
        elif j == 2:
            labels = SBM.multiThreaded(X, pn)

        results = []
        results.append(metrics.adjusted_rand_score(y, labels))
        results.append(metrics.adjusted_mutual_info_score(labels, y))

        # start of the NO-NOISE-POINTS (NNP) setting
        # we calculate only the accuracy of points that have been clustered(labeled as non-noise)
        adj = labels > 0
        yNN = y[adj]
        labelsNN = labels[adj]

        results.append(metrics.adjusted_rand_score(yNN, labelsNN))
        results.append(metrics.adjusted_mutual_info_score(labelsNN, yNN))

        print(results)


