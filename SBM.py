import numpy as np
from sklearn import preprocessing
import time
import sys
sys.setrecursionlimit(100000)

import SBM_functions as fs

def sequential(X, pn, ccThreshold = 5, version=2):
    """
    Sequential version of SBM
    :param X: matrix - the points of the dataset
    :param pn: integer - the number of partitions on columns and rows
    :param version: integer - the version of SBM (0-original version, 1=license, 2=modified with less noise)
    :param ccThreshold: integer - the minimum number of points needed in a partition/chunk for it to be considered a possible cluster center

    :returns labels: vector -  the vector of labels for each point of dataset X
    """
    X = preprocessing.MinMaxScaler((0, pn)).fit_transform(X)
    ndArray = fs.chunkify_sequential(X, pn)

    clusterCenters = fs.find_cluster_centers(ndArray, ccThreshold)

    labelsMatrix = np.zeros_like(ndArray, dtype=int)
    for labelM1 in range(len(clusterCenters)):
        point = clusterCenters[labelM1]
        if labelsMatrix[point] != 0:
            continue  # cluster was already discovered
        labelsMatrix = fs.expand_cluster_center(ndArray, point, labelsMatrix, labelM1 + 1, clusterCenters, version=version)

    #bring cluster labels back to (-1) - ("nr of clusters"-2) range
    uniqueClusterLabels = np.unique(labelsMatrix)
    nrClust = len(uniqueClusterLabels)
    for label in range(len(uniqueClusterLabels)):
        if uniqueClusterLabels[label] == -1 or uniqueClusterLabels[label]==0: # don`t remark noise/ conflicta
            nrClust -=1
            continue

    labels = fs.dechunkify_sequential(X, labelsMatrix, pn)#TODO 2
    #print("number of actual clusters: ", nrClust)

    return labels


def parallel(X, pn, ccThreshold = 5, version=2):
    """
    Multi-threaded version of SBM
    :param X: matrix - the points of the dataset
    :param pn: integer - the number of partitions on columns and rows
    :param version: integer - the version of SBM (1=license, 2=modified with less noise)
    :param ccThreshold: integer - the minimum number of points needed in a partition/chunk for it to be considered a possible cluster center

    :returns labels: vector -  the vector of labels for each point of dataset X
    """
    # 1. normalization of the dataset to bring it to 0-pn on all axes
    X = preprocessing.MinMaxScaler((0, pn)).fit_transform(X)


    # 2. chunkification of the dataset into a matrix of "squares"(for 2d)
    # returns an array of pn for each dimension
    start = time.time()
    ndArray = fs.chunkify_parallel(X, pn)
    #rotated = rotateMatrix(np.copy(ndArray))
    #np.savetxt("csf.txt", rotated, fmt="%i", delimiter="\t")
    end = time.time()
    #print('CHUNKIFY: ' + str(end - start))

    # 3, search of cluster centers (based on a guassian distrubution of the clusters) (current > all neighbours)
    # returns a list of points that are the cluster centers
    start = time.time()
    clusterCenters = fs.find_cluster_centers(ndArray, ccThreshold)
    end = time.time()
    #print('CC: ' + str(end - start))

    # 4. expansion of the cluster centers using BFS
    # returns an array of the same lengths as the chunkification containing the labels
    start = time.time()
    labelsMatrix = np.zeros_like(ndArray, dtype=int)
    for labelM1 in range(len(clusterCenters)):
        point = clusterCenters[labelM1]
        if labelsMatrix[point] != 0:
            continue  # cluster was already discovered
        labelsMatrix = fs.expand_cluster_center(ndArray, point, labelsMatrix, labelM1 + 1, clusterCenters, version=version)
    end = time.time()
    #print('EXPAND: ' + str(end - start))


    # scatter_plot.plotCenters("centers", X, clusterCenters, pn, plot=True, marker='.')

    start = time.time()
    # 5. inverse of chunkification, from the labels array we get the label of each points
    # returns an array of the size of the initial dataset each containing the label for the corresponding point
    labels = fs.dechunkify_parallel(X, labelsMatrix, pn)#TODO 2
    end = time.time()
    #print('DECHUNKIFY: ' + str(end - start))
    #print("number of actual clusters: ", nrClust)

    return labels



