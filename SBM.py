import numpy as np
from sklearn import preprocessing
import time
import sys
sys.setrecursionlimit(100000)

import SBM_functions as fs

# X = dataset
# pn = partioning number, number of segments for each dimension
# version = no need to know
# ccThreshold = cluster center threshold, minimum count the cluster center needs to be valid
def sequential(X, pn, version=1, ccThreshold = 5):
    X = preprocessing.MinMaxScaler((0, pn)).fit_transform(X)
    ndArray = fs.chunkify(X, pn)

    clusterCenters = fs.findClusterCenters(ndArray, ccThreshold)

    labelsMatrix = np.zeros_like(ndArray, dtype=int)
    #labelsMatrix2 = np.zeros_like(ndArray, dtype=int)
    for labelM1 in range(len(clusterCenters)):
        point = clusterCenters[labelM1]
        if labelsMatrix[point] != 0:
            continue  # cluster was already discovered
        labelsMatrix = fs.expand(ndArray, point,labelsMatrix,labelM1+1,clusterCenters, version=version)

    #bring cluster labels back to (-1) - ("nr of clusters"-2) range
    uniqueClusterLabels = np.unique(labelsMatrix)
    nrClust = len(uniqueClusterLabels)
    for label in range(len(uniqueClusterLabels)):
        if uniqueClusterLabels[label] == -1 or uniqueClusterLabels[label]==0: # don`t remark noise/ conflicta
            nrClust -=1
            continue
        #labelsMatrix2[labelsMatrix == uniqueClusterLabels[label]] = label

    labels = fs.dechunkify(X,labelsMatrix,pn)#TODO 2

    #print("number of actual clusters: ", nrClust)

    return labels


def multiThreaded(X, pn, version=1,  ccThreshold = 5):
    # 1. normalization of the dataset to bring it to 0-pn on all axes
    X = preprocessing.MinMaxScaler((0, pn)).fit_transform(X)


    # 2. chunkification of the dataset into a matrix of "squares"(for 2d)
    # returns an array of pn for each dimension
    start = time.time()
    ndArray = fs.chunkifyMT(X, pn)
    muie = rotateMatrix(np.copy(ndArray))
    np.savetxt("csf.txt", muie, fmt="%i", delimiter="\t")
    end = time.time()
    #print('CHUNKIFY: ' + str(end - start))

    # 3, search of cluster centers (based on a guassian distrubution of the clusters) (current > all neighbours)
    # returns a list of points that are the cluster centers
    start = time.time()
    clusterCenters = fs.findClusterCenters(ndArray, ccThreshold)
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
        labelsMatrix = fs.expand(ndArray, point, labelsMatrix, labelM1+1, clusterCenters, version=version)
    end = time.time()
    #print('EXPAND: ' + str(end - start))



    #scatter.plotCenters("centers", X, clusterCenters, pn, plot=True, marker='.')

    start = time.time()
    # 5. inverse of chunkification, from the labels array we get the label of each points
    # returns an array of the size of the initial dataset each containing the label for the corresponding point
    labels = fs.dechunkifyMT(X, labelsMatrix, pn)#TODO 2
    end = time.time()
    #print('DECHUNKIFY: ' + str(end - start))
    #print("number of actual clusters: ", nrClust)

    return labels


def rotateMatrix(mat):
    # Consider all squares one by one
    N = len(mat)
    for x in range(0, int(N / 2)):

        # Consider elements in group
        # of 4 in current square
        for y in range(x, N - x - 1):
            # store current cell in temp variable
            temp = mat[x][y]

            # move values from right to top
            mat[x][y] = mat[y][N - 1 - x]

            # move values from bottom to right
            mat[y][N - 1 - x] = mat[N - 1 - x][N - 1 - y]

            # move values from left to bottom
            mat[N - 1 - x][N - 1 - y] = mat[N - 1 - y][x]

            # assign temp to left
            mat[N - 1 - y][x] = temp

    return mat
