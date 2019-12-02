import sys
import math
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from joblib import Parallel, delayed
sys.setrecursionlimit(1000000)

import multiprocessing
num_cores = multiprocessing.cpu_count() - 2


def adjust(x):
    if x < 0:
        x = np.floor(x)
        x = np.floor(x / 5)
    else:
        x = np.ceil(x)
        x = np.ceil(x / 5)
    x = 5 * x
    return x


def pad_with(vector, pad_width, kwargs):
    pad_value = kwargs.get('padder', 10)
    vector[:pad_width[0]] = pad_value
    vector[-pad_width[1]:] = pad_value
    return vector


# TODO
def isValidCenter(value):
    return True  # TODO set the min value acceptable


def getNeighbours(p, shape):
    ndim = len(p)
    offsetIndexes = np.indices((3,) * ndim).reshape(ndim, -1).T
    offsets = np.r_[-1, 0, 1].take(offsetIndexes)
    offsets = offsets[np.any(offsets, 1)]

    neighbours = p + offsets

    valid = np.all((neighbours < np.array(shape)) & (neighbours >= 0), axis=1)
    neighbours = neighbours[valid]

    return neighbours


def isMaxima(array, point):
    neighbours = getNeighbours(point, np.shape(array))
    for neighbour in neighbours:
        if array[tuple(neighbour)] > array[point]:
            return False
    #print('CENTER {} VALUE {} NEIGHBOURS {}'.format(point, array[point], [array[tuple(n)] for n in neighbours]))
    return True


def findClusterCenters(array, threshold=5):
    clusterCenters = []

    # print('\n'.join([''.join(['{:10}'.format(item) for item in row]) for row in array]))

    for index, value in np.ndenumerate(array):
        if value >= threshold and isMaxima(array, index):  # TODO exclude neighbour centers
            clusterCenters.append(index)
    return clusterCenters


# noinspection SpellCheckingInspection
def getDropoff(ndArray, location):
    neighbours = getNeighbours(location, np.shape(ndArray))
    dropoff = 0
    for neighbour in neighbours:
        neighbourLocation = tuple(neighbour)
        dropoff += ((ndArray[location] - ndArray[neighbourLocation]) ** 2) / ndArray[location]
    if dropoff > 0:
        return math.sqrt(dropoff / len(neighbours))
    return 0


def getStrength(ndArray, clusterCenter, questionPoint, expansionPoint):
    dist = distance(clusterCenter, questionPoint)
    #strength = ndArray[expansionPoint] / ndArray[questionPoint] / dist

    #BEST FOR NOW
    strength = ndArray[questionPoint] / dist / ndArray[clusterCenter]

    return strength


def expand(array, start, labels, currentLabel, clusterCenters, version=1):  # TODO
    visited = np.zeros_like(array, dtype=bool)
    expansionQueue = []
    if labels[start] == 0:
        expansionQueue.append(start)
        labels[start] = currentLabel
    # else:
    #     oldLabel = labels[start]
    #     disRez = disambiguate(array,
    #                          start,
    #                          clusterCenters[currentLabel - 1],
    #                          clusterCenters[oldLabel - 1])
    #     if disRez == 1:
    #         labels[start] = currentLabel
    #         expansionQueue.append(start)
    #     elif disRez == 11:
    #         labels[labels == oldLabel] = currentLabel
    #         expansionQueue.append(start)
    #     elif disRez == 22:
    #         labels[labels == currentLabel] = oldLabel
    #         currentLabel = oldLabel
    #         expansionQueue.append(start)

    visited[start] = True

    dropoff = getDropoff(array, start)

    while expansionQueue:
        point = expansionQueue.pop(0)
        neighbours = getNeighbours(point, np.shape(array))
        for neighbour in neighbours:
            location = tuple(neighbour)
            if version == 1:
                number = dropoff * math.sqrt(distance(start, location))
            elif version == 2:
                number = math.floor(math.sqrt(dropoff * distance(start, location)))
            # print(number)
            if array[location] == 0:
                pass
            if (not visited[location]) and (number < array[location] <= array[point]):
                visited[location] = True
                if labels[location] == currentLabel:
                    expansionQueue.append(location)
                elif labels[location] == 0:
                    expansionQueue.append(location)
                    labels[location] = currentLabel
                else:
                    if version == 0:
                        labels[location] = -1
                    else:
                        oldLabel = labels[location]
                        disRez = disambiguate(array,
                                              location,
                                              point,
                                              clusterCenters[currentLabel - 1],
                                              clusterCenters[oldLabel - 1],
                                              version)
                        # print("choice"+str(disRez))
                        if disRez == 1:
                            labels[location] = currentLabel
                            expansionQueue.append(location)
                        elif disRez == 2 and version == 2:
                            labels[location] = oldLabel
                            expansionQueue.append(location)
                        elif disRez == 11:
                            # current label wins
                            labels[labels == oldLabel] = currentLabel
                            expansionQueue.append(location)
                        elif disRez == 22:
                            # old label wins
                            labels[labels == currentLabel] = oldLabel
                            currentLabel = oldLabel
                            expansionQueue.append(location)

    return labels

def disambiguate(array, questionPoint, expansionPoint, clusterCenter1, clusterCenter2, version):
    # CHOOSE CLUSTER FOR ALREADY ASSIGNED POINT
    # usually wont get to this
    if (clusterCenter1 == questionPoint) or (clusterCenter2 == questionPoint):
        # here the point in question as already been assigned to one cluster, but another is trying to accumulate it
        # we check which of the 2 has a count and merged them
        if array[clusterCenter1] > array[clusterCenter2]:
            return 11
        else:
            return 22

    # MERGE
    # cluster 2 was expanded first, but it is actually connected to a bigger cluster
    if array[clusterCenter2] == array[questionPoint]:
        return 11
    if version == 2:
        # cluster 1 was expanded first, but it is actually connected to a bigger cluster
        if array[clusterCenter1] == array[questionPoint]:
            return 22

    # XANNY
    if version == 1:
        distanceToC1 = distance(questionPoint, clusterCenter1)
        distanceToC2 = distance(questionPoint, clusterCenter2)
        pointStrength = array[questionPoint]

        c1Strength = array[clusterCenter1] / pointStrength - getDropoff(array, clusterCenter1) * distanceToC1
        c2Strength = array[clusterCenter2] / pointStrength - getDropoff(array, clusterCenter2) * distanceToC2

    # RICI
    elif version == 2:
        c1Strength = getStrength(array, clusterCenter1, questionPoint, expansionPoint)
        c2Strength = getStrength(array, clusterCenter2, questionPoint, expansionPoint)


    # RICI VERSION
    # neighbours = getNeighbours(questionPoint, np.shape(array))
    # maxN = 0
    # for n in neighbours:
    #     if labels[tuple(n)] == oldLabel and array[tuple(n)] > maxN:
    #         maxN = array[tuple(n)]
    #
    # c1Strength = array[questionPoint] / array[cluster1]
    # c2Strength = array[questionPoint] / array[cluster2]

    # distanceToC1 = distance(questionPoint, clusterCenter1)
    # distanceToC2 = distance(questionPoint, clusterCenter2)
    # pointStrength = array[questionPoint]
    # c1Strength = array[cluster1]*distanceToC1/(array[cluster1] - pointStrength)
    # c2Strength = array[cluster2]*distanceToC2/(array[cluster2] - pointStrength)
    # c2Strength = (array[cluster2] / pointStrength) / distanceToC2

    # if (abs(c1Strength - c2Strength) < threshold):
    #     return 0
    if c1Strength > c2Strength:
        return 1
    else:
        return 2


def chunkify(X, pn):
    nrDim = np.shape(X)[1]
    nArray = np.zeros((pn,) * nrDim, dtype=int)

    for point in X:
        if np.all(point < pn):
            location = tuple(np.floor(point).astype(int))
            nArray[location] += 1
        else:  # TODO
            # print(point)
            pass
    return nArray


def chunkifyMT(X, pn, nrThreads=num_cores):
    splittedX = np.array_split(X, nrThreads)

    results = Parallel(n_jobs=nrThreads)(delayed(chunkify)(x, pn) for x in splittedX)

    finalArray = sum(results)

    return finalArray


def dechunkifyMT(X, labelsArray, pn, nrThreads=num_cores):
    splittedX = np.array_split(X, nrThreads)

    results = Parallel(n_jobs=nrThreads)(delayed(dechunkify)(x, labelsArray, pn) for x in splittedX)

    finalLabels = np.concatenate(results, axis=0)
    return finalLabels


"""return array of "number of points" length with the label for each point"""
def dechunkify(X, labelsArray, pn):
    pointLabels = np.zeros(len(X), dtype=int)

    # import threading
    # print("Thread ID[{}] gets {}".format(threading.get_ident(), X[:1]))

    for index in range(0, len(X)):
        point = X[index]
        if np.all(point < pn):
            location = tuple(np.floor(point).astype(int))
            pointLabels[index] = labelsArray[location]
        else:  # TODO
            pointLabels[index] = -1

    return pointLabels


def distance(pointA, pointB):
    sum = 0
    for i in range(0, len(pointA)):
        sum += (pointA[i] - pointB[i]) ** 2
    return math.sqrt(sum)


