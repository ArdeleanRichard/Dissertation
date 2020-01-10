import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from sklearn import preprocessing
import math
import constants as cs




def plot(title, X, labels=None, plot=True, marker='o'):
    """
    Plots the dataset with or without labels
    :param title: string - the title of the plot
    :param X: matrix - the points of the dataset
    :param labels: vector - optional, contains the labels of the points/X (has the same length as X)
    :param plot: boolean - optional, whether the plot function should be called or not (for ease of use)
    :param marker: character - optional, the marker of the plot

    :returns None
    """
    if plot:
        plt.figure()
        plt.title(title)
        if labels is None:
            plt.scatter(X[:, 0], X[:, 1], marker=marker, edgecolors='k')
        else:
            label_color = [cs.LABEL_COLOR_MAP[l] for l in labels]
            plt.scatter(X[:, 0], X[:, 1], c=label_color, marker=marker, edgecolors='k')


def plot_centers(title, X, clusterCenters, pn, plot=True, marker='o'):
    """
    Plots the dataset with the cluster centers highlighted in red (the others white)
    :param title: string - the title of the plot
    :param X: matrix - the points of the dataset
    :param clusterCenters: list - list with the coordinates in the matrix of the cluster centers
    :param pn: integer - the number of partitions on columns and rows
    :param plot: boolean - optional, whether the plot function should be called or not (for ease of use)
    :param marker: character - optional, the marker of the plot

    :returns None
    """

    if plot:
        fig = plt.figure()
        plt.title(title)
        labels = np.zeros(len(X))
        for i in range(len(X)):
            for c in range(len(clusterCenters)):
                if math.floor(X[i, 0]) == clusterCenters[c][0] and math.floor(X[i, 1]) == clusterCenters[c][1]:
                    labels[i] = 1
        label_color = [cs.LABEL_COLOR_MAP[l] for l in labels]
        ax = fig.gca()
        ax.set_xticks(np.arange(0, pn, 1))
        ax.set_yticks(np.arange(0, pn, 1))
        plt.grid(True)
        plt.scatter(X[:, 0], X[:, 1], c=label_color, marker=marker, edgecolors='k')


def plot_grid(title, X, pn, labels=None, plot=True, marker='o'):
    """
    Plots the dataset with grid
    :param title: string - the title of the plot
    :param X: matrix - the points of the dataset
    :param pn: integer - the number of partitions on columns and rows
    :param labels: vector - optional, contains the labels of the points/X (has the same length as X)
    :param plot: boolean - optional, whether the plot function should be called or not (for ease of use)
    :param marker: character - optional, the marker of the plot

    :returns None
    """

    X = preprocessing.MinMaxScaler((0, pn)).fit_transform(X)
    if plot:
        nrDim = len(X[0])
        label_color = [cs.LABEL_COLOR_MAP[l] for l in labels]
        fig = plt.figure()
        plt.title(title)
        if nrDim == 2:
            ax = fig.gca()
            ax.set_xticks(np.arange(0, pn, 1))
            ax.set_yticks(np.arange(0, pn, 1))
            plt.scatter(X[:, 0], X[:, 1], marker=marker, c=label_color, s=25, edgecolor='k')
            plt.grid(True)
        if nrDim == 3:
            ax = Axes3D(fig)
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_zlabel("z")
            # ax.set_xticks(np.arange(0, pn, 1))
            # ax.set_zticks(np.arange(0, pn, 1))
            # ax.set_yticks(np.arange(0, pn, 1))
            ax.scatter(X[:, 0], X[:, 1], X[:, 2], marker=marker, c=label_color, s=25, )
            # plt.grid(True)


def plot_spikes(spikes, title=""):
    """"
    Plots spikes from a simulation
    :param spikes: matrix - the list of spikes in a simulation
    :param title: string - the title of the plot
    """
    for i in range(0, len(spikes), 300):
        plt.plot(np.arange(79), spikes[i])
    plt.title(title)
    plt.show()
