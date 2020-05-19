import math

import numpy as np
from scipy.signal import argrelextrema
from scipy.stats import iqr, kurtosis, skew
from sklearn.decomposition import PCA

import derivatives


def get_closest_index(signal, index, target):
    """Returns either index or index + 1, the one that is closest to target"""
    return index if abs(signal[index] - target) < abs(signal[index + 1] - target) else index + 1


def get_max(signal):
    max_index = np.argmax(signal)
    max_value = signal[max_index]
    return max_value, max_index


def get_min(signal):
    min_index = np.argmin(signal)
    min_value = signal[min_index]
    return min_value, min_index


def get_half_width(spike):
    left_width_index = 0
    right_width_index = 0

    spike_max, spike_max_index = get_max(spike)
    spike_min, spike_min_index = get_min(spike)

    for index in range(spike_max_index, len(spike) - 1):
        if spike[index] > spike_max / 2 > spike[index + 1]:
            right_width_index = get_closest_index(spike, index, spike_max)
            break
    for index in range(spike_max_index, 0, -1):
        if spike[index] < spike_max / 2 < spike[index + 1]:
            left_width_index = get_closest_index(spike, index, spike_max)
            break

    spike_half_width = abs(spike[right_width_index] - spike[left_width_index])
    return spike_half_width, right_width_index, left_width_index


def get_valleys_near_peak(spike):
    spike_max, spike_max_index = get_max(spike)
    try:
        left_min_index = 1 + argrelextrema(spike[1:spike_max_index], np.less)[0][0]
    except IndexError:
        left_min_index = 0
    try:
        right_min_index = spike_max_index + argrelextrema(spike[spike_max_index:], np.less)[0][0]
    except IndexError:
        right_min_index = len(spike) - 1
    return left_min_index, spike[left_min_index], right_min_index, spike[right_min_index]


def get_features(spikes):
    """:returns array representing extracted features"""
    features = []

    for spike in spikes:
        fd = np.array(derivatives.compute_derivative5stencil(spike))
        sd = np.array(derivatives.compute_derivative5stencil(fd))

        spike_max, spike_max_index = get_max(spike)
        spike_min, spike_min_index = get_min(spike)

        fd_max, fd_max_index = get_max(fd)
        fd_min, fd_min_index = get_min(fd)

        left_min_index, left_min, right_min_index, right_min = get_valleys_near_peak(spike)
        valley_diff = abs(right_min - left_min)

        spike_half_width, right_width_index, left_width_index = get_half_width(spike)

        # plt.plot(np.arange(79), spike)
        # plt.plot(left_min_index, left_min, marker='o')
        # plt.plot(right_min_index, right_min, marker='o')
        # plt.plot(spike_max_index, spike_max, marker='o')
        # # plt.plot(np.arange(79), fd)
        # plt.plot([0, 80], [0, 0])
        # plt.plot(spike_max_index, spike_max / 2, marker='x')
        # plt.plot(spike_min_index, spike_min, marker='x')
        # plt.plot(left_width_index, spike[left_width_index], marker='o')
        # plt.plot(right_width_index, spike[right_width_index], marker='o')
        # plt.axvline(x=spike_max_index)
        # plt.show()

        features.append([fd_min, fd_max])
    # exit()

    # pca_2d = PCA(n_components=2)
    # waveform_pca2d = pca_2d.fit_transform(spikes)
    # pca_2d_features = pca_2d.fit_transform(features)

    # return np.array(np.concatenate((waveform_pca2d, np.array(features)), axis=1))
    return np.array(features)


def get_derivative_features(spikes):
    """
    :returns derivative based features
    P1	First zero-crossing of the FD before the action potential has been detected
    P2	Valley of the FD of the action potential
    P3	Second zero-crossing of the FD of the action potential that has been detected
    P4	Peak of the FD of the action potential
    P5	Third zero-crossing of the FD after the action potential has been detected
    P6	Valley of the FD after the action potential
    """

    features = []

    p1 = p2 = p3 = p4 = p5 = p6 = 0

    for spike in spikes:

        fd = np.array(derivatives.compute_derivative5stencil(spike))
        sd = np.array(derivatives.compute_derivative5stencil(fd))

        spike_max, spike_max_index = get_max(spike)
        spike_min, spike_min_index = get_min(spike)

        fd_max, fd_max_index = get_max(fd)
        fd_min, fd_min_index = get_min(fd)

        p2 = np.argmin(fd)

        for index in range(p2 - 1, 0, -1):
            if fd[index] > 0 > fd[index + 1]:
                p1 = get_closest_index(fd, index, 0)
                break

        for index in range(p2 + 1, len(fd) - 1):
            if fd[index] < 0 < fd[index + 1]:
                p3 = get_closest_index(fd, index, 0)
                break

        for index in range(p2 + 1, len(fd) - 1):
            if fd[index] > fd[index + 1] and fd[index] > 0:
                p4 = index
                break

        for index in range(p4 + 1, len(fd) - 1):
            if fd[index] > 0 > fd[index + 1]:
                p5 = get_closest_index(fd, index, 0)
                break

        p6 = np.argmin(fd[p5:p5 + 20]) + p5

        # # plt.plot(np.arange(79), spike)
        # # plt.plot(spike_max_index, spike_max, marker='o')
        # plt.plot(np.arange(79), fd)
        # plt.plot(p1, fd[p1], marker='o')
        # plt.plot(p2, fd[p2], marker='o')
        # plt.plot(p3, fd[p3], marker='o')
        # plt.plot(p4, fd[p4], marker='o')
        # plt.plot(p5, fd[p5], marker='o')
        # plt.plot(p6, fd[p6], marker='o')
        # plt.axvline(x=spike_max_index)
        # # plt.plot(np.arange(79), sd[0])
        # plt.plot([0, 80], [0, 0])
        # plt.show()

        f1 = p5 - p1
        f2 = fd[p4] - fd[p2]
        f3 = fd[p6] - fd[p2]

        f5 = math.log2(abs((fd[p4] - fd[p2]) / (p4 - p2)))
        f6 = (fd[p6] - fd[p4]) / (p6 - p4)
        f7 = math.log2(abs((fd[p6] - fd[p2]) / (p6 - p2)))
        f9 = ((fd[p2] - fd[p1]) / (p2 - p1)) / ((fd[p3] - fd[p2]) / (p3 - p2))
        f10 = ((fd[p4] - fd[p3]) / (p4 - p3)) / ((fd[p5] - fd[p4]) / (p5 - p4))
        f11 = fd[p2] / fd[p4]

        f12 = fd[p1]
        f13 = fd[p3]
        f14 = fd[p4]
        f15 = fd[p5]
        f16 = fd[p6]
        f17 = sd[p1]
        f18 = sd[p3]
        f19 = sd[p5]
        f20 = iqr(fd)
        f21 = iqr(sd)
        f22 = kurtosis(fd)
        f23 = skew(fd)
        f24 = skew(sd)

        features.append(
            [spike_max, fd_max, fd_min, f2])

    return np.array(features)
