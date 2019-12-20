import math

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import iqr, kurtosis, skew

import derivatives


def get_closest_index(signal, index, target):
    """Returns either index or index + 1, the one that is closest to height"""
    return index if abs(signal[index] - target) < abs(signal[index + 1] - target) else index + 1


def shape_features(spikes):
    features = []

    fdList = derivatives.compute_derivative(spikes)

    first_derivative = np.array(fdList)
    second_derivative = np.array(derivatives.compute_derivative(fdList))

    for spike_index in range(len(spikes)):
        spike = spikes[spike_index]

        fd = first_derivative[spike_index]
        sd = second_derivative[spike_index]

        spike_max_index = np.argmax(spike)
        spike_max = spike[spike_max_index]

        left_width_index = 0
        right_width_index = 0
        for index in range(spike_max_index, len(spike) - 1):
            if spike[index] > spike_max / 2 > spike[index + 1]:
                right_width_index = get_closest_index(spike, index, spike_max)
                break
        for index in range(spike_max_index, 0, -1):
            if spike[index] < spike_max / 2 < spike[index + 1]:
                left_width_index = get_closest_index(spike, index, spike_max)
                break

        spike_half_width = abs(spike[right_width_index] - spike[left_width_index])

        spike_min_index = np.argmin(spike)
        spike_min = spike[spike_min_index]

        fd_min_index = np.argmin(fd)
        fd_min = fd[fd_min_index]
        fd_max_index = np.argmax(fd)
        fd_max = fd[fd_max_index]

        plt.plot(np.arange(79), spike)
        plt.plot([0, 80], [0, 0])
        plt.plot(spike_max_index, spike_max / 2, marker='x')
        plt.plot(left_width_index, spike[left_width_index], marker='o')
        plt.plot(right_width_index, spike[right_width_index], marker='o')
        plt.axvline(x=spike_max_index)
        plt.show()

        features.append([spike_max, spike_half_width, spike_min, fd_min, fd_max])

    return np.array(features)


def derivative_features(spikes):
    features = []

    fdList = derivatives.compute_derivative(spikes)

    first_derivative = np.array(fdList)
    second_derivative = np.array(derivatives.compute_derivative(fdList))

    p1 = p2 = p3 = p4 = p5 = p6 = 0

    for spike_index in range(len(spikes)):
        fd = first_derivative[spike_index]
        sd = second_derivative[spike_index]

        spike = spikes[spike_index]

        peak_index = np.argmax(spike)

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

        # for index in range(p5, len(fd) - 1):
        #     if fd[index] < fd[index + 1]:
        #         p6 = get_closest_index(fd, index, 0) # TODO: FIX
        #         break

        p6 = np.argmin(fd[p5:p5 + 20]) + p5

        if spike_index % 20 == 0:
            plt.plot(np.arange(79), spike)
            plt.plot(peak_index, spike[peak_index], marker='o')
            # plt.plot(np.arange(79), fd)
            # plt.plot(p1, fd[p1], marker='o')
            # plt.plot(p2, fd[p2], marker='o')
            # plt.plot(p3, fd[p3], marker='o')
            # plt.plot(p4, fd[p4], marker='o')
            # plt.plot(p5, fd[p5], marker='o')
            # plt.plot(p6, fd[p6], marker='o')
            # plt.plot(np.arange(79), sd[0])
            plt.plot([0, 80], [0, 0])
            plt.show()

        f1 = p5 - p1
        f2 = fd[p4] - fd[p2]
        f3 = fd[p6] - fd[p2]
        f5 = math.log2(abs((fd[p4] - fd[p2]) / (p4 - p2)))
        f6 = (fd[p6] - fd[p4]) / (p6 - p4)
        f7 = math.log2(abs((fd[p6] - fd[p2]) / (p6 - p2)))
        f9 = ((fd[p2] - fd[p1]) / (p2 - p1)) / ((fd[p3] - fd[p2]) / (p3 - p2))
        f10 = ((fd[p4] - fd[p3]) / (p4 - p3)) / ((fd[p5] - fd[p4]) / (p5 - p4))
        f11 = fd[p2] / fd[p4]  # strange
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
            [f1, f2, f3, f12, f13, f14, f15, f16, f17, f18, f19, f20, f21, f22, f23, f24])
    return np.array(features)
