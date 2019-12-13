import matplotlib.pyplot as plt
import numpy as np


def get_closest_index(spike, index, height):
    """Returns either index or index + 1, the one that is closest to height"""
    return index if spike[index] - height < spike[index + 1] - height else index + 1


def get_spike_features(spikes):
    spike_height_width = []
    for spike in spikes:

        # plt.plot(np.arange(79), spike)

        spike_height_index = np.argmax(spike)
        spike_height = spike[spike_height_index]

        spike_width = 0

        left_width_index = 0
        right_width_index = 0
        for index in range(spike_height_index, len(spike) - 1):
            if spike[index] > spike_height / 2 > spike[index + 1]:
                right_width_index = get_closest_index(spike, index, spike_height)
                break
        for index in range(spike_height_index, 0, -1):
            if spike[index] < spike_height / 2 < spike[index + 1]:
                left_width_index = get_closest_index(spike, index, spike_height)
                break

        spike_width = abs(spike[right_width_index] - spike[left_width_index])
        # plt.plot([0, 80], [0, 0])
        # plt.plot(spike_height_index, spike_height / 2, marker='x')
        # plt.plot(left_width_index, spike[left_width_index], marker='o')
        # plt.plot(right_width_index, spike[right_width_index], marker='o')
        # plt.axvline(x=spike_height_index)
        # plt.show()
        spike_height_width.append([spike_height, spike_width])
    print(np.array(spike_height_width))
    return np.array(spike_height_width)
