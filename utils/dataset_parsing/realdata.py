from utils import scatter_plot
import numpy as np
import struct
import matplotlib.pyplot as plt

spikes_per_channel = np.array([0, 11837, 2509, 1443, 2491, 18190, 9396, 876, 9484, 9947, 10558, 2095, 3046, 898,
                               1284, 5580, 6409, 9274, 13625, 419, 193, 3220, 2128, 281, 219, 4111, 1108, 5045, 6476,
                               973, 908,
                               787, 10734])
units_per_channel5 = [
    [],
    [1629, 474, 5951, 255],
    [],
    [],
    [686],
    [15231, 1386, 678],
    [1269, 1192, 3362, 2263, 192],
    [79],
    [684, 2053, 3125],
    [4313, 160, 123, 2582, 211],
    [1303, 6933, 1298],
    [],
    [285],
    [],
    [],
    [2658, 1489, 461],
    [1742, 150, 277],
    [5845, 542],
    [8762, 886, 699],
    [],
    [],
    [],
    [252],
    [],
    [],
    [1480, 745, 203],
    [],
    [2397, 512],
    [658, 1328, 138],
    [],
    [],
    [],
    [5899, 239],
]

units_per_channel = [
    [],
    [1773, 483, 2282],
    [2149, 280],
    [],
    [993, 2828],
    [32565, 200],
    [1061, 1362, 135, 1102],
    [],
    [2085, 3056, 692],
    [145, 349, 220],
    [1564],
    [9537],
    [14264],
    [4561],
    [6926],
    [1859, 439, 1359],
    [309, 1877],
    [1379, 242],
    [2739],
    [],
    [],
    [],
    [],
    [],
    [],
    [1149],
    [201, 244],
    [],
    [109, 209],
    [413],
    [377],
    [421],
    [276, 19014],
]


def read_waveforms(filename):
    with open(filename, 'rb') as file:
        waveforms = []
        read_val = file.read(4)
        waveforms.append(struct.unpack('f', read_val)[0])

        while read_val:
            read_val = file.read(4)
            try:
                waveforms.append(struct.unpack('f', read_val)[0])
            except struct.error:
                break

        return np.array(waveforms)


def sum_until_channel(channel):
    ch_sum = 0
    for i in units_per_channel[:channel]:
        ch_sum += np.sum(np.array(i)).astype(int)

    return ch_sum


def get_spike_units(waveform, channel, plot_spikes=False):
    spikes = []
    new_spikes = np.zeros((np.sum(units_per_channel[channel]), 58))

    for i in range(0, len(waveform), 58):
        spikes.append(waveform[i: i + 58])

    left_limit_spikes = sum_until_channel(channel)
    right_limit_spikes = left_limit_spikes + np.sum(np.array(units_per_channel[channel]))
    # print(left_limit_spikes, right_limit_spikes)
    spikes = spikes[left_limit_spikes: right_limit_spikes]

    if plot_spikes:
        for i in range(0, len(spikes), 1000):
            plt.plot(np.arange(58), -spikes[i])
        plt.show()

    labels = np.array([])
    for i, units in enumerate(units_per_channel[channel]):
        labels = np.append(labels, np.repeat(i, units))

    for i, units in enumerate(units_per_channel[channel]):
        left_lim = sum_until_channel(channel)
        right_lim = left_lim + units

        spike_index = 0
        for j in range(len(spikes)):
            # print(j, left_lim, right_lim)
            new_spikes[spike_index] = spikes[j]
            spike_index += 1
    return new_spikes, labels.astype(int)


def plot_spikes_per_unit(waveforms):
    for channel in range(0, len(units_per_channel)):
        if len(units_per_channel[channel]) > 0:
            spikes, labels = get_spike_units(waveforms, channel)
            start = 0
            unit_pos = 0
            for unit in units_per_channel[channel]:
                print(unit)
                print(start + unit)
                scatter_plot.plot_spikes(-spikes[start:start + unit],
                                         "Channel" + str(channel) + "Cluster" + str(unit_pos), )
                start += unit
                unit_pos += 1
