import numpy as np
import pywt
import matplotlib.pyplot as plt
from matplotlib import cm
import scaleogram as scg
import math
import derivatives as deriv
from scaleogram import CWT
# from oct2py import octave, Oct2Py
# from multiprocessing.dummy import Pool as ThreadPool

#
# def plot_superlet(spikes, spike_pos, sim_nr):
#     spike = spikes[spike_pos]
#     oct = Oct2Py()
#     result = oct.call_aslt(spike)
#     values = np.abs(result)
#     scales_freq = np.arange(50, 100)
#     df = scales_freq[-1] / scales_freq[-2]
#     ymesh = np.concatenate([scales_freq, [scales_freq[-1] * df]])
#     im = plt.pcolormesh(np.arange(len(spike)), ymesh, values, cmap="jet")
#     plt.colorbar(im)
#     plt.title("Sim" + str(sim_nr) + "_spike" + str(spike_pos) + "_superlets")
#     plt.xlabel("Time")
#     plt.ylabel("Frequency - linear")
#     plt.show()


def plot_wavelet(spikes, spike_pos, sim_nr):
    # wav = pywt.ContinuousWavelet('cmor')
    # print(wav.center_frequency)
    # print(wav.bandwidth_frequency)
    n = len(spikes[spike_pos])
    spike = spikes[spike_pos]
    time = np.arange(n)
    # wavelet = 'cmor0.1-1.5'
    wavelet = 'morl'
    scales = np.arange(1, min(len(time) / 10, 100))
    # scales = np.arange(10, 100)

    coeffs, scales_freq = pywt.cwt(spike, scales, wavelet)
    coeffs = np.abs(coeffs)
    df = scales_freq[-1] / scales_freq[-2]
    ymesh = np.concatenate([scales_freq, [scales_freq[-1] * df]])
    values = np.abs(coeffs)
    im = plt.pcolormesh(time, ymesh, values, cmap="jet")
    plt.colorbar(im)
    plt.title("Sim" + str(sim_nr) + "_spike" + str(spike_pos) + "wavelets")
    plt.xlabel("Time")
    plt.ylabel("Frequency - linear")
    plt.show()

#
# def superlet_1spike(spike):
#     oct = Oct2Py()
#     result = oct.call_aslt(spike)
#     print("MATLAB:")
#     print(result)
#     # coeffs = deriv.compute_fdmethod(result)
#     return np.ndarray.flatten(result)
#
#
# def superlets_parallel(spikes, derivative=True):
#     octave.addpath('C:\poli\year4\licenta\codeGoodGit\Dissertation')
#     pool = ThreadPool(4)
#     result = pool.map(superlet_1spike, spikes)
#     return result
#
#
# def superlets(spikes):
#     octave.addpath('C:\poli\year4\licenta\codeGoodGit\Dissertation')
#     results = []
#     i = 0
#     for spike in spikes:
#         result = octave.call_aslt(spike)
#         pool = ThreadPool(8)
#         coeffs = pool.map(deriv.compute_fdmethod_1spike, result)
#         print(i)
#         results.append(np.ndarray.flatten(np.array(coeffs)))
#         i = i + 1
#     return results


def fd_wavelets(spikes):
    # wavelet = 'morl'
    # wavelet  = 'cmor0.8-1.5'
    wavelet = 'cmor0.1-1.5'
    result = []
    # time = np.arange(len(spikes[0]))
    # scales = np.arange(1, min(len(time) / 10, 100))
    scales = np.arange(10,100)

    # scales = np.arange(1, min(len(spikes) / 10, 100))
    for spike in spikes:
        coeffs, scales_freq = pywt.cwt(spike, scales, wavelet)
        coeffs = deriv.compute_fdmethod(np.abs(coeffs))
        result.append(np.ndarray.flatten(coeffs))
    return result


def wavelets(spikes):
    # wavelet = 'morl'
    # wavelet  = 'cmor0.8-1.5'
    wavelet = 'cmor0.1-1.5'
    result = []
    # time = np.arange(len(spikes[0]))
    # scales = np.arange(1, min(len(time) / 10, 100))
    scales = np.arange(10,100)

    scales = np.arange(1, min(len(spikes) / 10, 100))
    for spike in spikes:
        coeffs, scales_freq = pywt.cwt(spike, scales, wavelet)
        # coeffs = deriv.compute_fdmethod(np.abs(coeffs))
        result.append(np.ndarray.flatten(np.abs(coeffs)))
    return result
