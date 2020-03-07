import numpy as np
import pywt
import matplotlib.pyplot as plt
from matplotlib import cm
import scaleogram as scg
import math
import derivatives as deriv
from scaleogram import CWT


def wavelet_coeff_height(spikes):
    wavelet = 'morl'
    result = []
    time = np.arange(len(spikes[0]))
    scales = np.arange(1,70)
    for spike in spikes:
        res = []
        coeffs, scales_freq = pywt.cwt(spike, scales, wavelet)
        pos = 0
        max = np.max(coeffs[0])
        posmax = 0
        for coeff in coeffs:
            if np.max(coeff)>max:
                max = np.max(coeff)
                posmax = pos
            pos = pos+1
        res.append(posmax)
        res.append(max)
        result.append(res)
    return np.array(result)


def fd_wavelets(spikes):
    wavelet = 'morl'
    result = []
    time = np.arange(len(spikes[0]))
    # scales = np.arange(1, min(len(time) / 10, 100))
    # scales = np.arange(1,50)
    scales = np.arange(1,80)
    for spike in spikes:
        coeffs, scales_freq = pywt.cwt(spike, scales, wavelet)
        coeffs = deriv.compute_fdmethod(coeffs)
        result.append(np.ndarray.flatten(coeffs))
    return result


def logarithmiccwt(spikes, i, log=True):
    data = spikes[i]
    n = len(data)
    time = np.arange(n)
    wavelet = 'morl'
    # data = list(map((lambda x: math.log10(math.fabs(x) + 0.01)), data))

    dt = time[1] - time[0]
    scales = np.arange(1, min(len(time) / 10, 100))
    coeffs, scales_freq = pywt.cwt(spikes[0], scales, wavelet)
    # coeffs = list(map((lambda x: math.log10(math.fabs(x) + 0.01)), coeffs))

    if log:
        for i in np.arange(0, len(coeffs)):
            for j in np.arange(0, len(coeffs[i])):
                # if coeffs[i, j] != 0:
                coeffs[i, j] = math.log10(math.fabs(coeffs[i, j] + 0.00001))

    df = scales_freq[-1] / scales_freq[-2]
    ymesh = np.concatenate([scales_freq, [scales_freq[-1] * df]])
    values = np.abs(coeffs)
    im = plt.pcolormesh(time, ymesh, values, cmap="jet")
    plt.colorbar(im)
    if log:
        plt.title("Log(abs(coefficients))")
    else:
        plt.title("abs(coefficients)")
    plt.xlabel("Time")
    plt.ylabel("Frequency - linear")
    # plt.savefig("morl_scaleauto_frequencyyaxis_sim15_spike0_log=true.png")
    plt.show()

    plt.plot(np.arange(79), spikes[0])
    plt.title("Spike0 Sim15")
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    # plt.savefig("plot_sim15_spike0.png")
    plt.show()


def test():
    n = 300
    time = np.arange(n)
    p1 = 20
    f1 = 1. / p1
    p2 = 60
    f2 = 1. / p2
    data = np.cos((2 * np.pi * f1) * time) + 0.6 * np.cos((2 * np.pi * f2) * time)
    wavelet = 'cmor0.7-1.5'
    #
    # cmor0.7-1.5'
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 6))
    fig.subplots_adjust(hspace=0.3)
    ax1.plot(time, data)
    ax1.set_xlim(0, n)
    ax1.set_title('Example 1: time domain signal with two cos() waves with period p1=%ds and p2=%ds' % (p1, p2))
    ax2 = scg.cws(time, data, scales=np.arange(1, 150), wavelet=wavelet,
                  ax=ax2, cmap="jet", cbar="vertical", ylabel="Period [seconds]", xlabel="Time [seconds]",
                  title='Example 1: scaleogram with linear period yscale', yaxis="period")
    txt = ax2.annotate("p1=%ds" % p1, xy=(n / 2, p1), xytext=(n / 2 - 10, p1), bbox=dict(boxstyle="round4", fc="w"))
    txt = ax2.annotate("p2=%ds" % p2, xy=(n / 2, p2), xytext=(n / 2 - 10, p2), bbox=dict(boxstyle="round4", fc="w"))
    plt.show()


def cwt(spikes, i):
    # scales = np.arange(1, 128)
    # [coefficients, frequencies] = pywt.cwt(spikes[0], scales, wavelet='mexh')
    # plt.plot(np.arange(0,len(coefficients)), coefficients)
    # plt.plot(np.arange(0,len(frequencies)), frequencies)
    # plt.show()

    t0 = 0
    dt = 0.24
    # scales = np.arange(1, 128)
    # [coefficients, frequencies] = pywt.cwt(spikes[0], scales, wavelet='cmor')
    # shape of coefficieints is 127, 79
    # shape of frequencies is 127,1
    # signal = np.ndarray.flatten(spikes)
    data = spikes[i]
    n = len(data)
    time = np.arange(n)
    wavelet = 'cmor0.7-1.5'
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 6))
    fig.subplots_adjust(hspace=0.3)
    ax1.plot(time, data)
    ax1.set_xlim(0, n)
    ax1.set_title('Plot for spike %ds' % i)
    # data = list(map((lambda x: math.log10(math.fabs(x) + 0.01)), data))
    ax2 = scg.cws(time, data, wavelet=wavelet, coi=True,
                  ax=ax2, cmap="jet", cbar="horizontal", xlabel="Time",
                  title='Scaleogram with log frequency yscale', yaxis="frequency")
    # plt.savefig("cmor0.7-1.5_scale1-150_frequencyyaxis_sim15_spike%d.png" % i)
    plt.show()


def plot_wavelet(time, signal, scales,
                 waveletname='cmor',
                 # cmap=cm.cmaps_listed['seismic'],
                 title='Wavelet Transform (Power Spectrum) of signal',
                 ylabel='Period (years)',
                 xlabel='Time'):
    dt = time[1] - time[0]
    [coefficients, frequencies] = pywt.cwt(signal, scales, waveletname, dt)
    power = (abs(coefficients)) ** 2
    period = 1. / frequencies
    levels = [0.0625, 0.125, 0.25, 0.5, 1, 2, 4, 8]
    contourlevels = np.log2(levels)

    fig, ax = plt.subplots(figsize=(15, 10))
    im = ax.contourf(time, np.log2(period), np.log2(power), contourlevels, extend='both')

    ax.set_title(title, fontsize=20)
    ax.set_ylabel(ylabel, fontsize=18)
    ax.set_xlabel(xlabel, fontsize=18)

    yticks = 2 ** np.arange(np.ceil(np.log2(period.min())), np.ceil(np.log2(period.max())))
    ax.set_yticks(np.log2(yticks))
    ax.set_yticklabels(yticks)
    ax.invert_yaxis()
    ylim = ax.get_ylim()
    ax.set_ylim(ylim[0], -1)

    cbar_ax = fig.add_axes([0.95, 0.5, 0.03, 0.25])
    fig.colorbar(im, cax=cbar_ax, orientation="vertical")
    plt.show()


def compute_haar(spikes):
    # without approx coeff
    # coeffsmatrix[0] - approximation coefficient
    # coeffsmatrix[1] - details coefficient

    result = []

    for spike in spikes:
        coeffsmatrix = haardecomposition(spike, 4)
        ca, cd4, cd3, cd2, cd1 = coeffsmatrix
        coeffs = np.concatenate((cd4, cd3, cd2, cd1))
        variance = np.var(coeffs)
        std = np.std(coeffs)
        mean = np.mean(coeffs)
        coeffs = coeffs[(coeffs >= mean - 3 * std) & (coeffs <= mean + 3 * std)]  # gets rid of outliers
        gauss = np.random.normal(mean, std, len(coeffs))
        cdf = np.sort(coeffs)
        cdfGauss = np.sort(gauss)
        lilliefors = np.zeros(len(cdf))
        for i in np.arange(0, len(cdf)):
            lilliefors[i] = (abs(np.argwhere(cdf == coeffs[i])[0] - np.argwhere(cdfGauss == gauss[i])[0]))
        respos = take_max_10positions(lilliefors)
        res1 = []
        respos = np.ndarray.flatten(respos)
        for i in range(0, len(respos)):
            res1.append(coeffs[respos[i]])
        result.append(res1)

    return result


def compute_haar2(spikes):
    # with approx coeff
    # coeffsmatrix[0] - approximation coefficient
    # coeffsmatrix[1] - details coefficient

    result = []

    for spike in spikes:
        coeffsmatrix = haardecomposition(spike, 4)
        ca, cd4, cd3, cd2, cd1 = coeffsmatrix
        coeffs = np.concatenate((ca, cd4, cd3, cd2, cd1))
        variance = np.var(coeffs)
        std = np.std(coeffs)
        mean = np.mean(coeffs)
        coeffs = coeffs[(coeffs >= mean - 3 * std) & (coeffs <= mean + 3 * std)]  # gets rid of outliers
        gauss = np.random.normal(mean, std, len(coeffs))
        cdf = np.sort(coeffs)
        cdfGauss = np.sort(gauss)
        lilliefors = np.zeros(len(cdf))
        for i in np.arange(0, len(cdf)):
            lilliefors[i] = (abs(np.where(cdf == coeffs[i])[0] - np.where(cdfGauss == gauss[i])[0]))
        respos = take_max_10positions(lilliefors)
        res1 = []
        respos = np.ndarray.flatten(respos)
        for i in range(0, len(respos)):
            res1.append(coeffs[respos[i]])
        result.append(res1)
    return np.array(result)


def haardecomposition(spike, level):
    coeffsmatrix = pywt.wavedec(spike, 'haar', mode='per', level=level)
    return coeffsmatrix


def take_max_10positions(coeff):
    res = np.argsort(coeff)[-10:][::-1]
    return res


def returnCoefficients(spikes, lvl):
    result = []
    for spike in spikes:
        coeffsmatrix = haardecomposition(spike, lvl)
        coeffs = []
        for i in np.arange(0, lvl + 1):
            coeffs = np.concatenate((coeffs, coeffsmatrix[i]))
        result.append(coeffs)
    return result


def returnCoefficient(spikes, lvl):
    result = []
    for spike in spikes:
        coeffsmatrix = haardecomposition(spike, lvl)
        result.append(coeffsmatrix[2])
    return result


def testplots(spikes):
    coeffsmatrix = pywt.dwt(spikes[0], 'haar')
    coeffs = np.concatenate((coeffsmatrix[0], coeffsmatrix[1]))
    variance = np.var(coeffs)
    std = np.std(coeffs)
    mean = np.mean(coeffs)
    gauss = np.random.normal(mean, std, 80)
    count, bins, ignored = plt.hist(gauss, 50, density=True)
    plt.plot(bins, 1 / (std * np.sqrt(2 * np.pi)) * np.exp(- (bins - mean) ** 2 / (2 * std ** 2)), linewidth=2,
             color='r')
    plt.show()

    plt.plot(np.arange(len(gauss)), gauss)
    plt.plot(np.arange(len(coeffs)), coeffs)
    plt.show()

    # for i in range(0, len(spikes[0]), 300):
    # plt.plot(np.arange(len(spikes[0])), spikes[0])  # blue
    # coeff0 = pywt.dwt(spikes[0], 'haar')[0]  # orange approx
    # plt.plot(np.arange(40), coeff0)
    # coeff0 = pywt.dwt(spikes[0], 'haar')[1]  # green details
    # plt.plot(np.arange(40), coeff0)
    # plt.show()


def plotspike0coeffs(spikes):
    coeffsmatrix = haardecomposition(spikes[0], 4)
    ca, cd4, cd3, cd2, cd1 = coeffsmatrix
    coeffs = np.concatenate((ca, cd4, cd3, cd2, cd1))
    plt.plot(np.arange(0, len(spikes[0])), spikes[0])
    plt.plot(np.arange(0, len(ca)), ca)
    plt.plot(np.arange(len(ca) - 1, len(cd4) + len(ca) - 1), cd4)
    plt.plot(np.arange(len(cd4) + len(ca) - 2, len(cd4) + len(ca) - 2 + len(cd3)), cd3)
    plt.plot(np.arange(len(cd4) + len(ca) - 3 + len(cd3), len(cd4) + len(ca) - 3 + len(cd3) + len(cd2)), cd2)
    plt.plot(np.arange(len(cd4) + len(ca) - 4 + len(cd3) + len(cd2),
                       len(cd4) + len(ca) - 4 + len(cd3) + len(cd2) + len(cd1)), cd1)

    plt.show()
