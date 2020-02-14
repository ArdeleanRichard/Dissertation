import numpy as np
import pywt
import matplotlib.pyplot as plt


def compute_haar(spikes):
    # coeffs[0] - approximation coefficient
    # coeffs[1] - details coefficient
    result = []
    for spike in spikes:
        coeffs = pywt.dwt(spike, 'haar')

#        max1, max2 = take_max_2(coeffs[1])
#        res = [max1, max2]
        res = take_max_10p(coeffs[1])
        result.append(res)

    # for i in range(0, len(spikes[0]), 300):
    plt.plot(np.arange(len(spikes[0])), spikes[0])  # blue
    coeff0 = pywt.dwt(spikes[0], 'haar')[0]  # orange approx
    plt.plot(np.arange(40), coeff0)
    coeff0 = pywt.dwt(spikes[0], 'haar')[1]  # green details
    plt.plot(np.arange(40), coeff0)
    plt.show()

    return np.array(result)


def take_max_10p(coeff):
    coeff = np.sort(coeff)
    res = []
    for i in range(0, len(coeff) // 10):
        res.append(coeff[len(coeff) - i -1])

    return res


def take_max_15p(coeff):
    coeff = np.sort(coeff)
    res = []
    for i in range(0, len(coeff) // 15):
        res.append(coeff[len(coeff) - i -1])

    return res


def take_max_2(coeff):
    max1 = max(coeff)
    arr = coeff
    arr = arr[arr != max1]
    max2 = max(arr)

    return max1, max2
