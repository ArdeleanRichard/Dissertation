import numpy as np
import pywt
import matplotlib.pyplot as plt


def compute_haar(spikes):
    # coeffs[0] - approximation coefficient
    # coeffs[1] - details coefficient
    result = []
    for spike in spikes:
        coeffs = pywt.dwt(spike, 'haar')
        result.append(coeffs[1])

    # for i in range(0, len(spikes[0]), 300):
    plt.plot(np.arange(len(spikes[0])), spikes[0])  # blue
    coeff0 = pywt.dwt(spikes[0], 'haar')[0]  # orange
    plt.plot(np.arange(40), coeff0)
    coeff0 = pywt.dwt(spikes[0], 'haar')[1]  # green
    plt.plot(np.arange(40), coeff0)
    plt.show()
    return np.array(result)
