import matplotlib.pyplot as plt
import numpy as np
import pywt

import derivatives as deriv


def dwt_fd_method(spikes):
    result = []
    for spike in spikes:
        coeffsmatrix = haardecomposition(spike, 4)
        ca, cd4, cd3, cd2, cd1 = coeffsmatrix
        res1 = []
        res1.append(deriv.compute_fdmethod_1spike(ca))
        res1.append(deriv.compute_fdmethod_1spike(cd4))
        res1.append(deriv.compute_fdmethod_1spike(cd3))
        res1.append(deriv.compute_fdmethod_1spike(cd2))
        res1.append(deriv.compute_fdmethod_1spike(cd1))
        result.append(np.ndarray.flatten(np.array(res1)))
    return result


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
