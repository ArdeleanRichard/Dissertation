import numpy as np
from scipy.ndimage import gaussian_filter, gaussian_filter1d, laplace, gaussian_laplace
from scipy.signal import savgol_filter


def compute_fdmethod(spikes):
    final_result = []

    for x in spikes:
        first_derivative = compute_derivative5stencil(x)
        f_min_pos = compute_min_pos(first_derivative)
        f_max_pos = compute_max_pos(first_derivative)
        result = []
        result.append(x[f_max_pos] - x[f_min_pos])
        result.append(max(x))
        final_result.append(result)

    return np.array(final_result)


def compute_fdmethod3d(spikes):
    final_result = []

    for x in spikes:
        first_derivative = compute_derivative5stencil(x)
        f_min_pos = compute_min_pos(first_derivative)
        f_max_pos = compute_max_pos(first_derivative)
        result = []
        result.append(x[f_max_pos])
        result.append(x[f_min_pos])
        result.append(max(x))
        final_result.append(result)

    return np.array(final_result)


def compute_first_second_derivative3d(spikes):
    final_result = []

    for x in spikes:
        first_derivative = compute_derivative5stencil(x)

        f_max_pos = compute_max_pos(first_derivative)

        second_derivative = compute_derivative5stencil(first_derivative)

        s_min_pos = compute_min_pos(second_derivative)
        s_max_pos = compute_max_pos(second_derivative)

        result = []
        result.append(x[f_max_pos])
        result.append(x[s_max_pos])
        result.append(x[s_min_pos])
        final_result.append(result)

    return np.array(final_result)


def compute_first_second_derivative(spikes):
    final_result = []

    for x in spikes:
        first_derivative = compute_derivative5stencil(x)

        f_min_pos = compute_min_pos(first_derivative)
        f_max_pos = compute_max_pos(first_derivative)

        second_derivative = compute_derivative5stencil(first_derivative)

        s_min_pos = compute_min_pos(second_derivative)
        s_max_pos = compute_max_pos(second_derivative)

        result = []
        f_pos, s_pos = compute_positionMethod6(x[f_min_pos], x[f_max_pos], x[s_min_pos], x[s_max_pos])
        result.append(f_pos)
        result.append(s_pos)
        final_result.append(result)

    return np.array(final_result)


# for fsde
def compute_positionMethod6(f_min, f_max, s_min, s_max):
    first = (f_min + f_max) / 2
    second = (s_min + s_max) / 2
    return first, second


# for fsde
def compute_position_method5(f_min_pos, f_max_pos, s_min_pos, s_max_pos):
    f_pos = abs(f_max_pos - f_min_pos)
    s_pos = abs(s_max_pos - s_min_pos)
    return f_pos, s_pos


def gaussian_filter(spikes):
    return gaussian_filter1d(spikes, sigma=3, order=0, truncate=9.0)


def compute_min_pos(array):
    # print("min fd")
    # print(min(array))
    return array.index(min(array))


def compute_max_pos(array):
    # print("max fd")
    # print(max(array))
    return array.index(max(array))


def compute_derivative(function):
    first_derivative = []

    for i in range(1, len(function)):
        first_derivative.append(function[i] - function[i - 1])
    first_derivative.append(0)
    return first_derivative


def compute_derivative5stencil(function):
    first_derivative = []

    for i in range(2, len(function) - 3):
        x = (-function[i + 2] + 8 * function[i + 1] - 8 * function[i - 1] + function[i - 2]) / 12
        first_derivative.append(x)

    first_derivative.append(0)
    first_derivative.append(0)
    first_derivative.append(0)
    first_derivative.append(0)
    first_derivative.append(0)

    return first_derivative
