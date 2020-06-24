import numpy as np
import scipy.signal as signal
from PyEMD import EMD
from scipy.fftpack import fft
from scipy.signal import hilbert
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import derivatives
import derivatives as deriv
import discretewlt as dwt
import shape_features
import stft_dpss
import superlets as slt
import wavelets as wlt


def continuous_wavelet_transform(spikes):
    """
    Load the dataset after applying continuous wavelets on 2 dimensions
    :returns result_spikes: matrix - the 2-dimensional points resulted
    """
    cwt_features = wlt.fd_wavelets(spikes)

    scaler = StandardScaler()
    features = scaler.fit_transform(cwt_features)

    return features


def discrete_wavelet_transform(spikes):
    """
    Load the dataset after dwt on 2 dimensions

    :returns result_spikes: matrix - the 2-dimensional points resulted
    """
    dwt_features = dwt.dwt_fd_method(spikes)

    scaler = StandardScaler()
    features = scaler.fit_transform(dwt_features)

    return features


def superlets(spikes):
    slt_features = slt.slt(spikes, 2, 1.1)
    # slt_features = slt.slt2(spikes, 5, 1.5)

    scaler = StandardScaler()
    features = scaler.fit_transform(slt_features)

    return features


def derivatives2d(spikes):
    """
    Extract derivatives on 2 dimensions
    :returns result_spikes: matrix - the 2-dimensional points resulted
    """
    derivative_features = deriv.compute_fdmethod(spikes)

    scaler = StandardScaler()
    features = scaler.fit_transform(derivative_features)

    return features


def derivatives3d(spikes):
    result_spikes = deriv.compute_fdmethod3d(spikes)

    return result_spikes


def shape_phase_distribution(spikes):
    features = shape_features.get_shape_phase_distribution_features(spikes)

    return features


def hilbert_envelope(spikes):
    spikes_hilbert = hilbert(spikes)
    envelope = np.abs(spikes_hilbert)

    return envelope


def emd_signal_no_residuum(spikes):
    emd = EMD()

    spikes = np.array(spikes)
    features = np.zeros((spikes.shape[0], spikes.shape[1]))
    for i, spike in enumerate(spikes):
        emd(spike)
        IMFs, res = emd.get_imfs_and_residue()
        features[i] = np.sum(IMFs, axis=0)

    return features


def emd_imf_derivatives(spikes):
    emd = EMD()

    features = np.zeros((spikes.shape[0], 8))
    for i, spike in enumerate(spikes):
        emd(spike)
        IMFs, res = emd.get_imfs_and_residue()

        f = np.array(deriv.compute_fdmethod(IMFs))

        if IMFs.shape[0] >= 4:
            features[i] = np.concatenate((f[0], f[1], f[2], f[3]))
        elif IMFs.shape[0] >= 3:
            features[i] = np.concatenate((f[0], f[1], f[2], [0, 0]))
        else:
            features[i] = np.concatenate((f[0], f[1], [0, 0], [0, 0]))

    return features


def stft(spikes):
    sampled_frequencies, time_segments, Zxx = signal.stft(spikes, window='blackman', fs=1, nperseg=52)
    stft_signal = Zxx.reshape(*Zxx.shape[:1], -1)
    amplitude = np.abs(stft_signal)
    return amplitude


def stft_d(spikes):
    sampled_frequencies, time_segments, Zxx = signal.stft(spikes, window='blackman', fs=1, nperseg=35)
    amplitude = np.abs(Zxx)
    amplitude = np.apply_along_axis(derivatives.compute_fdmethod_1spike, 2, amplitude)
    amplitude = amplitude.reshape(*amplitude.shape[:1], -1)
    return amplitude


def stft_multitaper(spikes):
    win = stft_dpss.generate_dpss_windows()
    f0, t0, zxx0 = signal.stft(spikes, window=win[0], nperseg=58, fs=1)
    f2, t2, zxx2 = signal.stft(spikes, window=win[2], nperseg=58, fs=1)
    Zxx = np.concatenate([zxx0, zxx2], axis=2)
    stft_signal = Zxx.reshape(*Zxx.shape[:1], -1)
    amplitude = np.abs(stft_signal)
    return amplitude


def stft_multitaper_w(spikes):
    win = stft_dpss.generate_dpss_windows()
    f, t, Zxx = signal.stft(spikes, window=win[1], nperseg=58, fs=1)
    stft_signal = Zxx.reshape(*Zxx.shape[:1], -1)
    amplitude = np.abs(stft_signal)
    return amplitude


def fourier_real(spikes):
    fft_signal = fft(spikes)
    X = [x.real for x in fft_signal[:, 0:40]]
    return X


def fourier_imaginary(spikes):
    fft_signal = fft(spikes)
    Y = [x.imag for x in fft_signal[:, 0:40]]
    return Y


def fourier_amplitude(spikes):
    fft_signal = fft(spikes)
    X = [x.real for x in fft_signal[:, 0:40]]
    Y = [x.imag for x in fft_signal[:, 0:40]]
    amplitude = np.sqrt(np.add(np.multiply(X, X), np.multiply(Y, Y)))
    return amplitude


def fourier_phase(spikes):
    fft_signal = fft(spikes)
    X = [x.real for x in fft_signal[:, 0:40]]
    Y = [x.imag for x in fft_signal[:, 0:40]]
    phase = np.arctan2(Y, X)
    return phase


def fourier_power(spikes):
    fft_signal = fft(spikes)
    X = [x.real for x in fft_signal[:, 0:40]]
    Y = [x.imag for x in fft_signal[:, 0:40]]
    amplitude = np.sqrt(np.add(np.multiply(X, X), np.multiply(Y, Y)))
    power = amplitude * amplitude
    return power


def reduce_dimensionality(n_features, method='PCA2D'):
    if method.lower() == 'pca2d':
        pca_2d = PCA(n_components=2)
        features = pca_2d.fit_transform(n_features)
    elif method.lower() == 'pca3d':
        pca_3d = PCA(n_components=3)
        features = pca_3d.fit_transform(n_features)
    elif method.lower() == 'derivatives2d':
        features = deriv.compute_fdmethod(n_features)
    elif method.lower() == 'derivatives3d':
        features = deriv.compute_fdmethod3d(n_features)
    elif method.lower() == 'derivatives_pca2d':
        features = deriv.compute_fdmethod(n_features)
        pca_2d = PCA(n_components=2)
        features = pca_2d.fit_transform(features)
    else:
        features = []
    return features


def apply_feature_extraction_method(spikes, feature_extraction_method=None, dim_reduction_method=None):
    spikes = np.array(spikes)

    if feature_extraction_method.lower() == 'pca2d':
        features = reduce_dimensionality(spikes, feature_extraction_method)
    elif feature_extraction_method.lower() == 'pca3d':
        features = reduce_dimensionality(spikes, feature_extraction_method)
    elif feature_extraction_method.lower() == 'derivatives2d':
        features = derivatives2d(spikes)
    elif feature_extraction_method.lower() == 'derivatives3d':
        features = derivatives3d(spikes)
    elif feature_extraction_method.lower() == 'superlets':
        features = superlets(spikes)
    elif feature_extraction_method.lower() == 'cwt':
        features = continuous_wavelet_transform(spikes)
    elif feature_extraction_method.lower() == 'dwt':
        features = discrete_wavelet_transform(spikes)
    elif feature_extraction_method.lower() == 'hilbert':
        features = hilbert_envelope(spikes)
    elif feature_extraction_method.lower() == 'emd':
        features = emd_imf_derivatives(spikes)
    elif feature_extraction_method.lower() == 'stft':
        features = stft(spikes)
    elif feature_extraction_method.lower() == 'stft_d':
        features = stft_d(spikes)
    elif feature_extraction_method.lower() == 'stft_dpss':
        features = stft_multitaper(spikes)
        # features = stft_multitaper_w(spikes)
    elif feature_extraction_method.lower() == 'fourier_real':
        features = fourier_real(spikes)
    elif feature_extraction_method.lower() == 'fourier_imaginary':
        features = fourier_imaginary(spikes)
    elif feature_extraction_method.lower() == 'fourier_amplitude':
        features = fourier_amplitude(spikes)
    elif feature_extraction_method.lower() == 'fourier_phase':
        features = fourier_phase(spikes)
    elif feature_extraction_method.lower() == 'fourier_power':
        features = fourier_power(spikes)
    elif feature_extraction_method.lower() == 'shape':
        features = shape_features.get_shape_phase_distribution_features(spikes)
    else:
        features = reduce_dimensionality(spikes, 'PCA2D')

    if dim_reduction_method is not None:
        features = reduce_dimensionality(features, dim_reduction_method)

    return features
