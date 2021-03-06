from scipy.fftpack import fft
import numpy as np

from utils.dataset_parsing import datasets as ds

"""
Scipy FFT implementation is faster if array is of length power of 2
=> padding with/without rolling
=> reducing by deletion
"""
def apply_fft(case, alignment, range_min, range_max):
    spikes, labels = ds.stack_simulations_range(range_min, range_max, True, True, alignment=alignment)

    # ORIGINAL SPIKE
    if case == "original":
        fft_real, fft_imag = fft_original_spike(spikes)
    # PADDED SPIKE
    elif case == "padded":
        fft_real, fft_imag = fft_padded_spike(spikes)
    # ROLLED SPIKE (also padded before)
    elif case == "rolled":
        fft_real, fft_imag = fft_rolled_spike(spikes)
    elif case == "reduced":
        fft_real, fft_imag = fft_reduced_spike(spikes)

    return fft_real, fft_imag

def fft_original_spike(spikes):
    fft_signal = fft(spikes)

    fft_real = [[point.real for point in fft_spike] for fft_spike in fft_signal]
    fft_imag = [[point.imag for point in fft_spike] for fft_spike in fft_signal]

    # fft_real_spike = [x.real for x in fft_signal[0]]
    # fft_imag_spike = [x.imag for x in fft_signal[0]]

    # plt.plot(np.arange(len(spikes[0])), spikes[0])
    # plt.title(f"padded spike")
    # plt.savefig(f'figures/autoencoder/fft_test/orig_spike')
    # plt.cla()
    #
    # plt.plot(np.arange(len(fft_real_spike)), fft_real_spike)
    # plt.title(f"FFT real part")
    # plt.savefig(f'figures/autoencoder/fft_test/orig_fft_real')
    # plt.cla()

    # plt.plot(np.arange(len(fft_imag_spike)), fft_imag_spike)
    # plt.title(f"FFT imag part")
    # plt.savefig(f'figures/autoencoder/fft_test/orig_fft_imag')
    # plt.cla()

    return fft_real, fft_imag


"""
PADDING AT END WITH 0
"""
def fft_padded_spike(spikes):
    spikes = np.pad(spikes, ((0, 0), (0, 128 - len(spikes[0]))), 'constant')
    fft_signal = fft(spikes)

    fft_real = [[point.real for point in fft_spike] for fft_spike in fft_signal]
    fft_imag = [[point.imag for point in fft_spike] for fft_spike in fft_signal]

    # fft_real_spike = [x.real for x in fft_signal[0]]
    # fft_imag_spike = [x.imag for x in fft_signal[0]]

    # plt.plot(np.arange(len(spikes[0])), spikes[0])
    # plt.title(f"padded spike")
    # plt.savefig(f'figures/autoencoder/fft_test/padded_spike')
    # plt.cla()
    #
    # plt.plot(np.arange(len(fft_real_spike)), fft_real_spike)
    # plt.title(f"FFT real part")
    # plt.savefig(f'figures/autoencoder/fft_test/padded_fft_real')
    # plt.cla()
    #
    # plt.plot(np.arange(len(fft_imag_spike)), fft_imag_spike)
    # plt.title(f"FFT imag part")
    # plt.savefig(f'figures/autoencoder/fft_test/padded_fft_imag')
    # plt.cla()

    return fft_real, fft_imag


"""
PADDING AT END WITH 0 and shifting the beginning (before amplitude) to the end
"""
def fft_rolled_spike(spikes):
    spikes = np.pad(spikes, ((0, 0), (0, 128 - len(spikes[0]))), 'constant')
    peak_ind = np.argmax(spikes, axis=1)

    spikes = [np.roll(spikes[i], -peak_ind[i]) for i in range(len(spikes))]
    spikes = np.array(spikes)

    fft_signal = fft(spikes)

    fft_real = [[point.real for point in fft_spike] for fft_spike in fft_signal]
    fft_imag = [[point.imag for point in fft_spike] for fft_spike in fft_signal]

    # fft_real_spike = [x.real for x in fft_signal[0]]
    # fft_imag_spike = [x.imag for x in fft_signal[0]]
    #
    # plt.plot(np.arange(len(spikes[0])), spikes[0])
    # plt.title(f"padded spike")
    # plt.savefig(f'figures/autoencoder/fft_test/rolled_woA_spike')
    # plt.cla()
    #
    # plt.plot(np.arange(len(fft_real_spike)), fft_real_spike)
    # plt.title(f"FFT real part")
    # plt.savefig(f'figures/autoencoder/fft_test/rolled_woA_fft_real')
    # plt.cla()
    #
    # plt.plot(np.arange(len(fft_imag_spike)), fft_imag_spike)
    # plt.title(f"FFT imag part")
    # plt.savefig(f'figures/autoencoder/fft_test/rolled_woA_fft_imag')
    # plt.cla()

    return fft_real, fft_imag


"""
DELETE THE LAST 15 points (out of 79) to get 64 (because power of 2)
"""
def fft_reduced_spike(spikes):
    spikes = [spike[0:64] for spike in spikes]
    spikes = np.array(spikes)

    fft_signal = fft(spikes)

    fft_real = [[point.real for point in fft_spike] for fft_spike in fft_signal]
    fft_imag = [[point.imag for point in fft_spike] for fft_spike in fft_signal]

    # fft_real_spike = [x.real for x in fft_signal[0]]
    # fft_imag_spike = [x.imag for x in fft_signal[0]]
    #
    # plt.plot(np.arange(len(spikes[0])), spikes[0])
    # plt.title(f"padded spike")
    # plt.savefig(f'figures/autoencoder/fft_test/reduced_woA_spike')
    # plt.cla()
    #
    # plt.plot(np.arange(len(fft_real_spike)), fft_real_spike)
    # plt.title(f"FFT real part")
    # plt.savefig(f'figures/autoencoder/fft_test/reduced_woA_fft_real')
    # plt.cla()
    #
    # plt.plot(np.arange(len(fft_imag_spike)), fft_imag_spike)
    # plt.title(f"FFT imag part")
    # plt.savefig(f'figures/autoencoder/fft_test/reduced_woA_fft_imag')
    # plt.cla()

    return fft_real, fft_imag
