import numpy as np
import feature_extraction.feature_extraction as fe
from utils import scatter_plot

CELL = 37
PATH = 'D:/Github/Dissertation2/datasets/ssd_studenti/c'

print("SPIKE TIMESTAMPS:")
spike_timestamps = np.fromfile(
    PATH + str(CELL) + '/units/c' + str(CELL) + '_npx.ssdst',
    dtype='int32')
print(spike_timestamps)

print("\nSPIKE WAVEFORMS:")
spike_waveforms = np.fromfile(
    PATH + str(CELL) + '/units/c' + str(CELL) + '_npx.ssduw',
    dtype='float32')
print(len(spike_waveforms))

print("\nEVENT TIMESTAMPS:")
event_timestamps = np.fromfile(
    PATH + str(CELL) + '/units/c' + str(CELL) + '_npx.ssdet',
    dtype='int32')
print(event_timestamps)
print(len(event_timestamps))

print("\nEVENT CODES:")
event_codes = np.fromfile(
    PATH + str(CELL) + '/units/c' + str(CELL) + '_npx.ssdec',
    dtype='int32')
print(event_codes)

labels = np.zeros(spike_timestamps.shape[0])
spike_count = 0
for s_ts_index, s_ts_val in enumerate(spike_timestamps):
    for e_ts in event_timestamps:
        if abs(s_ts_val - e_ts) < 6:
            spike_count += 1
            labels[s_ts_index] = 1

print(spike_count)

spikes = spike_waveforms.reshape((spike_waveforms.shape[0] // 54, 54))
features = fe.apply_feature_extraction_method(spikes,
                                              feature_extraction_method='cwt',
                                              dim_reduction_method='pca2d'
                                              )
scatter_plot.plot_clusters(features, labels, title="cell" + str(CELL),
                           save_folder='')
