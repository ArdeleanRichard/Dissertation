import datasets as ds
import numpy as np


timestamps_ = ds.real_data_read_timestamps()
print("timestamps read")
waveforms_ = ds.real_data_read_waveforms()
print("waveforms read")
ds.real_data_extract_spikes_specific_channel(timestamps_, waveforms_, channel=2)