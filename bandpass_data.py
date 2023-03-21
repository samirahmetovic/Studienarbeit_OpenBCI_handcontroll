'''
This script will import raw data and cuts of unwanted frequencies using a bandpass filter.
'''

import numpy as np
import pandas as pd
import os
from brainflow.board_shim import BoardShim, BoardIds
from brainflow.data_filter import DataFilter, FilterTypes, AggOperations, WaveletTypes, NoiseEstimationLevelTypes, \
    WaveletExtensionTypes, ThresholdTypes, WaveletDenoisingTypes
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mne
import seaborn as sns

# get filename
CURRDIR = os.path.dirname(os.path.abspath("create_training_data.py"))
filepath = os.path.join(CURRDIR, "training_data", "right")

# safe files
PLOT_DIR = os.path.join(CURRDIR, "plots", "bandpass")


# get data from csv file
df = pd.read_csv(os.path.join(filepath, "data_training.csv"), header=None)

# transpose data back to original format
data = df.values.transpose()

# get eeg data before bandpass filter
eeg_channels = BoardShim.get_eeg_channels(BoardIds.SYNTHETIC_BOARD.value)
eeg_data = data[eeg_channels, :]
eeg_data = eeg_data / 1000000  # BrainFlow returns uV, convert to V for MNE

# get plot information
ch_types = ['eeg'] * len(eeg_channels)
ch_names = BoardShim.get_eeg_names(BoardIds.SYNTHETIC_BOARD.value)
sfreq = BoardShim.get_sampling_rate(BoardIds.SYNTHETIC_BOARD.value)
info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)

# plot before bandpass filter
raw = mne.io.RawArray(eeg_data, info)
raw.compute_psd().plot(average=True)
# raw.plot_psd(average=True)
plt.savefig(os.path.join(PLOT_DIR, "before_bandpass.png"))

# Apply a bandpass filter to the EEG data
low_cut = 8  # Lower frequency bound (Hz)
high_cut = 12  # Upper frequency bound (Hz)
order = 4  # Filter order
sampling_rate = BoardShim.get_sampling_rate(BoardIds.CYTON_DAISY_BOARD.value)
filter_type = FilterTypes.BUTTERWORTH.value  # Choose the filter type
eeg_channels = BoardShim.get_eeg_channels(BoardIds.CYTON_DAISY_BOARD.value)

for channel in eeg_channels:
    DataFilter.perform_bandpass(data[channel-1], sampling_rate, low_cut, high_cut, order, filter_type, 0)

# get eeg data after bandpass filter
eeg_data = data[eeg_channels, :]
eeg_data = eeg_data / 1000000  # BrainFlow returns uV, convert to V for MNE

# plot after bandpass filter
raw = mne.io.RawArray(eeg_data, info)
raw.compute_psd().plot(average=True)
# raw.plot_psd(average=True)
plt.savefig(os.path.join(PLOT_DIR, "after_bandpass.png"))

# safe new data to csv file
df = pd.DataFrame(data.transpose())
df.to_csv(os.path.join(filepath, "data_bandpass.csv"), header=False, index=False)
