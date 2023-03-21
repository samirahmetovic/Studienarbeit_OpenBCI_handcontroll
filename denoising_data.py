'''
This script will import eeg data from a csv file and denoise it using Butterworth.
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

# get current datetime
now = time.strftime("%d-%m-%Y_%H-%M-%S")

# get filename
CURRDIR = os.path.dirname(os.path.abspath("create_training_data.py"))
filename = os.path.join(CURRDIR, "training_data", "right", "data_training.csv")

# safe files
SAVE_DIR = os.path.join(CURRDIR, "plots", "denoising")

# get data from csv file
df = pd.read_csv(filename, header=None)

# transpose data back to original format
data = df.values.transpose()

# get eeg channels
eeg_channels = BoardShim.get_eeg_channels(BoardIds.CYTON_DAISY_BOARD.value)

# figure size
figsize = (100, 30)

# create plot before denoising
plt.figure()
df[eeg_channels].plot(subplots=True, figsize=figsize)
plt.savefig(os.path.join(SAVE_DIR, f'psd_before_denoising_{now}.png'))


# demo for denoising, apply different methods to different channels for demo
for count, channel in enumerate(eeg_channels):
    DataFilter.perform_rolling_filter(data[channel], 3, AggOperations.MEAN.value)
    # DataFilter.perform_rolling_filter(data[channel], 3, AggOperations.MEDIAN.value)
    # DataFilter.perform_wavelet_denoising(data[channel], WaveletTypes.BIOR3_9, 16,WaveletDenoisingTypes.SURESHRINK, ThresholdTypes.HARD, WaveletExtensionTypes.SYMMETRIC, NoiseEstimationLevelTypes.FIRST_LEVEL)

'''
# Apply a bandpass filter to the EEG data
low_cut = 8  # Lower frequency bound (Hz)
high_cut = 12  # Upper frequency bound (Hz)
order = 4  # Filter order
sampling_rate = BoardShim.get_sampling_rate(BoardIds.CYTON_DAISY_BOARD.value)
filter_type = FilterTypes.BUTTERWORTH.value  # Choose the filter type

for channel in eeg_channels:
    DataFilter.perform_bandpass(eeg_data[channel-1], sampling_rate, low_cut, high_cut, order, filter_type, 0)
'''

# Creating plot after denoising
plt.figure()
df[eeg_channels].plot(subplots=True, figsize=figsize)
plt.savefig(os.path.join(SAVE_DIR, f'psd_after_denoising_{now}.png'))