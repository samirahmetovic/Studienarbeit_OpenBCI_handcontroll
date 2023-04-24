# -------------------------------------------------------------------
# this is a test file to test some functions
# -------------------------------------------------------------------


from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds, BrainFlowPresets
from brainflow.data_filter import DataFilter, FilterTypes, AggOperations, WindowOperations
import argparse
import time
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mne
import datetime
import os
import keyboard


def to_eeg(data, eeg_channels = BoardShim.get_eeg_channels(BoardIds.CYTON_DAISY_BOARD.value)):
    # eeg_channels = BoardShim.get_eeg_channels(BoardIds.CYTON_DAISY_BOARD.value)
    # eeg_data = data[eeg_channels, :]
    eeg_data = data
    eeg_data = eeg_data / 1000000  # BrainFlow returns uV, convert to V for MNE
    return eeg_data

# get filename
CURRDIR = os.path.dirname(os.path.abspath("create_training_data.py"))
filepath = os.path.join(CURRDIR, "training_data", "right")
# safe files
PLOT_DIR = os.path.join(CURRDIR, "plots")

# Board specific data
REC_DURATION = 3  # duration of each recording in seconds
sampling_rate = BoardShim.get_sampling_rate(BoardIds.CYTON_DAISY_BOARD.value)
filter_type = FilterTypes.BUTTERWORTH.value  # Choose the filter type
eeg_channels = BoardShim.get_eeg_channels(BoardIds.CYTON_DAISY_BOARD.value)

# Creating MNE objects from brainflow data arrays
ch_types = ['eeg'] * len(eeg_channels)
ch_names = BoardShim.get_eeg_names(BoardIds.SYNTHETIC_BOARD.value)
sfreq = BoardShim.get_sampling_rate(BoardIds.SYNTHETIC_BOARD.value)
info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)


df = pd.read_csv(os.path.join(filepath, f"data_training_marcel1 copy.csv"), header=None)
# transpose data back to original format
data = df.values.transpose()

fft_data = np.ndarray((16, 3563))
for count, channel in enumerate(eeg_channels):
    fft_data[count] = DataFilter.perform_fft(data[channel], WindowOperations.NO_WINDOW.value)
    # fft_data[count] = np.fft.fft(data[channel])
    # fft_data = np.fft.fft(data[channel])
    print("FFT shape")
    print(fft_data.shape)

plt.plot(fft_data)
plt.show()
# raw = mne.io.RawArray(to_eeg(fft_data, eeg_channels=16), info)
# raw.plot_psd(average=True)
plt.savefig('psd_fft.png')