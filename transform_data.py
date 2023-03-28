import time

from brainflow.board_shim import BoardShim, BrainFlowInputParams, LogLevels, BoardIds
from brainflow.data_filter import DataFilter, WindowOperations, WaveletTypes
import numpy as np
import pandas as pd
import os
import seaborn as sns

# get filename
filename = os.path.dirname(os.path.abspath("create_training_data.py"))
filename = os.path.join(filename, "training_data", "right", "data_training.csv")

# get data from csv file
df = pd.read_csv(filename, header=None)

# transpose data back to original format
data = df.values.transpose()

# get eeg channels
eeg_channels = BoardShim.get_eeg_channels(BoardIds.CYTON_DAISY_BOARD.value)

# demo for transforms
for count, channel in enumerate(eeg_channels):
    print('Original data for channel %d:' % channel)
    print(data[channel])
    # demo for wavelet transforms
    # wavelet_coeffs format is[A(J) D(J) D(J-1) ..... D(1)] where J is decomposition level, A - app coeffs, D - detailed coeffs
    # lengths array stores lengths for each block
    wavelet_coeffs, lengths = DataFilter.perform_wavelet_transform(data[channel], WaveletTypes.DB5, 3)
    print("Wavelet coeffs for channel %d:" % channel)
    print(wavelet_coeffs)
    app_coefs = wavelet_coeffs[0: lengths[0]]
    detailed_coeffs_first_block = wavelet_coeffs[lengths[0]: lengths[1]]
    # you can do smth with wavelet coeffs here, for example denoising works via thresholds 
    # for wavelets coefficients
    # restored_data = DataFilter.perform_inverse_wavelet_transform((wavelet_coeffs, lengths), data[channel].shape[0], WaveletTypes.DB5, 3)
    # print('Restored data after wavelet transform for channel %d:' % channel)
    # print(restored_data)

    # demo for fft, len of data must be a power of 2
    # fft_data = DataFilter.perform_fft(data[channel], WindowOperations.NO_WINDOW.value)
    # len of fft_data is N / 2 + 1
    # restored_fft_data = DataFilter.perform_ifft(fft_data)
    # print('Restored data after fft for channel %d:' % channel)
    # print(restored_fft_data)