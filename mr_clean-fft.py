'''
This MR. Clean script is like Mr. Wash, but for cleaning data not cars. It will take in raw data and clean it up using different strategies.
'''

import numpy as np
import pandas as pd
import os
from brainflow.board_shim import BoardShim, BoardIds
from brainflow.data_filter import DataFilter, FilterTypes, AggOperations, WaveletTypes, NoiseEstimationLevelTypes, \
    WaveletExtensionTypes, ThresholdTypes, WaveletDenoisingTypes, WindowOperations
import time
import argparse

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mne
import seaborn as sns
import pywt

def to_eeg(data, eeg_channels = BoardShim.get_eeg_channels(BoardIds.CYTON_DAISY_BOARD.value)):
    # eeg_channels = BoardShim.get_eeg_channels(BoardIds.CYTON_DAISY_BOARD.value)
    eeg_data = data[eeg_channels, :]
    eeg_data = eeg_data / 1000000  # BrainFlow returns uV, convert to V for MNE
    return eeg_data


# ------------------ Argparse Data ------------------
# configure argparse
parser = argparse.ArgumentParser()
parser.add_argument('--f', type=str, help='filename', required=True, default='')
args = parser.parse_args()

# ------------------ Bandpass filter ------------------

# Apply a bandpass filter to the EEG data
LOW_CUT = 8  # Lower frequency bound (Hz)
HIGH_CUT = 12  # Upper frequency bound (Hz)
ORDER = 4  # Filter order


# get filename
CURRDIR = os.path.dirname(os.path.abspath("create_training_data.py"))
filepath = os.path.join(CURRDIR, "training_data", "right")
plot_path = os.path.join(CURRDIR, "plots", "fft")
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

# get data
filename = args.f
df = pd.read_csv(os.path.join(filepath, f"{filename}.csv"), header=None)
# transpose data back to original format
data = df.values.transpose()


# plot before bandpass filter
BANDPASS_DIR = os.path.join(PLOT_DIR, "bandpass")

# its time to plot something!
#raw = mne.io.RawArray(to_eeg(data), info)
#raw.plot_psd(average=True)
#plt.savefig('psd.png')

# Apply Bandpass filter
for channel in eeg_channels:
    DataFilter.perform_bandpass(data[channel-1], sampling_rate, LOW_CUT, HIGH_CUT, ORDER, filter_type, 0)


# plot after bandpass
#raw = mne.io.RawArray(to_eeg(data), info)
#raw.plot_psd(average=True)
#plt.savefig('psd_bandpass.png')


# ------------------ Delete not used EEG Pins ------------------

needed_eeg_channels = [3, 4, 9, 10, 11, 12, 32]

data = np.take(data, needed_eeg_channels, 0)
'''
# ------------------ Fast Fourier transformation ------------------

# Create an empty list to store the wavelet coefficients for all channels
fft = pd.DataFrame()


# calcute batch
batch_size = REC_DURATION * sampling_rate
num_batches = data.shape[1] // batch_size

for batch_idx in range(num_batches):
    
    batch_df = pd.DataFrame()

    batch_target = data[-1,batch_idx*batch_size]

    for idx, channel in enumerate(eeg_channels):

        # Get the data for the current channel
        batch_inputs = data[channel:channel+1, batch_idx * batch_size:(batch_idx + 1) * batch_size]
        batch_inputs = batch_inputs[0]
        print("Batch input shape", batch_inputs.shape)
        fft_data = np.fft.fft(batch_inputs)
        print("FFT data shape", fft_data.shape)
        fft_data_abs = np.abs(fft_data)
        print("FFT data abs shape", fft_data_abs.shape)
        # print(batch_inputs.shape)
        # fft_data = DataFilter.perform_fft(batch_inputs.transpose()[0], WindowOperations.NO_WINDOW.value)

        batch_df[idx] = fft_data_abs

    # batch_df[17] = [batch_target] * 400

    fft = pd.concat([fft, batch_df], ignore_index=True)

print(fft.shape)
'''
# ------------------ Fast Fourier transformation ------------------

def fft_abs(data):
    fft_data = np.fft.fft(data, axis=1)
    fft_data_abs = np.abs(fft_data)
    return fft_data_abs

# calcute batch
batch_size = REC_DURATION * sampling_rate
num_batches = data.shape[1] // batch_size

#delete last value fo every second dimension
targets = data[-1, :]
data = data[:-1,:]


batches = np.array_split(data, num_batches, axis=1)

tmp_eeg = [0, 1, 2, 3, 4, 5]

fft_abs_batches = [fft_abs(to_eeg(batch, eeg_channels = tmp_eeg)) for batch in batches]

#fft_abs_batches = [fft_abs(batch) for batch in batches]

# Creating MNE objects from brainflow data arrays
ch_types = ['eeg'] * len(tmp_eeg)
ch_names = ['eeg'] * len(tmp_eeg)
sfreq = BoardShim.get_sampling_rate(BoardIds.SYNTHETIC_BOARD.value)
info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)


print(len(fft_abs_batches))
for idx, fft_abs_batch in enumerate(fft_abs_batches):
    plt.figure()
    plt.title(f'Batch {idx+1}')
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Amplitude')

    # plt.plot(fft_abs_batch[idx])
    # sns.lineplot(fft_abs_batch[idx])
    raw = mne.io.RawArray(fft_abs_batch, info)
    raw.compute_psd(fmin=2.0, fmax=15.0).plot()

    plt.savefig(os.path.join(plot_path, f"plot_fft_batch{idx}.png"))
    print("saved")

# build data array back and safe it in csv
data = np.concatenate(batches, axis=1)
data = np.concatenate((data, targets.reshape(1, -1)), axis=0)
df = pd.DataFrame(data.transpose())
df.to_csv(os.path.join(filepath, "cleaned", "fft", f"{filename}-fft.csv"))
# np.savetxt("text.csv", all_wavelet_coeffs_array, delimiter=",")
# Reshape the combined wavelet coefficients to match the input shape of your model
# Determine the shape based on your model's requirements
# input_shape = (all_wavelet_coeffs_array.shape[0], all_wavelet_coeffs_array.shape[1])
# reshaped_wavelet_coeffs = all_wavelet_coeffs_array.reshape(input_shape)

# print(input_shape)
# print(reshaped_wavelet_coeffs)

# plot after wavelet transformation
#raw = mne.io.RawArray(to_eeg(fft.values.transpose()), info)
#raw.plot_psd(average=True)
#plt.savefig('psd_wavelet.png')


'''# ------------------ Wavelet transformation ------------------

# Define wavelet parameters
WAVELET_TYPE = WaveletTypes.DAUBECHIES.value
DECOMPOSITION_LEVEL = 4

# Apply wavelet transformation to the filtered data
for channel in eeg_channels:
    coeffs = DataFilter.perform_wavelet_transform(data[channel-1], WAVELET_TYPE, DECOMPOSITION_LEVEL)
    reconstructed = DataFilter.perform_inverse_wavelet_transform(coeffs, WAVELET_TYPE, DECOMPOSITION_LEVEL)
    
    # Plot the raw and reconstructed signals for visualization
    fig, ax = plt.subplots(2, 1, figsize=(12, 6))
    ax[0].plot(data[channel-1])
    ax[0].set_title(f'Raw EEG Channel {channel}')
    ax[1].plot(reconstructed)
    ax[1].set_title(f'Reconstructed EEG Channel {channel}')
    fig.savefig(os.path.join(PLOT_DIR, f'eeg_channel_{channel}_wavelet.png'))
    plt.close(fig)'''


# ------------------ Delete not used EEG Pins ------------------

# needed_eeg_channels = [3, 4, 9, 10, 11, 12, 17]

# cleaned_df = fft[needed_eeg_channels]

# fft.to_csv(os.path.join(filepath, "cleaned", "fft", f"{filename}-clean-fft.csv"), header=None, index=False)
'''
Plot dont work with less EEG Channels


# change info
# change Channel typed to length of needed_eeg_channels
ch_types = ['eeg'] * len(needed_eeg_channels)
ch_names = BoardShim.get_eeg_names(BoardIds.SYNTHETIC_BOARD.value)
# change Channel names to needed_eeg_channels
ch_names = [ch_names[i-1] for i in needed_eeg_channels]
sfreq = BoardShim.get_sampling_rate(BoardIds.SYNTHETIC_BOARD.value)
info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=needed_eeg_channels)

raw = mne.io.RawArray(to_eeg(wavelet.values.transpose(), eeg_channels=needed_eeg_channels), info)
raw.plot_psd(average=True)
plt.savefig('psd_finished.png')

'''
