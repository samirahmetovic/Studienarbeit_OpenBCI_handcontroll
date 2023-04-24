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


def perform_data_cleaning_from_file(filename):
    # get filename
    CURRDIR = os.path.dirname(os.path.abspath("create_training_data.py"))
    filepath = os.path.join(CURRDIR, "training_data", "right")

    df = pd.read_csv(os.path.join(filepath, f"{filename}.csv"), header=None)
    data = df.values.transpose()

    cleaned_df = perform_data_cleaning(data=data)

    cleaned_df.to_csv(os.path.join(filepath, "cleaned", f"{filename}-cleaned.csv"), header=None, index=False)


def perform_data_cleaning(data: np.ndarray, training_data: bool = True):
    # ------------------ Bandpass filter ------------------

    # Apply a bandpass filter to the EEG data
    LOW_CUT = 8  # Lower frequency bound (Hz)
    HIGH_CUT = 12  # Upper frequency bound (Hz)
    ORDER = 4  # Filter order

    # get current time
    now = time.strftime("%d-%m-%Y_%H-%M-%S")

    # get filename
    # CURRDIR = os.path.dirname(os.path.abspath("create_training_data.py"))
    # safe files
    # PLOT_DIR = os.path.join(CURRDIR, "plots")

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


    # plot before bandpass filter
    # BANDPASS_DIR = os.path.join(PLOT_DIR, "bandpass")

    # its time to plot something!
    # raw = mne.io.RawArray(to_eeg(data), info)
    # raw.plot_psd(average=True)
    # plt.savefig('psd.png')

    # Apply Bandpass filter
    for channel in eeg_channels:
        DataFilter.perform_bandpass(data[channel], sampling_rate, LOW_CUT, HIGH_CUT, ORDER, filter_type, 0)


    # plot after bandpass
    # raw = mne.io.RawArray(to_eeg(data), info)
    # raw.plot_psd(average=True)
    # plt.savefig('psd_bandpass.png')


    # ------------------ Denoising ------------------
    # TODO: Add denoising




    # ------------------ Wavelet transformation / Fast Fourier transformation ------------------

    # Create an empty list to store the wavelet coefficients for all channels
    wavelet = pd.DataFrame()


    # calcute batch
    batch_size = REC_DURATION * sampling_rate
    num_batches = data.shape[1] // batch_size

    for batch_idx in range(num_batches):
        
        batch_df = pd.DataFrame()

        if training_data:
            # Get the target for the current batch
            batch_target = data[-1,batch_idx*batch_size]

        for idx, channel in enumerate(eeg_channels):

            # Get the data for the current channel
            batch_inputs = data[channel:channel+1, batch_idx * batch_size:(batch_idx + 1) * batch_size]

            # print(batch_inputs.shape)
            wavelet_coeffs, lengths = DataFilter.perform_wavelet_transform(batch_inputs[0], WaveletTypes.DB5, 3)
            # pywt.dwt(data[channel], 'db5', 'smooth')
            # print(wavelet_coeffs)
            # app_coefs = wavelet_coeffs[0: lengths[0]]
            # detailed_coeffs_first_block = wavelet_coeffs[lengths[0]: lengths[1]]
            batch_df[idx] = wavelet_coeffs
            # fft_data = DataFilter.perform_fft(data[channel], WindowOperations.NO_WINDOW.value)

        if training_data:
            # Add the target to the dataframe
            batch_df[17] = [batch_target] * 400

        wavelet = pd.concat([wavelet, batch_df], ignore_index=True)

    print(f"Wavelet Shape: {wavelet.shape}")

    # np.savetxt("text.csv", all_wavelet_coeffs_array, delimiter=",")
    # Reshape the combined wavelet coefficients to match the input shape of your model
    # Determine the shape based on your model's requirements
    # input_shape = (all_wavelet_coeffs_array.shape[0], all_wavelet_coeffs_array.shape[1])
    # reshaped_wavelet_coeffs = all_wavelet_coeffs_array.reshape(input_shape)

    # print(input_shape)
    # print(reshaped_wavelet_coeffs)

    # plot after wavelet transformation
    # raw = mne.io.RawArray(to_eeg(wavelet.values.transpose()), info)
    # raw.plot_psd(average=True)
    # plt.savefig('psd_wavelet.png')


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
    # TODO: Delete not used EEG Pins
    needed_eeg_channels = [3, 4, 9, 10, 11, 12, 17]

    if not training_data:
        needed_eeg_channels.remove(17)

    cleaned_df = wavelet[needed_eeg_channels]

    return cleaned_df
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
if __name__ == "__main__":
    # ------------------ Argparse Data ------------------
    # configure argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--f', type=str, help='filename', required=True, default='')
    args = parser.parse_args()
    perform_data_cleaning_from_file(filename=args.f)
