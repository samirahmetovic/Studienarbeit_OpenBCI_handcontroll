from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds, BrainFlowPresets
from brainflow.data_filter import DataFilter, FilterTypes, AggOperations
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

# BoardShim.enable_dev_board_logger()

# File path
CURR_DIR = os.path.dirname(os.path.abspath("create_training_data.py"))
CURR_DIR = os.path.join(CURR_DIR, "training_data")

# configure argparse
parser = argparse.ArgumentParser()
parser.add_argument('--serial-port', type=str, help='serial port', required=False, default='')
parser.add_argument('--hand', type=str, help='open / closed?', required=True, default='')
parser.add_argument('--arm', type=str, help='which hand are you using (left/right)?', required=True, default='')
args = parser.parse_args()

params = BrainFlowInputParams()
params.serial_port = args.serial_port

# left or right hand
# if not -> exit
if not (args.arm == "left" or args.arm == "right"):
    print("Your not specifying the arm you're training for. use --arm left OR --arm right")
    exit(1)

# open or closed hand
# if not -> exit
if not (args.hand == "open" or args.hand == "closed"):
    print("Your not specifying the hand status you're training for. use --hand open OR --hand closed")
    exit(1)

CURR_DIR = os.path.join(CURR_DIR, args.arm)

board = BoardShim(BoardIds.CYTON_DAISY_BOARD, params)
board.prepare_session()
board.start_stream()
time.sleep(5)
data = board.get_board_data()
board.stop_stream()
board.release_session()

eeg_channels = BoardShim.get_eeg_channels(BoardIds.CYTON_DAISY_BOARD.value)
eeg_data = data[eeg_channels, :]
eeg_data = eeg_data / 1000000  # BrainFlow returns uV, convert to V for MNE

# Apply a bandpass filter to the EEG data
low_cut = 8  # Lower frequency bound (Hz)
high_cut = 12  # Upper frequency bound (Hz)
order = 4  # Filter order
sampling_rate = BoardShim.get_sampling_rate(BoardIds.CYTON_DAISY_BOARD.value)
filter_type = FilterTypes.BUTTERWORTH.value  # Choose the filter type

for channel in eeg_channels:
    DataFilter.perform_bandpass(eeg_data[channel], sampling_rate, low_cut, high_cut, order, filter_type, 0)

# Creating MNE objects from brainflow data arrays
ch_types = ['eeg'] * len(eeg_channels)
ch_names = BoardShim.get_eeg_names(BoardIds.CYTON_DAISY_BOARD.value)
sfreq = BoardShim.get_sampling_rate(BoardIds.CYTON_DAISY_BOARD.value)
info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
raw = mne.io.RawArray(eeg_data, info)

# its time to plot something!
raw.plot_psd(average=True)
plt.savefig('psd.png')

# convert it to pandas DF
df = pd.DataFrame(np.transpose(eeg_data))

# add column for hand closed / open
if args.hand == "closed":
    fist = 1
else:
    fist = 0

df[17] = [fist] * len(df)

# Get the current date and time in German timezone
#current_time = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=1)))

# Format the date and time as a string
#date_time_string = current_time.strftime("%d-%m-%Y_%H-%M-%S")

# Create the filename using the formatted date and time
filename = os.path.join(CURR_DIR, f"eeg_df.csv")
print(filename)
# write to file
df.to_csv(filename, index=False, mode="a", header=False)

# DataFilter.write_file(data, "data.csv", "w")
# DataFilter.write_file(df, filename, "a")