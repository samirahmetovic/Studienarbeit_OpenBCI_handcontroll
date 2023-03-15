# -------------------------------------------------------------------
# this python script will generate the training data for pytorch
# it safes the .csv files in /training_data
# 
# use --serial SERIAL for the serial port the OpenBCI USB is plugged in
# use --arm left / right to spezify the arm you're training for
# use --hand open / closed to speozify if hand is open or closed
# -------------------------------------------------------------------


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
    print("Your not specifiing the arm your training for. use --arm left OR --arm right")
    exit(1)

# open or closed hand
# if not -> exit
if not (args.hand == "open" or args.hand == "closed"):
    print("Your not specifiing the hand status your training for. use --hand open OR --hand closed")
    exit(1)


CURR_DIR = os.path.join(CURR_DIR, args.arm)

board = BoardShim(BoardIds.CYTON_DAISY_BOARD, params)
board.prepare_session()
board.start_stream()
time.sleep(5)
# data = board.get_current_board_data (256) # get latest 256 packages or less, doesnt remove them from internal buffer
data = board.get_board_data()  # get all data and remove it from internal buffer
board.stop_stream()
board.release_session()

eeg_channels = BoardShim.get_eeg_channels(BoardIds.CYTON_DAISY_BOARD.value)
eeg_data = data[eeg_channels, :]
eeg_data = eeg_data / 1000000  # BrainFlow returns uV, convert to V for MNE


# Creating MNE objects from brainflow data arrays
ch_types = ['eeg'] * len(eeg_channels)
ch_names = BoardShim.get_eeg_names(BoardIds.CYTON_DAISY_BOARD.value)
sfreq = BoardShim.get_sampling_rate(BoardIds.CYTON_DAISY_BOARD.value)
info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
raw = mne.io.RawArray(eeg_data, info)
# its time to plot something!
raw.plot_psd(average=True)
# raw.compute_psd().plot()
plt.savefig('psd.png')


# convert it to pandas DF
# df = pd.DataFrame(np.transpose(eeg_data), columns=[i+1 for i in range(eeg_data.shape[0])])
df = pd.DataFrame(np.transpose(eeg_data))


# add column for hand closed / open
# hand closed = 0
# hand open = 1

if args.hand == "closed":
    fist = 1
    #CURR_DIR = os.path.join(CURR_DIR, "hand-closed")
else:
    fist = 0
    #CURR_DIR = os.path.join(CURR_DIR, "hand-open")

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