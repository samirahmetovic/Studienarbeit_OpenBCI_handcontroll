# -------------------------------------------------------------------
# this is a test file to test some functions
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
import keyboard

'''
params = BrainFlowInputParams()
params.serial_port = "COM3"


board = BoardShim(BoardIds.CYTON_DAISY_BOARD, params)
eeg_channels = BoardShim.get_eeg_channels(BoardIds.CYTON_DAISY_BOARD.value)
board.prepare_session()
board.start_stream()


# data = board.get_current_board_data (256) # get latest 256 packages or less, doesnt remove them from internal buffer
while True:
    if keyboard.is_pressed('q'):
        break
    # data = board.get_current_board_data(BoardShim.get_sampling_rate(board.get_board_id()))  # get all data and remove it from internal buffer
    data = board.get_current_board_data(1)
    eeg_data = data[eeg_channels, :]
    eeg_data = eeg_data / 1000000  # BrainFlow returns uV, convert to V for MNE
    print(eeg_data)
    eeg_data = eeg_data.transpose()
    print(eeg_data)
    time.sleep(0.1)

board.stop_stream()
board.release_session()
'''
'''
sampling_rate = BoardShim.get_sampling_rate(BoardIds.CYTON_DAISY_BOARD)
print(sampling_rate)

params = BrainFlowInputParams()
params.serial_port = "/dev/cu.usbserial-DM03H72A"

board = BoardShim(BoardIds.CYTON_DAISY_BOARD, params)
eeg_channels = BoardShim.get_eeg_channels(BoardIds.CYTON_DAISY_BOARD.value)

board.prepare_session()
board.start_stream(375)
time.sleep(4)
data1 = board.get_current_board_data(375)
time.sleep(1)
data2 = board.get_current_board_data(375)
board.stop_stream()
board.release_session()

# create DataFrame
df1 = pd.DataFrame(data1.transpose())
df2 = pd.DataFrame(data2.transpose())

df1.to_csv("test1.csv", index=False)
df2.to_csv("test2.csv", index=False)
'''

def test(np_array):
    np_array = np_array + 1
    return np_array



arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

test2 = test(arr)

print(arr)
print(test2)
print(arr)