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

d1 = {"Name": ["Pankaj", "Lisa"], "ID": [1, 2]}
d2 = {"Name": "David", "ID": 3}
df1 = pd.DataFrame(d1, index={1, 2})
df2 = pd.DataFrame(d2, index={3})
print(df1)
print(df2)

df3 = pd.concat([df1, df2], ignore_index=True)
print(df3)