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

BoardShim.enable_dev_board_logger()

# configure argparse
parser = argparse.ArgumentParser()
parser.add_argument('--serial-port', type=str, help='serial port', required=False, default='')
args = parser.parse_args()

params = BrainFlowInputParams()
params.serial_port = args.serial_port


board = BoardShim(BoardIds.CYTON_DAISY_BOARD, params)
board.prepare_session()
board.start_stream()
time.sleep(10)
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
plt.savefig('psd.png')


# convert it to pandas DF and plot data
df = pd.DataFrame(np.transpose (data))
# df.columns = [i for i in range(len(data))]



print('Data From the Board')
print (df.head(2))

# write to file
DataFilter.write_file(data, "data.csv", "w")

print(data)