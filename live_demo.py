import torch
from nn_model import EEGClassifier

from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds, BrainFlowPresets
from brainflow.data_filter import DataFilter, FilterTypes, AggOperations

import time
import argparse
import keyboard

# get the trained Pytorch NN model
model = EEGClassifier(16, 32, 1)
model.load_state_dict(torch.load('model.pt'))
model.eval()

parser = argparse.ArgumentParser()
parser.add_argument('--serial-port', type=str, help='serial port', required=False, default='')
args = parser.parse_args()

params = BrainFlowInputParams()
params.serial_port = args.serial_port

board = BoardShim(BoardIds.CYTON_DAISY_BOARD, params)
eeg_channels = BoardShim.get_eeg_channels(BoardIds.CYTON_DAISY_BOARD.value)
board.prepare_session()
board.start_stream()

while True:
    if keyboard.is_pressed('q'):
        break
    # data = board.get_current_board_data(BoardShim.get_sampling_rate(board.get_board_id()))  # get all data and remove it from internal buffer
    # get current data
    # get NumPy array with 1 sample and 16 channsls
    data = board.get_current_board_data(1)
    eeg_data = data[eeg_channels, :]
    eeg_data = eeg_data / 1000000
    # transpose to get 16x1 array
    eeg_data = eeg_data.transpose()

    # check if numpy array is empty
    if eeg_data.size == 0:
        continue
    input = torch.tensor(eeg_data, dtype=torch.float32)

    # get prediction
    prediction = model(input)
    if prediction >= 0.5:
        print(f'Predicted {prediction}: closing hand')
    else:
        print(f'Predicted {prediction}: opening hand')
    
    # wait for 0.1 seconds
    time.sleep(0.1)

board.stop_stream()
board.release_session()
