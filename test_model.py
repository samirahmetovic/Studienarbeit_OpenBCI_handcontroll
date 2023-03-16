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
board.prepare_session()

while True:
    if keyboard.is_pressed('q'):
        break
    board.start_stream()
    time.sleep(0.5)
    data = board.get_current_board_data() # get latest 256 packages or less, doesnt remove them from internal buffer
    board.stop_stream()
    eeg_channels = BoardShim.get_eeg_channels(BoardIds.CYTON_DAISY_BOARD.value)
    eeg_data = data[eeg_channels, :]
    eeg_data = eeg_data / 1000000
    test_input = torch.tensor([[0.5, 0.6, 0.4, 0.2, 0.1, 0.7]])
    if prediction >= 0.5:
        print('Predicted class: closing hand')
    else:
        print('Predicted class: opening hand')

board.release_session()
