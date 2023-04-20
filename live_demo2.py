import tkinter as tk
import torch
import time
from nn_model import EEGClassifier
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds, BrainFlowPresets
from brainflow.data_filter import DataFilter, FilterTypes, AggOperations
import argparse

def update_hand_status():
    data = board.get_current_board_data(1)
    eeg_data = data[eeg_channels, :]
    eeg_data = eeg_data / 1000000
    eeg_data = eeg_data.transpose()

    if eeg_data.size == 0:
        root.after(100, update_hand_status)
        return

    input = torch.tensor(eeg_data, dtype=torch.float32)
    prediction = model(input)

    if prediction >= 0.5:
        label.config(text="Closing hand")
    else:
        label.config(text="Opening hand")

    root.after(100, update_hand_status)

parser = argparse.ArgumentParser()
parser.add_argument('--serial-port', type=str, help='serial port', required=False, default='')
args = parser.parse_args()

params = BrainFlowInputParams()
params.serial_port = args.serial_port

board = BoardShim(BoardIds.CYTON_DAISY_BOARD, params)
eeg_channels = BoardShim.get_eeg_channels(BoardIds.CYTON_DAISY_BOARD.value)
board.prepare_session()
board.start_stream()

model = EEGClassifier(16, 32, 1)
model.load_state_dict(torch.load('model.pt'))
model.eval()

root = tk.Tk()
root.title("Hand status")

label = tk.Label(root, text="Hand status", font=("Arial", 14))
label.pack(padx=100, pady=100, fill="both", expand=True)

update_hand_status()

root.mainloop()

board.stop_stream()
board.release_session()
