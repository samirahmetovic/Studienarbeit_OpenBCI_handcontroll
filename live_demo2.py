import tkinter as tk
import torch
import time
from nn_model7 import EEGClassifier
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds, BrainFlowPresets
from brainflow.data_filter import DataFilter, FilterTypes, AggOperations
import argparse
from mr_clean import perform_data_cleaning

# get start parameters
parser = argparse.ArgumentParser()
parser.add_argument('--serial-port', type=str, help='serial port', required=False, default='')
args = parser.parse_args()


# get Board data
params = BrainFlowInputParams()
params.serial_port = args.serial_port

board = BoardShim(BoardIds.CYTON_DAISY_BOARD, params)
eeg_channels = BoardShim.get_eeg_channels(BoardIds.CYTON_DAISY_BOARD.value)

sampling_rate = BoardShim.get_sampling_rate(BoardIds.CYTON_DAISY_BOARD)
SAMPLING_DURATION = 3

def update_hand_status():
    data = board.get_current_board_data(SAMPLING_DURATION * sampling_rate)
    
    # perform cleaning like mr_clean.py
    df = perform_data_cleaning(data=data, training_data=False)
    data = df.values.transpose()

    if data.size == 0:
        root.after(100, update_hand_status)
        return

    input = torch.tensor(data, dtype=torch.float32)
    prediction = model(input)
    
    print(prediction.shape)
    print(prediction)

    if prediction >= 0.5:
        label.config(text="Closing hand")
    else:
        label.config(text="Opening hand")

    root.after(100, update_hand_status)


print("loading model...")
model = EEGClassifier()
model.load_state_dict(torch.load('model_cnn_wavelet.pt'))
model.eval()

print("Starting streaming...")
# start streaming
board.prepare_session()
board.start_stream()
time.sleep(5)

root = tk.Tk()
root.title("Hand status")

label = tk.Label(root, text="Hand status", font=("Arial", 14))
label.pack(padx=100, pady=100, fill="both", expand=True)

update_hand_status()

root.mainloop()

board.stop_stream()
board.release_session()
