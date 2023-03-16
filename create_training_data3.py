from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds, BrainFlowPresets
from brainflow.data_filter import DataFilter, FilterTypes, AggOperations
import time
import os
from datetime import datetime
import argparse
import pandas as pd
import numpy as np

RECORDING_DURATION = 3  # duration of each recording in seconds
PAUSE_DURATION = 3  # pause between recordings in seconds
REPEATS = 2 # 5
SETS = 2 # 20


# File path
CURR_DIR = os.path.dirname(os.path.abspath("create_training_data.py"))
CURR_DIR = os.path.join(CURR_DIR, "training_data")

# configure argparse
parser = argparse.ArgumentParser()
parser.add_argument('--serial-port', type=str, help='serial port', required=False, default='')
parser.add_argument('--arm', type=str, help='which hand are you using (left/right)?', required=True, default='')
args = parser.parse_args()

# if not -> exit
if not (args.arm == "left" or args.arm == "right"):
    print("Your not specifiing the arm your training for. use --arm left OR --arm right")
    exit(1)

CURR_DIR = os.path.join(CURR_DIR, args.arm)


params = BrainFlowInputParams()
params.serial_port = args.serial_port

board = BoardShim(BoardIds.CYTON_DAISY_BOARD, params)
eeg_channels = BoardShim.get_eeg_channels(BoardIds.CYTON_DAISY_BOARD.value)

def countdown_timer(seconds):
    for remaining_seconds in range(seconds, 0, -1):
        print(f"Remaining time: {remaining_seconds} seconds")
        time.sleep(1)


def main():
    
    board.prepare_session()

    print("Starting EEG data recording for hand state classification...")

    for set_number in range(SETS):
        print(f"\nSet {set_number + 1} of {SETS}")

        for _ in range(REPEATS):
            countdown_timer(PAUSE_DURATION)
            # Record data for hand opening
            print("\nRecording data for hand opening...")
            record_eeg_data("open")
            countdown_timer(PAUSE_DURATION)
            # Record data for hand closing
            print("\nRecording data for hand closing...")
            record_eeg_data("closed")

        print("Set completed. Taking a short break...")
        time.sleep(30)  # Take a break between sets

    board.release_session()
    print("Finished recording EEG data for hand state classification.")

def record_eeg_data(hand_state):
    board.start_stream()
    time.sleep(RECORDING_DURATION)
    data = board.get_board_data()
    board.stop_stream()
    
    eeg_data = data[eeg_channels, :]
    eeg_data = eeg_data / 1000000

    # Save the data with the appropriate label (e.g., "open" or "closed")
    save_data(hand_state, eeg_data)

def save_data(hand_state, eeg_data):

    df = pd.DataFrame(np.transpose(eeg_data))

    if hand_state == "closed":
        fist = 1
        #CURR_DIR = os.path.join(CURR_DIR, "hand-closed")
    else:
        fist = 0
        #CURR_DIR = os.path.join(CURR_DIR, "hand-open")

    df[17] = [fist] * len(df)

    filename = os.path.join(CURR_DIR, f"eeg_df.csv")

    # write to file
    df.to_csv(filename, index=False, mode="a", header=False)

main()
