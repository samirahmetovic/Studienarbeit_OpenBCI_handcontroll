from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds, BrainFlowPresets
from brainflow.data_filter import DataFilter, FilterTypes, AggOperations
from sklearn.model_selection import train_test_split
import time
import os
from datetime import datetime
import argparse
import pandas as pd
import numpy as np

RECORDING_DURATION = 3  # duration of each recording in seconds
PAUSE_DURATION = 1  # pause between recordings in seconds
REPEATS = 2 # 5
SETS = 2 # 20
PAUSE_BETWEEN_SETS = 2  # pause between sets in seconds
TRAINING_SIZE = 0.8 # percentage of data used for training


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

# create DataFrame
df_training = pd.DataFrame()
df_test = pd.DataFrame()

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

            # Record data for hand closing
            print("\nRecording data for hand closing...")
            eeg_data = record_eeg_data()
            write_in_df("closed", eeg_data)

            countdown_timer(PAUSE_DURATION)

            # Record data for hand opening
            print("\nRecording data for hand opening...")
            eeg_data = record_eeg_data()
            write_in_df("open", eeg_data)

        # Take a break between sets
        if(set_number != SETS - 1):
            print("Set completed. Taking a short break...")
            time.sleep(PAUSE_BETWEEN_SETS) 

    # close board session
    board.release_session()

    # safe DataFrame with EEG data to file
    save_to_file()
    print("Finished recording EEG data for hand state classification.")

def record_eeg_data():
    # start streaming for RECORDING_DURATION seconds
    board.start_stream()
    time.sleep(RECORDING_DURATION)

    # get data from board
    data = board.get_board_data()
    board.stop_stream()
    
    # convert to numpy array EEG Data
    eeg_data = data[eeg_channels, :]
    eeg_data = eeg_data / 1000000

    return eeg_data

def write_in_df(hand_state, eeg_data):
    # get global df_training and df_test
    global df_training
    global df_test
    # create temp df
    temp_df = pd.DataFrame(eeg_data.transpose())
    # add handstate to df
    if hand_state == "closed":
        fist = 1
    else:
        fist = 0

    temp_df[17] = [fist] * len(temp_df)

    # split in training and test data
    temp_df_training, temp_df_test = train_test_split(temp_df, train_size=TRAINING_SIZE)

    # append to df
    # df_training = df_training.append(temp_df_training, ignore_index=True)
    # df_test = df_test.append(temp_df_test, ignore_index=True)

    # concat to df bcause append is deprecated
    df_training = pd.concat([df_training, temp_df_training], ignore_index=True)
    df_test = pd.concat([df_test, temp_df_test], ignore_index=True)

    return df_training, df_test


def save_to_file():
    # get global df_training and df_test
    global df_training
    global df_test

    # create directory
    filename_training = os.path.join(CURR_DIR, f"eeg_traininf.csv")
    filename_test = os.path.join(CURR_DIR, f"eeg_test.csv")

    # write to file
    df_training.to_csv(filename_training, index=False, mode="a", header=False)
    df_test.to_csv(filename_test, index=False, mode="a", header=False)

main()
