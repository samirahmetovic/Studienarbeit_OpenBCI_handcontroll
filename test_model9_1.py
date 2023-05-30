import torch
from nn_model9_1 import EEGNET

from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds, BrainFlowPresets
from brainflow.data_filter import DataFilter, FilterTypes, AggOperations

import time
import argparse
import keyboard
import pandas as pd
import os
import numpy as np

# BCI Data
sampling_rate = BoardShim.get_sampling_rate(BoardIds.CYTON_DAISY_BOARD)
duration = 3

# get the trained Pytorch NN model
model = EEGNET(1, 0.5, 1)
model.load_state_dict(torch.load('EEGNET.pt'))
model.eval()

# directory
CURR_DIR = os.path.dirname(os.path.abspath("pytorch.py"))
MODEL_DIR = os.path.join(CURR_DIR, "models", "EEGNET_9_1.pt")
file = os.path.join(CURR_DIR, "training_data", "right", "cleaned", "fft", "splitted", "data_test.csv")

data = pd.read_csv(file, header=None)

inputs = data.iloc[:, :6].values
output = data.iloc[:,-1:].values

# get batch infos
batch_size = sampling_rate * duration

num_batches = inputs.shape[0] // batch_size


inputs = torch.tensor(inputs.transpose(), dtype=torch.float32)
targets = torch.tensor(output, dtype=torch.float32)

'''for batch_idx in range(num_batches):

    batch_inputs = inputs[:, batch_idx * batch_size:(batch_idx + 1) * batch_size]
    batch_targets = targets[batch_idx * batch_size:(batch_idx + 1) * batch_size]

    prediction = model(batch_inputs)
    print(prediction.shape)
    
    if prediction >= 0.5:
        print(f'Predicted {prediction}: closing hand. Real value: {batch_targets[0]}')
    else:
        print(f'Predicted {prediction}: opening hand. Real value: {batch_targets[0]}')
    
    time.sleep(0.5)'''

hits = 0
misses = 0

for batch_idx in range(num_batches):
    batch_inputs = inputs[:, batch_idx * batch_size:(batch_idx + 1) * batch_size]
    batch_targets = targets[batch_idx * batch_size:(batch_idx + 1) * batch_size]

    reshaped_data = batch_inputs.view(1, 6, 1, 375)

    prediction = model(reshaped_data)
    round_prediction = prediction.round().item()
    prediction = prediction.item()
    '''
    for idx, pred in enumerate(predicted_class):
        print(idx)
        if pred == [1]:
            print(f'Predicted {pred}: closing hand. Real value: {batch_targets[idx]}')
        else:
            print(f'Predicted {pred}: opening hand. Real value: {batch_targets[idx]}')
      ''' 
    if batch_targets[0] == round_prediction:
        print(f'Prediction was correct.')
        hits += 1
    else:
        print(f'Prediction wrong: {prediction}')
        misses += 1

    print(f"Predicted {prediction} -> {round_prediction}. Real value: {batch_targets[0].item()} \n")

print("*****************************************")
print(f"Hits: {hits}, Misses: {misses}")
print(f"Accuracy: {hits / (hits + misses)}")


