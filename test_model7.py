import torch
from nn_model7 import EEGClassifier

from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds, BrainFlowPresets
from brainflow.data_filter import DataFilter, FilterTypes, AggOperations

import time
import argparse
import keyboard
import pandas as pd
import os

# BCI Data
sampling_rate = BoardShim.get_sampling_rate(BoardIds.CYTON_DAISY_BOARD)
duration = 3

# get the trained Pytorch NN model
model = EEGClassifier()
model.load_state_dict(torch.load('model_cnn_wavelet.pt'))
model.eval()

# directory
CURR_DIR = os.path.dirname(os.path.abspath("pytorch.py"))
file = os.path.join(CURR_DIR, "training_data", "right", "data_test2-cleaned.csv")

data = pd.read_csv(file, header=None)

inputs = data.iloc[:, :6].values
output = data.iloc[:,-1:].values

# get batch infos
batch_size = sampling_rate * duration
# +25 due to wavelet transformation
batch_size = batch_size + 25

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


for batch_idx in range(num_batches):
    batch_inputs = inputs[:, batch_idx * batch_size:(batch_idx + 1) * batch_size]
    batch_targets = targets[batch_idx * batch_size:(batch_idx + 1) * batch_size]

    prediction = model(batch_inputs)
    predicted_class = torch.argmax(prediction, dim=1)

    for idx, pred in enumerate(predicted_class):
        if pred == 1:
            print(f'Predicted {pred}: closing hand. Real value: {batch_targets[idx]}')
        else:
            print(f'Predicted {pred}: opening hand. Real value: {batch_targets[idx]}')

    time.sleep(0.5)
