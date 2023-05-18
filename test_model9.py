import torch
from nn_model9 import EEGNET

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
file = os.path.join(CURR_DIR, "training_data", "right", "cleaned", "data_training_marcel1-cleaned.csv")

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

    reshaped_data = batch_inputs.view(1, 6, 1, 400)

    prediction = model(reshaped_data)
    prediction = prediction.round().item()
    '''
    for idx, pred in enumerate(predicted_class):
        print(idx)
        if pred == [1]:
            print(f'Predicted {pred}: closing hand. Real value: {batch_targets[idx]}')
        else:
            print(f'Predicted {pred}: opening hand. Real value: {batch_targets[idx]}')
      ''' 
    if batch_targets[0] == prediction:
        print(f'Prediction correct: {prediction}')
    else:
        print(f'Prediction wrong: {prediction}')

    time.sleep(0.5)
