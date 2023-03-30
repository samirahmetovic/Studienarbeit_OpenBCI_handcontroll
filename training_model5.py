# -------------------------------------------------------------------
# this python script will train with the data from /training_data
# it will safe the traind model in /models
# -------------------------------------------------------------------

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import os
from nn_model5 import EEGClassifier
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds, BrainFlowPresets

# get sampling rate
sampling_rate = BoardShim.get_sampling_rate(BoardIds.CYTON_DAISY_BOARD)
duration = 3

# Load data from CSV file
CURR_DIR = os.path.dirname(os.path.abspath("pytorch.py"))
CURR_DIR = os.path.join(CURR_DIR, "training_data", "right", "eeg_training_backup.csv")

# read data from csv file
data = pd.read_csv(CURR_DIR, header=None)

input = data.iloc[:, :16].values
output = data.iloc[:,-1:].values


# Split data into inputs (EEG signals) and targets (hand state)
inputs = torch.tensor(input.transpose(), dtype=torch.float32)
targets = torch.tensor(output, dtype=torch.int64)


# Define hyperparameters
input_size = 16
hidden_size = 32
output_size = 1
lr = 0.001
num_epochs = 200

# define bach size
# bache size is sampling rate times recorded seconds
batch_size = sampling_rate * duration
num_batches = len(inputs) // batch_size

# Initialize model, loss function, and optimizer
model = EEGClassifier()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

# Train the model
for epoch in range(num_epochs):

    # Split data into batches
    for batch_idx in range(num_batches):

        # batch_inputs = inputs[:, batch_idx * batch_size:(batch_idx + 1) * batch_size].unsqueeze(0)
        batch_inputs = inputs[:, batch_idx * batch_size:(batch_idx + 1) * batch_size]
        batch_targets = targets[batch_idx * batch_size:(batch_idx + 1) * batch_size]
        # Forward pass
        outputs = model(batch_inputs)
        # loss = criterion(outputs, batch_targets.unsqueeze(1))
        loss = criterion(outputs, batch_targets)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Print loss for monitoring training progress
    if (epoch + 1) % 20 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')


# safe model to file
torch.save(model.state_dict(), "model_cnn.pt")

# Test the model on a new input
#test_input = torch.tensor([[0.5, 0.6, 0.4, 0.2, 0.1, 0.7]])
#prediction = model(test_input)
#if prediction >= 0.5:
#    print('Predicted class: closing hand')
#else:
#    print('Predicted class: opening hand')
