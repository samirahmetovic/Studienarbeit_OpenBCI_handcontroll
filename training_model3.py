import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import os
from nn_model4 import EEGClassifier
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds, BrainFlowPresets
from sklearn.model_selection import train_test_split

# get sampling rate
sampling_rate = BoardShim.get_sampling_rate(BoardIds.CYTON_DAISY_BOARD)
duration = 3

# Load data from CSV file
CURR_DIR = os.path.dirname(os.path.abspath("pytorch.py"))
CURR_DIR = os.path.join(CURR_DIR, "training_data", "right", "eeg_training_backup.csv")

# read data from csv file
data = pd.read_csv(CURR_DIR, header=None)

input = data.iloc[:, :16].values
print(input.shape)
output = data.iloc[:, -1].values

# Reshape input data to match the model input shape
# Calculate the number of trials and timesteps per trial
num_trials = int(len(output) / (sampling_rate * duration))
timesteps_per_trial = sampling_rate * duration

# Create a list of trials, where each trial contains a list of timesteps
trials = []
targets = []
for i in range(num_trials):
    trial_data = input[i * timesteps_per_trial: (i + 1) * timesteps_per_trial]
    trial_targets = output[i * timesteps_per_trial: (i + 1) * timesteps_per_trial]
    trials.append(trial_data)
    targets.append(int(np.mean(trial_targets).round()))

# Convert lists to tensors
inputs = torch.tensor(trials, dtype=torch.float32).unsqueeze(1)  # Add an extra dimension for input_channels
targets = torch.tensor(targets, dtype=torch.long)

# Split data into training and testing sets
train_inputs, test_inputs, train_targets, test_targets = train_test_split(inputs, targets, test_size=0.2, random_state=42)

# Define hyperparameters
lr = 0.001
num_epochs = 200

# Initialize model, loss function, and optimizer
model = EEGClassifier(dropout=0.5)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

# Train the model
for epoch in range(num_epochs):
    # Forward pass
    outputs = model(train_inputs)
    loss = criterion(outputs, train_targets)

    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Print loss for monitoring training progress
    if (epoch + 1) % 20 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# safe model to file
torch.save(model.state_dict(), "model_cnn.pt")

# Test the model on test data
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    outputs = model(test_inputs)
    _, predicted = torch.max(outputs.data, 1)
    total += test_targets.size(0)
    correct += (predicted == test_targets).sum().item()

    print(f'Accuracy of the model on test inputs: {100 * correct / total:.2f}%')
