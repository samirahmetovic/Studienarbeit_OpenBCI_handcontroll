# -------------------------------------------------------------------
# this python script will train fft data from /training_data/cleaned/fft
# nn_model9_1.py is used as model. IT is the same as nn_model9.py but for FFT Data
# -------------------------------------------------------------------

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import os
from nn_model9_1 import EEGNET
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds, BrainFlowPresets
import glob
import matplotlib.pyplot as plt
import seaborn as sns
import time

# get sampling rate
sampling_rate = BoardShim.get_sampling_rate(BoardIds.CYTON_DAISY_BOARD)
duration = 3
# Load data from CSV file
CURR_DIR = os.path.dirname(os.path.abspath("pytorch.py"))
MODEL_DIR = os.path.join(CURR_DIR, "models")
CURR_DIR = os.path.join(CURR_DIR, "training_data", "right", "cleaned", "fft")

MODEL_NAME = "EEGNET_9_1.pt"

# create empty df
data = pd.DataFrame()

# get all files in folder starting with data_training
# file_list = glob.glob( os.path.join(CURR_DIR, 'data_training*'))
file_list = glob.glob( os.path.join(CURR_DIR, 'data_training*'))
# read all files and append to df
for file in file_list:
    tmpdf = pd.read_csv(file, header=None)
    data = pd.concat([data, tmpdf],ignore_index=True)
    print(data.shape)


input = data.iloc[:, :6].values
output = data.iloc[:,-1:].values


# Split data into inputs (EEG signals) and targets (hand state)
inputs = torch.tensor(input.transpose(), dtype=torch.float32)
#targets = torch.tensor(output, dtype=torch.int64)
# targets = torch.tensor(output.transpose(), dtype=torch.float32)
targets = torch.tensor(output, dtype=torch.float32)


# Define hyperparameters
input_size = 6
# hidden_size = 32
# output_size = 1
lr = 0.001
num_epochs = 200

# define bach size
# bache size is sampling rate times recorded seconds
batch_size = sampling_rate * duration
# +25 due to wavelet transformation from Brainflow Library
# batch_size = batch_size + 25

num_batches = inputs.shape[1] // batch_size

# Initialize model, loss function, and optimizer
model = EEGNET(1, 0.5, 1)
# criterion = nn.CrossEntropyLoss()
criterion = nn.BCEWithLogitsLoss()
# criterion = nn.BCELoss()
# criterion = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

loss_list = []
acc_list = []

# Train the model
model.train()
print("Model training started...")

# start time
start_time = time.time()

for epoch in range(num_epochs):
    train_corrects = 0

    # safe loss for each batch
    loss_for_batch = []
    
    # Split data into batches
    for batch_idx in range(num_batches):
        # batch_inputs = inputs[:, batch_idx * batch_size:(batch_idx + 1) * batch_size].unsqueeze(0)
        batch_inputs = inputs[:, batch_idx * batch_size:(batch_idx + 1) * batch_size]
        batch_targets = targets[batch_idx * batch_size:(batch_idx + 1) * batch_size]
        first_target = batch_targets[0].unsqueeze(0)

        # reshape
        reshaped_data = batch_inputs.view(1, 6, 1, 375)

        # Forward pass
        outputs = model(reshaped_data)
        # loss = criterion(outputs, batch_targets.unsqueeze(1))
        loss = criterion(outputs, first_target)

        # safe loss to list
        loss_for_batch.append(loss.item())

        # safe accuracy to list
        # preds = torch.sigmoid(output) >= 0.5
        # train_corrects += torch.sum(preds == target.data)
        # acc_list.append(train_corrects.double())


        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Print loss for monitoring training progress
        # if (epoch + 1) % 20 == 0 and batch_idx == 0:
        if batch_idx % 400 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')
    
    # safe loss to list
    loss_list.append(np.mean(loss_for_batch))

print("Model training finished...")

end_time = time.time()
print(f"Training took {end_time - start_time} seconds")

# Plot loss over time
print("plotting...")
sns.lineplot(x=list(range(1, len(loss_list)+1)), y=loss_list).set(xlabel='Epoch', ylabel='Loss', title='Loss over Epochs')
plt.savefig(f"loss_{MODEL_NAME}.png")

# sns.lineplot(x=list(range(1, len(acc_list)+1)), y=acc_list)
# plt.savefig("accuracy.png")

# safe model to file
print("saving model")
torch.save(model.state_dict(), os.path.join(MODEL_DIR, MODEL_NAME))

# Test the model on a new input
#test_input = torch.tensor([[0.5, 0.6, 0.4, 0.2, 0.1, 0.7]])
#prediction = model(test_input)
#if prediction >= 0.5:
#    print('Predicted class: closing hand')
#else:
#    print('Predicted class: opening hand')
