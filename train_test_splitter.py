# -------------------------------------------------------------------
# this python script can split data to training and test data
# because the batches ar correlated, we need to split the data without breaking the batches
# therefore the date is reshaped and then splited
# -------------------------------------------------------------------

import pandas as pd
import os
from brainflow.board_shim import BoardShim, BoardIds
import glob
from sklearn.model_selection import train_test_split

# get sampling rate
sampling_rate = BoardShim.get_sampling_rate(BoardIds.CYTON_DAISY_BOARD)
duration = 3

# calcute batch size
batch_size = sampling_rate * duration

# +25 for wavelet
# batch_size += 25

# Load data from CSV file
CURR_DIR = os.path.dirname(os.path.abspath("pytorch.py"))
CURR_DIR = os.path.join(CURR_DIR, "training_data", "right", "cleaned")

# create empty df
data = pd.DataFrame()

# get all files in folder starting with data_training
# file_list = glob.glob( os.path.join(CURR_DIR, 'data_training*'))
file_list = glob.glob( os.path.join(CURR_DIR, 'data_training*'))

# read all files and append to df
for file in file_list:
    tmpdf = pd.read_csv(file, header=None)
    data = pd.concat([data, tmpdf],ignore_index=True)

# get numpy array from df
data = data.values.transpose()
print("original shape: " + str(data.shape))

# calculate number of batches for reshape
num_batches = data.shape[1] // batch_size

# reshape data to batches
data = data.reshape((data.shape[0], batch_size, num_batches))
print("after reshape: " + str(data.shape))

data = data.transpose((2, 0, 1))

splited_data = train_test_split(data, test_size=0.2, random_state=42)
# splited_data = train_test_split(data, test_size=0.2, shuffle=False)

for idx, data in enumerate(splited_data):
    data = data.transpose((1, 2, 0))
    print(f"data {idx}: {str(data.shape)}")
    data = data.reshape(data.shape[0], -1)
    df = pd.DataFrame(data.transpose())
    df.to_csv("data_" + ("training" if idx == 0 else "test") + ".csv", index=False, header=False)





train_data, test_data = splited_data