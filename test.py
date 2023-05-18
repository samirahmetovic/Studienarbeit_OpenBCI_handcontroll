# -------------------------------------------------------------------
# this is a test file to test some functions
# -------------------------------------------------------------------


from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds, BrainFlowPresets
from brainflow.data_filter import DataFilter, FilterTypes, AggOperations, WindowOperations
import argparse
import time
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import mne
import datetime
import os
import keyboard

loss_list = [0.435, 0.3344, 0.745, 0.64343, 0.554]
# Plot loss over time
print("plotting...")
sns.lineplot(loss_list)
plt.savefig("loss.png")