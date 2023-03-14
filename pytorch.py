import numpy as np
import torch
import torch.nn as nn

# Load data from CSV file
data = np.loadtxt('bci_data.csv', delimiter=',', skiprows=1)

# Split data into inputs (EEG signals) and targets (hand state)
inputs = torch.tensor(data[:, :-1], dtype=torch.float32)
targets = torch.tensor(data[:, -1], dtype=torch.float32)

class BciNet(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x

net = BciNet(input_size=64, hidden_size=32)

class BciNet(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x

net = BciNet(input_size=64, hidden_size=32)