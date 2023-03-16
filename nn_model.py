import torch.nn as nn

# Define neural network architecture
class EEGClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(EEGClassifier, self).__init__()
        self.hidden = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.output = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.hidden(x)
        x = self.relu(x)
        x = self.output(x)
        x = self.sigmoid(x)
        return x
