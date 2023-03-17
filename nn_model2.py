import torch
import torch.nn as nn
import torch.optim as optim

class EEGClassifier(nn.Module):
    def __init__(self):
        super(EEGClassifier, self).__init__()
        self.conv1 = nn.Conv1d(1, 64, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(128, 256, kernel_size=5, padding=2)
        self.conv4 = nn.Conv1d(256, 512, kernel_size=5, padding=2)
        self.conv5 = nn.Conv1d(512, 3, kernel_size=16)
        self.activation = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.activation(self.conv1(x))
        x = self.activation(self.conv2(x))
        x = self.activation(self.conv3(x))
        x = self.activation(self.conv4(x))
        x = self.conv5(x).squeeze(2)
        x = self.softmax(x)
        return x

# Instantiate the model
#model = EEGClassifier()

# Define the loss function and the optimizer
#criterion = nn.CrossEntropyLoss()
#optimizer = optim.Adam(model.parameters())

# Print the model summary
# print(model)