import torch
import torch.nn as nn

class EEGClassifier(nn.Module):
    def __init__(self, num_classes=2):
        super(EEGClassifier, self).__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=6, out_channels=24, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=3)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels=24, out_channels=48, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=3)
        )

        self.conv3 = nn.Sequential(
            nn.Conv1d(in_channels=48, out_channels=96, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=3)
        )

        self.conv4 = nn.Sequential(
            nn.Conv1d(in_channels=96, out_channels=400, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=3)
        )
        self.flatten = nn.Flatten()
        
        self.fc1 = nn.Linear(3, 800)  # Anpassen der Größe basierend auf den vorherigen Schichten
        self.fc2 = nn.Linear(800, 300)
        self.fc3 = nn.Linear(300, 1)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        # x = x.view(x.size(0), -1)  # Umwandlung in 1D für Fully Connected Layer
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        # x = nn.Softmax(dim=1)(x)
        return x

# Erstellen einer Instanz des Modells
model = EEGClassifier()
