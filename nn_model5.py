import torch 
import torch.nn as nn   

class EEGClassifier(nn.Module):
    def __init__(self, num_classes=2, dropout=0.0):
        super(EEGClassifier, self).__init__()
        self.layer1 = nn.Sequential(
            nn.ConstantPad1d((15, 15), 0),
            nn.Conv1d(in_channels=16, out_channels=64, kernel_size=31, stride=1, padding=0),
            nn.LeakyReLU(),
            nn.Dropout(p=dropout))
        self.layer2 = nn.Sequential(
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=2, stride=2, padding=0),
            nn.BatchNorm1d(41634, affine=False),
            nn.LeakyReLU(),
            nn.MaxPool1d(kernel_size=3, stride=2)
        )
        self.layer3 = nn.Sequential(
            nn.Conv1d(in_channels=128, out_channels=128, kernel_size=21, stride=1),
            nn.LeakyReLU(),
            nn.Dropout(p=dropout))
        self.pool2 = nn.Sequential(
            nn.MaxPool1d(kernel_size=2, stride=2))
        self.layer4 = nn.Sequential(
            nn.Conv1d(in_channels=128, out_channels=64, kernel_size=11, stride=1),
            nn.BatchNorm1d(10388, affine=False),
            nn.LeakyReLU(),
            nn.Dropout(p=dropout))
        self.pool3 = nn.Sequential(
            nn.MaxPool1d(kernel_size=3, stride=3))
        self.layer5 = nn.Sequential(
            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=7, stride=7),
            nn.BatchNorm1d(494, affine=False),
            nn.LeakyReLU())
        self.pool4 = nn.Sequential(
            nn.MaxPool1d(kernel_size=3, stride=3))
        self.linear1 = nn.Sequential(
            # nn.Linear(164, num_classes),
            nn.Linear(164, 64),
            nn.Linear(64, 32),
            nn.Linear(32, 16),
            nn.Linear(16, 2),
            nn.LogSoftmax(dim=1))

    def forward(self, x):
        x = self.layer1(x)
        print(x.shape)
        x = self.layer2(x)
        print(x.shape)
        x = self.layer3(x)
        print(x.shape)
        x = self.pool2(x)
        print(x.shape)
        x = self.layer4(x)
        print(x.shape)
        x = self.pool3(x)
        print(x.shape)
        x = self.layer5(x)
        print(x.shape)
        x = self.pool4(x)
        print(x.shape)
        #x = x.view(x.size(0), -1)
        x = self.linear1(x)
        print(x.shape)
        return x