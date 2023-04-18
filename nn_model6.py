'''
This is Model 6
This model uses COnv2d instead of Conv1d because we want to train the model on a time-frequenz aspect
that means we train our cleaned (wavelet transformed) data on the model so that it learn time based frequency patterns
'''

'''import torch 
import torch.nn as nn   

class EEGClassifier(nn.Module):
    def __init__(self, num_classes=2, dropout=0.0, print_shapes=False, time_dimension=400, freq_dimension=6):
        self.print_shapes = print_shapes

        super(EEGClassifier, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=6, out_channels=12, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=12, out_channels=24, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.fc1 = nn.Linear(24 * (time_dimension // 4) * (freq_dimension // 4), 48)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(48, num_classes)

    def forward(self, x):
        # Bool of printing shapes
        print_shapes = self.print_shapes

        x = self.conv1(x)
        if print_shapes:
            print(x.shape)

        x = self.relu1(x)
        if print_shapes:
            print(x.shape)

        x = self.pool1(x)
        if print_shapes:
            print(x.shape)

        x = self.conv2(x)
        if print_shapes:
            print(x.shape)

        x = self.relu2(x)
        if print_shapes:
            print(x.shape)

        x = self.pool2(x)
        if print_shapes:
            print(x.shape)

        x = x.view(x.size(0), -1)

        x = self.fc1(x)
        if print_shapes:
            print(x.shape)

        x = self.relu3(x)
        if print_shapes:
            print(x.shape)
        
        x = self.fc2(x)
        if print_shapes:
            print(x.shape)
        return x'''

import torch
import torch.nn as nn

class EEGClassifier(nn.Module):
    def __init__(self, num_classes=2, dropout=0.0, print_shapes=False):
        self.print_shapes = print_shapes
        super(EEGClassifier, self).__init__()
        self.layer1 = nn.Sequential(
            nn.ConstantPad2d((15, 15, 0, 0), 0),
            nn.Conv2d(in_channels=6, out_channels=24, kernel_size=(1, 3), stride=1, padding=0),
            nn.LeakyReLU(),
            nn.Dropout(p=dropout))
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=24, out_channels=48, kernel_size=(1, 3), stride=(1, 2), padding=0),
            nn.BatchNorm2d(48, affine=False),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 2))
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=48, out_channels=48, kernel_size=(1, 3), stride=1),
            nn.LeakyReLU(),
            nn.Dropout(p=dropout))
        self.pool2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)))
        self.layer4 = nn.Sequential(
            nn.Conv2d(in_channels=48, out_channels=96, kernel_size=(1, 3), stride=1),
            nn.BatchNorm2d(96, affine=False),
            nn.LeakyReLU(),
            nn.Dropout(p=dropout))
        self.pool3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 3)))
        self.layer5 = nn.Sequential(
            nn.Conv2d(in_channels=96, out_channels=400, kernel_size=(1, 3), stride=(1, 7)),
            nn.BatchNorm2d(400, affine=False),
            nn.LeakyReLU())

        self.linear1 = nn.Sequential(
            nn.Linear(400, 800),
            nn.Linear(800, 200),
            nn.Linear(200, 1),
            )

    def forward(self, x):
        x = x.unsqueeze(2)  # Add a height dimension
        print_shapes = self.print_shapes

        x = self.layer1(x)
        if print_shapes:
            print(x.shape)
        x = self.layer2(x)
        if print_shapes:
            print(x.shape)
        x = self.layer3(x)
        if print_shapes:
            print(x.shape)
        x = self.pool2(x)
        if print_shapes:
            print(x.shape)
        x = self.layer4(x)
        if print_shapes:
            print(x.shape)
        x = self.pool3(x)
        if print_shapes:
            print(x.shape)
        x = self.layer5(x)
        if print_shapes:
            print(x.shape)

        x = x.view(x.size(0), -1)
        x = self.linear1(x)
        if print_shapes:
            print(x.shape)
        return x
