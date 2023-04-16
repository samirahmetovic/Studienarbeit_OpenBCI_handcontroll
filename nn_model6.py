'''
This is Model 6
This model uses COnv2d instead of Conv1d because we want to train the model on a time-frequenz aspect
that means we train our cleaned (wavelet transformed) data on the model so that it learn time based frequency patterns
'''

import torch 
import torch.nn as nn   

class EEGClassifier(nn.Module):
    def __init__(self, num_classes=2, dropout=0.0, print_shapes=False):
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
        return x