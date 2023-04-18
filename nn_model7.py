import torch 
import torch.nn as nn   

class EEGClassifier(nn.Module):
    def __init__(self, num_classes=2, dropout=0.0, print_shapes=False):
        self.print_shapes = print_shapes
        super(EEGClassifier, self).__init__()
        self.layer1 = nn.Sequential(
            nn.ConstantPad1d((15, 15), 0),
            nn.Conv1d(in_channels=6, out_channels=24, kernel_size=3, stride=1, padding=0),
            nn.LeakyReLU(),
            nn.Dropout(p=dropout))
        self.layer2 = nn.Sequential(
            nn.Conv1d(in_channels=24, out_channels=48, kernel_size=3, stride=2, padding=0),
            nn.BatchNorm1d(213, affine=False),
            nn.LeakyReLU(),
            nn.MaxPool1d(kernel_size=3, stride=2)
        )
        self.layer3 = nn.Sequential(
            nn.Conv1d(in_channels=48, out_channels=48, kernel_size=3, stride=1),
            nn.LeakyReLU(),
            nn.Dropout(p=dropout))
        self.pool2 = nn.Sequential(
            nn.MaxPool1d(kernel_size=2, stride=2))
        self.layer4 = nn.Sequential(
            nn.Conv1d(in_channels=48, out_channels=96, kernel_size=3, stride=1),
            nn.BatchNorm1d(50, affine=False),
            nn.LeakyReLU(),
            nn.Dropout(p=dropout))
        self.pool3 = nn.Sequential(
            nn.MaxPool1d(kernel_size=3, stride=3))
        self.layer5 = nn.Sequential(
            nn.Conv1d(in_channels=96, out_channels=400, kernel_size=3, stride=7),
            nn.BatchNorm1d(2, affine=False),
            nn.LeakyReLU())
            
        '''self.pool4 = nn.Sequential(
            nn.MaxPool1d(kernel_size=3, stride=3))'''
        self.linear1 = nn.Sequential(
            # nn.Linear(164, num_classes),
            nn.Linear(2, 400),
            nn.Linear(400, 800),
            nn.Linear(800, 200),
            nn.Linear(200, 1),
            # nn.LogSoftmax(dim=1)
            )

    def forward(self, x):
        # Bool of printing shapes
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
        # x = self.pool4(x)
        x = x.view(x.size(0), -1)
        x = self.linear1(x)
        if print_shapes:
            print(x.shape)
        return x