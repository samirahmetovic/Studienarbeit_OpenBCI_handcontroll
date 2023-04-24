import torch 
import torch.nn as nn   

class EEGClassifier(nn.Module):
    def __init__(self, num_classes=1, dropout=0.0, print_shapes=False):
        self.print_shapes = print_shapes
        super(EEGClassifier, self).__init__()
        self.layer1 = nn.Sequential(
            nn.ConstantPad1d((15, 15), 0),
            nn.Conv1d(in_channels=6, out_channels=48, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Dropout(p=dropout))
        self.layer2 = nn.Sequential(
            nn.Conv1d(in_channels=48, out_channels=192, kernel_size=3, stride=2, padding=0),
            nn.BatchNorm1d(213, affine=False),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=2)
        )
        self.pool1 = nn.Sequential(
            nn.MaxPool1d(kernel_size=2, stride=2))
        self.layer3 = nn.Sequential(
            nn.Conv1d(in_channels=192, out_channels=400, kernel_size=3, stride=1),
            nn.BatchNorm1d(51, affine=False),
            nn.ReLU(),
            nn.Dropout(p=dropout))
            
        '''self.pool4 = nn.Sequential(
            nn.MaxPool1d(kernel_size=3, stride=3))'''
        self.linear1 = nn.Sequential(
            # nn.Linear(164, num_classes),
            nn.Linear(51, 400),
            nn.Linear(400, 200),
            nn.Linear(200, 100),
            nn.Linear(100, num_classes),
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
        x = self.pool1(x)
        if print_shapes:
            print(x.shape)
        x = self.layer3(x)
        if print_shapes:
            print(x.shape)
        
        x = x.view(x.size(0), -1)
        x = self.linear1(x)
        if print_shapes:
            print(x.shape)
        x = nn.Softmax(dim=1)(x)
        if print_shapes:
            print(x.shape)
        return x