# -------------------------------------------------------------------
# A Convolutional Neural Network for EEG data
# It uses 2D convolutional layers to extract features from the EEG signals.
# same as nn_model9.py bot for FFT Data
# -------------------------------------------------------------------

import torch
import torch.nn as nn

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class EEGNET(nn.Module):
    def __init__(self, filter_sizing, dropout, D, receptive_field=2, mean_pool=5):
        super(EEGNET,self).__init__()
        channel_amount = 6
        num_classes = 1
        self.temporal=nn.Sequential(
            nn.Conv2d(6,filter_sizing,kernel_size=[1,receptive_field],stride=1, bias=False, padding='same'), 
            nn.BatchNorm2d(filter_sizing),
        )
        self.spatial=nn.Sequential(
            # nn.Conv2d(filter_sizing,filter_sizing*D,kernel_size=[channel_amount,1],bias=False, groups=filter_sizing),
            nn.Conv2d(filter_sizing,filter_sizing*D,kernel_size=[1, channel_amount],bias=False, groups=filter_sizing),
            nn.BatchNorm2d(filter_sizing*D),
            nn.ELU(True),
        )

        self.seperable=nn.Sequential(
            nn.Conv2d(filter_sizing*D,filter_sizing*D,kernel_size=[1,16], padding='same',groups=filter_sizing*D, bias=False),
            nn.Conv2d(filter_sizing*D,filter_sizing*D,kernel_size=[1,1], padding='same',groups=1, bias=False),
            nn.BatchNorm2d(filter_sizing*D),
            nn.ELU(True),
        )
        self.avgpool1 = nn.AvgPool2d([1, 5], stride=[1, 5], padding=0)   
        self.avgpool2 = nn.AvgPool2d([1, 5], stride=[1, 5], padding=0)
        self.dropout = nn.Dropout(dropout)
        self.view = nn.Sequential(Flatten())

        endsize = 14
        self.fc2 = nn.Linear(endsize, num_classes)

    def forward(self,x):
        # print(x.shape)
        out = self.temporal(x)
        # print(x.shape)
        out = self.spatial(out)
        # print(x.shape)
        out = self.avgpool1(out)
        # print(x.shape)
        out = self.dropout(out)
        # print(x.shape)
        out = self.seperable(out)
        # print(x.shape)
        out = self.avgpool2(out)
        # print(x.shape)
        out = self.dropout(out)
        # print(x.shape)
        out = out.view(out.size(0), -1)
        # print(x.shape)
        prediction = self.fc2(out)
        return prediction
