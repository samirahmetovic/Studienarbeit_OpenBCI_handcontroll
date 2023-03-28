import torch.nn as nn

class EEGClassifier(nn.Module):
    def __init__(self, input_size, input_length):
        super(EEGClassifier, self).__init__()
        self.conv1 = nn.Conv1d(input_size, 64, kernel_size=3)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=2)
        self.conv3 = nn.Conv1d(128, 128, kernel_size=2)
        self.conv4 = nn.Conv1d(128, 64, kernel_size=2)
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        self.conv5 = nn.Conv1d(64, 64, kernel_size=16)
        self.pool2 = nn.MaxPool1d(kernel_size=2)
        self.flatten = nn.Flatten()

        # Calculate the output size after passing through the convolutional and pooling layers
        out_length = (((((input_length - 2) // 2) - 1) - 1) - 1) // 2 - 15
        num_features = 64 * out_length

        self.fc1 = nn.Linear(num_features, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 3)
        self.activation = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.activation(self.conv1(x))
        x = self.activation(self.conv2(x))
        x = self.activation(self.conv3(x))
        x = self.activation(self.conv4(x))
        x = self.pool1(x)
        x = self.activation(self.conv5(x))
        x = self.pool2(x)
        x = self.flatten(x)
        
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.activation(self.fc3(x))
        x = self.fc4(x)
        x = self.softmax(x)
        return x
