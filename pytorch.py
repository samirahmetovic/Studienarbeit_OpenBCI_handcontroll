import numpy as np
import torch
import torch.nn as nn

# Load data from CSV file
data = np.loadtxt('bci_data.csv', delimiter=',', skiprows=1)

# Split data into inputs (EEG signals) and targets (hand state)
inputs = torch.tensor(data[:, :-1], dtype=torch.float32)
targets = torch.tensor(data[:, -1], dtype=torch.float32)


class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

# Define hyperparameters
input_size = inputs.shape[1]
hidden_size = 10
output_size = 2
learning_rate = 0.001
num_epochs = 100

# Initialize model and loss function
model = NeuralNet(input_size, hidden_size, output_size)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Train model
for epoch in range(num_epochs):
    # Forward pass
    outputs = model(inputs)
    loss = criterion(outputs, targets.long())

    # Backward and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Print progress
    if (epoch+1) % 10 == 0:
        print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))
        
        
# Predict hand state for new EEG signals
new_inputs = torch.tensor([[0.5, 0.4, 0.6, ..., 0.2]])
outputs = model(new_inputs)
_, predicted = torch.max(outputs.data, 1)
if predicted.item() == 0:
    print('Hand is closed')
else:
    print('Hand is open')