import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Load data from CSV file
data = np.loadtxt('bci_data.csv', delimiter=',', skiprows=1)

# Split data into inputs (EEG signals) and targets (hand state)
inputs = torch.tensor(data[:, :-1], dtype=torch.float32)
targets = torch.tensor(data[:, -1], dtype=torch.float32)

# Define neural network architecture
class EEGClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(EEGClassifier, self).__init__()
        self.hidden = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.output = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.hidden(x)
        x = self.relu(x)
        x = self.output(x)
        x = self.sigmoid(x)
        return x

# Define hyperparameters
input_size = inputs.shape[1]
hidden_size = 32
output_size = 1
lr = 0.001
num_epochs = 200

# Initialize model, loss function, and optimizer
model = EEGClassifier(input_size, hidden_size, output_size)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

# Train the model
for epoch in range(num_epochs):
    # Forward pass
    outputs = model(inputs)
    loss = criterion(outputs, targets)

    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Print loss for monitoring training progress
    if (epoch + 1) % 20 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')


# safe model to file
torch.save(EEGClassifier.state_dict(), "model.pth")

# Test the model on a new input
test_input = torch.tensor([[0.5, 0.6, 0.4, 0.2, 0.1, 0.7]])
prediction = model(test_input)
if prediction >= 0.5:
    print('Predicted class: closing hand')
else:
    print('Predicted class: opening hand')
