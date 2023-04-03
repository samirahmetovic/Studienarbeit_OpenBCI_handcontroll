from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import torch 
import torch.nn as nn   
from nn_model5 import EEGClassifier

state_dict = torch.load("model_cnn.pt")

# Create a new instance of the model
model = EEGClassifier()

# Load the state dictionary into the model
model.load_state_dict(state_dict)

# Define a hook function to extract the internal activations
activations = []
def get_activations(name):
    def hook(model, input, output):
        activations.append(output.detach())
    return hook

# Register the hook function for the first layer of the model
model.conv1.register_forward_hook(get_activations('conv1'))

# Pass some input through the model to trigger the hook
model.eval()
with torch.no_grad():
    inputs = torch.randn(1, 16, 128)
    outputs = model(inputs)

# Visualize the internal activations
plt.figure(figsize=(10, 10))
for i in range(32):
    plt.subplot(8, 4, i+1)
    plt.imshow(activations[0][0, i].cpu(), cmap='gray')
    plt.axis('off')
plt.show()