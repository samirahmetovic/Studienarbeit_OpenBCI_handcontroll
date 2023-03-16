import torch
from nn_model import EEGClassifier

model = EEGClassifier(16, 32, 1)

model.load_state_dict(torch.load('model.pt'))
model.eval()

