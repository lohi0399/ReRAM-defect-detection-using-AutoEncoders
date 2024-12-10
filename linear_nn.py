import torch
import torch.nn as nn
import torch.optim as optim

# Define the neural network
class NN(nn.Module):
    def __init__(self, input_size, output_size):
        super(NN, self).__init__()
        self.fc1 = nn.Linear(input_size, 588)  # Combine inputs and outputs
        self.fc2 = nn.Linear(588, 410)
        self.fc3 = nn.Linear(410, output_size)  # Predict corrected outputs

    def forward(self, x):

        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
