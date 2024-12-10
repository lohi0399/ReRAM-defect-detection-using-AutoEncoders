import torch
import torch.nn as nn
import torch.optim as optim

# Define the neural network
class NN(nn.Module):
    def __init__(self, input_size, output_size):
        super(NN, self).__init__()
        self.fc1 = nn.Linear(input_size, 588)  # Combine inputs and outputs
        self.bn1 = nn.BatchNorm1d(588)  # Batch normalization for fc1

        self.fc2 = nn.Linear(588, 490)
        self.bn2 = nn.BatchNorm1d(490)   # Batch normalization for fc2

        self.fc3 = nn.Linear(490, 420)
        self.bn3 = nn.BatchNorm1d(420)  # Batch normalization for fc3

        self.fc4 = nn.Linear(420, output_size)  # Predict corrected outputs

    def forward(self, x):

        x = torch.relu(self.bn1(self.fc1(x)))
        x = torch.relu(self.bn2(self.fc2(x)))
        x = torch.relu(self.bn3(self.fc3(x)))
        x = self.fc4(x)
        return x
