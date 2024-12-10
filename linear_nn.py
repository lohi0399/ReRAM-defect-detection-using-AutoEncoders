import torch
import torch.nn as nn
import torch.optim as optim

# Define the neural network
class NN(nn.Module):
    def __init__(self, input_size, output_size):
        super(NN, self).__init__()
        self.fc1 = nn.Linear(input_size, 197)  # Combine inputs and outputs
        self.bn1 = nn.BatchNorm1d(197)  # Batch normalization for fc1

        self.fc2 = nn.Linear(197, 98)
        self.bn2 = nn.BatchNorm1d(98)   # Batch normalization for fc2

        self.fc3 = nn.Linear(98, 32)
        self.bn3 = nn.BatchNorm1d(32)  # Batch normalization for fc3

        self.fc4 = nn.Linear(32, 98)
        self.bn4 = nn.BatchNorm1d(98)  # Batch normalization for fc4

        self.fc5 = nn.Linear(98, 197)
        self.bn5 = nn.BatchNorm1d(197) # Batch normalization for fc5

        self.fc6 = nn.Linear(197, output_size)  # Predict corrected outputs

    def forward(self, x):

        x = torch.relu(self.bn1(self.fc1(x)))
        x = torch.relu(self.bn2(self.fc2(x)))
        x = torch.relu(self.bn3(self.fc3(x)))
        x = torch.relu(self.bn4(self.fc4(x)))
        x = torch.relu(self.bn5(self.fc5(x)))
        x = self.fc6(x)
        return x
