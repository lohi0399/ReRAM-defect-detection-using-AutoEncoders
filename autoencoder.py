import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt


class AE(nn.Module):
    def __init__(self, input_dim=392):
        super(AE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 196),
            nn.ReLU(),
            nn.Linear(196, 98),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(98, 196),
            nn.ReLU(),
            nn.Linear(196, input_dim),
            # nn.Sigmoid()  # Assuming your input data is normalized [0, 1]
        )

    def forward(self, x):
        latent = self.encoder(x)
        decoded = self.decoder(latent)
        return decoded