import numpy as np
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from autoencoder import AE


def main():
        
    #---------------------------------------------------------DATA PREPARATION------------------------------------------------------------------#

    clean_data_path = "crossbar_clean_dynamic.csv"
    defective_data_path = "crossbar_defective_dynamic.csv" 

    clean_data = torch.from_numpy(np.loadtxt(clean_data_path, dtype=np.float32, delimiter=",", skiprows=1))
    defective_data = torch.from_numpy(np.loadtxt(defective_data_path, dtype=np.float32, delimiter=",", skiprows=1))

    clean_dataloader = torch.utils.data.DataLoader(dataset=clean_data, batch_size=16,shuffle=False, drop_last=True)
    defective_dataloader = torch.utils.data.DataLoader(dataset=defective_data, batch_size=16,shuffle=False, drop_last=True)

    #---------------------------------------------------------MODEL TRAINING------------------------------------------------------------------#
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AE(input_dim = 392).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    epochs = 20

    for epoch in range(epochs):
        loss = 0
        for batch in clean_dataloader:

            # Forward Pass
            outputs = model(batch)
            train_loss = criterion(outputs, batch)
            
            # Backward Pass
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()
            
            # add the mini-batch training loss to epoch loss
            loss += train_loss.item()
        
        # compute the epoch training loss
        loss = loss / len(clean_dataloader)
        
        # display the epoch training loss
        print("epoch : {}/{}, loss = {:.6f}".format(epoch + 1, epochs, loss))


if __name__ == "__main__":
    main()