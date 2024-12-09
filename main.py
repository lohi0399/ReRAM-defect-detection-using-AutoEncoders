import numpy as np
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from autoencoder import AE



#---------------------------------------------------------DATA PREPARATION------------------------------------------------------------------#

clean_data_path = "crossbar_clean_dynamic.csv"
defective_data_path = "crossbar_defective_dynamic.csv" 

clean_data = torch.from_numpy(np.loadtxt(clean_data_path, dtype=np.float32, delimiter=",", skiprows=1))
defective_data = torch.from_numpy(np.loadtxt(defective_data_path, dtype=np.float32, delimiter=",", skiprows=1))

print(clean_data.shape)











#---------------------------------------------------------MODEL TRAINING------------------------------------------------------------------#
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AE(input_shape = 392).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()


# for epoch in range(epochs):
#     loss = 0
#     for batch_features, _ in train_loader:
#         # reshape mini-batch data to [N, 784] matrix
#         # load it to the active device
#         batch_features = batch_features.view(-1, 784).to(device)
        
#         # reset the gradients back to zero
#         # PyTorch accumulates gradients on subsequent backward passes
#         optimizer.zero_grad()
        
#         # compute reconstructions
#         outputs = model(batch_features)
        
#         # compute training reconstruction loss
#         train_loss = criterion(outputs, batch_features)
        
#         # compute accumulated gradients
#         train_loss.backward()
        
#         # perform parameter update based on current gradients
#         optimizer.step()
        
#         # add the mini-batch training loss to epoch loss
#         loss += train_loss.item()
    
#     # compute the epoch training loss
#     loss = loss / len(train_loader)
    
#     # display the epoch training loss
#     print("epoch : {}/{}, loss = {:.6f}".format(epoch + 1, epochs, loss))