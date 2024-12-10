import numpy as np
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from autoencoder import AE

# Train-test split
def split_dataset(dataset, train_ratio=0.8):
    train_size = int(train_ratio * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    return train_dataset, test_dataset

def custom_collate_fn(batch):
    batch = np.array(batch)
    # Normalize the batch
    batch = (batch - batch.mean(axis=0)) / batch.std(axis=0)
    return torch.tensor(batch, dtype=torch.float32)

def main():
        
    #---------------------------------------------------------DATA PREPARATION------------------------------------------------------------------#

    clean_data_path = "crossbar_clean_dynamic.csv"
    clean_data = torch.from_numpy(np.loadtxt(clean_data_path, dtype=np.float32, delimiter=",", skiprows=1))

    clean_train , clean_test = split_dataset(clean_data, train_ratio=0.8)

    clean_train_dataloader = torch.utils.data.DataLoader(dataset=clean_train, batch_size=16,shuffle=False, drop_last=True,collate_fn=custom_collate_fn)
    clean_test_dataloader = torch.utils.data.DataLoader(dataset=clean_test, batch_size=16,shuffle=False, drop_last=True,collate_fn=custom_collate_fn)
    
    
    #---------------------------------------------------------MODEL TRAINING------------------------------------------------------------------#
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AE(input_dim = 392).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    epochs = 300

    for epoch in range(epochs):
        loss = 0
        for batch in clean_train_dataloader:

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
        loss = loss / len(clean_train_dataloader)
        
        # display the epoch training loss
        print("epoch : {}/{}, loss = {:.9f}".format(epoch + 1, epochs, loss))

    #----------------------------------------------------EVALUATING THE MODEL--------------------------------------------------------------#
    model.eval()
    with torch.no_grad():
        total_loss = 0
        for batch in clean_test_dataloader:
            outputs = model(batch)
            loss = criterion(outputs,batch)
            total_loss += loss.item()
    print(f"Test Loss: {total_loss / len(clean_test_dataloader): .9f}")

    #----------------------------------------------------SAVING THE MODEL-------------------------------------------------------------------#
    torch.save(model.state_dict(), 'autoencoder.pth')
    print("Model saved!")

if __name__ == "__main__":
    main()