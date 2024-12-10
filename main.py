import numpy as np
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from autoencoder import AE
from linear_nn import NN

data = np.load("mock_dataset.npz")
inputs = data['inputs'].astype(np.float32)
labels = data['labels'].astype(np.float32)

num_samples = inputs.shape[0]
input_size = inputs.shape[1]
output_size = labels.shape[1]

inputs = torch.from_numpy(inputs)
labels = torch.from_numpy(labels)

#Train-test split
train_inputs, train_labels = inputs[:800], labels[:800]
test_inputs, test_labels = inputs[800:], labels[800:]

train_labels_test = labels[:800]
test_labels_test = labels[800:]

#Initialize the neural network
model = NN(394,394)
criterion = nn.MSELoss()
optmizier = optim.Adam(model.parameters(), lr=1e-3)

#Training loop
for epoch in range(5000):
    optmizier.zero_grad()
    # predictions = model(train_inputs[:,-394:])
    # loss = criterion(predictions,train_labels)

    predictions = model(train_labels_test)
    loss = criterion(predictions,train_labels_test)
    

    loss.backward()
    optmizier.step()

    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}, Loss: {loss.item():.9f}")

#Evaluate on test data
with torch.no_grad():
    model.eval()
    test_predictions = model(test_labels_test)
    test_loss = criterion(test_predictions, test_labels_test)
    print(f"Test Loss: {test_loss.item():.9f}")

torch.save(model.state_dict(), 'linear_nn.pth')
print("Model saved!")

















#-------------------------------------------------------------AE------------------------------------------------------------------------#
# def custom_collate_fn(batch):
#     batch = np.array(batch)
#     # Normalize the batch
#     batch = (batch - batch.mean(axis=0)) / batch.std(axis=0)
#     return torch.tensor(batch, dtype=torch.float32)

# def main():
        
#     #---------------------------------------------------------DATA PREPARATION------------------------------------------------------------------#

#     clean_data_path = "crossbar_clean_dynamic.csv"
#     clean_data = torch.from_numpy(np.loadtxt(clean_data_path, dtype=np.float32, delimiter=",", skiprows=1))

#     clean_train , clean_test = split_dataset(clean_data, train_ratio=0.8)

#     clean_train_dataloader = torch.utils.data.DataLoader(dataset=clean_train, batch_size=16,shuffle=False, drop_last=True,collate_fn=custom_collate_fn)
#     clean_test_dataloader = torch.utils.data.DataLoader(dataset=clean_test, batch_size=16,shuffle=False, drop_last=True,collate_fn=custom_collate_fn)
    
    
#     #---------------------------------------------------------MODEL TRAINING------------------------------------------------------------------#
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model = AE(input_dim = 392).to(device)
#     optimizer = optim.Adam(model.parameters(), lr=1e-3)
#     criterion = nn.MSELoss()
#     epochs = 300

#     for epoch in range(epochs):
#         loss = 0
#         for batch in clean_train_dataloader:

#             # Forward Pass
#             outputs = model(batch)
#             train_loss = criterion(outputs, batch)
            
#             # Backward Pass
#             optimizer.zero_grad()
#             train_loss.backward()
#             optimizer.step()
            
#             # add the mini-batch training loss to epoch loss
#             loss += train_loss.item()
        
#         # compute the epoch training loss
#         loss = loss / len(clean_train_dataloader)
        
#         # display the epoch training loss
#         print("epoch : {}/{}, loss = {:.9f}".format(epoch + 1, epochs, loss))

#     #----------------------------------------------------EVALUATING THE MODEL--------------------------------------------------------------#
#     model.eval()
#     with torch.no_grad():
#         total_loss = 0
#         for batch in clean_test_dataloader:
#             outputs = model(batch)
#             loss = criterion(outputs,batch)
#             total_loss += loss.item()
#     print(f"Test Loss: {total_loss / len(clean_test_dataloader): .9f}")

#     #----------------------------------------------------SAVING THE MODEL-------------------------------------------------------------------#
#     torch.save(model.state_dict(), 'autoencoder.pth')
#     print("Model saved!")

# if __name__ == "__main__":
#     main()