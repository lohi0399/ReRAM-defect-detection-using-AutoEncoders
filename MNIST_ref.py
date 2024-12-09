import numpy as np
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from autoencoder import AE



#---------------------------------------------------------DATA PREPARATION------------------------------------------------------------------#
seed = 42
torch.manual_seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

batch_size = 512
epochs = 20
learning_rate = 1e-3

transform = transforms.ToTensor()

mnist_data = datasets.MNIST(root='./MNIST_data',train=True,download=False,transform=transform)
data_loader = torch.utils.data.DataLoader(dataset=mnist_data, batch_size=64,shuffle=True)
print(len(data_loader))


for (img,_) in data_loader:
    print('Image size: ',img.shape)
    img = img.view(-1, 784)
    print('New image shape: ',img.shape)
    break


# #---------------------------------------------------------MODEL TRAINING------------------------------------------------------------------#
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = AE(input_shape = 784).to(device)
# optimizer = optim.Adam(model.parameters(), lr=1e-3)
# criterion = nn.MSELoss()
# epochs = 20

# for epoch in range(epochs):
#     loss = 0
#     for batch_features, _ in data_loader:
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

# #-------------------------------------------------------------------Testing the data---------------------------------------------------------------------------------#

# test_dataset = datasets.MNIST(
#     root="./torch_datasets", train=False, transform=transform, download=True
# )

# test_loader = torch.utils.data.DataLoader(
#     test_dataset, batch_size=10, shuffle=False
# )

# test_examples = None

# with torch.no_grad():
#     for batch_features in test_loader:
#         batch_features = batch_features[0]
#         test_examples = batch_features.view(-1, 784).to(device)
#         reconstruction = model(test_examples)
#         break

# #-----------------------------------------------------------------Visualizing the data----------------------------------------------------------------------------------#

# with torch.no_grad():
#     number = 10
#     plt.figure(figsize=(20, 4))
#     for index in range(number):
#         # display original
#         ax = plt.subplot(2, number, index + 1)
#         plt.imshow(test_examples[index].numpy().reshape(28, 28))
#         plt.gray()
#         ax.get_xaxis().set_visible(False)
#         ax.get_yaxis().set_visible(False)

#         # display reconstruction
#         ax = plt.subplot(2, number, index + 1 + number)
#         plt.imshow(reconstruction[index].numpy().reshape(28, 28))
#         plt.gray()
#         ax.get_xaxis().set_visible(False)
#         ax.get_yaxis().set_visible(False)
#     plt.show()
