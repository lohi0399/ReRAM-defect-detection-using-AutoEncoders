import torch
import numpy as np
from autoencoder import AE
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns

def custom_collate_fn(batch):
    batch = np.array(batch)
    # Normalize the batch
    batch = (batch - batch.mean(axis=0)) / batch.std(axis=0)
    return torch.tensor(batch, dtype=torch.float32)



clean_data_path = "crossbar_clean_dynamic.csv"
clean_data = torch.from_numpy(np.loadtxt(clean_data_path, dtype=np.float32, delimiter=",", skiprows=1))
print(clean_data.shape)
clean_dataloader = defective_dataloader = torch.utils.data.DataLoader(dataset=clean_data, batch_size=16,shuffle=False, drop_last=True,collate_fn=custom_collate_fn)

defective_data_path = "crossbar_defective_dynamic.csv" 
defective_data = torch.from_numpy(np.loadtxt(defective_data_path, dtype=np.float32, delimiter=",", skiprows=1))
defective_dataloader = torch.utils.data.DataLoader(dataset=defective_data, batch_size=16,shuffle=False, drop_last=True,collate_fn=custom_collate_fn)


def reload_model(model_path, input_dim=392):
    # Initialize the model with the same architecture
    model = AE(input_dim=input_dim)
    
    # Load the state dictionary
    model.load_state_dict(torch.load(model_path,weights_only=True))
    
    # Set the model to evaluation mode
    model.eval()
    
    return model



model_path = 'autoencoder.pth'
model = reload_model(model_path)

# If you want to move the model to GPU
if torch.cuda.is_available():
    model = model.to('cuda')
    print("CUDA available")

criterion = nn.MSELoss()

#----------------------------------------------------EVALUATING THE MODEL--------------------------------------------------------------#
model.eval()
with torch.no_grad():
    total_loss_clean = 0
    for batch_clean in clean_dataloader:
        outputs_clean = model(batch_clean)
        delta_clean = outputs_clean - batch_clean
        loss = criterion(outputs_clean,batch_clean)
        print('Clean Loss: ',loss)
        break
        
        # total_loss_clean += loss.item()
    # print(f"Test Loss: {total_loss_clean / len(clean_dataloader): .9f}")

    total_loss_defective = 0
    for batch_defective in defective_dataloader:
        outputs_defective = model(batch_defective)
        delta_defective = outputs_defective - batch_defective
        loss = criterion(outputs_defective,batch_defective)
        print('Defective Loss: ',loss)
        break
        # total_loss_defective += loss.item()
    # print(f"Test Loss: {total_loss_defective / len(defective_dataloader): .9f}")

#Heatmap
# plt.figure(figsize=(14, 6))
# sns.heatmap((outputs_clean - outputs_defective), cmap="coolwarm", cbar=True, xticklabels=False, yticklabels=False)
# plt.title("Delta Heatmap of clean data")
# plt.grid()
# plt.show()

# Assuming outputs_clean and outputs_defective are numpy arrays or similar
delta = outputs_clean[13, :] - outputs_defective[13, :]  # Calculate the difference

plt.figure(figsize=(14, 6))
plt.plot(batch_clean[13, :], label="Clean Output", color="blue", linestyle="-", linewidth=2)
plt.plot(batch_defective[13, :], label="Defective Output", color="red", linestyle="--", linewidth=2)
plt.plot(delta, label="Delta (Difference)", color="green", linestyle=":", linewidth=2)
plt.xlabel("Index")
plt.ylabel("Output Value")
plt.title("Comparison of Clean and Defective Outputs with Delta")
plt.legend(loc="upper right")
plt.grid(True)
plt.show()

