import torch
import numpy as np
from autoencoder import AE
import torch.nn as nn



clean_data_path = "crossbar_clean_dynamic.csv"
clean_data = torch.from_numpy(np.loadtxt(clean_data_path, dtype=np.float32, delimiter=",", skiprows=1))
clean_dataloader = defective_dataloader = torch.utils.data.DataLoader(dataset=clean_data, batch_size=16,shuffle=False, drop_last=True)

defective_data_path = "crossbar_defective_dynamic.csv" 
defective_data = torch.from_numpy(np.loadtxt(defective_data_path, dtype=np.float32, delimiter=",", skiprows=1))
defective_dataloader = torch.utils.data.DataLoader(dataset=defective_data, batch_size=16,shuffle=False, drop_last=True)


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
    total_loss = 0
    for batch in clean_dataloader:
        outputs = model(batch)
        loss = criterion(outputs,batch)
        total_loss += loss.item()
    print(f"Test Loss: {total_loss / len(defective_dataloader): .9f}")
