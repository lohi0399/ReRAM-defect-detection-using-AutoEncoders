import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
from linear_nn import NN

#------------------------------------------------HELPER FUNCTIONS------------------------------------------------------------------#

def reload_model(model_path, input_dim=394, output_dim=394):
    # Initialize the model with the same architecture
    model = NN(1178,394)
    
    # Load the state dictionary
    model.load_state_dict(torch.load(model_path,weights_only=True))
    
    # Set the model to evaluation mode
    model.eval()
    
    return model

# Compare predictions with defective outputs and detect faults
def detect_faults(defective_outputs, predicted_outputs, threshold=1e-5):
    # Calculate the absolute difference between defective and corrected outputs
    differences = torch.abs(defective_outputs - predicted_outputs)
    
    # Detect faults where the difference exceeds the threshold
    fault_locations = torch.where(differences > threshold, 1, 0)
    return fault_locations, differences

#------------------------------------------------------------------------------------------------------------------------------------#
model_path = 'linear_nn.pth'
model = reload_model(model_path)
data = np.load("mock_dataset.npz")
inputs = data['inputs'].astype(np.float32)
labels = data['labels'].astype(np.float32)
inputs_tensor = torch.from_numpy(inputs)
labels_tensor = torch.from_numpy(labels)

# If you want to move the model to GPU
if torch.cuda.is_available():
    model = model.to('cuda')
    print("CUDA available")

criterion = nn.MSELoss()

#----------------------------------------------------EVALUATING THE MODEL--------------------------------------------------------------#
model.eval()

with torch.no_grad():
    predictions = model(inputs_tensor)


# Use the first 10 samples for testing
defective_outputs = inputs_tensor[:, -394:]  # Extract defective outputs from inputs
predicted_outputs = predictions

threshold = 1e-5
fault_locations, differences = detect_faults(defective_outputs, predicted_outputs, threshold=threshold)
fault_locations1, differences1 = detect_faults(defective_outputs, labels_tensor, threshold=threshold)

sample_idx = 0  # Index of the sample to visualize


plt.figure(figsize=(10, 5))

plt.subplot(1,2,1)
plt.plot(defective_outputs[sample_idx].numpy(), label="Defective Outputs", color="red", alpha=0.7)
plt.plot(predicted_outputs[sample_idx].numpy(), label="Model Outputs", color="green", alpha=0.7)
plt.title(f"Sample {sample_idx} - Outputs")
plt.xlabel("Output Index")
plt.ylabel("Output Value")
plt.legend()

plt.subplot(1,2,2)
plt.plot(defective_outputs[sample_idx].numpy(), label="Defective Outputs", color="red", alpha=0.7)
plt.plot(labels_tensor[sample_idx].numpy(), label="Ideal Outputs", color="green", alpha=0.7)
plt.title(f"Sample {sample_idx} - Outputs")
plt.xlabel("Output Index")
plt.ylabel("Output Value")
plt.legend()

plt.grid()
plt.show()

print(defective_outputs.shape)
print(predicted_outputs.shape)

plt.figure(figsize=(14, 6))

plt.subplot(1,2,1)
sns.heatmap(defective_outputs[:50,:50] - labels_tensor[:50,:50], cmap="coolwarm", cbar=True, xticklabels=False, yticklabels=False)
plt.title("Delta Heatmap with predicted outputs(First 50 Columns and rows)")
plt.grid()


plt.subplot(1,2,2)
sns.heatmap(defective_outputs[:50,:50] - predicted_outputs[:50,:50], cmap="coolwarm", cbar=True, xticklabels=False, yticklabels=False)
plt.title("Delta Heatmap with labels (First 50 Columns and rows)")
plt.grid()

plt.show()


# Select a subset of samples to visualize (e.g., the first 10)
samples_to_visualize = 10
binary_fault_map = fault_locations[:samples_to_visualize].numpy()
binary_fault_map1 = fault_locations1[:samples_to_visualize].numpy()
# Plot binary fault map
plt.figure(figsize=(12, 6))

plt.subplot(1,2,1)
plt.imshow(binary_fault_map1, cmap="gray", interpolation="nearest", aspect="auto")
plt.colorbar(label="Fault Detected (1 = Fault, 0 = No Fault)")
plt.title("Binary Fault Map - (defective & labels)")
plt.xlabel("Output Columns")
plt.ylabel("Sample Index")

plt.subplot(1,2,2)
plt.imshow(binary_fault_map, cmap="gray", interpolation="nearest", aspect="auto")
plt.colorbar(label="Fault Detected (1 = Fault, 0 = No Fault)")
plt.title("Binary Fault Map - (defective & predicted)")
plt.xlabel("Output Columns")
plt.ylabel("Sample Index")

plt.show()
