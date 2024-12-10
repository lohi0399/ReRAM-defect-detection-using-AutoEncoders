import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

data = np.load("mock_dataset.npz")
inputs = data['inputs'].astype(np.float32)
labels = data['labels'].astype(np.float32)
defective_outputs = inputs[:, -394:]  # Extract defective outputs from inputs
inputs_only = inputs[:,:784]
print(inputs_only.shape)

inputs_only = inputs_only* 1e3
labels = labels* 1e3
defective_outputs = defective_outputs* 1e3

delta = defective_outputs- labels

plt.figure(figsize=(14, 6))
sns.heatmap(defective_outputs[:50,:50]- labels[:50,:50], cmap="Blues", cbar=True, xticklabels=False, yticklabels=False)
plt.title("Delta Heatmap (First 50 Columns)")
plt.grid()
plt.show()



# Histogram for deltas
plt.figure(figsize=(14, 6))
plt.hist((delta).flatten(), bins=50,alpha=0.7,color='orange',label='Delta')
plt.title("Distribution of Delta Values in mA")
plt.xlabel("Delta (mA)")
plt.ylabel("Frequency")
plt.grid()
plt.show()


# Histogram for Clean Dataset
plt.subplot(1, 2, 1)
plt.hist(labels.flatten(), bins=50, alpha=0.7, color="blue", label="Clean Data")
plt.title("Distribution of Clean Dataset Values in mA")
plt.xlabel("Current (mA)")
plt.ylabel("Frequency")
plt.grid()
plt.legend()

# Histogram for Defective Dataset
plt.subplot(1, 2, 2)
plt.hist(defective_outputs.flatten(), bins=50, alpha=0.7, color="red", label="Defective Data")
plt.title("Distribution of Defective Dataset Values in mA")
plt.xlabel("Current (mA)")
plt.ylabel("Frequency")
plt.grid()
plt.legend()

plt.tight_layout()
plt.show()

# Calculate mean and standard deviation for both datasets
clean_mean = np.mean(labels)
clean_std = np.std(labels)
defective_mean = np.mean(defective_outputs)
defective_std = np.std(defective_outputs)

print(f"Mean of the clean data: {clean_mean} ; Std.Dev of clean data: {clean_std}")
print(f"Mean of the randomly defective data: {defective_mean} ; Std.Dev of randomly defective data: {defective_std}")


# Flatten the datasets for comparison
clean_flattened = labels.flatten()
defective_flattened = defective_outputs.flatten()

# Calculate MAE and MSE
mae = mean_absolute_error(clean_flattened, defective_flattened)
mse = mean_squared_error(clean_flattened, defective_flattened)

# Calculate correlation coefficient
correlation = np.corrcoef(clean_flattened, defective_flattened)[0, 1]

print(f'MAE: {mae}')
print(f'MSE: {mse}')
print(f'correlation: {correlation}')















# # Convert currents to µA for better readability
# clean_currents_ua = clean_currents * 1e3  # Convert from A to mA
# defective_currents_ua = defective_currents * 1e3  # Convert from A to mA

# # Heatmap for deltas
# plt.figure(figsize=(14, 6))
# sns.heatmap(clean_currents_ua[:, :20] - defective_currents_ua[:, :20], cmap="coolwarm", cbar=True, xticklabels=False, yticklabels=False)
# plt.title("Delta Heatmap (First 20 Columns)")
# plt.grid()
# plt.show()

# # Histogram for deltas
# plt.figure(figsize=(14, 6))
# plt.hist((clean_currents_ua[:, :20] - defective_currents_ua[:, :20]).flatten(), bins=50,alpha=0.7,color='orange',label='Delta')
# plt.title("Distribution of Delta Values in mA")
# plt.xlabel("Delta (mA)")
# plt.ylabel("Frequency")
# plt.grid()
# plt.show()

# # Histograms
# plt.figure(figsize=(12, 6))



# # Histogram for Clean Dataset
# plt.subplot(1, 2, 1)
# plt.hist(clean_currents_ua.flatten(), bins=50, alpha=0.7, color="blue", label="Clean Data")
# plt.title("Distribution of Clean Dataset Values in mA")
# plt.xlabel("Current (mA)")
# plt.ylabel("Frequency")
# plt.grid()
# plt.legend()

# # Histogram for Defective Dataset
# plt.subplot(1, 2, 2)
# plt.hist(defective_currents_ua.flatten(), bins=50, alpha=0.7, color="red", label="Defective Data")
# plt.title("Distribution of Defective Dataset Values in mA")
# plt.xlabel("Current (mA)")
# plt.ylabel("Frequency")
# plt.grid()
# plt.legend()

# plt.tight_layout()
# plt.show()


# # Calculate mean and standard deviation for both datasets
# clean_mean = np.mean(clean_currents_ua)
# clean_std = np.std(clean_currents_ua)
# defective_mean = np.mean(defective_currents_ua)
# defective_std = np.std(defective_currents_ua)

# print(f"Mean of the clean data: {clean_mean} ; Std.Dev of clean data: {clean_std}")
# print(f"Mean of the randomly defective data: {defective_mean} ; Std.Dev of randomly defective data: {defective_std}")


# # Flatten the datasets for comparison
# clean_flattened = clean_currents_ua.flatten()
# defective_flattened = defective_currents_ua.flatten()

# # Calculate MAE and MSE
# mae = mean_absolute_error(clean_flattened, defective_flattened)
# mse = mean_squared_error(clean_flattened, defective_flattened)

# # Calculate correlation coefficient
# correlation = np.corrcoef(clean_flattened, defective_flattened)[0, 1]

# print(f'MAE: {mae}')
# print(f'MSE: {mse}')
# print(f'correlation: {correlation}')










