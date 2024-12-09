import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

clean_currents = pd.read_csv('crossbar_clean_dynamic.csv').to_numpy()
defective_currents = pd.read_csv('crossbar_defective_dynamic.csv').to_numpy()

# Convert currents to µA for better readability
clean_currents_ua = clean_currents * 1e6  # Convert from A to µA
defective_currents_ua = defective_currents * 1e6  # Convert from A to µA

# Heatmaps
plt.figure(figsize=(14, 6))

# Heatmap for Clean Dataset
plt.subplot(1, 2, 1)
sns.heatmap(clean_currents_ua[:, :20], cmap="Blues", cbar=True, xticklabels=False, yticklabels=False)
plt.title("Clean Dataset Heatmap (First 20 Columns) in µA")

# Heatmap for Defective Dataset
plt.subplot(1, 2, 2)
sns.heatmap(defective_currents_ua[:, :20], cmap="Reds", cbar=True, xticklabels=False, yticklabels=False)
plt.title("Defective Dataset Heatmap (First 20 Columns) in µA")

plt.tight_layout()
plt.show()

# Histograms
plt.figure(figsize=(12, 6))

# Histogram for Clean Dataset
plt.subplot(1, 2, 1)
plt.hist(clean_currents_ua.flatten(), bins=50, alpha=0.7, color="blue", label="Clean Data")
plt.title("Distribution of Clean Dataset Values in µA")
plt.xlabel("Current (µA)")
plt.ylabel("Frequency")
plt.legend()

# Histogram for Defective Dataset
plt.subplot(1, 2, 2)
plt.hist(defective_currents_ua.flatten(), bins=50, alpha=0.7, color="red", label="Defective Data")
plt.title("Distribution of Defective Dataset Values in µA")
plt.xlabel("Current (µA)")
plt.ylabel("Frequency")
plt.legend()

plt.tight_layout()
plt.show()

# Calculate mean and standard deviation for both datasets
clean_mean = np.mean(clean_currents_ua)
clean_std = np.std(clean_currents_ua)
defective_mean = np.mean(defective_currents_ua)
defective_std = np.std(defective_currents_ua)

print(f"Mean of the clean data: {clean_mean} ; Std.Dev of clean data: {clean_std}")
print(f"Mean of the randomly defective data: {defective_mean} ; Std.Dev of randomly defective data: {defective_std}")


# Flatten the datasets for comparison
clean_flattened = clean_currents_ua.flatten()
defective_flattened = defective_currents_ua.flatten()

# Calculate MAE and MSE
mae = mean_absolute_error(clean_flattened, defective_flattened)
mse = mean_squared_error(clean_flattened, defective_flattened)

# Calculate correlation coefficient
correlation = np.corrcoef(clean_flattened, defective_flattened)[0, 1]

print(f'MAE: {mae}')
print(f'MSE: {mse}')
print(f'correlation: {correlation}')










