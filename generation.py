"""
Artificial dataset created in to simulate random defects in memristor crossbars for on-chip training

"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Generating a mock dataset for training a neural network to detect and mitigate faults

# Parameters
num_samples = 1000  # Total number of samples
input_size = 784  # Number of inputs (e.g., pixels from MNIST)
output_size = 394  # Number of outputs (crossbar columns)

# Generating synthetic input data (normalized between 0 and 1)
np.random.seed(42)
inputs = np.random.rand(num_samples, input_size)

# Generating synthetic weights (conductance values)
weights = np.random.uniform(1e-6, 1e-3, (input_size, output_size))

# Generating expected outputs (ideal outputs without defects)
ideal_outputs = np.dot(inputs, weights)

# Introducing synthetic defects
# Read disturb: Gradual drift in some outputs
read_disturb = np.random.normal(0, 1e-6, ideal_outputs.shape)
# RTN: Random spikes in some outputs
rtn_spikes = np.random.choice([0, 1], size=ideal_outputs.shape, p=[0.995, 0.005]) * np.random.normal(0, 1e-5, ideal_outputs.shape)

# Defective outputs
defective_outputs = ideal_outputs + read_disturb + rtn_spikes

# Dataset
# Input: Concatenate inputs and defective outputs for the NN
nn_inputs = np.concatenate([inputs, defective_outputs], axis=1)

# Labels: Ideal outputs (to train the NN for correction)
labels = ideal_outputs


# Save mock dataset
np.savez("mock_dataset.npz", inputs=nn_inputs, labels=labels)

plt.figure(figsize=(14, 6))
sns.heatmap(defective_outputs[:50,:50]- ideal_outputs[:50,:50], cmap="Blues", cbar=True, xticklabels=False, yticklabels=False)
plt.title("Delta Heatmap (First 20 Columns)")
plt.grid()
plt.show()
