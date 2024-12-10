"""
Artificial dataset created in to simulate random defects in memristor crossbars for on-chip training

"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Parameters
num_samples = 1000  # Number of time steps (inference operations)
num_rows = 784 # Number of neurons in the input layer
num_columns = 392  # Number of neurons in the output layer 
learning_rate = 0.01  # Simulated weight update magnitude
defect_frequency = 0.2  # 20% of the total runtime
num_defective_time_steps = int(defect_frequency * num_samples)
columns_to_plot = [1, 2, 3, 4, 5,10,24,67,89,100,250,349]  # Columns to visualize
time_steps = np.arange(num_samples)

# Initialize weights (conductances)
conductance_matrix = np.random.uniform(1e-6, 1e-3, size=(num_rows,num_columns))  # HRS to LRS range
clean_currents_list = []
defective_currents_list = []

for i in range(num_samples):

    weight_updates = learning_rate * np.random.uniform(-1, 1, size=conductance_matrix.shape)
    conductance_matrix += weight_updates
    conductance_matrix = np.clip(conductance_matrix, 1e-6, 1e-3)  # Clip to valid range (HRS to LRS)

    # Generate clean currents for this iteration
    clean_current_matrix = conductance_matrix * 0.25  # Read voltage = 0.25V
    clean_currents = clean_current_matrix.sum(axis=0) # KCL --> Taking the sum of all the column currents
    clean_currents_list.append(clean_currents)

    # Generate defective current for this time step
    defective_conductance_matrix = conductance_matrix.copy()

    if np.random.rand() < defect_frequency:  # Introduce defects for 20% of the time

        defect_indices = np.random.choice(num_rows * num_columns, size=int(0.2 * num_rows * num_columns), replace=False)
        row_indices, col_indices = np.unravel_index(defect_indices, (num_rows, num_columns))

        # Apply defects to selected indices
        for row, col in zip(row_indices, col_indices):
            defective_conductance_matrix[row, col] = np.random.uniform(1e-6, 1e-3)  # Random conductance within valid range
    
    defective_current_matrix = defective_conductance_matrix * 0.25 # Read voltage = 0.25V
    defective_currents =  defective_current_matrix.sum(axis=0) # KCL --> Taking the sum of all the column currents
    defective_currents_list.append(defective_currents)

# Convert to arrays
clean_currents_dynamic = np.array(clean_currents_list)
defective_currents_dynamic = np.array(defective_currents_list)

# Save the dynamically generated datasets
clean_data_dynamic = pd.DataFrame(clean_currents_dynamic, columns=[f"Column_{i}" for i in range(1, num_columns + 1)])
defective_data_dynamic = pd.DataFrame(defective_currents_dynamic, columns=[f"Column_{i}" for i in range(1, num_columns + 1)])

clean_dataset_dynamic_path = "crossbar_clean_dynamic.csv"
defective_dataset_dynamic_path = "crossbar_defective_dynamic.csv"

clean_data_dynamic.to_csv(clean_dataset_dynamic_path, index=False)
defective_data_dynamic.to_csv(defective_dataset_dynamic_path, index=False)

# Visualize a subset of the columns
plt.figure(figsize=(14, 6))

for i, column in enumerate(columns_to_plot):
    plt.plot(
        time_steps,
        clean_currents_dynamic[:, column - 1] * 1e6 + i * 10000,  # Offset for spacing, convert to µA
        label=f"Clean Column {column}",
        color="blue",
        alpha=0.7,
    )
    plt.plot(
        time_steps,
        defective_currents_dynamic[:, column - 1] * 1e6 + i * 10000,  # Offset for spacing, convert to µA
        label=f"Defective Column {column}",
        color="red",
        alpha=0.7,
    )

plt.title("Clean vs Defective Currents with Dynamic Weight Updates (µA)")
plt.xlabel("Time Steps")
plt.ylabel("Output Current (µA, Offset for Spacing)")
plt.legend(loc="upper left", fontsize="small")
plt.grid()
plt.tight_layout()
plt.show()

# Print paths to the saved datasets
print("Clean dataset saved to:", clean_dataset_dynamic_path)
print("Defective dataset saved to:", defective_dataset_dynamic_path)