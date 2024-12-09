# Re-importing necessary libraries and re-loading datasets after environment reset
import pandas as pd
import matplotlib.pyplot as plt

# File paths
clean_dataset_path = "crossbar_clean_dynamic.csv"
defective_dataset_path = "crossbar_defective_dynamic.csv"

# Load the clean and defective datasets
clean_data = pd.read_csv(clean_dataset_path)
defective_data = pd.read_csv(defective_dataset_path)

# Select a subset of columns for visualization
columns_to_plot = clean_data.columns[:20]

# Create a scatter plot
plt.figure(figsize=(12, 8))
for column in columns_to_plot:
    plt.scatter(
        clean_data.index,
        clean_data[column],
        alpha=0.6,
        c='red',
        s=10,
    )
    plt.scatter(
        defective_data.index,
        defective_data[column],
        alpha=0.6,
        c='blue',
        s=10,
    )

plt.title("Scatter Plot of Clean vs Defective Currents")
plt.xlabel("Sample Index")
plt.ylabel("Current (A)")
plt.grid()
plt.tight_layout()
plt.show()
