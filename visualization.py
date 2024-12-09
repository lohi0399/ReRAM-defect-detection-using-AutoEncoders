import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Visualize a subset of rows for both clean and defective datasets
sample_rows = 50  # Number of rows to visualize


clean_dataset =  pd.read_csv('clean_memristor_dataset_regenerated.csv')
defective_dataset = pd.read_csv('defective_memristor_dataset_regenerated.csv')


# Clean dataset visualization
plt.figure(figsize=(10, 6))
plt.plot(clean_dataset.iloc[:sample_rows, :-1].T, alpha=0.7)
plt.title("Clean Dataset: Column Currents")
plt.xlabel("Columns")
plt.ylabel("Current (A)")
plt.show()

# Defective dataset visualization
plt.figure(figsize=(10, 6))
plt.plot(defective_dataset.iloc[:sample_rows, :-1].T, alpha=0.7)
plt.title("Defective Dataset: Column Currents")
plt.xlabel("Columns")
plt.ylabel("Current (A)")
plt.show()

# Comparison of Output Currents
plt.figure(figsize=(10, 6))
plt.plot(clean_dataset['Output_Current'][:sample_rows], label="Clean Data", marker='o')
plt.plot(defective_dataset['Output_Current'][:sample_rows], label="Defective Data", marker='x')
plt.title("Output Current Comparison")
plt.xlabel("Sample Index")
plt.ylabel("Output Current (A)")
plt.legend()
plt.show()
