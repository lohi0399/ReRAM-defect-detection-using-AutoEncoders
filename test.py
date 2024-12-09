import torch
import numpy as np

defective_data_path = "crossbar_defective_dynamic.csv" 
defective_data = torch.from_numpy(np.loadtxt(defective_data_path, dtype=np.float32, delimiter=",", skiprows=1))
defective_dataloader = torch.utils.data.DataLoader(dataset=defective_data, batch_size=16,shuffle=False, drop_last=True)
