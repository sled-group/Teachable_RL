import torch
import os
import numpy as np
import torch
import torch
import numpy as np
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer

import h5py
import torch

def save_dataset(dataset, path):
    with h5py.File(path, 'w') as hdf:
        for i, data_dict in enumerate(dataset):
            group = hdf.create_group(f'data_{i}')
            for key, value in data_dict.items():
                if isinstance(value, torch.Tensor):
                    value = value.numpy()
                elif isinstance(value, list):
                    value = np.array(value)
                group.create_dataset(key, data=value)
                
def read_hdf5(file_path):
    data_list = []
    with h5py.File(file_path, 'r') as hdf:
        for group_name in hdf:
            group = hdf[group_name]
            data_dict = {}
            for key, dataset in group.items():
                if key.endswith('_embedding'):  
                    data_dict[key] = torch.tensor(dataset[...])
                else:
                    data_dict[key] = dataset[...]
            data_list.append(data_dict)
    return data_list

def getDataset(data_size,trajectory_dir):
    encoded_empty_language=torch.tensor(SentenceTransformer("sentence-transformers/paraphrase-TinyBERT-L6-v2").encode("")).unsqueeze(dim=0).to(device='cuda')
    states, traj_lens, returns,languages = [], [], [],[]
    dataset=[]
    print(trajectory_dir)
    for i in range(data_size):
            print(i)
            file=f"{trajectory_dir}/trajectory_{i+1}.pth"
            if os.path.exists(file):
                data=torch.load(file)
                states.append(data["state"])
                traj_lens.append(len(data["state"]))
                returns.append(data["return_to_go"][0])
                # languages.append(data["languages"])
                dataset.append(data)

    print("=" * 50)
    print(f"Starting new experiment")
    print(f"Average return: {np.mean(returns):.2f}, std: {np.std(returns):.2f}")
    print(f"Max return: {np.max(returns):.2f}, min: {np.min(returns):.2f}")
    print("=" * 50)
    print("dataset trajectories: ",len(dataset))
    import pdb;pdb.set_trace()
    return dataset[0:data_size]


    