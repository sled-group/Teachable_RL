import torch
import os
import numpy as np
from transformers import RobertaModel, RobertaTokenizer
import torch
import math
import argparse
import h5py


def save_dataset(dataset, path):
    try:
        with h5py.File(path, 'w') as hdf:
            for i, data_dict in enumerate(dataset):
                group = hdf.create_group(f'data_{i}')
                for key, value in data_dict.items():
                    if isinstance(value, torch.Tensor):
                        value = value.numpy()
                    elif isinstance(value, list):
                        value = np.array(value)
                    group.create_dataset(key, data=value)
    except:
        import pdb;pdb.set_trace()
                
def read_hdf5(file_path):
    data_list = []
    with h5py.File(file_path, 'r') as hdf:
        for group_name in hdf:
            group = hdf[group_name]
            data_dict = {}
            import pdb;pdb.set_trace()
            for key, dataset in group.items():
                if key.endswith('_embedding'):  
                    data_dict[key] = torch.tensor(dataset[...])
                else:
                    data_dict[key] = dataset[...]
            data_list.append(data_dict)
    
    return data_list

def process_states(states):
    for i in range(len(states)):
        # print(states[i])
        if len(states[i])!=10:
            states[i]=states[i] + [0.0] * (10 - len(states[i]))
        # print(states[i])

def process_all_trajectories(taskName,dir,start,end):
    trajectory_dir = dir+'/trajectory_'
    processed_dir = dir+'/trajectory_'
    if not os.path.exists(dir):
        os.makedirs(dir)
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    # Load pre-trained model (weights)
    model = RobertaModel.from_pretrained('roberta-base').to('cuda')
    model.eval()
    # Predict hidden states features for each layer
    for i in range(start,end):
        file=trajectory_dir+str(i)+".pth"
        if os.path.exists(file):
            trajectory=torch.load(file)
            if(taskName=="metaworld"):
                process_states(trajectory["state"])
            rewards = trajectory["reward"]
            returns = []
            if (len(trajectory["languages"])==0):
                trajectory["languages"]=[""]*len(rewards)
            returns = []
            gamma=1
            cumulative_return = 0
            for reward in reversed(rewards):
                cumulative_return = reward + gamma * cumulative_return
                returns.append(cumulative_return)
            if("encoded_manual" in trajectory):
                print(i," already processed")
                continue
            # Since we calculated the returns in reverse (to efficiently use the previous computation),
            # we need to reverse them back to match the original order of rewards
            returns.reverse()
            trajectory["return_to_go"] = returns
            hindsight_future_language_embedding=[]
            future_language_embedding=[]
            for text in trajectory["languages"]:
                if(text=={}):
                    text={"hindsight positive":"", "hindsight negative":"", "foresight positive":"","foresight negative":""}
                hindsight_future_language="hindsight positive: "+text["hindsight positive"]+"; hindsight negative: "+text["hindsight negative"]+"; foresight positive: "+text["foresight positive"]+"; foresight negative: "+text["foresight negative"]
                future_language="foresight positive: "+text["foresight positive"]+"; foresight negative: "+text["foresight negative"]
                with torch.no_grad():  # No gradient is needed (inference mode)
                    encoded_input = tokenizer(hindsight_future_language, return_tensors='pt').to('cuda')  # "pt" for PyTorch tensors
                    outputs = model(**encoded_input)
                    last_hidden_state = outputs.last_hidden_state
                    cls_embedding = last_hidden_state[:, 0, :]
                    hindsight_future_language_embedding.append(cls_embedding)

                    encoded_input = tokenizer(future_language, return_tensors='pt').to('cuda')  # "pt" for PyTorch tensors
                    outputs = model(**encoded_input)
                    last_hidden_state = outputs.last_hidden_state
                    cls_embedding = last_hidden_state[:, 0, :]
                    future_language_embedding.append(cls_embedding)
                    
            hindsight_future_language_embedding = torch.stack(hindsight_future_language_embedding)
            future_language_embedding = torch.stack(future_language_embedding)
            assert future_language_embedding.shape == (len(trajectory["languages"]), 1, 768)
            encoded_input = tokenizer(trajectory["manual"], 
                                    return_tensors='pt',
                                    padding=True,  # Adds padding
                                    truncation=True,  # Truncates to max model length
                                    max_length=model.config.max_position_embeddings).to('cuda')  # "pt" for PyTorch tensors
            # Forward pass, get hidden states output
            outputs = model(**encoded_input)
            # The last_hidden_state is the last layer hidden states, which are the contextualized word embeddings
            last_hidden_state = outputs.last_hidden_state
            # To get the embeddings for the `[CLS]` token, which can be used as a sentence representation
            cls_embedding = last_hidden_state[:, 0, :]
            print(i,"manual: ",cls_embedding.shape)
            trajectory["future_embedding"]=future_language_embedding
            trajectory["hindsight_future_embedding"]=hindsight_future_language_embedding
            trajectory["encoded_manual"]=cls_embedding
            torch.save(trajectory,processed_dir+str(i)+".pth")


def getDataset(data_size,trajectory_dir):
    states, traj_lens, returns,languages = [], [], [],[]
    all_states=[]
    dataset=[]
    for i in range(data_size):
        file=f"{trajectory_dir}/trajectory_{i+1}.pth"
        if os.path.exists(file):
            data=torch.load(file)
            del data["manual"]
            del data["languages"]
            del data["f_language"]
            del data["h_language"]
            del data["hf_language"]
            del data["rhf_language"]
            del data["nonExpertTime"]
            import pdb;pdb.set_trace()
            dataset.append(torch.load(file))
    for path in dataset:
            process_states(path["state"])
            states.append(path["state"])
            for state in path["state"]:
                all_states.append(state)
            traj_lens.append(len(path["state"]))
            returns.append(path["return_to_go"][0])
            languages.append(path["languages"])
    traj_lens, returns = np.array(traj_lens), np.array(returns)
    num_timesteps = sum(traj_lens)
    state_mean=np.zeros(10)
    state_std=np.ones(10)
    print("=" * 50)
    print(f"Starting new experiment")
    print(f"{len(traj_lens)} trajectories, {num_timesteps} timesteps found")
    # print(f"Average State: {state_mean}, std: {state_std}")
    print(f"Average return: {np.mean(returns):.2f}, std: {np.std(returns):.2f}")
    print(f"Max return: {np.max(returns):.2f}, min: {np.min(returns):.2f}")
    print("=" * 50)
    print("dataset trajectories: ",len(dataset))
    import pdb;pdb.set_trace()
    return dataset,state_mean,state_std

if __name__ == "__main__":
   getDataset(20,"metaworld_hammer_dataset/hammer-v2-goal-observable")
    