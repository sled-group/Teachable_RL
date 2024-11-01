import torch
import numpy as np
from utils.evaluate import evaluate_episode_rtg
import argparse
import torch.nn.functional as F
import os
from train import MultimodalTransformer
import random
def set_seed(seed=42):
    """reproduce"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  
    np.random.seed(seed)
    random.seed(seed)
    
if __name__ == '__main__':
    set_seed(42)
    env_name="metaworld"
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", type=str, default='metaworld')
    # parser.add_argument("--train_tasks", type=list, default=['hammer-v2-goal-observable'])
    parser.add_argument("--train_tasks", type=list, default=['assembly-v2-goal-observable'])
    parser.add_argument("--load_ckpt", type=str, default=None)
    parser.add_argument("--gpu_id", type=str, default='0')
    parser.add_argument("--scale", type=float, default=10) 
    parser.add_argument("--target_return", type=float, default=25) 
    parser.add_argument("--withLanguage", type=bool, default=True)
    parser.add_argument("--eval_interval", type=int, default=5) 
    parser.add_argument("--num_eval_episodes", type=int, default=100) 
    parser.add_argument("--embed_dim", type=int, default=256)
    parser.add_argument("--n_layer", type=int, default=5)
    parser.add_argument("--n_head", type=int, default=2)
    parser.add_argument("--activation_function", type=str, default="relu")
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--K", type=int, default=12) 
    parser.add_argument("--state_dim", type=int, default=10)
    parser.add_argument("--act_dim", type=int, default=2)
    parser.add_argument("--action_category", type=int, default=5)
    args = parser.parse_args()
    variant=vars(args)
    os.environ['CUDA_VISIBLE_DEVICES'] = "1"
    torch.cuda.empty_cache()
    result={}
    state_mean=torch.tensor(np.zeros(10)).cuda()
    state_std=torch.tensor(np.ones(10)).cuda()
    model = MultimodalTransformer(state_mean,state_std,variant=variant).to("cuda")
    filename=variant["load_ckpt"]
    model.load_state_dict(torch.load(filename,map_location='cuda')["state_dict"])
    model.eval()
    total_tasks_success=0
    success_rate=[]
    lengths=[]
    success_list=[]
    print(filename)
    eval_data=[]
    for i in range(5):
        for task_name in variant["train_tasks"]:
            return_list=[]
            total_success=0
            for i in range(variant["num_eval_episodes"]):
                print("Seed: ",i)
                returns,success,length=evaluate_episode_rtg(
                    model=model.decision_transformer,
                    device='cuda',
                    target_return=variant["target_return"],
                    eval_mode="testing",
                    probability_threshold=100,
                    task_name=task_name,
                    language_model=model.language_model,
                    seed=i,
                    isGPT=False
                )
                total_success+=success
                lengths.append(length)
                success_list.append(success)
                eval_data.append([returns,length,success])
                print("Accumulated Success Rate: ",total_success/(i+1))
            total_tasks_success+=total_success
            success_rate.append(total_success/variant["num_eval_episodes"])
    print(success_rate)
