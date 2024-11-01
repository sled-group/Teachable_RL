import torch
from torch import nn
import pytorch_lightning as pl
from transformers import RobertaTokenizer
from torch.utils.data import Dataset ,DataLoader
from LTDT_test.messenger.utils.model import VLDT
from lightning.pytorch.loggers import WandbLogger
import wandb
from torch.nn.functional import pad
from utils.data_process import getDataset
import numpy as np
from utils.observation_process import observationProcessor
from LTDT_test.messenger.utils.evaluate import evaluate_episode_rtg
import argparse
from transformers import RobertaModel, RobertaTokenizer
import torch.nn.functional as F
import os
from train import MultimodalDataset,MultimodalTransformer
import random
def set_seed(seed=42):
    """reproduce"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 如果使用多GPU
    np.random.seed(seed)
    random.seed(seed)
if __name__ == '__main__':
    set_seed(42)
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", type=str, default='messenger')
    # parser.add_argument("--gpu_id", type=str, default='0')
    parser.add_argument("--load_ckpt", type=str, default=None)
    parser.add_argument("--realLang", type=bool, default=True)
    parser.add_argument("--withLanguage", type=bool, default=True)
    parser.add_argument("--LanguageType", type=str, default="rhf") # language
    parser.add_argument("--newTask", type=bool, default=False)
    parser.add_argument("--num_eval_episodes", type=int, default=100)
    parser.add_argument("--log_artifact_interval", type=int, default=15)
    parser.add_argument("--roberta_encoder_len", type=int, default=768)
    parser.add_argument("--embed_dim", type=int, default=256)
    parser.add_argument("--n_layer", type=int, default=5)
    parser.add_argument("--n_head", type=int, default=2)
    parser.add_argument("--activation_function", type=str, default="relu")
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--K", type=int, default=6) 
    parser.add_argument("--max_ep_len", type=int, default=50) 
    parser.add_argument("--state_dim", type=int, default=10)
    parser.add_argument("--act_dim", type=int, default=4)
    parser.add_argument("--env", type=str, default="msgr-train-v3")
    parser.add_argument(
        "--mode", type=str, default="normal"
    )  # normal for standard setting, delayed for sparse
    parser.add_argument("--action_category", type=int, default=5)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()
    variant=vars(args)
    os.environ['CUDA_VISIBLE_DEVICES'] = "0"
    torch.cuda.empty_cache()
    variant["state_dim"]=18
    variant["act_dim"]=1
    variant["scale"]=100
    variant["target_return"]=200
    state_mean=torch.load("data/messenger_state_mean.pth")
    state_std=torch.load("data/messenger_state_std.pth")
    model = MultimodalTransformer(state_mean=state_mean,state_std=state_std,variant=variant).to("cuda")
    filename=variant["load_ckpt"]
    model.load_state_dict(torch.load(filename,map_location='cuda')["state_dict"])
    model.eval()
    goal_rate_list=[]
    eval_data=[]
    split=[]
    for i in range(5):
        result={}
        return_list=[]
        length_list=[]
        subgoal_completes=0
        goal_completes=0
        for seed in range(0,variant["num_eval_episodes"]):
            print("Evaluating seed: ",seed)
            episode_return, episode_length,subgoal_complete,goal_complete,data=evaluate_episode_rtg(
                model=model.decision_transformer,
                language_model=model.language_model,
                max_ep_len=variant["max_ep_len"],
                scale=variant["scale"],
                device='cuda',
                state_mean=state_mean,
                state_std=state_std,
                target_return=variant["target_return"],
                mode='normal',
                newTask=False,
                eval_mode="testing",
                probability_threshold=100,
                seed=seed,
                gpt=False,
                disturb=True
            )
            subgoal_completes+=subgoal_complete
            goal_completes+=goal_complete
            eval_data.append([episode_return,episode_length,subgoal_complete,goal_complete,data])
        result["ave_return"]=sum(return_list)/variant["num_eval_episodes"]
        result["goal_complete_rate"]=goal_completes/variant["num_eval_episodes"]
        goal_rate_list.append(result["goal_complete_rate"])
        print(result)
    print(goal_rate_list)
    print(split)
