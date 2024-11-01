import torch
from torch import nn
import pytorch_lightning as pl
from torch.utils.data import Dataset ,DataLoader
from model import VLDT
from lightning.pytorch.loggers import WandbLogger
import wandb
from torch.nn.functional import pad
from utils.data_process import getDataset
import numpy as np
from utils.observation_process import observationProcessor
from evaluate import evaluate_episode_rtg
import argparse
import torch.nn.functional as F
import random
from sentence_transformers import SentenceTransformer
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.callbacks import ModelCheckpoint
import os
import torch.distributed as dist
from tqdm import tqdm
from utils.data_process import read_hdf5
class MultimodalTransformer(pl.LightningModule):
    def __init__(self,state_mean,state_std,variant):
        super().__init__()
        self.state_mean=state_mean
        self.state_std=state_std
        self.language_model = SentenceTransformer("sentence-transformers/paraphrase-TinyBERT-L6-v2")
        self.stateProcessor=observationProcessor()
        with torch.no_grad():
                self.encoded_empty_language=torch.tensor(self.language_model.encode("")).unsqueeze(dim=0).to(device='cuda')
        self.variant=variant
        model = VLDT(
            empty_language_embedding=self.encoded_empty_language,
            state_std=state_std,
            state_mean=state_mean,
            state_dim=self.variant["state_dim"],
            act_dim=self.variant["act_dim"],
            hidden_size=self.variant["embed_dim"],
            max_length=self.variant["K"],
            n_layer=self.variant["n_layer"],
            n_head=self.variant["n_head"],
            n_inner=4 * self.variant["embed_dim"],
            activation_function=self.variant["activation_function"],
            n_positions=1024,
            resid_pdrop=self.variant["dropout"],
            attn_pdrop=self.variant["dropout"],
            segment_length=self.variant["K"],
            roberta_encoder_len=self.variant["roberta_encoder_len"],
            category=self.variant["action_category"],
            env_name=self.variant["env_name"])
        self.decision_transformer = model  # Decision Transformer with trajectory input
        
    def create_cnn(self, cnn_channels):
        # This is a simple CNN architecture example. You might want to design your own.
        layers = []
        for in_channels, out_channels in zip(cnn_channels[:-1], cnn_channels[1:]):
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
            layers.append(nn.ReLU())
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        return nn.Sequential(*layers)

    def forward(self, text_input, action,reward,state,language,return_to_go,timesteps,mask):
        # Forward pass through the decision transformer
        decision_output = self.decision_transformer(text_input,state, action, reward, return_to_go,timesteps,language,mask)
        return decision_output


        
    def training_step(self, batch, batch_idx):
        self.train()
        text_input, action,reward,state,language,return_to_go,timesteps,mask= batch['text'], batch["action"],batch["reward"],batch["state"],batch["language"],batch["return_to_go"],batch["time_stamp"],batch["mask"]  
        action_pred,_,_ = self.forward(text_input, action,reward,state,language,return_to_go,timesteps,mask)
        loss_function = nn.CrossEntropyLoss()
        loss = loss_function(action_pred.view(-1, self.variant["action_category"]), action.long().view(-1))
        self.log("training_loss", loss, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def on_train_epoch_end(self):
        self.eval()
        if ((self.current_epoch+1)%self.variant["eval_interval"]==0 and self.current_epoch > 100):
        # if ((self.current_epoch+1)%self.variant["eval_interval"]==0):
            if self.variant["env_name"]=="messenger":
                for probability in [100]:
                    return_list=[]
                    length_list=[]
                    subgoal_completes=0
                    goal_completes=0
                    for seed in tqdm(range(self.variant["num_eval_episodes"])):
                        episode_return, episode_length,subgoal_complete,goal_complete,_=evaluate_episode_rtg(
                            model=self.decision_transformer,
                            language_model=self.language_model,
                            max_ep_len=self.variant["max_ep_len"],
                            scale=self.variant["scale"],
                            state_mean=self.state_mean,
                            state_std=self.state_std,
                            device='cuda',
                            target_return=self.variant["target_return"],
                            mode='normal',
                            newTask=self.variant["newTask"],
                            eval_mode="validation",
                            probability_threshold=probability,
                            seed=seed
                            # seed=seed%5
                        )
                        subgoal_completes+=subgoal_complete
                        goal_completes+=goal_complete
                        return_list.append(episode_return)
                        length_list.append(episode_length)
                    print(f"Epoch {self.current_epoch}: ,Language Probability {probability}, ave return: ",sum(return_list)/self.variant["num_eval_episodes"], " subgoal rate: ",subgoal_completes/self.variant["num_eval_episodes"]," goal rate: ",goal_completes/self.variant["num_eval_episodes"])                
                    self.log(f"Evaluation_ave_return with language probability {probability}", sum(return_list)/self.variant["num_eval_episodes"],sync_dist=True, logger=True)
                    self.log(f"Subgoal rate with language probability {probability}", subgoal_completes/self.variant["num_eval_episodes"],sync_dist=True, logger=True)
                    self.log(f"Success rate with language probability {probability}", goal_completes/self.variant["num_eval_episodes"],sync_dist=True, logger=True)
                    self.log("success_rate", goal_completes/self.variant["num_eval_episodes"], on_epoch=True, prog_bar=True,sync_dist=True, logger=True)
        else:
            self.log("success_rate", 0, on_epoch=True, prog_bar=True,sync_dist=True, logger=True)
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=self.variant["learning_rate"], weight_decay=self.variant["weight_decay"])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=self.variant["T_max"],  # Ensure this is set to the desired epochs, e.g., 15 for reset
            eta_min=self.variant["eta_min"]  # Make sure this is set to 1e-4 for the minimum learning rate
        )

        # PyTorch Lightning scheduler configuration
        scheduler_config = {
            'scheduler': scheduler,
            'interval': 'epoch',  # Scheduler steps every epoch
            'frequency': 1,  # Scheduler is applied every epoch
        }
        return [optimizer],[scheduler_config]
        
class MultimodalDataset(Dataset):
    def __init__(self, data,state_mean,state_std,variant):
        self.trajectories=data
        self.segmentLength=variant["K"]
        self.state_mean=state_mean
        self.state_std=state_std
        self.variant=variant
        self.language_model = SentenceTransformer("sentence-transformers/paraphrase-TinyBERT-L6-v2")
        with torch.no_grad():
                self.encoded_empty_language=torch.tensor(self.language_model.encode("")).unsqueeze(dim=0).unsqueeze(dim=0).to(device='cuda')
    
    def __len__(self):
        return len(self.trajectories)

    def __getitem__(self, idx):
        trajectory=self.trajectories[idx]
        max_length=torch.tensor(trajectory["state"], device='cuda').shape[0]
        starting_point=0 if(max_length<=self.segmentLength) else torch.randint(0, max_length-2, (1,), device='cuda')[0]
        ending_point=starting_point+self.segmentLength if(starting_point+self.segmentLength<max_length-1) else max_length-1
        pad_length = self.segmentLength - (ending_point - starting_point)
        # For text and language, we can use the tokenizer's padding functionality directly
        text = trajectory["encoded_manual"]
        rewards = torch.tensor(trajectory["reward"][starting_point:ending_point], device='cuda').float().unsqueeze(1)
        if pad_length > 0:
            rewards = pad(rewards, (0, 0, pad_length, 0), "constant", 0)
        # Divide the scale
        rewards=rewards/self.variant["scale"]
        actions = torch.tensor(trajectory["action"][starting_point:ending_point], device='cuda').float().unsqueeze(1)
        if pad_length > 0:
            actions = pad(actions, (0, 0, pad_length, 0), "constant", 0)
        return_to_go = torch.tensor(trajectory["return_to_go"][starting_point:ending_point], device='cuda').float().unsqueeze(1)
        if pad_length > 0:
            return_to_go = pad(return_to_go, (0, 0, pad_length, 0), "constant", 0)

        # Divide the scale
        return_to_go=return_to_go/self.variant["scale"]

        mask = torch.zeros(self.segmentLength, device='cuda').bool()
        mask[-max_length:] = True  # We shift the 'True' values to the end part of the mask

        # For states, since it's a numpy array, you'll need to adjust the concatenation:
        states = torch.tensor(trajectory["state"][starting_point:ending_point], dtype=torch.float32).to(device='cuda')
        padding = torch.zeros(pad_length, states.shape[1], device='cuda',dtype=torch.float32)
        states = torch.cat((padding, states), dim=0)
        self.state_mean=self.state_mean.to(device=states.device)
        self.state_std=self.state_std.to(device=states.device)
        states = (states - self.state_mean) / self.state_std
        
        # For language and text, adjust the concatenation logic to add padding at the front:
        if("h" in self.variant["LanguageType"] or "f" in self.variant["LanguageType"] or "r" in self.variant["LanguageType"]):
            if("r" in self.variant["LanguageType"]):
                if("h" in self.variant["LanguageType"] and "f" in self.variant["LanguageType"]):  
                    key="rhf_embedding"
                    language = trajectory[key][starting_point:ending_point].to("cuda")
                elif("h" in self.variant["LanguageType"]):
                    language = trajectory["rh_embedding"][starting_point:ending_point].to("cuda")
                elif("f" in self.variant["LanguageType"]):
                    language = trajectory["rf_embedding"][starting_point:ending_point].to("cuda")
            elif("h" in self.variant["LanguageType"] and "f" in self.variant["LanguageType"]):
                language = trajectory["hf_embedding"][starting_point:ending_point].to("cuda")
            elif("f" in self.variant["LanguageType"]):
                language=trajectory["f_embedding"][starting_point:ending_point].to("cuda")
            elif("h" in self.variant["LanguageType"]):
                language=trajectory["h_embedding"][starting_point:ending_point].to("cuda")
            if pad_length>0:
                padding=self.encoded_empty_language.repeat(pad_length,1,1).to(device=language.device)
                language = torch.cat((padding, language), dim=0).to(device='cuda')
        else:
            with torch.no_grad():
                language=self.encoded_empty_language.repeat(self.segmentLength,1,1).to('cuda')
        time_stamps = torch.arange(starting_point, ending_point, device='cuda').long()
        time_stamps = pad(time_stamps, (pad_length, 0), "constant", 0)
        language=language.squeeze(1)
        return {
            'text': text.to("cuda").unsqueeze(1),
            'state': states,
            'reward': rewards,
            'action': actions,
            'return_to_go': return_to_go,
            'language': language.to("cuda"),
            'time_stamp': time_stamps,
            'mask': mask.to("cuda")
        }
def experiment(variant):
    import os
    torch.multiprocessing.set_start_method('spawn')
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    # os.environ['CUDA_VISIBLE_DEVICES'] = variant["gpu_id"]
    torch.cuda.empty_cache()
    if(variant["log"]):
        wandb.login()
        wandb_logger = WandbLogger(project="Messenger-MetaWorld-Environment", name=variant["project_name"],log_model=True)
    else:
        wandb_logger=None
    if(variant["env_name"]=="messenger"):
        variant["state_dim"]=18
        variant["act_dim"]=1
        variant["scale"]=100
        variant["target_return"]=200
    
    # Load data
    data=read_hdf5(variant["data_path"])[0:variant["data_size"]]
    state_mean=torch.load("data/messenger_state_mean.pth")
    state_std=torch.load("data/messenger_state_std.pth")
    model = MultimodalTransformer(state_mean=state_mean,state_std=state_std,variant=variant).to("cuda")
    
    if variant["load_ckpt"] is not None:
        model.load_state_dict(torch.load(variant["load_ckpt"],map_location='cuda')["state_dict"])
    dataset = MultimodalDataset(data,state_mean,state_std,variant)
    train_loader = DataLoader(dataset, batch_size=variant.get("batch_size"), shuffle=True)
    log_dir=variant.get("exp_name")
    checkpoint_callback = ModelCheckpoint(
        dirpath=f"checkpoint/messenger/{log_dir}",  # Change this to your desired path
        filename="model-{epoch:02d}-{success_rate:.2f}",  # Customize filename format
        save_top_k=3,  # Save only the best model
        monitor="success_rate",  # Metric to monitor for checkpointing
        mode="max"  # Save checkpoints when `val_loss` decreases
    )
    trainer = pl.Trainer(
        max_epochs=variant.get("max_epoch"), 
        accelerator="gpu", 
        devices='auto',
        logger=False,
        log_every_n_steps=variant.get("log_every_n_steps"),
        strategy=DDPStrategy(find_unused_parameters=True),
        callbacks=[checkpoint_callback]  # Use the custom checkpoint callback
    )
    # Train the model
    trainer.fit(model, train_loader)
    
    wandb.finish()

def set_seed(seed=42):
    """reproduce"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 如果使用多GPU
    np.random.seed(seed)
    random.seed(seed)


if __name__ == '__main__':
    set_seed(42)
    env_name="messenger"############################################################################ Must check
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", type=str, default='messenger')
    parser.add_argument("--newTask", type=int, default=0)
    parser.add_argument("--load_ckpt", type=str, default=None)
    parser.add_argument("--LanguageType", type=str, default="f") # language
    parser.add_argument("--exp_name", type=str, default='msgr_f_h1') 
    newTask=parser.parse_args().newTask
    parser.add_argument("--data_size", type=int, default=5) 
    import pdb;pdb.set_trace()
    if newTask:
        parser.add_argument("--data_path", type=str, default='data/messenger_adaptation_dataset.h5') # folder
        parser.add_argument("--learning_rate", "-lr", type=float, default=1e-4)
        parser.add_argument("--max_epoch", type=int, default=100) 
        parser.add_argument("--eval_interval", type=int, default=1) 
        parser.add_argument("--batch_size", type=int, default=5) 
    else:
        parser.add_argument("--data_path", type=str, default='data/messenger_dataset.h5') # folder
        parser.add_argument("--learning_rate", "-lr", type=float, default=1e-3)
        parser.add_argument("--max_epoch", type=int, default=500) 
        parser.add_argument("--eval_interval", type=int, default=5) 
        parser.add_argument("--batch_size", type=int, default=256) 
    # parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--log", type=bool, default=False)
    parser.add_argument("--T_max", type=float, default=500) 
    parser.add_argument("--eta_min", type=float, default=1e-5)
    parser.add_argument("--num_eval_episodes", type=int, default=50) 
    parser.add_argument("--log_every_n_steps", type=int, default=25) 
    parser.add_argument("--roberta_encoder_len", type=int, default=768)
    parser.add_argument("--embed_dim", type=int, default=256)
    parser.add_argument("--n_layer", type=int, default=5)
    parser.add_argument("--n_head", type=int, default=2)
    parser.add_argument("--activation_function", type=str, default="relu")
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--K", type=int, default=6) # max length of trajectory sent to transformer, can change.
    parser.add_argument("--max_ep_len", type=int, default=50)
    parser.add_argument("--state_dim", type=int, default=10)
    parser.add_argument("--act_dim", type=int, default=4)
    parser.add_argument("--weight_decay", "-wd", type=float, default=1e-5)
    parser.add_argument("--log_to_wandb", "-w", type=bool, default=True)
    parser.add_argument("--env", type=str, default="msgr-train-v3")
    parser.add_argument(
        "--mode", type=str, default="normal"
    )  # normal for standard setting, delayed for sparse
    parser.add_argument("--action_category", type=int, default=5)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()
    import pdb;pdb.set_trace()
    experiment(variant=vars(args))
    
    