import torch
from torch import nn
import pytorch_lightning as pl
from torch.utils.data import Dataset ,DataLoader
from utils.model import VLDT
from lightning.pytorch.loggers import WandbLogger
import wandb
from torch.nn.functional import pad
from utils.data_process import getDataset
import numpy as np
from utils.evaluate import evaluate_episode_rtg
import argparse
import torch.nn.functional as F
import random
import os
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.callbacks import ModelCheckpoint
num_pos_classes,num_gripper_classes=7,3
class MultimodalTransformer(pl.LightningModule):
    def __init__(self,state_mean,state_std,variant):
        super().__init__()
        self.state_mean=state_mean
        self.state_std=state_std
        self.language_model = SentenceTransformer("sentence-transformers/paraphrase-TinyBERT-L6-v2")
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
        decision_output = self.decision_transformer(text_input,state, action,return_to_go,timesteps,language,mask)
        return decision_output

    def training_step(self, batch, batch_idx):
        self.train()
        text_input, action,reward,state,language,return_to_go,timesteps,mask= batch['text'], batch["action"],batch["reward"],batch["state"],batch["language"],batch["return_to_go"],batch["time_stamp"],batch["mask"]  
        pred_pos,pred_gripper = self.forward(text_input, action,reward,state,language,return_to_go,timesteps,mask)
        action=action.to(torch.int64)
        labels_pos = action[:,:,0].squeeze() 
        labels_gripper = action[:,:,1].squeeze() 
        criterion = nn.CrossEntropyLoss()
        loss_pos = criterion(pred_pos.view(-1, num_pos_classes), labels_pos.view(-1))
        loss_gripper = criterion(pred_gripper.view(-1, num_gripper_classes), labels_gripper.view(-1))
        loss = (loss_pos + loss_gripper)/2
        self.log("training_loss", loss, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def on_train_epoch_end(self):
        self.eval()
            
        if (self.current_epoch+1)%self.variant["eval_interval"]==0 and self.current_epoch >=0:
            for probability in [100]:
                list_successes=[]
                list_returns=[]
                for task in self.variant["train_tasks"]:
                    returns_list=[]
                    success_list=[]
                    for i in tqdm(range(self.variant["num_eval_episodes"])):
                        returns,success,length=evaluate_episode_rtg(
                            model=self.decision_transformer,
                            device='cuda',
                            target_return=self.variant["target_return"],
                            eval_mode="validation",
                            probability_threshold=probability,
                            task_name=task,
                            language_model=self.language_model,
                            seed=i,
                        )
                        returns_list.append(returns)
                        success_list.append(success)
                    list_successes+=success_list
                    list_returns+=returns_list
                    self.log(f"Success rate for {task} with language probability {probability}", np.array(success_list).mean(), logger=True)
                    print(f"Success rate for {task} with language probability {probability}", np.array(success_list).mean())
                    print(f"Average return for {task} with language probability {probability}", np.array(returns_list).mean())
                self.log(f"Success rate for all tasks with language probability {probability}", np.array(list_successes).mean(), logger=True)
                self.log("success_rate", np.array(list_successes).mean() , on_epoch=True, prog_bar=True,sync_dist=True, logger=True)
        else:
            self.log("success_rate", 0, on_epoch=True, prog_bar=True,sync_dist=True, logger=True)
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.variant["learning_rate"], weight_decay=self.variant["weight_decay"])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=self.variant["T_max"],  
            eta_min=self.variant["eta_min"] 
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
        ending_point=torch.randint(1,max_length+1,(1,),device='cuda')[0]
        starting_point=0 if ending_point<=self.segmentLength else ending_point-self.segmentLength
        pad_length = self.segmentLength - (ending_point - starting_point)
        text = trajectory["encoded_manual"]
        rewards = torch.tensor(trajectory["reward"][starting_point:ending_point], device='cuda').float().unsqueeze(1)
        if pad_length > 0:
            rewards = pad(rewards, (0, 0, pad_length, 0), "constant", 0)
        rewards=rewards/self.variant["scale"]
        actions = torch.tensor(np.array(trajectory["action"][starting_point:ending_point]), device='cuda').float()
        if pad_length > 0:
            actions = pad(actions, (0, 0, pad_length, 0), "constant", 0)
        return_to_go = torch.tensor(trajectory["return_to_go"][starting_point:ending_point], device='cuda').float().unsqueeze(1)
        if pad_length > 0:
            return_to_go = pad(return_to_go, (0, 0, pad_length, 0), "constant", 0)
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
        return {
            'text': text.to("cuda").unsqueeze(1),
            'state': states.float(),
            'reward': rewards.float(),
            'action': actions.float(),
            'return_to_go': return_to_go.float(),
            'language': language.to("cuda"),
            'time_stamp': time_stamps,
            'mask': mask.to("cuda")
        }
def experiment(variant):
    import os
    torch.multiprocessing.set_start_method('spawn')
    os.environ["CUDA_VISIBLE_DEVICES"] = str(variant.get("gpu_id"))
    # os.environ['CUDA_VISIBLE_DEVICES'] = "0"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    torch.cuda.empty_cache()
    if(variant["log"]):
        wandb.login()
        wandb_logger = WandbLogger(project="Messenger-MetaWorld-Environment", name=variant["project_name"],log_model=True)
    else:
        wandb_logger=None
    
     # Load data
    data,state_mean,state_std=getDataset(variant["data_size"],trajectory_dir = variant["trajectory_dir"],env=variant["env_name"],train_tasks=variant["train_tasks"])
    state_mean=torch.tensor(state_mean).cuda()
    state_std=torch.tensor(state_std).cuda()
    # Create dataset
    dataset = MultimodalDataset(data,state_mean,state_std,variant)

    # Create a DataLoader
    train_loader = DataLoader(dataset, batch_size=variant.get("batch_size"), shuffle=True)

    model = MultimodalTransformer(state_mean=state_mean,state_std=state_std,variant=variant).to("cuda")
    # model.load_state_dict(torch.load("/data/heyinong/checkpoint/metaworld/metaworld_tool_rhf_assembly/model_89.pth"))
    # model.load_state_dict(torch.load("/data/heyinong/checkpoint/metaworld/metaworld_tool_rf_assembly/model_44.pth"))
    # model.load_state_dict(torch.load("checkpoint/metaworld/h1/meta_rhf_h1/model-epoch=69-success_rate=0.66.ckpt",map_location='cuda')['state_dict'])
    # model.load_state_dict(torch.load("checkpoint/metaworld/h1/meta_no_h1/model-epoch=49-success_rate=0.49.ckpt",map_location='cuda')['state_dict'])
    model.load_state_dict(torch.load("checkpoint/metaworld/h2/meta_rhf_pretrain/model-epoch=29-success_rate=0.51.ckpt",map_location='cuda')['state_dict'])
    # model.load_state_dict(torch.load("checkpoint/metaworld/h1/meta_rf_h1/model-epoch=29-success_rate=0.55.ckpt",map_location='cuda')['state_dict'])
    # model.load_state_dict(torch.load("checkpoint/metaworld/h1/meta_rh_h1/model-epoch=69-success_rate=0.51.ckpt",map_location='cuda')['state_dict'])
    log_dir=variant.get("exp_name")
    
    checkpoint_callback = ModelCheckpoint(
        dirpath=f"checkpoint/metaworld/{log_dir}",  # Change this to your desired path
        filename="model-{epoch:02d}-{success_rate:.2f}", 
        save_top_k=3,
        monitor="success_rate",  # Metric to monitor for checkpointing
        mode="max" 
    )
    trainer = pl.Trainer(
        max_epochs=variant.get("max_epoch"), 
        accelerator="gpu", 
        devices='auto',
        # devices=[variant.get("gpu_id")], 
        logger=False,
        log_every_n_steps=variant.get("log_every_n_steps"),
        strategy=DDPStrategy(find_unused_parameters=True),
        callbacks=[checkpoint_callback] 
    )
    # Train the model
    trainer.fit(model, train_loader)
    
    wandb.finish()

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
    newTask=True
    parser.add_argument("--exp_name", type=str, default='meta_rhf_h2_5')
    if newTask:
        parser.add_argument("--LanguageType", type=str, default="rhf") # language type
        parser.add_argument("--train_tasks", type=list, default=['hammer-v2-goal-observable'])
        parser.add_argument("--trajectory_dir", type=str, default='metaworld_hammer_dataset') 
        parser.add_argument("--learning_rate", "-lr", type=float, default=1e-5) # lr 1e-5 for rhf,rf,rh ; lr 1e-6 for no lang (no lang easy to overfit and perform badly at lr=1e-5, so give it a smaller learning rate)
        
        parser.add_argument("--data_size", type=int, default=5) 
        parser.add_argument("--batch_size", type=int, default=5) 
        parser.add_argument("--max_epoch", type=int, default=500) 
        parser.add_argument("--eval_interval", type=int, default=1) 
        parser.add_argument("--num_eval_episodes", type=int, default=100)
    else:
        parser.add_argument("--LanguageType", type=str, default="rhf") # language type
        parser.add_argument("--train_tasks", type=list, default=['assembly-v2-goal-observable'])
        parser.add_argument("--trajectory_dir", type=str, default='metaworld_assembly_dataset') 
        parser.add_argument("--learning_rate", "-lr", type=float, default=1e-5)
        parser.add_argument("--data_size", type=int, default=10000) 
        parser.add_argument("--batch_size", type=int, default=128) 
        parser.add_argument("--max_epoch", type=int, default=200 ) 
        parser.add_argument("--eval_interval", type=int, default=1)  # or 2, optionally
        parser.add_argument("--num_eval_episodes", type=int, default=100)
    parser.add_argument("--gpu_id", type=int, default=1)
    parser.add_argument("--env_name", type=str, default='metaworld')
    parser.add_argument("--log", type=bool, default=False)
    parser.add_argument("--eta_min", type=float, default=1e-6)
    parser.add_argument("--T_max", type=float, default=500) 
    parser.add_argument("--scale", type=float, default=10) 
    parser.add_argument("--target_return", type=float, default=20) 
    parser.add_argument("--log_every_n_steps", type=int, default=25) 
    parser.add_argument("--embed_dim", type=int, default=256)
    parser.add_argument("--n_layer", type=int, default=5)
    parser.add_argument("--n_head", type=int, default=2)
    parser.add_argument("--activation_function", type=str, default="relu")
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--K", type=int, default=12)
    parser.add_argument("--max_ep_len", type=int, default=50)
    parser.add_argument("--state_dim", type=int, default=10)
    parser.add_argument("--act_dim", type=int, default=2)
    parser.add_argument("--weight_decay", "-wd", type=float, default=1e-5)
    parser.add_argument("--log_to_wandb", "-w", type=bool, default=True)
    parser.add_argument(
        "--mode", type=str, default="normal"
    )  # normal for standard setting, delayed for sparse
    parser.add_argument("--action_category", type=int, default=5)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()
    experiment(variant=vars(args))
    
    
