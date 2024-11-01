import numpy as np
import torch
import torch.nn as nn
from transformers import RobertaTokenizer
import transformers
from decision_transformer.models.model import TrajectoryModel
from decision_transformer.models.trajectory_gpt2 import GPT2Model
from utils.data_process import getDataset
from utils.inferenceDataManager import inferenceDataManager
class VLDT(TrajectoryModel):
    def __init__(
            self,
            state_dim,
            act_dim,
            state_mean,
            state_std,
            empty_language_embedding,
            hidden_size=768,
            num_embeddings=52096,
            max_length=None,
            max_ep_len=4096,
            action_tanh=True,
            category=5,
            roberta_encoder_len=768,
            env_name="messenger",
            **kwargs
    ):
        super().__init__(state_dim, act_dim, max_length=max_length)
        self.hidden_size = hidden_size
        config = transformers.GPT2Config(
            vocab_size=1,  # doesn't matter -- we don't use the vocab
            n_embd=hidden_size,
            **kwargs
        )
        self.act_category=category
        self.transformer = GPT2Model(config)
        self.inferenceDataManager=inferenceDataManager(state_mean,state_std,empty_language_embedding,max_length)
        self.embed_timestep = nn.Embedding(max_ep_len, hidden_size)
        self.embed_lang1 = torch.nn.Linear(roberta_encoder_len, hidden_size)
        self.embed_lang2 = torch.nn.Linear(hidden_size, hidden_size)
        self.embed_return = torch.nn.Linear(1, hidden_size)
        self.embed_state = torch.nn.Linear(self.state_dim, hidden_size)
        # self.embed_state2 = torch.nn.Linear(hidden_size, hidden_size)
        self.embed_action = torch.nn.Linear(self.act_dim, hidden_size)
        self.task_description = torch.nn.Linear(roberta_encoder_len, hidden_size)
        self.embed_ln = nn.LayerNorm(hidden_size)
        self.predict_reward=torch.nn.Linear(hidden_size,1)
        self.predict_rtg=torch.nn.Linear(hidden_size,1)
        self.predict_action_pos = nn.Sequential(
                nn.Linear(hidden_size, 7),  # Maps from hidden_size to the number of actions
            )
        self.predict_action_gripper = nn.Sequential(
                nn.Linear(hidden_size, 3),  # Maps from hidden_size to the number of actions
            )
    
    def forward(self,text_input,states, actions, returns_to_go,timesteps,languages,attention_mask=None):
        batch_size, seq_length = states.shape[0], states.shape[1]
        if attention_mask is None:
            # attention mask for GPT: 1 if can be attended to, 0 if not
            attention_mask = torch.ones((batch_size, seq_length), dtype=torch.long)
        state_embeddings = self.embed_state(states)
        action_embeddings = self.embed_action(actions)
        returns_embeddings = self.embed_return(returns_to_go)
        time_embeddings = self.embed_timestep(timesteps)
        language_embeddings=self.embed_lang2(torch.relu(self.embed_lang1(languages).mean(dim=2)))
        # # time embeddings are treated similar to positional embeddings
        state_embeddings = state_embeddings + time_embeddings
        action_embeddings = action_embeddings + time_embeddings
        returns_embeddings = returns_embeddings + time_embeddings
        language_embeddings=language_embeddings+time_embeddings
        task_desc_embeddings=self.task_description(text_input).mean(dim=2)
        stacked_inputs = torch.stack(
            (returns_embeddings, state_embeddings,language_embeddings, action_embeddings), dim=1
        ).permute(0, 2, 1, 3).reshape(batch_size, 4*seq_length, self.hidden_size)
        stacked_inputs=torch.cat((task_desc_embeddings,stacked_inputs),dim=1)
        stacked_inputs = self.embed_ln(stacked_inputs)
        # to make the attention mask fit the stacked inputs, have to stack it as well
        stacked_attention_mask = torch.ones(batch_size,4*seq_length+1, dtype=torch.int64, device='cuda')
        # # we feed in the input embeddings (not word indices as in NLP) to the model
        transformer_outputs = self.transformer(
            inputs_embeds=stacked_inputs,
            attention_mask=stacked_attention_mask,
        )
        x = transformer_outputs['last_hidden_state']
        rest_of_sequence=x[:,1:]
        rest_of_sequence = rest_of_sequence.reshape(batch_size, seq_length, 4, self.hidden_size).permute(0, 2, 1, 3)
        # (0) return to go (1) state (2) language (3) action
        action_preds = self.predict_action_pos(rest_of_sequence[:,1])
        action_gripper = self.predict_action_gripper(rest_of_sequence[:,1])
        return action_preds,action_gripper
    
    def get_action(self,task_description, states, actions, returns_to_go,languages,env_name="metaworld"):
            trajectory={
            "state":states,
            "action":actions,
            "return_to_go":returns_to_go,
            "languages":languages,
            "manual":task_description
            }
            batch=self.inferenceDataManager.prepareTrajectory(trajectory,env_name)
            # Forward pass
            pred_pos,pred_gripper= self.forward(**batch)
            action_pos=pred_pos.argmax(dim=2)[0,-1]
            action_gripper=pred_gripper.argmax(dim=2)[0,-1]     
            return (action_pos,action_gripper)