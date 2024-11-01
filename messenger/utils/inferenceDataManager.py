import torch
from transformers import RobertaTokenizer
from torch.nn.functional import pad
from utils.data_process import getDataset
import numpy as np

class inferenceDataManager():
    def __init__(self,state_mean,state_std,empty_language_encode, segment_length=20):
        self.segmentLength=segment_length
        self.state_mean=state_mean
        self.state_std=state_std
        self.empty_language_encode=empty_language_encode
    def prepareTrajectory(self, trajectory,env_name="messenger"):
            # print(trajectory["state"])
            max_length=torch.tensor(trajectory["state"], device='cuda').shape[0]
            ending_point=max_length
            starting_point=0 if(max_length<=self.segmentLength) else ending_point-self.segmentLength
            # Calculate padding length
            pad_length = self.segmentLength - (ending_point - starting_point)

            # For text and language, we can use the tokenizer's padding functionality directly
            text = trajectory["manual"]
            rewards = torch.tensor(trajectory["reward"][starting_point:ending_point], device='cuda').float().unsqueeze(1)

            if pad_length > 0:
                rewards = pad(rewards, (0, 0, pad_length, 0), "constant", 0)
            if (env_name=="messenger"):
                actions = torch.tensor(trajectory["action"][starting_point:ending_point], device='cuda').float().unsqueeze(1)
                if pad_length > 0:
                    actions = pad(actions, (0, 0, pad_length, 0), "constant", 0)
            elif (env_name=="metaworld"):
                actions=torch.tensor(trajectory["action"][starting_point:ending_point], device='cuda').float().unsqueeze(0)
                if pad_length > 0:
                    actions = pad(actions, (0, 0, pad_length, 0), "constant", 0)               
            return_to_go = torch.tensor(trajectory["return_to_go"][starting_point:ending_point], device='cuda').float().unsqueeze(1)
            if pad_length > 0:
                return_to_go = pad(return_to_go, (0, 0, pad_length, 0), "constant", 0)

            mask = torch.zeros(self.segmentLength, device='cuda').bool()
            mask[-max_length:] = True  # We shift the 'True' values to the end part of the mask

            # For states, since it's a numpy array, you'll need to adjust the concatenation:
            states = torch.tensor(trajectory["state"][starting_point:ending_point], dtype=torch.float32).to(device='cuda')
            padding = torch.zeros(pad_length, states.shape[1], device='cuda',dtype=torch.float32)
            states = torch.cat([padding, states], dim=0)
            self.state_mean=self.state_mean.to(device=states.device)
            self.state_std=self.state_std.to(device=states.device)
            states = (states - self.state_mean) / self.state_std

            # For language and text, adjust the concatenation logic to add padding at the front:
            language = trajectory["languages"][starting_point:ending_point]
            if pad_length>0:
                # padding = torch.zeros(pad_length, 1, 768, device='cuda',dtype=torch.float32)
                padding=self.empty_language_encode.repeat(pad_length,1,1).to(device=language.device)
                language = torch.cat((padding, language), dim=0).to(device='cuda')

            # For text, do the same if necessary. The example is not complete as the previous text padding isn't shown.

            # For time_stamps, you'll need to pad at the beginning as well:
            time_stamps = torch.arange(starting_point, ending_point, device='cuda').long()
            time_stamps = pad(time_stamps, (pad_length, 0), "constant", 0)
            if(env_name=="messenger"):
                actions=actions.unsqueeze(0).to("cuda")
            language=language.squeeze(1)
            return {
                'text_input': text.unsqueeze(0).unsqueeze(0).to("cuda"),
                'states': states.unsqueeze(0).to("cuda"),
                'rewards': rewards.unsqueeze(0).to("cuda"),
                'actions': actions,
                'returns_to_go': return_to_go,
                'languages': language.to("cuda").unsqueeze(0),
                'timesteps': time_stamps.unsqueeze(0).to("cuda"),
                'attention_mask': mask.unsqueeze(0).to("cuda")
            }
           

def test():
    checkpoint = torch.load('./artifacts/Multimodal_Decision_Transformer:v3/MDT.pth')
    torch.cuda.empty_cache()
    encoder = RobertaTokenizer.from_pretrained("roberta-base")
    cnn_channels = [3, 64, 128, 256]
    # Load data
    data,state_mean,state_std=getDataset()
    manager=inferenceDataManager(encoder,state_mean,state_std)
    trajectory=data[0]
    print(trajectory.keys())

if __name__ == "__main__":
    test()