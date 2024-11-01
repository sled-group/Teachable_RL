import torch
import numpy as np
import random
import numpy as np
import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
metaworld_dir = os.path.join(current_dir, 'metaworld')
if metaworld_dir not in sys.path:
    sys.path.insert(0, metaworld_dir)
import torch
from metaworld import policies
from sentence_transformers import SentenceTransformer
from metaworld.envs import ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE as env_dict
from typing import List
import random
from utils.data_generation import get_policy,get_task_text,get_action,get_h_language,get_f_language
from utils.utils import agent_step,get_state
import json
import openai
from openai import OpenAI
client = OpenAI(
    #     # This is the default and can be omitted
        api_key="API_KEY",
    )
GPT_version="gpt4" # else "gpt3.5"
def evaluate_episode_rtg(
        model,
        device='cuda',
        target_return=None,
        eval_mode="validation",
        probability_threshold=33,
        task_name=None,
        language_model=None,
        seed=0,
        isGPT=False
    ):
        return evaluate_episode_rtg_metaworld(model,device,target_return=target_return,mode='normal',env_name=task_name,eval_mode=eval_mode,probability_threshold=probability_threshold,language_model=language_model,seed=seed, isGPT=isGPT)

def set_seed(seed=42):
    """Control the seed for the environment, critical for fair comparison during validation/testing evaluation."""
    np.random.seed(seed)
    random.seed(seed)
    
def evaluate_episode_rtg_metaworld(
        model,
        device='cuda',
        target_return=None,
        mode='normal',
        env_name=None,
        eval_mode="validation",
        probability_threshold=100,
        language_model=None,
        seed=None,
        isGPT=False
):    
    set_seed(seed)
    model.eval()
    model.to(device=device)
    benchmark_env = env_dict[env_name]
    task_name=get_task_text(env_name)
    policy=get_policy(env_name)
    manual=task_name
    env = benchmark_env(seed)
    observation,info=env.reset()
    success=0
    with torch.no_grad():
        encoded_manual = torch.unsqueeze(torch.tensor(language_model.encode(manual)), dim=0).to('cuda')
        encoded_language=torch.unsqueeze(torch.tensor(language_model.encode("")), dim=0).to('cuda')
    state=get_state(observation,policy,env_name)
    states,actions,target_return_list,languages=[state],[],[target_return],encoded_language.unsqueeze(0)
    episode_return,success=0,0
    # Disturbation from the environment. In this task, the agent is disturbed at random timesteps, and the agent is expected to recover from the disturbance.
    duration=0
    max_distort=3
    non_expert_step=random.randint(2,6)
    non_expert_step_list=[] # This record the agent's steps that are not optimal for giving hindsight languages.
    length=0
    for t in range(30):
        actions.append([0,0])
        if (t==non_expert_step and max_distort>0) or duration > 0:
            if (duration==0):
                duration=random.randint(0,2)
                action=(random.randint(0,6), pred_gripper)
                non_expert_step=random.randint(t+duration+4,t+duration+8)
                max_distort-=1
            else:
                duration-=1
            pred_pos,pred_gripper=action
            pred_pos=torch.tensor(pred_pos)
        else:
            action = model.get_action(encoded_manual,states,actions,target_return_list,languages,   env_name="metaworld")
            pred_pos,pred_gripper=(action[0].detach().cpu(),action[1].detach().cpu())
                
        (expert_action_id,expert_gripper_id),f_language_id=policy.get_action(observation)
        observation, reward, terminated, truncated, info = agent_step(policy,env,observation,get_action(policy,action,observation,env_name),env_name)
        reward=((pred_pos,pred_gripper)==(expert_action_id,expert_gripper_id))-0.5
        state=get_state(observation,policy,env_name)
        states.append(state)
        actions[-1]=[pred_pos,pred_gripper]
        if mode != 'delayed':
            pred_return = target_return_list[-1] - reward
        else:
            pred_return = target_return_list[-1]
        target_return_list.append(pred_return)
        if(info["success"]):
            reward+=20
        episode_return += reward
        done = terminated or truncated
        length+=1
        if(info["success"]):
            success=1
            break
        if done:
            break
        random_probability=random.randint(0,100)
        if((pred_pos,pred_gripper)!=(expert_action_id,expert_gripper_id)):
            non_expert_step_list.append(t)
        it=0
        while(t-it in non_expert_step_list):
            it+=1
        it=it if it<3 else 3
        h_language=get_h_language(env_name,it,eval_mode,pred_pos)[0]
        (expert_action_id,expert_gripper_id),f_language_id=policy.get_action(observation)
        f_language=get_f_language(env_name,f_language_id,eval_mode)[0]
        if not isGPT:
            if(random_probability<probability_threshold):
                if random_probability<100:
                    f_language=f_language
                    h_language=h_language
            else:
                f_language=""
                h_language=""      
            language=h_language+f_language
        else:
            language,isSpeak=get_GPT_response(h_language,f_language)
            print("Original Language: ",h_language,f_language)
            print("GPT: ",language, isSpeak)
            print("")
        language=""
        with torch.no_grad():
            encoded_language=torch.unsqueeze(torch.tensor(language_model.encode(language)), dim=0).to('cuda')
        languages=torch.cat((languages,encoded_language.unsqueeze(0)),dim=0)
    return episode_return,success,length


PROMPT_GPT4="""
You are a human expert that teaches non-expert agent to improve its performance in a robotic task.

Here's the action space of the robot:
["raise_gripper", "open_gripper", "place_gripper_above_tool", "aim_gripper_at_goal", "grasp_tool", "raise_tool", "get_to_goal"]

The game simulator provides the following hint due to the last action:
(1) hindsight feedback: {hindsight}
(2) foresight feedback: {foresight}

As a human expert, fully consider the given information and hint. 

You are expected to give the agent hindsight compliment/criticism on the agent's previous actions, and give instructions for the agent's future actions. Don't only give foresight instructions, hindsight feedback is also important. 

Simply give the hindsight and foresight information is enough, you don't need to include other supporting information.

Please translate the hindsight and foresight languages slightly to mimic the diverse human languages. Please be brief, at similar length to the original sentences.

You can freely decide to output hindsight only or foresight only or both. 

If the hindsight is compliment and appraise, you should be reluctant to give instructions if you think the agent is on the right track and doesn't need any help, just like humans that only give instructions when necessary. 

You should only respond in a json format as described below:

{
   "response": "(Strictly less than 25 words) your hindsight compliment/criticism and foresight instruction to give to the robot. (empty string if_give_response is false)",
   "if_give_response": true/false (Python Boolean), true if you feel necessary to give response, otherwise false
}

Make sure the response contains all keys listed in the above example and must be parsed by Python json.loads().
"""

def get_GPT_response(hindsight,foresight):
    try:
        prompt = (
                        PROMPT_GPT4.replace("{hindsight}",hindsight).replace("{foresight}",foresight)
                )
        response = client.chat.completions.create(
                    model="gpt-4",
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": prompt},
                    ],
                    max_tokens=100,  
                    temperature=0.7,  # Adjust this for more/less randomness
                )
        response = response.choices[0].message.content
        parsed_json = json.loads(response)
        lang = parsed_json["response"]
        isSpeak=parsed_json["if_give_response"]
    except:
        return "",False
    return lang,isSpeak