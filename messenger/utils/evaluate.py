import torch
import numpy as np
import argparse
import json
import torch
import numpy as np
import gym
import messenger
from messenger.models.emma import EMMA
from messenger.models.utils import ObservationBuffer
from utils.observation_process import observationProcessor, numpy_formatter
import torch
from transformers import RobertaTokenizer
from torch.nn.functional import pad
from utils.data_process import getDataset
import numpy as np
from utils.inferenceDataManager import inferenceDataManager
from utils.deepCopy import copier
from utils.pathSolver import pathSolver
import random
import re
# from prompt import LLMPrompter
import openai
from openai import OpenAI
client = OpenAI(
    #     # This is the default and can be omitted
        api_key="API_KEY",
    )
GPT_version="gpt4" # else "gpt3.5"

def evaluate_episode_rtg(
        model,
        language_model,
        max_ep_len=50,
        scale=100.,
        state_mean=0.,
        state_std=1.,
        device='cuda',
        target_return=None,
        mode='normal',
        newTask=True,
        eval_mode="validation",
        probability_threshold=33,
        seed=None,
        gpt=False,
        evalType="rhf",
        disturb=False
    ):
        episode_return, episode_length,subgoal_complete,goal_complete,data=evaluate_episode_rtg_messenger(
            model=model,
            language_model=language_model,
            max_ep_len=max_ep_len,
            scale=scale,
            state_mean=state_mean,
            state_std=state_std,
            device=device,
            target_return=target_return,
            mode=mode,
            newTask=newTask,
            eval_mode=eval_mode,
            probability_threshold=probability_threshold,
            seed=seed,
            gpt=gpt,
            evalType=evalType,
            disturb=disturb
        )            
            
        return episode_return, episode_length,subgoal_complete,goal_complete,data

def evaluate_episode_rtg_messenger(
        model,
        language_model,
        max_ep_len=50,
        scale=100.,
        state_mean=0.,
        state_std=1.,
        device='cuda',
        target_return=None,
        mode='normal',
        newTask=True,
        eval_mode="validation",
        probability_threshold=100,
        seed=None,
        gpt=False,
        evalType="rhf",
        disturb=False
    ):
    if(seed is None):
        seed=random.randint(0,1000)
    np.set_printoptions(formatter={'int': numpy_formatter})
    model.eval()
    model.to(device=device)
    language_model.eval()
    language_model.to(device=device)
    stateProcessor=observationProcessor()
    state_mean = state_mean.to(device=device)
    state_std = state_std.to(device=device)
    tempenv = gym.make("msgr-train-v3")
    obs, manual = tempenv.reset(seed=seed)
    if(newTask):
        manual.insert(0,"First go to the goal, then go to the message. ")
    else:
        manual.insert(0,"First go to the message, then go to the goal. ")
    envCopier=copier(tempenv)
    env=envCopier.newTask(tempenv,newTask)
    stateProcessor=observationProcessor()
    state=stateProcessor.generate_trajectory_state(obs)
    states,actions,rewards = [state],[],[]
    currentState=stateProcessor.generate_state(env)
    envCopier=copier(env)
    expertEnv=envCopier.deep_copy(env,newTask)
    stateContainer=[]
    expertEnv=envCopier.deep_copy(env,newTask)
    fp,fn,distance_to_target,distance_to_enemy,fnFlag,expert_action=get_foresight_language(expertEnv,stateProcessor,newTask,mode=eval_mode)
    language_feedback=fp
    with torch.no_grad():
        encoded_manual = torch.unsqueeze(torch.tensor(language_model.encode(" ".join(manual))), dim=0).to('cuda')
        encoded_language=torch.unsqueeze(torch.tensor(language_model.encode(language_feedback)), dim=0).to('cuda')
    target_return_list = [target_return/scale]
    languages=encoded_language.unsqueeze(0)
    episode_return, episode_length = 0,0
    subgoal_complete=0
    goal_complete=0
    stateContainer.append(stateProcessor.simplifyState(currentState))
    data={"languages":[]}
    for t in range(max_ep_len):
        actions.append(4)
        rewards.append(0)
        action = model.get_action(encoded_manual,states,actions,rewards,target_return_list,languages)
        actions[-1] = action
        action = action.detach().cpu().numpy()
        obs, reward, done, _ = env.step(action)
        currentState=stateProcessor.generate_state(env)
        reward=reward*100+stateProcessor.process_reward(currentState)-0.5
        ### Update Reward. Achieving the subgoal will have 50 reward, and achieving the final goal has 100 reward.
        if(reward>45 and reward<70):
            subgoal_complete+=1
        if(reward>95):
            goal_complete+=1
        state=stateProcessor.generate_trajectory_state(obs)
        stateContainer.append(stateProcessor.simplifyState(currentState))
        states.append(state)
        rewards[-1] = reward
        if mode != 'delayed':
            pred_return = target_return_list[-1] - (reward/scale)
        else:
            pred_return = target_return_list[-1]
        target_return_list.append(pred_return)
        episode_return += reward
        episode_length += 1
        if done:
            # print("done")
            break
        randomProbability=random.randint(0,100)
        if(randomProbability<probability_threshold) :
            if disturb is False:
                disturb_hindsight,disturb_foresight=(False,False)
            else:
                # Disturb the hindsight or foresight with the probability.
                disturb_prob=random.uniform(0,1)
                if disturb_prob < 0.3:
                    # Disturb both hindsight and foresight
                    disturb_hindsight,disturb_foresight=(True,True)
                elif disturb_prob < 0.65:
                    # Disturb foresight
                    disturb_hindsight,disturb_foresight=(False,True)
                else:
                    # Disturb hindsight
                    disturb_hindsight,disturb_foresight=(True,False)
            hindsight_raw,hnFlag=stateProcessor.generate_hindsight_language(stateContainer[t:t+2],mode=eval_mode,moreInfo=True,expert_action=expert_action,disturb=disturb_hindsight)
            hindsight=""
            hp,hn="",""
            if ("hindsight positive" in hindsight_raw):
                hp=hindsight_raw["hindsight positive"]["human"]
            if ("hindsight negative" in hindsight_raw):
                hn=hindsight_raw["hindsight negative"]["human"]
            hindsight=hp+hn
            expertEnv=envCopier.deep_copy(env,newTask)
            fp,fn,distance_to_target,distance_to_enemy,_,expert_action=get_foresight_language(expertEnv,stateProcessor,newTask,mode=eval_mode, disturb=disturb_foresight)
            foresight=fp+fn
        else:
            hindsight=""
            foresight=""
        if "h" not in evalType:
            hindsight=""
        if "f" not in evalType:
            foresight=""
        with torch.no_grad():
            if gpt:
                language,isSpeak=get_GPT_response(hindsight,foresight,distance_to_target,distance_to_enemy)
                print("Original: ",hindsight+" "+foresight)
                print("Language: ",language)
                language=language if isSpeak else ""
            else:
                language=hindsight+foresight
            #################################################
            if language !="":
                segments = language.split('.')
                # Remove any empty segments or segments with only whitespace
                segments = [seg.strip() for seg in segments if seg.strip()]
                # Encode each segment and store the embeddings
                embeddings = []  
                for segment in segments:
                    # print(segment)
                    encoded_segment = torch.unsqueeze(torch.tensor(language_model.encode(segment+". ")), dim=0).to('cuda')
                    embeddings.append(encoded_segment)
                encoded_language = torch.mean(torch.stack(embeddings), dim=0)
            else:
                encoded_language=torch.unsqueeze(torch.tensor(language_model.encode(language)), dim=0).to('cuda')
        languages=torch.cat((languages,encoded_language.unsqueeze(0)),dim=0)
    print("Episode ends: ",goal_complete==1, seed)
    return episode_return, episode_length,subgoal_complete,goal_complete,data

def get_foresight_language(expertEnv,observation_Processor,newTask=False,mode="validation",disturb=False):
    path_Solver = pathSolver()
    stateList=[]
    currentState = observation_Processor.generate_state(expertEnv)
    stateList.append(observation_Processor.simplifyState(currentState))
    if("goal" not in currentState or "agent" not in currentState):
            return
    for i in range(1):
                currentState = observation_Processor.generate_state(expertEnv)
                path_Solver.update(currentState)
                action = path_Solver.get_action()
                # path_Solver.print_map()
                _,_, done, _ = expertEnv.step(action)
                currentState = observation_Processor.generate_state(expertEnv)
                stateList.append(observation_Processor.simplifyState(currentState))
                if done:
                    break
    future_pos=""
    try:
        instruct,distance_to_target,distance_to_enemy,fn=observation_Processor.generate_foresight_language(stateList,newTask,mode=mode,diversity='augmented',moreInfo=True,disturb=disturb)
        language_template_human="human" 
        future_pos="" if ("foresight positive" not in instruct) else instruct["foresight positive"][language_template_human]
        future_neg="" if ("foresight negative" not in instruct) else instruct["foresight negative"][language_template_human]
    except:
        import pdb;pdb.set_trace()
        future_pos=""
        future_neg=""
        distance_to_enemy,distance_to_target=np.inf,np.inf
        fn=False
    return future_pos,future_neg,distance_to_target,distance_to_enemy,fn,action

PROMPT_GPT4="""
You are a human expert that teaches non-expert agent to improve its performance in a grid world. The robot task is to find the message and then send it to the goal.

Here's the action space of the robot:
["right", "left", "up", "down", "noMotion"]

The game simulator provides the following hint due to the last action:
(1) hindsight feedback: {hindsight}
(2) foresight feedback: {foresight}
(3) Supporting state information: {states}

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

def get_GPT_response(hindsight,foresight,distance_to_target,distance_to_enemy):
    state_info=f"The distance from the agent to the target is {distance_to_target}, the distance from the agent to the enemy is {distance_to_enemy}."
    try:
        prompt = (
                        PROMPT_GPT4.replace("{hindsight}",hindsight).replace("{foresight}",foresight).replace("{states}",state_info)
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

