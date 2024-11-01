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
import argparse
from typing import List
import imageio
import random
from utils.utils import agent_step,get_state,get_action,get_action_space,get_f_language,get_h_language

def get_task_text(env_name):
    name = " ".join(env_name.split('-')[:-3])
    return name

def transform(strings:List[str])->List[str]:
    for i in range(len(strings)):
        strings[i]=strings[i][0].upper()+strings[i][1:]
    return strings        

def get_policy(env_name):
    name = "".join(transform(get_task_text(env_name).split(" ")))
    name=name.replace("Insert","Insertion")
    policy_name = "Sawyer" + name + "V2Policy"   
    try:
        policy = getattr(policies, policy_name)()
    except:
        policy = None
    return policy

os.environ["TOKENIZERS_PARALLELISM"] = "false"
def process_data(trajectory,name):
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
        returns.reverse()
        trajectory["return_to_go"] = returns
        f,h,hf,rhf=[],[],[],[]
        rf,rh=[],[]
        for text in trajectory["languages"]:
            f.append(text["f"][1])
            h.append(text["h"][1])
            rf.append(text["f"][0])
            rh.append(text["h"][0])
            random_probability=random.randint(0,100)
            # (Optional) Randomly drop some hindsight or foresight information only for the H+F setting when collecting the dataset.
            if random_probability<70:
                hf.append(text["h"][1]+text["f"][1])
                rhf.append(text["h"][0]+text["f"][0])
            elif random_probability <85:
                rhf.append(text["h"][0])
                hf.append(text["h"][1])
            else:
                rhf.append(text["f"][0])
                hf.append(text["f"][1])
        trajectory["f_language"]=f
        trajectory["h_language"]=h
        trajectory["hf_language"]=hf
        trajectory["rhf_language"]=rhf
        l=f+h+hf+rhf+rf+rh
        with torch.no_grad():  # No gradient is needed (inference mode)
            l_embedding=torch.tensor(model.encode(l)).reshape(6,len(trajectory["languages"]),1,768)
            f_embedding=l_embedding[0]
            h_embedding=l_embedding[1]
            hf_embedding=l_embedding[2]
            rhf_embedding=l_embedding[3]
            rf_embedding=l_embedding[4]
            rh_embedding=l_embedding[5]
        assert f_embedding.shape == (len(trajectory["languages"]), 1, 768)
        manual_embedding = torch.unsqueeze(torch.tensor(model.encode(trajectory["manual"])), dim=0)
        trajectory["encoded_manual"]=manual_embedding
        trajectory["f_embedding"]=f_embedding
        trajectory["h_embedding"]=h_embedding
        trajectory["hf_embedding"]=hf_embedding
        trajectory["rhf_embedding"]=rhf_embedding
        trajectory["rf_embedding"]=rf_embedding
        trajectory["rh_embedding"]=rh_embedding
        torch.save(trajectory,name)

def calculate_angle(curr_pos,expert_target_pos,agent_target_pos):
    vector_expert = expert_target_pos - curr_pos
    vector_agent = agent_target_pos - curr_pos
    if (np.linalg.norm(vector_expert))!=0 and np.linalg.norm(vector_agent)!=0:
        vector_expert_normalized = vector_expert / np.linalg.norm(vector_expert)
        vector_agent_normalized = vector_agent / np.linalg.norm(vector_agent)
        dot_product = np.dot(vector_expert_normalized, vector_agent_normalized)
        angle_cosine = round(dot_product,3)
        angle = np.arccos(angle_cosine)  # 结果是弧度
        angle_degrees = np.degrees(angle)  # 转换为度数
        return angle_degrees
    elif((np.linalg.norm(vector_expert)+np.linalg.norm(vector_agent))!=0):
        return 90
    else:
        return 0
        

class simulator():
    def __init__(self,env_name,seed=0,verbose=False):
        self.verbose=verbose
        self.nonExpert_steps=0
        self.nonExpert_steps_max=3
        self.env_name=env_name
        benchmark_env = env_dict[env_name]
        task_name=get_task_text(env_name)
        self.policy=get_policy(env_name)
        self.env = benchmark_env(seed)
        self.observation,self.info=self.env.reset()
        self.success,self.cumulative_reward,self.timestep=0,0,0
        (self.expert_action_id,self.expert_gripper_id),f_language_id=self.policy.get_action(self.observation)
        (self.action_id,self.gripper_id)=(self.expert_action_id,self.expert_gripper_id)
        state=get_state(self.observation,self.policy,env_name)
        self.observation, self.reward, self.terminate,self.truncate, self.info= agent_step(self.policy,self.env,self.observation,get_action(self.policy,(self.action_id,self.gripper_id),self.observation,self.env_name),self.env_name)
        self.task_description=task_name
        self.non_expert_step_list=[]
        self.images=[]
        self.reward=-0.5
        self.timestep=0
        # print("Expert Step: ","action selection: ",(self.action_id,self.gripper_id)," ",self.timestep,"",f_language,"reward: ",self.reward)
        self.timestep+=1
        f_language=get_f_language(self.env_name,f_language_id,"training")
        # self.trajectory={"state":[init_state],"action":[self.action],"reward":[self.reward],"manual":self.task_description,"languages":[{"h":"","f":""}]}
        self.trajectory={"state":[state],"action":[np.array([self.action_id,self.gripper_id])],"reward":[self.reward],"manual":self.task_description,"languages":[{"h":("",""),"f":f_language}]}
        self.done=False
        self.success=0
        self.disturbed=False
    
    def run_expert_episode(self,steps):
        for _ in range(steps):
            if (self.done):
                break
            # h language
            if((self.action_id,self.gripper_id)!=(self.expert_action_id,self.expert_gripper_id)):
                self.non_expert_step_list.append(self.timestep-1)
            it=0
            while(self.timestep-it-1 in self.non_expert_step_list):
                it+=1            
            # h_language="You are disturbed. " if self.disturbed else "You are not disturbed. "
            h_language=get_h_language(self.env_name,self.disturbed,"training",self.action_id)
            self.disturbed=False
            # f language
            (self.expert_action_id,self.expert_gripper_id),f_language_id=self.policy.get_action(self.observation)
            f_language=get_f_language(self.env_name,f_language_id,"training")
            (self.action_id,self.gripper_id)=(self.expert_action_id,self.expert_gripper_id)
            self.trajectory["languages"].append({"h":h_language,"f":f_language})
            state=get_state(self.observation,self.policy,env_name)
            self.trajectory["state"].append(state)
            self.observation, self.reward, self.terminate,self.truncate, self.info= agent_step(self.policy,self.env,self.observation,get_action(self.policy,(self.action_id,self.gripper_id),self.observation,self.env_name),self.env_name)
            # image=self.env.render() # image shape: 240,320,3
            # self.images.append(image)
            # Design a better reward:
            self.reward=((self.action_id,self.gripper_id)==(self.expert_action_id,self.expert_gripper_id))*0.5
            self.trajectory["action"].append(np.array([self.action_id,self.gripper_id]))
            if self.info["success"]:
                self.reward+=20
            if self.verbose:
                print("Expert Step: ",self.timestep,h_language,f_language,"action selection: ",(self.action_id,self.gripper_id)," ","reward: ",self.reward)
            self.trajectory["reward"].append(self.reward)
            self.cumulative_reward += self.reward
            self.timestep+=1
            if self.info["success"]:
                self.success=1
                self.done=True
                break
            if self.terminate or self.truncate:
                self.done=True
                break
    
    def run_non_expert_episode(self,steps):
        action_space,gripper_space=get_action_space(self.env_name)
        self.disturbed=False
        for i in range(steps):
            if (self.done):
                break
            # h language
            if((self.action_id,self.gripper_id)!=(self.expert_action_id,self.expert_gripper_id)):
                self.non_expert_step_list.append(self.timestep-1)
            it=0
            while(self.timestep-it-1 in self.non_expert_step_list):
                it+=1
            # if it==steps or it==0:
            h_language=get_h_language(self.env_name,self.disturbed,"training",self.action_id)
            # h_language="You are disturbed. " if self.disturbed else "You are not disturbed. "
            self.disturbed=True
            # else:
                # h_language=""
            if (i==0):
                self.action_id=random.randint(0,action_space)
                # self.gripper_id=random.randint(0,gripper_space)
            # f language
            (self.expert_action_id,self.expert_gripper_id),f_language_id=self.policy.get_action(self.observation)
            f_language=get_f_language(self.env_name,f_language_id,"training")
            self.trajectory["languages"].append({"h":h_language,"f":f_language})
            state=get_state(self.observation,self.policy,env_name)
            self.trajectory["state"].append(state)
            self.observation, self.reward, self.terminate,self.truncate, self.info= agent_step(self.policy,self.env,self.observation,get_action(self.policy,(self.action_id,self.gripper_id),self.observation,self.env_name),self.env_name)
            # image=self.env.render() # image shape: 240,320,3
            # self.images.append(image)
            self.trajectory["action"].append(np.array([self.action_id,self.gripper_id]))
            self.reward=((self.action_id,self.gripper_id)==(self.expert_action_id,self.expert_gripper_id))*0.5-1
            if self.info["success"]:
                self.reward+=20
            if self.verbose:
                print("Non Expert Step: ",self.timestep,"h: ",h_language,"f: ",f_language,"expert: ", (self.expert_action_id,self.expert_gripper_id),"action selection: ",(self.action_id,self.gripper_id)," ","reward: ",self.reward)
            self.trajectory["reward"].append(self.reward)
            self.cumulative_reward += self.reward
            self.timestep+=1
            if self.info["success"]:
                self.done=True
                self.success+=1
                break
            if self.terminate or self.truncate:
                self.done=True
                break
    
    def run_episode(self):
        non_expert_step=random.randint(2,6)
        while(not self.done and self.timestep<30):
            if(self.timestep!=non_expert_step or self.nonExpert_steps>=self.nonExpert_steps_max):
                self.run_expert_episode(1)
            else:
                step1=random.randint(1,3)
                self.run_non_expert_episode(step1)
                non_expert_step=random.randint(self.timestep+step1+3,self.timestep+step1+7)
                self.nonExpert_steps+=1
        self.trajectory["languages"]=self.trajectory["languages"]
        self.trajectory["nonExpertTime"]=self.nonExpert_steps

def set_seed(seed=42):
    """reproduce"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 如果使用多GPU
    np.random.seed(seed)
    random.seed(seed)
    
if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--start",
        default=0,
        type=int,
    )
    parser.add_argument(
        "--end",
        default=1,
        type=int,
    )
    args=parser.parse_args()
    model = SentenceTransformer("sentence-transformers/paraphrase-TinyBERT-L6-v2")
    model.eval()
    set_seed()
    newTask=True
    trajectory_dir = './metaworld_hammer_dataset2' if newTask else "./metaworld_assembly_dataset"
    if not os.path.exists(trajectory_dir):
        os.makedirs(trajectory_dir)
    iter_begin, iter_end=(0,20) if newTask else (args.start,args.end)
    for i in range(iter_begin,iter_end):
        env_name='hammer-v2-goal-observable' if newTask else 'assembly-v2-goal-observable'
        try:
            success=0
            # We only want to collect the successful trajectories.
            while(success!=1):
                # Mod 5 since we are adapting with 5 shots / 10 shots/ 20 shots. When adapting with different number of shots, the tasks and scenes are identicalto each other, while the languages are augmented. We want to encourage the agent to learn from the languages.
                seed=i%5 if newTask else i
                metaworldSimulator=simulator(env_name=env_name,seed=seed,verbose=False)
                metaworldSimulator.run_episode()
                if not os.path.exists(f"{trajectory_dir}/{env_name}/"):
                    os.makedirs(f"{trajectory_dir}/{env_name}/")
                success+=metaworldSimulator.success
                print(f"Seed {seed}, {env_name} success: ",metaworldSimulator.success, " cumulative reward: ",metaworldSimulator.cumulative_reward,metaworldSimulator.timestep)
            process_data(metaworldSimulator.trajectory,f"{trajectory_dir}/{env_name}/trajectory_{i+1}.pth")
        except:
            print("Error")
    