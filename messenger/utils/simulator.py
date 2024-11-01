import torch
from utils.base import simulator
import random
import torch
import os
from sentence_transformers import SentenceTransformer
import torch

class msgr_simulator(simulator):
    def __init__(self):
        super().__init__()
    def run_episode_ground_truth_pathSolver(self, episode,newTask=False):
        if self.done:
            return
        for i in range(episode):
            with torch.no_grad():
                currentState = self.observationProcessor.generate_state(self.training_env)
                self.stateContainer.append(self.observationProcessor.simplifyState(currentState))
                self.pathSolver.update(currentState)
                action = self.pathSolver.get_action()
                self.trajectory["state"].append(self.observationProcessor.generate_trajectory_state(self.obs))
                self.envContainer.append(self.envCopier.deep_copy(self.training_env,newTask))
                self.trajectory["action"].append(action)
                if self.verbose:
                    print("Step: ", self.trainingStep+1)
                    print("Map:")
                    self.pathSolver.print_map()
                    print(self.observationProcessor.generate_grid(self.obs))
                    print(currentState)
                    actionList=["w","s","a","d",""]
                    print("action: ",actionList[action])
                self.obs, self.reward, self.done, _ = self.training_env.step(action)                # Calcualte new current state, and generate reward based on this state
                currentState = self.observationProcessor.generate_state(self.training_env)
                if self.verbose:
                    print(currentState)
                    print("After Step")
                    print(self.observationProcessor.generate_grid(self.obs))
                self.reward=self.reward*100+self.observationProcessor.process_reward(currentState)
                self.totalReward=self.totalReward+self.reward
                self.trajectory["reward"].append(self.reward)
                # self.subgoal.append(self.observationProcessor.generateSubgoal(currentState))
                self.trainingState.append(self.observationProcessor.generate_state(self.training_env))
                self.trainingStep=self.trainingStep+1
                if self.done:
                    break

    def run_episode_EMMA_train(self, episode,newTask=False): 
        if self.done:
            return
        # self.buffer.reset(self.obs)
        for i in range(episode):
            # with torch.no_grad():
            #     action = self.model(self.buffer.get_obs(), self.manual)
            self.trajectory["state"].append(self.observationProcessor.generate_trajectory_state(self.obs))
            self.envContainer.append(self.envCopier.deep_copy(self.training_env,newTask))
            action=random.randint(0,4)
            self.trajectory["action"].append(action)
            if self.verbose:
                print("Step: ", self.trainingStep+1)
                print(self.observationProcessor.generate_grid(self.obs))
            currentState=self.observationProcessor.generate_state(self.training_env)
            self.stateContainer.append(
                self.observationProcessor.simplifyState(
                    currentState
                )
            )
            # print(currentState)
            self.obs, self.reward, self.done, _ = self.training_env.step(action)
            if self.verbose:
                print(self.observationProcessor.generate_grid(self.obs))
            # self.buffer.update(self.obs)
            currentState=self.observationProcessor.generate_state(self.training_env)
            # print(currentState)
            self.reward=self.reward*100+self.observationProcessor.process_reward(currentState)
            self.totalReward=self.totalReward+self.reward
            self.trajectory["reward"].append(self.reward)
            # self.subgoal.append(self.observationProcessor.generateSubgoal(currentState))
            self.trainingState.append(currentState)
            self.trainingStep=self.trainingStep+1
            if self.done:
                break
            
    def run_trajectory(self,episode,newTask=False):
        self.nonExpertTimes=0
        self.nonExpertTick=0
        stride=2
        random_number=random.random()
        # self.run_episode_ground_truth_pathSolver(50,newTask)
        # print("debug: ",random_number)
        if (random_number<self.args.entire_expert_probability):
            print("This is entire expert trajectory")
            self.run_episode_ground_truth_pathSolver(50,newTask)
        else:
            for _ in range(episode):
                if(self.done):
                    break
                if(self.nonExpertTick<5 and self.random_training()):
                    self.nonExpertTimes+=stride
                    self.nonExpertTick+=1
                    self.run_episode_EMMA_train(stride,newTask)
                    self.run_episode_ground_truth_pathSolver(6,newTask)
                else:
                    self.run_episode_ground_truth_pathSolver(3,newTask)
        # # Add this step for the observation process to give reflection on the last step.
        # currentState=self.observationProcessor.generate_state(self.training_env)
        # self.stateContainer.append(
        #         self.observationProcessor.simplifyState(
        #             currentState
        #         )
        #     )
        print("Total Reward: " ,self.totalReward)

def process_data_sentenceBert(trajectory,name):
        rewards = trajectory["reward"]
        returns = []
        gamma=1
        cumulative_return = 0
        for reward in reversed(rewards):
            cumulative_return = reward + gamma * cumulative_return
            returns.append(cumulative_return)
        returns.reverse()
        trajectory["return_to_go"] = returns
        rh,rf,h,f=[],[],[],[]
        for text in trajectory["languages"]:
            if(text=={}):
                text={"hindsight positive":"", "hindsight negative":"", "foresight positive":"","foresight negative":""}
            rh.append(text["hindsight positive"]["augmented"]["human"]+text["hindsight negative"]["augmented"]["human"])
            rf.append(text["foresight positive"]["augmented"]["human"])
            h.append(text["hindsight positive"]["augmented"]["template"]+text["hindsight negative"]["augmented"]["template"])
            f.append(text["foresight positive"]["augmented"]["template"])
        l=rh+rf+h+f
        with torch.no_grad(): 
            l_embedding=torch.tensor(model.encode(l)).reshape(4,len(trajectory["languages"]),1,768)
            rh_embedding=l_embedding[0]
            rf_embedding=l_embedding[1]
            h_embedding=l_embedding[2]
            f_embedding=l_embedding[3]
            
        assert rh_embedding.shape == (len(trajectory["languages"]), 1, 768)
        manual_embedding = torch.unsqueeze(torch.tensor(model.encode(trajectory["manual"])), dim=0)
        trajectory["encoded_manual"]=manual_embedding
        trajectory["rhf_embedding"]=(rh_embedding+rf_embedding)/2 # Average Sentence Embedding.
        trajectory["rh_embedding"]=rh_embedding
        trajectory["rf_embedding"]=rf_embedding
        trajectory["h_embedding"]=h_embedding
        trajectory["f_embedding"]=f_embedding
        trajectory["hf_embedding"]=(h_embedding + f_embedding)/2
        del trajectory["checkpoints"]
        del trajectory["languages"]
        del trajectory["image"]
        del trajectory["manual"]
        torch.save(trajectory,name)
        

if __name__ == "__main__":
    newTask=False
    messengerSimulator = msgr_simulator()    
    if newTask:
        trajectory_dir = 'Messenger_Dataset_newTask'
        iter_begin,iter_end=(0,25)
    else:
        trajectory_dir = 'Messenger_Dataset'
        iter_begin,iter_end=(messengerSimulator.start,messengerSimulator.end)
    model = SentenceTransformer("sentence-transformers/paraphrase-TinyBERT-L6-v2")
    model.eval()
    if not os.path.exists(trajectory_dir):
        os.makedirs(trajectory_dir)
    for x in range(iter_begin,iter_end):
        try:
            seed=x
            messengerSimulator.reset(newTask=newTask,verbose=False,seed=seed)
            messengerSimulator.run_trajectory(20,newTask=newTask)
            messengerSimulator.generateLanguage()
            trajectory_name=trajectory_dir+"/trajectory_"+str(x+1)+".pth"
            process_data_sentenceBert(messengerSimulator.trajectory,trajectory_name)
        except:
            print("Program Error")
    
