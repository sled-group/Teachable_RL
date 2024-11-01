import argparse
import json
import torch
import numpy as np
import gym
from utils.observation_process import observationProcessor, numpy_formatter
from utils.deepCopy import copier
from utils.pathSolver import pathSolver
from PIL import Image
import numpy as np
from PIL import Image, ImageDraw
import random

def _symbolic_to_multihot(obs):
    # (h, w, 2)
    layers = np.concatenate((obs["entities"], obs["avatar"]),
                            axis=-1).astype(int)
    new_ob = np.maximum.reduce([np.eye(17)[layers[..., i]] for i
                                in range(layers.shape[-1])])
    new_ob[:, :, 0] = 0
    # assert new_ob.shape == self.observation_space["image"].shape
    return new_ob

def make_image(img):
    assert len(img.shape) == 3
    assert img.shape[2] == 17
    # Remove padding
    img = img[:10, :10]

    idx_to_letter = {
      2: 'A',
      3: 'M',
      4: 'D',
      5: 'B',
      6: 'F',
      7: 'C',
      8: 'T',
      9: 'H',
      10: 'B',
      11: 'R',
      12: 'Q',
      13: 'S',
      14: 'W',
      15: 'a',
      16: 'm'
    }
   
    scale = 256 / 10
    # fontpath = "/usr/share/fonts/truetype/liberation/LiberationMono-Bold.ttf"
    # font = ImageFont.truetype(fontpath, 12) if os.path.exists(fontpath) else None
    new_img = Image.new(size=(256, 256), mode="RGB", color=(31, 33, 50))
    draw = ImageDraw.Draw(new_img)
    idxs = img.argmax(-1)
    for i, row in enumerate(img):
      for j, col in enumerate(row):
        if idxs[i][j] == 0: continue
        letter = idx_to_letter[idxs[i][j]]
        # x,y canvas reversed
        color = (247, 193, 119) if letter in ("a", "m") else (238, 108, 133)
        draw.text((int(j * scale), int(i * scale)), letter, fill=color)
    new_img = np.asarray(new_img)
    return new_img

class simulatorParser:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument(
            "--window_size",
            default=5,
            type=int,
            help="The size of the window that the GPT will see",
        )
        self.parser.add_argument(
            "--trajectory_id",
            default="",
            type=str,
            help="The size of the window that the GPT will see",
        )
        self.parser.add_argument(
            "--non_expert_probability",
            default=0.4,
            type=float,
            help="The size of the window that the GPT will see",
        )
        self.parser.add_argument(
            "--entire_expert_probability",
            default=0,
            type=float,
            help="The size of the window that the GPT will see",
        )
        self.parser.add_argument(
            "--train_model_param",
            default="./EMMA_model_param/emma_s3_10.pth",
            type=str,
            help="Path to model states to evaluate.",
        )
        self.parser.add_argument(
            "--env_id",
            default="msgr-train-v3",
            type=str,
            help="Environment id used in gym.    make",
        )
        self.parser.add_argument(
            "--max_steps",
            default=128,
            type=int,
            help="Maximum number of steps for each    episode",
        )
        # Add new arguments for start and end of the loop
        self.parser.add_argument('--start', type=int, help='Start of the loop', default=0)
        self.parser.add_argument('--end', type=int, help='End of the loop', default=10)
    def parse(self):
        return self.parser.parse_args()

class simulator():
    def reset(self,newTask=False,verbose=False,seed=0):
        # Parser
        self.parser = simulatorParser()
        self.args = self.parser.parse()
        self.start=self.args.start
        self.end=self.args.end
        self.window_size = self.args.window_size
        self.expertState=[]
        self.trainingState=[]
        # Set the environment and observation processor
        self.training_env = gym.make(self.args.env_id)
        self.obs, self.manual = self.training_env.reset(seed=seed)
        self.done = False
        self.envCopier = copier(self.training_env)
        self.expertEnv = self.envCopier.deep_copy(self.training_env,newTask=newTask)
        self.training_env=self.envCopier.newTask(self.training_env,newTask=newTask)
        self.verbose = verbose
        self.input = ""
        self.output = ""
        # Store environments
        self.trainingStep=0
        self.envContainer = []
        self.subgoal = []
        self.train_totalStep = 0
        self.totalReward=0
        self.observationProcessor = observationProcessor()
        self.enemy_distanceList = []
        self.target_distanceList = []
        self.checkpoints = []
        self.stateContainer = []
        self.newTask=newTask
        # Initialize the LLM prompter
        # api_key = os.environ["OPENAI_API_KEY"]
        # self.prompterGPT3 = LLMPrompter("gpt-3.5-turbo", api_key)
        if (self.newTask):
            self.manual.insert(0,"First go to the goal, then go to the message.")
        else:
            self.manual.insert(0,"First go to the message, then go to the goal.")
        self.manual=" ".join(self.manual)
        self.trajectory={"state":[],"reward":[],"manual":self.manual,"action":[],"checkpoints":[],"languages":[],"image":make_image(_symbolic_to_multihot(self.obs))}
        self.promptList=[]
        # set the device
        if torch.cuda.is_available():
            self.args.device = torch.device("cuda:0")
        else:
            self.args.device = torch.device("cpu")

        # # Set up the training model
        # self.model.load_state_dict(
        #     torch.load(self.args.train_model_param, map_location=self.args.device)
        # )
        # self.buffer = ObservationBuffer(buffer_size=3, device=self.args.device)
        # self.model.eval()
        self.pathSolver = pathSolver()
        np.set_printoptions(formatter={"int": numpy_formatter})
        self.resultList_GPT3 = []
        self.trajectory_name="./trajectories/trajectory_"+self.args.trajectory_id+".pth"
    
    def __init__(self):
        # Parser
        self.parser = simulatorParser()
        self.args = self.parser.parse()
        self.start=self.args.start
        self.end=self.args.end
        self.window_size = self.args.window_size
        self.expertState=[]
        self.trainingState=[]
        # Set the environment and observation processor
        self.training_env = gym.make(self.args.env_id)
        self.obs, self.manual = self.training_env.reset()
        self.done = False
        self.envCopier = copier(self.training_env)
        self.expertEnv = self.envCopier.deep_copy(self.training_env)
        self.training_env=self.envCopier.newTask(self.training_env)
        self.verbose = False
        self.input = ""
        self.output = ""
        # Store environments
        self.trainingStep=0
        self.envContainer = []
        self.subgoal = []
        self.train_totalStep = 0
        self.totalReward=0
        self.observationProcessor = observationProcessor()
        self.enemy_distanceList = []
        self.target_distanceList = []
        self.checkpoints = []
        self.stateContainer = []
        # Initialize the LLM prompter
        # api_key = os.environ["OPENAI_API_KEY"]
        # self.prompterGPT3 = LLMPrompter("gpt-3.5-turbo", api_key)
        self.trajectory={"state":[],"reward":[],"manual":self.manual,"action":[],"checkpoints":[],"languages":[],"image":make_image(_symbolic_to_multihot(self.obs))}
        self.promptList=[]
        # set the device
        if torch.cuda.is_available():
            self.args.device = torch.device("cuda:0")
        else:
            self.args.device = torch.device("cpu")
        # # Set up the training model
        # self.model = EMMA().to(self.args.device)
        # self.model.load_state_dict(
        #     torch.load(self.args.train_model_param, map_location=self.args.device)
        # )
        # self.buffer = ObservationBuffer(buffer_size=3, device=self.args.device)
        # self.model.eval()
        self.pathSolver = pathSolver()
        np.set_printoptions(formatter={"int": numpy_formatter})
        self.resultList_GPT3 = []
        self.trajectory_name="./trajectories/trajectory_"+self.args.trajectory_id+".pth"
            
    def log(self):
        if self.verbose:
            if self.reward == 1:
                print("Win the Game")
            else:
                print("Lose the game")
            print(
                "\n***************************** Manual ************************************"
            )
            print(self.manual)
    
    def runForkEpisode(self, index):
        expertEnv = self.envContainer[index]
        done = False
        stateList = []
        stateList.append(self.observationProcessor.simplifyState(self.observationProcessor.generate_state(expertEnv)))
        currentState = self.observationProcessor.generate_state(expertEnv)
        if("goal" not in currentState):
            return
        for i in range(1):                
                currentState = self.observationProcessor.generate_state(expertEnv)
                self.pathSolver.update(currentState)
                action = self.pathSolver.get_action()
                self.expert_actions.append(action)
                _,_, done, _ = expertEnv.step(action)
                currentState = self.observationProcessor.generate_state(expertEnv)
                stateList.append(self.observationProcessor.simplifyState(currentState))
                if done:
                    break
        instruction={}
        hindsight={}
        diversities=["augmented"]
        for diversity in diversities:
            instruction[str(diversity)]=self.observationProcessor.generate_foresight_language(stateList,self.newTask,mode="training",diversity=diversity)
            if index>=1:
                hindsight[str(diversity)]=self.observationProcessor.generate_hindsight_language(self.stateContainer[index-1: index+1],self.newTask,mode="training",diversity=diversity,expert_action=self.expert_actions[-2])
            else:
                hindsight[str(diversity)]=""
        language={"hindsight positive":{},"hindsight negative":{},"foresight positive":{}}
        for diversity in diversities:
            language["hindsight positive"][str(diversity)]={"template":"","human":""} if ("hindsight positive" not in hindsight[str(diversity)]) else hindsight[str(diversity)]["hindsight positive"]
            language["hindsight negative"][str(diversity)]={"template":"","human":""} if ("hindsight negative" not in hindsight[str(diversity)]) else hindsight[str(diversity)]["hindsight negative"]
            language["foresight positive"][str(diversity)]={"template":"","human":""} if ("foresight positive" not in instruction[str(diversity)]) else instruction[str(diversity)]["foresight positive"]
        self.trajectory["languages"].append(language)
    
    def generateLanguage(self):
        self.expert_actions=[]
        for i in range(self.trainingStep):
                self.runForkEpisode(i)
        return
    
    def random_training(self):
        # return true is non_expert trajectory
        # retur false is expert trajectory
        random_number=random.random()
        if (random_number < self.args.non_expert_probability):
            return True
        else:
            return False
