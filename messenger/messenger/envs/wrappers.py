'''
Implements wrappers on top of the basic messenger environments
'''
import random

from messenger.envs.base import MessengerEnv
from messenger.envs.stage_one import StageOne
from messenger.envs.stage_two import StageTwo
from messenger.envs.stage_three import StageThree


class TwoEnvWrapper(MessengerEnv):
    '''
    Switches between two Messenger environments
    '''
    def __init__(self, stage:int, split_1:str,split_2:str, prob_env_1=0.5, **kwargs):
        super().__init__()
        if stage == 1:
            self.env_1 = StageOne(split=split_1, **kwargs)
            # self.env_2 = StageOne(split=split_2, **kwargs)
        elif stage == 2:
            self.env_1 = StageTwo(split=split_1, **kwargs)
            # self.env_2 = StageTwo(split=split_2, **kwargs)
        elif stage == 3:
            self.msgrEnv = StageThree(split=split_1,seed=None, **kwargs)
            # self.env_2 = StageThree(split=split_2, **kwargs)
        
        self.prob_env_1 = prob_env_1
        self.cur_env = None
    
    def deep_copy(self,newEnv,newTask=False):
        newEnv.msgrEnv=self.msgrEnv.deep_copy(newEnv.msgrEnv,newTask)
        # newEnv.env_2=self.env_2.deep_copy(newEnv.env_2)
        # Deep copy the current environment
        # Note: You might need to handle the case where cur_env is None
        if self.cur_env is self.msgrEnv:
            newEnv.cur_env = newEnv.msgrEnv
        # elif self.cur_env is self.env_2:
        #     newEnv.cur_env = newEnv.env_2
        else:
            newEnv.cur_env = None  # or handle appropriately
        return newEnv
    # def deep_copy(self):
    #     new_obj = TwoEnvWrapper(stage=1, split_1='train_mc', split_2='train_mc')  # Use any default values
    #     new_obj.env_1 = self.env_1.deep_copy()
    #     if self.cur_env is self.env_1:
    #         new_obj.cur_env = new_obj.env_1
    #     else:
    #         new_obj.cur_env = None
    #     return new_obj
    
    def reset(self,newTask=False, seed=None, **kwargs):
        # if random.random() < self.prob_env_1:
        #     self.cur_env = self.env_1
        # else:
        #     self.cur_env = self.env_2
        self.cur_env = self.msgrEnv
        return self.cur_env.reset(seed,**kwargs)
    
    def step(self, action):
        return self.cur_env.step(action)