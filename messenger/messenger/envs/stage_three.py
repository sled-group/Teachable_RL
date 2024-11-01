'''
Classes that follows a gym-like interface and implements stage three of the Messenger
environment.
'''
import gym
import json
import random
from collections import namedtuple
from pathlib import Path
from os import environ
import re
import copy
# hack to stop PyGame from printing to stdout
environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "hide"
import sys
sys.path.insert(0, "/home/heyinong/anaconda3/envs/msgr-emma/lib/python3.7/site-packages/vgdl/interfaces/gym/env.py")
import numpy as np
from vgdl.interfaces.gym import VGDLEnv

from messenger.envs.base import MessengerEnv, Grid, Position
import messenger.envs.config as config
from messenger.envs.manual import TextManual, Descr
from messenger.envs.utils import games_from_json


# specifies the game variant path is path to the vgdl domain file describing the variant.
GameVariant = namedtuple(
    "GameVariant",
    [
        "path",
        "enemy_type",
        "message_type",
        "goal_type",
        "decoy_message_type",
        "decoy_goal_type"
    ]
)

class StageThree(MessengerEnv):
    '''
    Similar to stage two Messenger, except with decoy objects that require
    disambiduation (e.g. chasing knight, vs immovable knight)
    '''
    

    def __init__(self, split:str, shuffle_obs=True,seed=None):
        super().__init__()
        self.stateFrame={}
        self.variant=None,
        self.seed=seed,
        self.shuffle_obs = shuffle_obs # shuffle the entity layers
        self.split=split
        self.game=None
        self.newTask=False
        this_folder = Path(__file__).parent
        # Get the games and manual
        games_json_path = this_folder.joinpath("games.json")
        if "train" in split and "mc" in split: # multi-combination games
            game_split = "train_multi_comb"
            text_json_path = this_folder.joinpath("texts", "text_train.json")
        elif "train" in split and "sc" in split: # single-combination games
            game_split = "train_single_comb"
            text_json_path = this_folder.joinpath("texts", "text_train.json")
        elif "val" in split:
            game_split = "val"
            text_json_path = this_folder.joinpath("texts", "text_val.json")
        elif "test" in split:
            game_split = "test"
            text_json_path = this_folder.joinpath("texts", "text_test.json")
        else:
            raise Exception(f"Split: {split} not understood.")

        # list of Game namedtuples
        self.all_games = games_from_json(json_path=games_json_path, split=game_split)
        self.text_manual = TextManual(json_path=text_json_path)

        vgdl_files = this_folder.joinpath("vgdl_files", "stage_3")
        
        # get the file paths to possible starting states
        self.init_states = [
            str(path) for path in vgdl_files.joinpath("init_states").glob("*.txt")
        ]
        # get all the game variants
        self.game_variants = [
            self._get_variant(path) for path in vgdl_files.joinpath("variants").glob("*dom_[345]_2.txt")
        ]

        # entities tracked by VGDLEnv
        self.notable_sprites = ["enemy", "message", "goal", "decoy_message", "decoy_goal", "no_message", "with_message"]
        self.env = None # the VGDLEnv
    
    def get_state(self):
        state = {
            "shuffle_obs": self.shuffle_obs,
            "all_games": self.all_games,
            "text_manual": self.text_manual,
            "init_states": self.init_states,
            "game_variants": self.game_variants,
            "notable_sprites": self.notable_sprites,
            "variant":self.variant,
            "game": self.game
            # ... any other attributes you need to save ...
        }

        # Only include _envargs if it has been initialized
        if hasattr(self, '_envargs'):
            state["_envargs"] = self._envargs
        return state
    
    def set_state(self, state):
        self.shuffle_obs = state["shuffle_obs"]
        self.all_games = state["all_games"]
        self.text_manual = state["text_manual"]
        self.init_states = state["init_states"]
        self.game_variants = state["game_variants"]
        self.notable_sprites = state["notable_sprites"]
        self.game = state["game"]
        self.variant=state["variant"]

        # Only set _envargs if it's in the state dictionary
        if "_envargs" in state:
            self._envargs = state["_envargs"]
            # If you have any post-state restoration setup or checks, add them here.
            # For example, if you need to reinitialize the environment using _envargs:
            self.env = VGDLEnv(**self._envargs)
        # ... restore any other attributes you saved ...
    
    def deep_copy(self,new_obj,newTask=False):
        state = self.get_state()
        new_obj.set_state(state)
        new_obj.newTask=newTask
        # print(new_obj.newTask)
        return new_obj

    def _get_variant(self, variant_file:Path) -> GameVariant:
        '''
        Return the GameVariant for the variant specified by variant_file. 
        Searches through the vgdl code to find the correct type:
        {chaser, fleeing, immovable}
        '''

        code = variant_file.read_text()
        return GameVariant(
            path = str(variant_file),
            enemy_type = re.search(r'enemy > (\S+)', code)[1].lower(),
            message_type = re.search(r'message > (\S+)', code)[1].lower(),
            goal_type = re.search(r'goal > (\S+)', code)[1].lower(),
            decoy_message_type = re.search(r'decoy_message > (\S+)', code)[1].lower(),
            decoy_goal_type = re.search(r'decoy_goal > (\S+)', code)[1].lower()
        )

    def _convert_obs(self, vgdl_obs):
        '''
        Return a grid built from the vgdl observation which is a
        KeyValueObservation object (see vgdl code for details).
        '''
        entity_locs = Grid(layers=5, shuffle=self.shuffle_obs)
        avatar_locs = Grid(layers=1)
        self.stateFrame={}
        
        # try to add each entity one by one, if it's not there move on.
        if 'enemy.1' in vgdl_obs:
            entity_locs.add(self.game.enemy, Position(*vgdl_obs['enemy.1']['position']))
            x,y=vgdl_obs['enemy.1']["position"]
            self.stateFrame["enemy"]={}
            self.stateFrame["enemy"]["pos"]=(x,9-y)
            self.stateFrame["enemy"]["e"]=self.game.enemy.name
            self.stateFrame["enemy"]["type"]=self.entity_type["enemy"]
            
        if 'message.1' in vgdl_obs:
            entity_locs.add(self.game.message, Position(*vgdl_obs['message.1']['position']))
            
            x,y=vgdl_obs['message.1']["position"]
            if(self.newTask):
                self.stateFrame["goal"]={}
                self.stateFrame["goal"]["e"]=self.game.message.name
                self.stateFrame["goal"]["pos"]=(x,9-y)
                self.stateFrame["goal"]["type"]=self.entity_type["message"]
            else:
                self.stateFrame["message"]={}
                self.stateFrame["message"]["e"]=self.game.message.name
                self.stateFrame["message"]["type"]=self.entity_type["message"]
                self.stateFrame["message"]["pos"]=(x,9-y)
            
        else:
            # advance the entity counter, Oracle model requires special order.
            # TODO: maybe used named layers to make this more understandable.
            entity_locs.entity_count += 1
        if 'goal.1' in vgdl_obs:
            entity_locs.add(self.game.goal, Position(*vgdl_obs['goal.1']['position']))
            
            x,y=vgdl_obs['goal.1']["position"]
            if(self.newTask):
                self.stateFrame["message"]={}
                self.stateFrame["message"]["e"]=self.game.goal.name
                self.stateFrame["message"]["pos"]=(x,9-y)
                self.stateFrame["message"]["type"]=self.entity_type["goal"]
            else:
                self.stateFrame["goal"]={}
                self.stateFrame["goal"]["e"]=self.game.goal.name
                self.stateFrame["goal"]["pos"]=(x,9-y)
                self.stateFrame["goal"]["type"]=self.entity_type["goal"]
           
        if 'decoy_message.1' in vgdl_obs:
            entity_locs.add(self.game.message, Position(*vgdl_obs['decoy_message.1']['position']))
            self.stateFrame["decoy_message"]={}
            x,y=vgdl_obs['decoy_message.1']["position"]
            self.stateFrame["decoy_message"]["pos"]=(x,9-y)
            self.stateFrame["decoy_message"]["e"]=self.game.message.name
            self.stateFrame["decoy_message"]["type"]=self.entity_type["decoy_message"]
           
        if 'decoy_goal.1' in vgdl_obs:
            entity_locs.add(self.game.goal, Position(*vgdl_obs['decoy_goal.1']['position']))
            self.stateFrame["decoy_goal"]={}
            x,y=vgdl_obs['decoy_goal.1']["position"]
            self.stateFrame["decoy_goal"]["pos"]=(x,9-y)
            self.stateFrame["decoy_goal"]["e"]=self.game.goal.name
            self.stateFrame["decoy_goal"]["type"]=self.entity_type["decoy_goal"]
           
        if 'no_message.1' in vgdl_obs:
            '''
            Due to a quirk in VGDL, the avatar is no_message if it starts as no_message
            even if the avatar may have acquired the message at a later point.
            To check, if it has a message, check that the class vector corresponding to
            with_message is == 1.
            '''
            avatar_pos = Position(*vgdl_obs['no_message.1']['position'])
            # with_key is last position, see self.notable_sprites
            if vgdl_obs['no_message.1']['class'][-1] == 1:
                avatar = config.WITH_MESSAGE
            else:
                avatar = config.NO_MESSAGE
            self.stateFrame["agent"]={}
            x,y=vgdl_obs['no_message.1']["position"]
            self.stateFrame["agent"]["pos"]=(x,9-y)
            self.stateFrame["agent"]["e"]="without_Message"
        elif "with_message.1" in vgdl_obs:
            # this case only occurs if avatar begins as with_message at start of episode
            avatar_pos = Position(*vgdl_obs['with_message.1']['position'])
            avatar = config.WITH_MESSAGE
            self.stateFrame["agent"]={}
            x,y=vgdl_obs['with_message.1']["position"]
            self.stateFrame["agent"]["pos"]=(x,9-y)
            self.stateFrame["agent"]["e"]="with_Message"
        else: # the avatar is not in observation, so is probably dead
            return {"entities": entity_locs.grid, "avatar": avatar_locs.grid}

        avatar_locs.add(avatar, avatar_pos) # if not dead, add it.

        return {"entities": entity_locs.grid, "avatar": avatar_locs.grid}


    def reset(self,seed=None,variant_id:int=None, **kwargs):
        '''
        Resets the current environment. NOTE: We remake the environment each time.
        This is a workaround to a bug in py-vgdl, where env.reset() does not
        properly reset the environment. kwargs go to get_document().
        '''
        if(seed !=None):
            random.seed(seed)
            np.random.seed(seed)
        # Fix some variants for simplicity, however the complexity is kept for the init state diversities and character entities diversities.
        if(self.game==None):
            self.game=self.all_games[15]
        self.variant=None
        if(self.variant==None):
            if variant_id is not None:
                self.variant = self.game_variants[variant_id]
            else:
                self.variant = random.choice(self.game_variants)
                # print(self.game_variants)
        self.entity_type={"enemy":self.variant[1],"message":self.variant[2],"goal":self.variant[3],"decoy_message":self.variant[4],"decoy_goal":self.variant[5]}
        init_state = random.choice(self.init_states) # inital state file
        # init_state=self.init_states[10]
        self._envargs = {
            'game_file': self.variant.path,
            'level_file': init_state,
            'notable_sprites': self.notable_sprites.copy(),
            'obs_type': 'objects', # track the objects
            'seed':1,
            'block_size': 34,  # rendering block size,
        }
        
        self.env = VGDLEnv(**self._envargs)
        self.env.seed(seed)
        self.vgdl_obs = self.env.reset()
        
        all_npcs_for_manual = (
            Descr(entity=self.game.enemy.name, role='enemy', type=self.variant.enemy_type),
            Descr(entity=self.game.message.name, role='message', type=self.variant.message_type),
            Descr(entity=self.game.goal.name, role='goal', type=self.variant.goal_type),
            Descr(entity=self.game.message.name, role='enemy', type=self.variant.decoy_message_type),
            Descr(entity=self.game.goal.name, role='enemy', type=self.variant.decoy_goal_type),
        )
        
        manual = self.text_manual.get_document_plus(*all_npcs_for_manual, **kwargs)
        return self._convert_obs(self.vgdl_obs), manual

    def step(self, action):
        self.vgdl_obs, reward, done, info = self.env.step(action)
        return self._convert_obs(self.vgdl_obs), reward, done, info