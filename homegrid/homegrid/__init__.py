from gym.envs.registration import register
from homegrid.homegrid_base import HomeGridBase
from homegrid.language_wrappers import MultitaskWrapper, LanguageWrapper
from homegrid.wrappers import RGBImgPartialObsWrapper, FilterObsWrapper
# from generate_json import generate_init_json
import homegrid.utils as utils

import warnings
warnings.filterwarnings("ignore", module="gym.utils.passive_env_checker")
warnings.filterwarnings("ignore", module="gym.spaces.box")
import random

class HomeGrid:

    def __init__(self, lang_types, *args, **kwargs):
        env = HomeGridBase(*args, **kwargs)
        env = RGBImgPartialObsWrapper(env)
        env = FilterObsWrapper(env, ["image"])
        env = MultitaskWrapper(env)
        env = LanguageWrapper(
            env,
            preread_max=28,
            repeat_task_every=20,
            p_language=0.2,
            lang_types=lang_types,
            
        )
        self.env = env
        self.info = utils.generate_init_json()
        self.if_has_reset = False
    def __getattr__(self, name):
        return getattr(self.env, name)
    
    def add_init_state_info(self, info, terminated, truncated):
        cur_info = info['symbolic_state']
        cur_info['task_description'] = self.env.task
        cur_info['terminated'] = terminated
        cur_info['truncated'] = truncated
        self.info['init_state_info'] = cur_info
        
    def add_intermediate_info(self, info, terminated, truncated):
        cur_info = info['symbolic_state']
        cur_info['task_description'] = self.env.task
        cur_info['terminated'] = terminated
        cur_info['truncated'] = truncated
        cur_info['success'] = info['success']
        cur_info['action'] = utils.action_num2name[info['action']]
        cur_info['action_status'] = info['action_status']
        self.info['intermediate_info'].append(cur_info)

    def reset(self, seed=None, only_one_reset=True):
        if seed:
            random.seed(seed)
        obs, info = self.env.reset(seed)
        # evaluate requires multiple reset, but planer doesn't
        if only_one_reset:
            if not self.if_has_reset:
                self.add_init_state_info(info, False, False)
                self.if_has_reset = True
        else:
            self.add_init_state_info(info, False, False)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        self.add_intermediate_info(info, terminated, truncated)

        return obs, reward, terminated, truncated, info

register(
    id="homegrid-task",
    entry_point="homegrid:HomeGrid",
    kwargs={"lang_types": ["task"]},
)

register(
    id="homegrid-future",
    entry_point="homegrid:HomeGrid",
    kwargs={"lang_types": ["task", "future"]},
)

register(
    id="homegrid-dynamics",
    entry_point="homegrid:HomeGrid",
    kwargs={"lang_types": ["task", "dynamics"]},
)

register(
    id="homegrid-corrections",
    entry_point="homegrid:HomeGrid",
    kwargs={"lang_types": ["task", "corrections"]}
)
