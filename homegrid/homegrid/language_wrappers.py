from enum import Enum
import random
import pathlib
import pickle

import gym
from gym import spaces
import numpy as np
from tokenizers import Tokenizer

from homegrid.base import Pickable, Storage
from homegrid.layout import room2name
from homegrid.planer import Planer, valid_position

def get_disturbed_hind(action_failed_reason):
    if action_failed_reason == "You are doing well so far.":
        disturbed_action_failed_reason = "no, turn around"
    else:
        disturbed_action_failed_reason = "You are doing well so far."
    return disturbed_action_failed_reason


def get_disturbed_action():
    return random.choice(["go left", "go right", "go up", "go down", "pick up", "drop", "pedal", "grasp", "lift"])

def get_disturbed_object():
    return random.choice(["cupboard",
	"stove",
	"fridge",
	"countertop",
	"chairl",
	"chairr",
	"table",
	"sofa",
	"sofa_side",
	"rugl",
	"rugr",
	"coffeetable",
	"cabinet",
	"plant",	
    "cupboard",
	"stove",
	"countertop",
	"chairl",
	"chairr",
	"table",
	"sofa",
	"sofa_side",
	"rugl",
	"rugr",
	"coffeetable",
    "bottle",
	"fruit",
	"papers",
	"plates",
	"tomato",
 	"recycling bin",
	"trashbin",
	"compost bin",
	"fridge bin",])

def get_disturbed_room():
    return random.choice(["kitchen", "living room", "dining room"])

class MultitaskWrapper(gym.Wrapper):
    """Continually sample tasks during an episode, rewarding the agent for
    completion."""

    Tasks = Enum("Tasks", ["find", "get", "cleanup", "rearrange", "open"], start=0)

    def __init__(self, env):
        super().__init__(env)
        self.tasks = list(MultitaskWrapper.Tasks)

    def sample_task(self, seed=None):
        if seed:
            random.seed(seed)
        task_type = random.choice(self.tasks)

        if task_type == MultitaskWrapper.Tasks.find:
            obj_name = random.choice(self.env.objs).name
            task = f"find the {obj_name}"

            def reward_fn(symbolic_state, info):
                return int(symbolic_state["front_obj"] == obj_name)

        elif task_type == MultitaskWrapper.Tasks.get:
            obj_name = random.choice(
                [ob for ob in self.env.objs if isinstance(ob, Pickable)]
            ).name
            task = f"get the {obj_name}"

            def reward_fn(symbolic_state, info):
                return int(symbolic_state["agent"]["carrying"] == obj_name)

        elif task_type == MultitaskWrapper.Tasks.open:
            obj_name = random.choice(
                [ob for ob in self.env.objs if isinstance(ob, Storage)]
            ).name
            task = f"open the {obj_name}"

            def reward_fn(symbolic_state, info):
                for obj in symbolic_state["objects"]:
                    if obj["name"] == obj_name:
                        return int(obj["state"] == "open")

        elif task_type == MultitaskWrapper.Tasks.cleanup:
            obj_name = random.choice(
                [ob for ob in self.env.objs if isinstance(ob, Pickable)]
            ).name
            bin_name = random.choice(
                [ob for ob in self.env.objs if isinstance(ob, Storage)]
            ).name
            task = f"put the {obj_name} in the {bin_name}"

            def reward_fn(symbolic_state, info):

                if symbolic_state["agent"]["carrying"] == obj_name:
                    info["if_carry_plate"] = ""
                    info["bin_status"] = "closed"
                    return 0.5
                else:
                    info["if_carry_plate"] = "not"
                    info["bin_status"] = "closed"
                for obj in symbolic_state["objects"]:
                    if obj["name"] == bin_name:
                        if obj["state"] == "open":
                            info["bin_status"] = "open"
                        else:
                            info["bin_status"] = "closed"
                        return int(obj_name in obj["contains"])

        elif task_type == MultitaskWrapper.Tasks.rearrange:
            room_code = random.choice(list(self.env.room_to_cells.keys()))
            obj_name = random.choice(
                [ob for ob in self.env.objs if isinstance(ob, Pickable)]
            ).name
            task = f"move the {obj_name} to the {room2name[room_code]}"

            def reward_fn(symbolic_state, info):
                if symbolic_state["agent"]["carrying"] == obj_name:
                    return 0.5
                for obj in symbolic_state["objects"]:
                    if obj["name"] == obj_name:
                        return int(obj["room"] == room_code)

        else:
            raise ValueError(f"Unknown task type {task_type}")

        def dist_goal(symbolic_state):
            goal_name = obj_name
            if task_type == MultitaskWrapper.Tasks.cleanup:
                goal_name = (
                    bin_name
                    if symbolic_state["agent"]["carrying"] == obj_name
                    else obj_name
                )
            pos = [o for o in symbolic_state["objects"] if o["name"] == goal_name][0][
                "pos"
            ]
            if (
                task_type == MultitaskWrapper.Tasks.rearrange
                and symbolic_state["agent"]["carrying"] == obj_name
            ):
                if room_code == "K":
                    pos = [5, 8]
                elif room_code == "L":
                    pos = [9, 4]
                elif room_code == "D":
                    pos = [9, 7]
                else:
                    raise NotImplementedError
            need_action_list = Planer.bfs_actions_to_adjacent(self.agent_pos, pos, valid_position)
            if need_action_list is None:
                return 100
            else:
                dist = len(need_action_list)
            return dist

        self.task = task
        self.reward_fn = reward_fn
        self.dist_goal = dist_goal
        self.subtask_done = False
        self.start_step = self.step_cnt
        self.flag = False

    def reset(self, seed=None):
        if seed:
            random.seed(seed)
        obs, info = self.env.reset(seed=seed)
        self.step_cnt = 0
        self.start_step = 0
        self.accomplished_tasks = []
        self.task_times = []
        self.sample_task(seed=seed)
        info.update(
            {
                "log_timesteps_with_task": self.step_cnt - self.start_step,
                "log_new_task": True,
                "log_dist_goal": self.dist_goal(info["symbolic_state"]),
            }
        )
        return obs, info

    def step(self, action):
        self.step_cnt += 1
        obs, rew, term, trunc, info = self.env.step(action)
        info.update(
            {
                "log_timesteps_with_task": self.step_cnt - self.start_step,
                "log_new_task": False,
                "log_dist_goal": self.dist_goal(info["symbolic_state"]),
            }
        )
        if term:
            return obs, rew, term, trunc, info
        rew = self.reward_fn(info["symbolic_state"], info)
        if rew == 1:
            self.accomplished_tasks.append(self.task)
            self.task_times.append(self.step_cnt - self.start_step)
            self.sample_task()
            info.update(
                {
                    "log_timesteps_with_task": self.step_cnt - self.start_step,
                    "log_accomplished_tasks": self.accomplished_tasks,
                    "log_task_times": self.task_times,
                    "log_new_task": True,
                    "log_dist_goal": self.dist_goal(info["symbolic_state"]),
                }
            )
        elif rew == 0.5:
            if self.subtask_done:
                rew = 0  # don't reward twice
            self.subtask_done = True
        info["success"] = rew == 1
        term = rew == 1
        return obs, rew, term, trunc, info


class LanguageWrapper(gym.Wrapper):
    """Provide the agent with language information one token at a time, using underlying
    environment state and task wrapper.

    Configures types of language available, and specifies logic for which language is provided at
    a given step, if multiple strings are available."""

    def __init__(
        self,
        env,
        # Max # tokens during prereading phase (for future/dynamics)
        preread_max=-1,
        # How often to repeat the task description
        repeat_task_every=20,
        # Prob of sampling descriptions when we don't have task language
        p_language=1,
        debug=False,
        lang_types=["task", "future", "dynamics", "corrections", "termination"],
        train_ratio=None,
        mode=None,
        
    ):
        super().__init__(env)
        assert (
            len(lang_types) >= 1 and "task" in lang_types
        ), f"Must have task language, {lang_types}"
        for t in lang_types:
            assert t in [
                "task",
                "future",
                "dynamics",
                "corrections",
                "termination",
            ], f"Unknown language type {t}"

        if "dynamics" in lang_types or "future" in lang_types:
            assert preread_max > -1, "Must have preread for dynamics/future language"

        self.instruction_only = len(lang_types) == 1 and lang_types[0] == "task"
        self.preread_max = preread_max
        self.repeat_task_every = repeat_task_every
        self.p_language = p_language
        self.debug = debug
        self.lang_types = lang_types
        self.preread = -1 if self.instruction_only else self.preread_max

        self.preread = -1
        self.train_ratio = train_ratio
        self.mode = mode
        
        assert (self.train_ratio and self.mode) or (
            not self.train_ratio and not self.mode
        )

        directory = pathlib.Path(__file__).resolve().parent
        with open(directory / "homegrid_embeds.pkl", "rb") as f:
            self.cache, self.embed_cache = pickle.load(f)
        self.empty_token = self.cache["<pad>"]
        # List of tokens of current utterance we're streaming
        self.tokens = [self.empty_token]
        # Index of self.tokens for current timestep
        self.cur_token = 0
        self.embed_size = 512
        self.observation_space = spaces.Dict(
            {
                **self.env.observation_space.spaces,
                "token": spaces.Box(0, 32100, shape=(), dtype=np.uint32),
                "token_embed": spaces.Box(
                    -np.inf, np.inf, shape=(self.embed_size,), dtype=np.float32
                ),
                "is_read_step": spaces.Box(
                    low=np.array(False), high=np.array(True), shape=(), dtype=bool
                ),
                "log_language_info": spaces.Text(
                    max_length=10000,
                ),
            }
        )
        if self.debug:
            self.tok = Tokenizer.from_pretrained("t5-small")

    def train_test_split(self, real_lang_list):
        if self.mode is None:
            return real_lang_list
        elif self.mode == "train":
            return real_lang_list[: int(len(real_lang_list) * self.train_ratio)]
        elif self.mode == "val":
            return real_lang_list[
                int(len(real_lang_list) * self.train_ratio) : int(
                    len(real_lang_list) * (self.train_ratio + self.val_ratio)
                )
            ]
        elif self.mode == "test":
            return real_lang_list[int(
                    len(real_lang_list) * (self.train_ratio + self.val_ratio)) :]
        else:
            raise NotImplementedError

    def get_descriptions(self, state, task=None):
        # facts:
        # - object locations (beginning only but also anytime)
        # - irreversible state (don't change)
        # - dynamics (don't change)
        descs = []
        agent_pos = state["agent"]["pos"]
        if task is None:
            for obj in state["objects"]:
                if "dynamics" in self.lang_types and obj["action"]:
                    if not self.gpt_pool:
                        if self.disturb_fore:
                            disturbed_action = get_disturbed_action()
                            disturbed_object = get_disturbed_object()
                            descs.append(
                                f"{disturbed_action.capitalize()} to open the {disturbed_object}."
                            )
                        else:
                            descs.append(
                                f"{obj['action'].capitalize()} to open the {obj['name']}."
                            )
                    else:
                        if self.disturb_fore:
                            disturbed_action = get_disturbed_action()
                            disturbed_object = get_disturbed_object()
                            descs.append(
                                random.choice(self.train_test_split(self.template["dynamic template"]))
                                .replace("{obj['name']}", disturbed_object)
                                .replace("{obj['action']}", disturbed_action)
                            )
                        else:
                            descs.append(
                                random.choice(self.train_test_split(self.template["dynamic template"]))
                                .replace("{obj['name']}", obj["name"])
                                .replace("{obj['action']}", obj["action"])
                            )
                if "future" in self.lang_types and obj["room"]:
                    if not self.gpt_pool:
                        if self.disturb_fore:
                            disturbed_object = get_disturbed_object()
                            disturbed_room = get_disturbed_room()
                            descs.append(
                                f"{disturbed_object.capitalize()} is in the {disturbed_room}."
                            )
                        else:
                            descs.append(
                                f"{obj['name'].capitalize()} is in the {room2name[obj['room']]}."
                            )
                    else:
                        if self.disturb_fore:
                            disturbed_object = get_disturbed_object()
                            disturbed_room = get_disturbed_room()
                            descs.append(
                                random.choice(self.train_test_split(self.template["future template"]))
                                .replace("{obj['name']}", disturbed_object)
                                .replace("{room2name[obj['room']]}", disturbed_room)
                            )
                        else:
                            descs.append(
                                random.choice(self.train_test_split(self.template["future template"]))
                                .replace("{obj['name']}", obj["name"])
                                .replace("{room2name[obj['room']]}", room2name[obj["room"]])
                            )
        else:
            for obj in state["objects"]:
                if obj["name"] in task:
                    dist = abs(agent_pos[0] - obj["pos"][0]) + abs(
                        agent_pos[1] - obj["pos"][1]
                    )
                    if "dynamics" in self.lang_types and obj["action"]:
                        # for openable object and agent is carrying correct object
                        if dist < 2 and (
                            task[0] == "o"
                            or (
                                state["agent"]["carrying"] is not None
                                and state["agent"]["carrying"] in task
                            )
                        ):
                            if obj["state"] in ["closed", "broken"]:
                                if not self.gpt_pool:
                                    if self.disturb_fore:
                                        disturbed_action = get_disturbed_action()
                                        disturbed_object = get_disturbed_object()
                                        descs.append(
                                            f"{disturbed_action.capitalize()} to open the {disturbed_object}."
                                        )
                                    else:
                                        descs.append(
                                            f"{obj['action'].capitalize()} to open the {obj['name']}."
                                        )
                                else:
                                    if self.disturb_fore:
                                        disturbed_action = get_disturbed_action()
                                        disturbed_object = get_disturbed_object()
                                        descs.append(
                                            random.choice(self.train_test_split(self.template["dynamic template"]))
                                            .replace("{obj['name']}", disturbed_object)
                                            .replace("{obj['action']}", disturbed_action)
                                        )
                                    else:
                                        descs.append(
                                            random.choice(self.train_test_split(self.template["dynamic template"]))
                                            .replace("{obj['name']}", obj["name"])
                                            .replace("{obj['action']}", obj["action"])
                                        )
                                # if just beside the bin, then must output this sentence
                            else:
                                if not self.gpt_pool:
                                    if self.disturb_fore:
                                        descs.append(f"Turn back.")
                                    else:
                                        descs.append(f"Drop the carried object.")
                                else:
                                    if self.disturb_fore:
                                        descs.append(
                                            random.choice(self.train_test_split(self.template["Turn back."]))
                                        )
                                    else:
                                        descs.append(
                                            random.choice(
                                                self.train_test_split(self.template["Drop the carried object."])
                                            )
                                        )
                            return descs, False
                    elif "dynamics" in self.lang_types:
                        # for pickable object
                        if dist < 2:
                            if not self.gpt_pool:
                                if self.disturb_fore:
                                    descs.append(f"Turn back.")
                                else:
                                    descs.append(f"Pick up the front object.")
                            else:
                                if self.disturb_fore:
                                    descs.append(
                                        random.choice(self.train_test_split(self.template["Turn back."]))
                                    )
                                else:
                                    descs.append(
                                        random.choice(self.train_test_split(self.template["Pick up the front object."]))
                                    )
                            # True here means "potentially" the agent will pick up the object
                            return descs, True

                    if "future" in self.lang_types and obj["room"]:
                        # when picking up the object, the next language has to be future
                        self.flag = False
                        if not self.gpt_pool:
                            if self.disturb_fore:
                                disturbed_object = get_disturbed_object()
                                disturbed_room = get_disturbed_room()
                                descs.append(
                                    f"{disturbed_object.capitalize()} is in the {disturbed_room}."
                                )
                            else:
                                descs.append(
                                    f"{obj['name'].capitalize()} is in the {room2name[obj['room']]}."
                                )
                        else:
                            if self.disturb_fore:
                                disturbed_object = get_disturbed_object()
                                disturbed_room = get_disturbed_room()
                                descs.append(
                                    random.choice(self.train_test_split(self.template["future template"]))
                                    .replace("{obj['name']}", disturbed_object)
                                    .replace("{room2name[obj['room']]}", disturbed_room)
                                )
                            else:
                                descs.append(
                                    random.choice(self.train_test_split(self.template["future template"]))
                                    .replace("{obj['name']}", obj["name"])
                                    .replace(
                                        "{room2name[obj['room']]}", room2name[obj["room"]]
                                    )
                                )
            if len(descs) == 0 and "move" in task:
                # that means the target is to the specific room
                if "kitchen" in task:
                    room_location = [5, 8]
                elif "living" in task:
                    room_location = [9, 4]
                elif "dining" in task:
                    room_location = [9, 7]
                else:
                    raise NotImplementedError
                dist = abs(agent_pos[0] - room_location[0]) + abs(
                    agent_pos[1] - room_location[1]
                )
                if dist < 2:
                    if not self.gpt_pool:
                        if self.disturb_fore:
                            descs.append(f"Turn back.")
                        else:
                            descs.append(f"Drop the carried object.")
                    else:
                        if self.disturb_fore:
                            descs.append(random.choice(self.train_test_split(self.template["Turn back."])))
                        else:
                            descs.append(
                                random.choice(self.train_test_split(self.template["Drop the carried object."]))
                            )
                    return descs, False

                if "kitchen" in task:
                    if not self.gpt_pool:
                        if self.disturb_fore:
                            descs.append(f"Go to the dining room.")
                        else:
                            descs.append(f"Go to the kitchen.")
                    else:
                        if self.disturb_fore:
                            descs.append(random.choice(self.train_test_split(self.template["Go to the dining room."])))
                        else:
                            descs.append(random.choice(self.train_test_split(self.template["Go to the kitchen."])))
                elif "living" in task:
                    if not self.gpt_pool:
                        if self.disturb_fore:
                            descs.append(f"Go to the kitchen.")
                        else:
                            descs.append(f"Go to the living room.")
                    else:
                        if self.disturb_fore:
                            descs.append(random.choice(self.train_test_split(self.template["Go to the kitchen."])))
                        else:
                            descs.append(random.choice(self.train_test_split(self.template["Go to the living room."])))
                elif "dining" in task:
                    if not self.gpt_pool:
                        if self.disturb_fore:
                            descs.append(f"Go to the living room.")
                        else:
                            descs.append(f"Go to the dining room.")
                    else:
                        if self.disturb_fore:
                            descs.append(random.choice(self.train_test_split(self.template["Go to the living room."])))
                        else:
                            descs.append(random.choice(self.train_test_split(self.template["Go to the dining room."])))
                else:
                    raise NotImplementedError

        return descs, False

    def _tokenize(self, string):
        if string in self.cache:
            return self.cache[string]
        if self.debug:
            return self.tok(string, add_special_tokens=False)["input_ids"]
        raise NotImplementedError(f"tokenize, string not preembedded: >{string}<")

    def _embed(self, string):
        if string in self.embed_cache:
            return self.embed_cache[string]
        if self.debug:
            return [5555] * len(self.tokens)
        raise NotImplementedError(f"embed, string not preembedded: >{string}<")

    def _set_current_string(self, string_or_strings):
        self.string = string_or_strings
        return

        # if isinstance(string_or_strings, list):
        #     self.string = " ".join(string_or_strings)
        #     self.tokens = [
        #         x for string in string_or_strings for x in self._tokenize(string)
        #     ]
        #     self.token_embeds = [
        #         x for string in string_or_strings for x in self._embed(string)
        #     ]
        #     self.cur_token = 0
        # elif isinstance(string_or_strings, str):
        #     string = string_or_strings
        #     self.string = string
        #     # self.tokens = self._tokenize(string)
        #     # self.token_embeds = self._embed(string)
        #     self.cur_token = 0

    def _increment_token(self):
        return
        # if self._lang_is_empty():
        #     return
        # self.cur_token += 1
        # # don't iterate over all tokens any more!!
        # self.cur_token = len(self.tokens)

        # if self.cur_token == len(self.tokens):
        #     self.string = "<pad>"
        #     self.tokens = [self.empty_token]
        #     self.token_embeds = [self._embed(self.string)]
        #     self.cur_token = 0

    def _lang_is_empty(self):
        return self.string == "<pad>"

    def add_language_to_obs(self, obs, info):
        """Adds language keys to the observation:
        - token (int): current token
        - token_embed (np.array): embedding of current token
        - log_language_info (str): human-readable info about language

        On each step, either
          describe new task (will interrupt other language)
          continue tokens that are currently being streamed
          repeat task if it's time
          describe something that changed or will happen (events)
          describe a fact (if not preread) - TODO
          correct the agent - TODO
        """
        if self._step_cnt >= self.preread and info["log_new_task"]:
            # on t=self._step_cnt, we will start streaming the new task
            self._set_current_string(self.env.task)

        describable_evts = []
        self.string = ""
        if (
            self.repeat_task_every > 0
            and self._step_cnt - self._last_task_repeat >= self.repeat_task_every
        ):
            self._set_current_string(self.env.task)
            self._last_task_repeat = self._step_cnt
        elif len(describable_evts) > 0:
            evt = random.choice(describable_evts)
            self._set_current_string(evt["description"])
        elif np.random.rand() < 1:
            if (
                "corrections" in self.lang_types
                and info["log_dist_goal"] > self.last_dist
                and not (
                    self.flag
                    and info["symbolic_state"]["agent"]["carrying"] is not None
                )
            ):
                if not self.gpt_pool:
                    if self.disturb_fore:
                        self._set_current_string("Go to the kitchen.")
                    else:
                        self._set_current_string("Turn back.")
                else:
                    if self.disturb_fore:
                        self._set_current_string(self.train_test_split(random.choice(self.template["Go to the kitchen."])))
                    else:
                        self._set_current_string(self.train_test_split(random.choice(self.template["Turn back."])))
            else:
                descs, self.flag = self.get_descriptions(
                    info["symbolic_state"], task=self.env.task
                )
                if len(descs) > 0:
                    self._set_current_string(random.choice(descs))

        obs.update(
            {
                "token": self.tokens[self.cur_token],
                # "token_embed": self.token_embeds[self.cur_token],
                "log_language_info": self.string,
            }
        )

        self._increment_token()
        return obs

    def reset(self, seed=None):
        if seed:
            random.seed(seed)
        obs, info = self.env.reset(seed=seed)
        obs["is_read_step"] = False
        self.last_dist = info["log_dist_goal"]
        self._step_cnt = 0
        self._last_task_repeat = 0
        obs = self.add_language_to_obs(obs, info)
        return obs, info

    def step(self, action):
        self._step_cnt += 1
        obs, rew, term, trunc, info = self.env.step(action)
        obs["is_read_step"] = False
        obs = self.add_language_to_obs(obs, info)
        self.last_dist = info["log_dist_goal"]
        if (
            obs["log_language_info"] == "Turn back."
            or obs["log_language_info"] in self.template["Turn back."]
        ):
            if not self.gpt_pool:
                if self.disturb_hind:
                    info["action_status"]["action_failed_reason"] = "You are doing well so far."
                else:
                    info["action_status"][
                        "action_failed_reason"
                    ] = "You have gone to the wrong direction."
            else:
                if self.disturb_hind:
                    info["action_status"]["action_failed_reason"] = random.choice(
                        self.train_test_split(self.template["You are doing well so far."])
                    )
                else:
                    info["action_status"]["action_failed_reason"] = random.choice(
                        self.train_test_split(self.template["You have gone to the wrong direction."])
                    )
        if rew == 1:
            info["action_status"]["action_failed_reason"] = ""
            if not self.gpt_pool:
                obs["log_language_info"] = "You complete the task."
            else:
                obs["log_language_info"] = random.choice(
                    self.train_test_split(self.template["You complete the task."])
                )
        return obs, rew, term, trunc, info
