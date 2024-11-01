from typing import Dict, Optional, List, Tuple
import gym
from gym import spaces
from collections import defaultdict
from enum import IntEnum
import random
import numpy as np

from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw

from homegrid.base import MiniGridEnv, Grid, Storage, Inanimate, Pickable
from homegrid.layout import ThreeRoom, CANS, TRASH, room2name
import json

def get_disturbed_hind(action_failed_reason):
    if action_failed_reason == "You are doing well so far.":
        disturbed_action_failed_reason = "no, turn around"
    else:
        disturbed_action_failed_reason = "You are doing well so far."
    return disturbed_action_failed_reason

class HomeGridBase(MiniGridEnv):

    class Actions(IntEnum):
        left = 0
        right = 1
        up = 2
        down = 3
        # item actions
        pickup = 4
        drop = 5
        # storage actions
        get = 6
        pedal = 7
        grasp = 8
        lift = 9

    ac2dir = {
        Actions.right: 0,
        Actions.down: 1,
        Actions.left: 2,
        Actions.up: 3,
    }

    def __init__(
        self,
        layout=ThreeRoom,
        num_trashcans=2,
        num_trashobjs=2,
        view_size=3,
        max_steps=50,
        p_teleport=0.0,
        max_objects=4,
        p_unsafe=0.0,
        fixed_state=None,
        gpt_pool=False,
        train_ratio=None,
        mode=None,
        val_ratio=None,
        disturb_hind=False,
        disturb_fore=False,
    ):
        self.layout = layout()
        self.textures = self.layout.textures
        super().__init__(
            width=self.layout.width,
            height=self.layout.height,
            render_mode="rgb_array",
            agent_view_size=view_size,
            max_steps=max_steps,
        )
        self.actions = HomeGridBase.Actions
        self.action_space = spaces.Discrete(len(self.actions))
        self.num_trashcans = num_trashcans
        self.num_trashobjs = num_trashobjs
        self.p_teleport = p_teleport
        self.max_objects = max_objects
        self.p_unsafe = p_unsafe
        self.fixed_state = fixed_state
        self.seed = None
        self.gpt_pool = gpt_pool
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.mode = mode
        self.disturb_hind = disturb_hind
        self.disturb_fore = disturb_fore
        assert (self.train_ratio and self.mode) or (
            not self.train_ratio and not self.mode
        )
        assert train_ratio + val_ratio < 1
        
        template_path = "./homegrid/template.json"
        with open(template_path) as f:
            self.template = json.load(f)

    @property
    def step_cnt(self):
        return self._step_cnt

    def init_from_state(self, state):
        """Initialize the env from a symbolic state."""
        self._create_layout(self.width, self.height)
        # place agent
        self.agent_pos = state["agent"]["pos"]
        self.agent_dir = state["agent"]["dir"]
        print(self.agent_pos, self.agent_dir)
        self.objs = []
        # place objects with appropriate state
        for ob in state["objects"]:
            if ob["pos"] == (-1, -1):
                print(f"Skipping carried object {ob['name']}")
                continue
            if ob["type"] == "Storage":
                pfx = ob["name"].replace(" ", "_")
                obj = Storage(
                    name=ob["name"],
                    textures={
                        "open": self.textures[f"{pfx}_open"],
                        "closed": self.textures[f"{pfx}_closed"],
                    },
                    state=ob["state"],
                    action=ob["action"],
                    contains=ob["contains"],
                )
            elif ob["type"] == "Pickable":
                obj = Pickable(
                    name=ob["name"],
                    texture=self.textures[ob["name"]],
                    invisible=ob["invisible"],
                )
            else:
                raise NotImplementedError("Obj type {ob['type']}")
            self.objs.append(obj)
            self.place_obj(obj, top=ob["pos"], size=(1, 1), max_tries=1)

    def _add_cans_to_house(self, seed=None):
        if seed:
            random.seed(seed)
        cans = random.sample(CANS, self.num_trashcans)
        poss = random.sample(self.layout.valid_poss["can"], self.num_trashcans)
        can_objs = []
        for i, can in enumerate(cans):
            obj = Storage(
                can,
                {
                    "open": self.textures[f"{can}_open"],
                    "closed": self.textures[f"{can}_closed"],
                },
                # Make one of the cans irreversibly broken
                reset_broken_after=200 if i == 0 else 1,
                seed=seed,
            )
            pos = self.place_obj(obj, top=poss[i], size=(1, 1), max_tries=5)
            can_objs.append(obj)
            self.objs.append(obj)

    def _add_objs_to_house(self, seed=None):
        if seed:
            random.seed(seed)
        trash_objs = random.sample(TRASH, self.num_trashobjs)
        poss = random.sample(self.layout.valid_poss["obj"], self.num_trashobjs)
        trashobj_objs = []
        for i, trash in enumerate(trash_objs):
            obj = Pickable(trash, self.textures[trash])
            pos = self.place_obj(obj, top=poss[i], size=(1, 1), max_tries=5)
            trashobj_objs.append(obj)
            self.objs.append(obj)

    def _gen_grid(self, width, height, seed=None):
        if seed:
            random.seed(seed)
        if self.fixed_state:
            print("Initializing from fixed state")
            self.init_from_state(self.fixed_state)
            return
        regenerate = True
        while regenerate:
            self._create_layout(width, height)
            regenerate = False

            self.objs = []
            self.goal = {"obj": None, "can": None}
            # Place objects
            self._add_cans_to_house(seed=seed)
            self._add_objs_to_house(seed=seed)

        # Place agent
        agent_poss = random.choice(self.layout.valid_poss["agent_start"])
        self.agent_pos = self.place_agent(top=agent_poss, size=(1, 1))

    def _create_layout(self, width, height):
        # Create grid with surrounding walls
        self.grid = Grid(width, height)
        self.layout.populate(self.grid)
        self.room_to_cells = self.layout.room_to_cells
        self.cell_to_room = self.layout.cell_to_room

    def _maybe_teleport(self, seed=None):
        if seed:
            random.seed(seed)
        if np.random.random() > self.p_teleport:
            return False
        objs = [
            o
            for o in self.objs
            if isinstance(o, Pickable) and o.cur_pos[0] != -1 and not o.invisible
        ]
        if len(objs) == 0:
            print(self.objs)
            print([o.cur_pos for o in self.objs])
            print(self.all_events)
            return False
        obj = random.choice(objs)
        # Choose a random new location with no object to place this
        poss = random.choice(
            [
                pos
                for pos in self.layout.valid_poss["obj"]
                if pos != obj.cur_pos
                and self.grid.get(*pos) is None
                and pos != self.agent_pos
            ]
        )
        self.grid.set(*obj.cur_pos, None)
        obj.cur_pos = poss
        self.grid.set(*poss, obj)
        return obj

    def _maybe_spawn(self, seed=None):
        if seed:
            random.seed(seed)
        new_objs = [t for t in TRASH if t not in [o.name for o in self.objs]]
        # no spawn in our project
        # if np.random.rand() < 0.1 * len(new_objs):
        #   trash = random.choice(new_objs)
        #   obj = Pickable(trash, self.textures[trash], invisible=True)
        #   poss = random.choice([
        #     pos for pos in self.layout.valid_poss["obj"] \
        #     if pos != obj.cur_pos and self.grid.get(*pos) is None \
        #     and pos != self.agent_pos])
        #   self.place_obj(obj, top=poss, size=(1,1), max_tries=5)
        #   self.objs.append(obj)
        #   return obj
        return None

    def _maybe_unsafe(self, seed=None):
        if seed:
            random.seed(seed)
        if len(self.unsafe_poss) == 0 and np.random.rand() < self.p_unsafe:
            can = random.choice([o for o in self.objs if isinstance(o, Storage)])
            self.unsafe_poss = set()
            self.unsafe_name = can.name
            for x in [can.cur_pos[0] - 1, can.cur_pos[0], can.cur_pos[0] + 1]:
                for y in [can.cur_pos[1] - 1, can.cur_pos[1], can.cur_pos[1] + 1]:
                    self.unsafe_poss.add((x, y))
            self.unsafe_end = self.step_count + random.randint(1, 10)
            return can
        return None

    def reset(self, *, seed=None, options=None):
        self.prev_action = "Reset"
        self.seed = seed
        obs, info = super().reset(seed=seed, options=options)
        # All events in the episode so far
        self.all_events = []
        self.step_count = 0
        self.unsafe_poss = {}
        self.unsafe_end = -1
        self.unsafe_name = None
        info = {"symbolic_state": self.get_full_symbolic_state(), "events": []}
        return obs, info

    def step(self, action):
        self.step_count += 1

        reward = 0
        terminated = False
        truncated = False
        success = None
        events = []
        if_action_failed = False
        action_failed_reason = None

        # Step all the object states
        for obj in self.objs:
            obj.tick()

        # Get the position in front of the agent
        fwd_pos = self.front_pos

        # Get the contents of the cell in front of the agent
        fwd_cell = self.grid.get(*fwd_pos)
        fwd_floor = self.grid.get_floor(*fwd_pos)

        if (
            action == self.actions.left
            or action == self.actions.right
            or action == self.actions.up
            or action == self.actions.down
        ):
            self.agent_dir = HomeGridBase.ac2dir[action]
            # Get the position in front of the agent after turning
            fwd_pos = self.front_pos

            # Get the contents of the cell in front of the agent
            fwd_cell = self.grid.get(*fwd_pos)
            fwd_floor = self.grid.get_floor(*fwd_pos)

            if (fwd_cell is None or fwd_cell.agent_can_overlap()) and (
                fwd_floor is None or fwd_floor.agent_can_overlap()
            ):
                self.agent_pos = tuple(fwd_pos)
            else:
                if_action_failed = True
                action_failed_reason = f"Agent cannot overlap on invalid grid."
        # Pick up an object
        elif action == self.actions.pickup:
            if isinstance(fwd_cell, Pickable) and fwd_cell.can_pickup():
                if self.carrying is None:
                    self.carrying = fwd_cell
                    self.carrying.cur_pos = (-1, -1)
                    self.grid.set(fwd_pos[0], fwd_pos[1], None)
                else:
                    if_action_failed = True
                    action_failed_reason = f"Cannot carry more than one object."
            elif fwd_cell is None:
                if_action_failed = True
                action_failed_reason = f"There is no front object."
            else:
                if_action_failed = True
                action_failed_reason = f"Front object is not a Pickable object."

        # Drop an object
        elif action == self.actions.drop:
            if not fwd_cell and self.carrying:
                self.grid.set(fwd_pos[0], fwd_pos[1], self.carrying)
                self.carrying.cur_pos = tuple(fwd_pos)
                self.carrying = None
            elif isinstance(fwd_cell, Storage) and self.carrying:
                succeeded, reason = fwd_cell.interact(
                    self.actions(action).name, obj=self.carrying
                )
                if succeeded:
                    self.carrying = None
                else:
                    if_action_failed = True
                    action_failed_reason = reason
            elif not self.carrying:
                if_action_failed = True
                action_failed_reason = f"No object carrying to drop."
            else:
                if_action_failed = True
                action_failed_reason = (
                    f"Cannot drop objects where there exists another object."
                )
        elif action == self.actions.get:
            if not self.carrying and isinstance(fwd_cell, Storage):
                obj, reason = fwd_cell.interact(self.actions(action).name)
                if obj:
                    self.carrying = obj
                    self.carrying.cur_pos = (-1, -1)
                elif reason:
                    if_action_failed = True
                    action_failed_reason = reason
                else:
                    if_action_failed = True
                    action_failed_reason = f"There is no object in Storage object."
            elif fwd_cell is None:
                if_action_failed = True
                action_failed_reason = f"There is no front object."
            elif not isinstance(fwd_cell, Storage):
                if_action_failed = True
                action_failed_reason = f"Front object is not a Storage object."
            else:
                if_action_failed = True
                action_failed_reason = f"Cannot carry more than one object."

        elif (
            action == self.actions.pedal
            or action == self.actions.grasp
            or action == self.actions.lift
        ):
            if isinstance(fwd_cell, Storage):
                succeeded, reason = fwd_cell.interact(self.actions(action).name)
                if not succeeded:
                    if_action_failed = True
                    action_failed_reason = reason
            elif fwd_cell is None:
                if_action_failed = True
                action_failed_reason = f"There is no front object."
            else:
                if_action_failed = True
                action_failed_reason = f"Front object is not a Storage object."

        else:
            raise ValueError(f"Unknown action: {action}")

        if self.step_count >= self.max_steps:
            truncated = True
        if (terminated or truncated) and success is None:
            success = False

        if self.render_mode == "human":
            self.render_with_text()
        obs = self.gen_obs()

        # For rendering purposes
        self.prev_action = HomeGridBase.Actions(action).name
        if success:
            self.done_condition = "success"
        elif truncated:
            self.done_condition = "truncated"

        # Random events
        if self.agent_pos in self.unsafe_poss:
            terminated = True
            reward = -1

        if len(self.unsafe_poss) > 0 and self.step_count == self.unsafe_end:
            self.unsafe_poss = {}
            self.unsafe_name = None
            self.unsafe_end = -1
            events.append(
                {
                    "type": "termination",
                    "description": f"i cleaned the spill",
                }
            )
        elif len(self.unsafe_poss) == 0:
            obj = self._maybe_unsafe(seed=self.seed)
            if obj:
                events.append(
                    {
                        "type": "termination",
                        "description": f"spill near the {self.unsafe_name}",
                    }
                )
        else:
            events.append(
                {
                    "type": "termination",
                    "description": f"spill near the {self.unsafe_name}",
                }
            )
        obj = self._maybe_teleport(seed=self.seed)
        if obj:
            room_code = self.cell_to_room[obj.cur_pos]
            room_name = room2name[room_code]
            events.append(
                {
                    "type": "future",
                    "obj": obj,
                    "room": room_code,
                    "description": f"i moved the {obj.name} to the {room_name}",
                }
            )
        obj = self._maybe_spawn(seed=self.seed)
        if obj:
            room_code = self.cell_to_room[obj.cur_pos]
            room_name = room2name[room_code]
            events.append(
                {
                    "type": "future",
                    "obj": obj,
                    "description": f"there will be {obj.name} in the {room_name} later",
                }
            )

        if action_failed_reason is None:
            action_failed_reason = "You are doing well so far."
        self.all_events.append(events)

        if self.gpt_pool:
            if not self.mode:
                action_failed_reason = np.random.choice(
                    self.template[action_failed_reason]
                )
            elif self.mode == "train":
                length = len(self.template[action_failed_reason])
                if self.disturb_hind:
                    action_failed_reason = get_disturbed_hind(action_failed_reason)
                action_failed_reason = np.random.choice(
                    self.template[action_failed_reason][
                        : int(length * self.train_ratio)
                    ]
                )
            elif self.mode == "test":
                length = len(self.template[action_failed_reason])
                if self.disturb_hind:
                    action_failed_reason = get_disturbed_hind(action_failed_reason)
                action_failed_reason = np.random.choice(
                    self.template[action_failed_reason][
                        int(length * (self.train_ratio + self.val_ratio)) :
                    ]
                )
            elif self.mode == "val":
                length = len(self.template[action_failed_reason])
                if self.disturb_hind:
                    action_failed_reason = get_disturbed_hind(action_failed_reason)
                action_failed_reason = np.random.choice(
                    self.template[action_failed_reason][
                        int(length * self.train_ratio) : int(
                            length * (self.train_ratio + self.val_ratio)
                        )
                    ]
                )

        info = {
            "success": success,
            "action": action,
            "symbolic_state": self.get_full_symbolic_state(),
            "events": events,
            "all_events": self.all_events,
            "action_status": {
                "if_action_failed": if_action_failed,
                "action_failed_reason": action_failed_reason,
            },
        }

        return obs, reward, terminated, truncated, info

    def get_full_symbolic_state(self) -> Dict:
        fwd_pos = self.front_pos
        fwd_cell = self.grid.get(*fwd_pos)
        if isinstance(fwd_cell, Pickable) or isinstance(fwd_cell, Storage):
            front_obj = fwd_cell.name
        else:
            front_obj = None

        state = {
            "step": self.step_count,
            "agent": {
                "pos": self.agent_pos,
                "room": (
                    self.cell_to_room[self.agent_pos]
                    if self.agent_pos in self.cell_to_room
                    else None
                ),
                "dir": self.agent_dir,
                "carrying": self.carrying.name if self.carrying else None,
            },
            "objects": [
                {
                    "name": obj.name,
                    "type": obj.__class__.__name__,
                    "pos": obj.cur_pos,
                    "room": (
                        self.cell_to_room[obj.cur_pos]
                        if (obj.cur_pos[0] != -1 and obj.cur_pos in self.cell_to_room)
                        else None
                    ),
                    "state": getattr(obj, "state", None),
                    "action": getattr(obj, "action", None),
                    "invisible": getattr(obj, "invisible", None),
                    "contains": (
                        [contained_obj.name for contained_obj in obj.contains]
                        if isinstance(obj, Storage)
                        else None
                    ),
                }
                for obj in self.objs
            ],
            "front_obj": front_obj,
            "unsafe": {
                "name": self.unsafe_name,
                "poss": self.unsafe_poss,
                "end": self.unsafe_end,
            },
        }
        return state

    def render_with_text(self, text):
        img = self._env.render(mode="rgb_array")
        img = Image.fromarray(img)
        draw = ImageDraw.Draw(img)
        draw.text((0, 0), text, (0, 0, 0))
        draw.text((0, 45), "Action: {}".format(self._env.prev_action), (0, 0, 0))
        img = np.asarray(img)
        return img
