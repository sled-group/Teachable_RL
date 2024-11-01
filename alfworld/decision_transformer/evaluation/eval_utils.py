import numpy as np
import torch
import re
import yaml

OBJECTS = [
    "NULL",
    "AlarmClock",
    "Apple",
    "ArmChair",
    "BaseballBat",
    "BasketBall",
    "Bathtub",
    "BathtubBasin",
    "Bed",
    "Blinds",
    "Book",
    "Boots",
    "Bowl",
    "Box",
    "Bread",
    "ButterKnife",
    "Cabinet",
    "Candle",
    "Cart",
    "CD",
    "CellPhone",
    "Chair",
    "Cloth",
    "CoffeeMachine",
    "CounterTop",
    "CreditCard",
    "Cup",
    "Curtains",
    "Desk",
    "DeskLamp",
    "DishSponge",
    "Drawer",
    "Dresser",
    "Egg",
    "FloorLamp",
    "Footstool",
    "Fork",
    "Fridge",
    "GarbageCan",
    "Glassbottle",
    "HandTowel",
    "HandTowelHolder",
    "HousePlant",
    "Kettle",
    "KeyChain",
    "Knife",
    "Ladle",
    "Laptop",
    "LaundryHamper",
    "LaundryHamperLid",
    "Lettuce",
    "LightSwitch",
    "Microwave",
    "Mirror",
    "Mug",
    "Newspaper",
    "Ottoman",
    "Painting",
    "Pan",
    "PaperTowel",
    "PaperTowelRoll",
    "Pen",
    "Pencil",
    "PepperShaker",
    "Pillow",
    "Plate",
    "Plunger",
    "Poster",
    "Pot",
    "Potato",
    "RemoteControl",
    "Safe",
    "SaltShaker",
    "ScrubBrush",
    "Shelf",
    "ShowerDoor",
    "ShowerGlass",
    "Sink",
    "SinkBasin",
    "SoapBar",
    "SoapBottle",
    "Sofa",
    "Spatula",
    "Spoon",
    "SprayBottle",
    "Statue",
    "StoveBurner",
    "StoveKnob",
    "DiningTable",
    "CoffeeTable",
    "SideTable",
    "TeddyBear",
    "Television",
    "TennisRacket",
    "TissueBox",
    "Toaster",
    "Toilet",
    "ToiletPaper",
    "ToiletPaperHanger",
    "ToiletPaperRoll",
    "Tomato",
    "Towel",
    "TowelHolder",
    "TVStand",
    "Vase",
    "Watch",
    "WateringCan",
    "Window",
    "WineBottle",
]


OBJ_MAP = {obj.lower(): i for i, obj in enumerate(OBJECTS)}

ACTIONS = ["look", "go", "take", "clean", "put", "heat", "open", "close", "inventory", "examine"]

ACTIONS_MAP = {action: i for i, action in enumerate(ACTIONS)}

MAX_IND = 30

STATE_DIM = 3
ACTION_DIM = len(ACTIONS) + 2 * len(OBJECTS) + 2 * MAX_IND


def object_one_hot_encode(obj):
    res = np.zeros(len(OBJECTS))
    res[OBJ_MAP[obj]] = 1
    return res


def action_one_hot_encode(action):
    res = np.zeros(len(ACTIONS))
    res[ACTIONS_MAP[action]] = 1
    return res


def index_one_hot_encode(ind):
    res = np.zeros(MAX_IND)
    res[ind] = 1
    return res


def parse_action(action_str):
    res_action, res_obj_1, res_obj_1_ind, res_obj_2, res_obj_2_ind = (
        None,
        "null",
        0,
        "null",
        0,
    )
    tokens = action_str.split()
    if tokens[0] in ["look", "inventory"]:
        res_action = tokens[0].lower()
    elif tokens[0] == "go":
        res_action, res_obj_1, res_obj_1_ind = (
            tokens[0].lower(),
            tokens[2].lower(),
            int(tokens[3]),
        )
    elif tokens[0] in ["take", "clean", "put", "heat"]:
        res_action, res_obj_1, res_obj_1_ind = (
            tokens[0].lower(),
            tokens[1].lower(),
            int(tokens[2]),
        )
        res_obj_2, res_obj_2_ind = tokens[4].lower(), int(tokens[5])
    elif tokens[0] in ["open", "close", "examine"]:
        res_action, res_obj_1, res_obj_1_ind = (
            tokens[0].lower(),
            tokens[1].lower(),
            int(tokens[2]),
        )
    else:
        raise ValueError(f"Unknown action {action_str}")

    res_action = action_one_hot_encode(res_action)
    res_obj_1 = object_one_hot_encode(res_obj_1)
    res_obj_2 = object_one_hot_encode(res_obj_2)
    res_obj_1_ind = index_one_hot_encode(res_obj_1_ind)
    res_obj_2_ind = index_one_hot_encode(res_obj_2_ind)

    return np.concatenate(
        (res_action, res_obj_1, res_obj_1_ind, res_obj_2, res_obj_2_ind)
    )


def action_loss_fn(action_pred, action):
    action_name_pred = action_pred[:, : len(ACTIONS)]
    obj_1_pred = action_pred[:, len(ACTIONS) : len(ACTIONS) + len(OBJECTS)]
    obj_1_ind_pred = action_pred[
        :, len(ACTIONS) + len(OBJECTS) : len(ACTIONS) + len(OBJECTS) + MAX_IND
    ]
    obj_2_pred = action_pred[:, 
        len(ACTIONS)
        + len(OBJECTS)
        + MAX_IND : len(ACTIONS)
        + 2 * len(OBJECTS)
        + MAX_IND
    ]
    obj_2_ind_pred = action_pred[:, 
        len(ACTIONS)
        + 2 * len(OBJECTS)
        + MAX_IND : len(ACTIONS)
        + 2 * len(OBJECTS)
        + 2 * MAX_IND
    ]

    action_name = action[:, : len(ACTIONS)]
    obj_1 = action[:, len(ACTIONS) : len(ACTIONS) + len(OBJECTS)]
    obj_1_ind = action[:, 
        len(ACTIONS) + len(OBJECTS) : len(ACTIONS) + len(OBJECTS) + MAX_IND
    ]
    obj_2 = action[:, 
        len(ACTIONS)
        + len(OBJECTS)
        + MAX_IND : len(ACTIONS)
        + 2 * len(OBJECTS)
        + MAX_IND
    ]
    obj_2_ind = action[:, 
        len(ACTIONS)
        + 2 * len(OBJECTS)
        + MAX_IND : len(ACTIONS)
        + 2 * len(OBJECTS)
        + 2 * MAX_IND
    ]

    loss_fn = torch.nn.BCEWithLogitsLoss()

    loss = (
        loss_fn(action_name_pred, action_name)
        + loss_fn(obj_1_pred, obj_1)
        + loss_fn(obj_1_ind_pred, obj_1_ind)
        + loss_fn(obj_2_pred, obj_2)
        + loss_fn(obj_2_ind_pred, obj_2_ind)
    )

    return loss


def form_action_str(action, task_str):
    task_type, task_obj_1, task_obj_2 = parse_task_name(task_str)
    # suppose the action is not yet sigmoid
    action_name = action[: len(ACTIONS)]
    obj_1 = action[len(ACTIONS) : len(ACTIONS) + len(OBJECTS)]
    obj_1_ind = action[
        len(ACTIONS) + len(OBJECTS) : len(ACTIONS) + len(OBJECTS) + MAX_IND
    ]
    obj_2 = action[
        len(ACTIONS)
        + len(OBJECTS)
        + MAX_IND : len(ACTIONS)
        + 2 * len(OBJECTS)
        + MAX_IND
    ]
    obj_2_ind = action[
        len(ACTIONS)
        + 2 * len(OBJECTS)
        + MAX_IND : len(ACTIONS)
        + 2 * len(OBJECTS)
        + 2 * MAX_IND
    ]

    action_name = ACTIONS[np.argmax(action_name).item()]
    obj_1 = OBJECTS[np.argmax(obj_1).item()]
    obj_1_ind = np.argmax(obj_1_ind).item()
    obj_2 = OBJECTS[np.argmax(obj_2).item()]
    obj_2_ind = np.argmax(obj_2_ind).item()
    
    if action_name in ["look", "inventory"]:
        return action_name
    elif action_name == "go":
        return f"go to {obj_1} {obj_1_ind}"
    elif action_name == "take":
        return f"take {obj_1} {obj_1_ind} from {obj_2} {obj_2_ind}"
    elif action_name == "clean":
        return f"clean {obj_1} {obj_1_ind} with {obj_2} {obj_2_ind}"
    elif action_name == "put":
        return f"put {obj_1} {obj_1_ind} in/on {obj_2} {obj_2_ind}"
    elif action_name == "heat":
        return f"heat {obj_1} {obj_1_ind} with {obj_2} {obj_2_ind}"
    elif action_name == "open":
        return f"open {obj_1} {obj_1_ind}"
    elif action_name == "close":
        return f"close {obj_1} {obj_1_ind}"
    elif action_name == "examine":
        return f"examine {obj_1} {obj_1_ind}"
    else:
        raise ValueError(f"Unknown action name {action_name}")

def parse_task_name(task_str):
    tokens = task_str.split()
    if "some" not in tokens:
        ind = 2
        if tokens[2] in ["clean", "hot"]:
            ind = 3
        obj_1 = tokens[ind]
        obj_2 = tokens[ind + 2]

        if tokens[2] == "clean":
            task_type = "clean"
        elif tokens[2] == "hot":
            task_type = "heat"
        else:
            task_type = "put"
    else:
        ind = 2
        obj_1 = tokens[ind]

        if tokens[0] == "clean":
            obj_2 = tokens[ind + 5]
            task_type = "clean"
        elif tokens[0] == "heat":
            obj_2 = tokens[ind + 5]
            task_type = "heat"
        else:
            obj_2 = tokens[ind + 2]
            task_type = "put"
    return task_type, obj_1, obj_2


def parse_observation(observation_str, task_str, subgoal_1_finished, subgoal_2_finished, subgoal_3_finished):
    object_found = 0
    task_type, task_obj_1, task_obj_2 = parse_task_name(task_str)

    target_str = f"take {task_obj_1}"
    _target_str = f"take {task_obj_1.lower()}"
    if subgoal_1_finished:
        object_found = 0
    elif (target_str in observation_str or _target_str in observation_str) and "Your task is to" not in observation_str:
        object_found = 1
    
    
    tool_found = 0
    if subgoal_1_finished:
        if task_type in ["heat"]:
            if "heat" in observation_str:
                tool_found = 1
        elif task_type in ["clean"]:
            if "clean" in observation_str:
                tool_found = 1
    if subgoal_2_finished:
        tool_found = 0

    place_found = 0
    target_str = f"put {task_obj_1}"
    target_str_ = f"put {task_obj_1.lower()}"
    target_str_2 = f"in/on {task_obj_2}"
    target_str_2_ = f"in/on {task_obj_2.lower()}"
    if (subgoal_1_finished and subgoal_2_finished) or (not subgoal_2_finished and task_type in ["put"]):
        if (target_str in observation_str or target_str_ in observation_str) and (target_str_2 in observation_str or target_str_2_ in observation_str):
            place_found = 1
    if subgoal_3_finished:
        place_found = 0
    
    object_found = np.array([object_found])
    tool_found = np.array([tool_found])
    place_found = np.array([place_found])
    subgoal_1_finished = np.array([subgoal_1_finished])
    subgoal_2_finished = np.array([subgoal_2_finished])
    subgoal_3_finished = np.array([subgoal_3_finished])
    

    task_type = action_one_hot_encode(task_type)
    task_obj_1 = object_one_hot_encode(task_obj_1)
    task_obj_2 = object_one_hot_encode(task_obj_2)

    return np.concatenate((object_found, tool_found, place_found))


def extract_task_name(observation_str):
    pattern = r"Your task is to: (.*?)\."

    match = re.search(pattern, observation_str)

    if match:
        return match.group(1)
    else:
        raise ValueError(f"Task not found in observation string {observation_str}")

def get_subgoal_finished(task_name, obs_str):
    subgoal_1 = 0
    subgoal_2 = 0
    subgoal_3 = 0
    task_type, task_obj_1, task_obj_2 = parse_task_name(task_name)
    
    pattern_1 = r"You pick up the " + task_obj_1
    pattern_2 = r"You " + task_type + r" the " + task_obj_1
    pattern_3 = r"You put the " + task_obj_1 + r" \d in/on the " + task_obj_2
    
    match_1 = re.search(pattern_1, obs_str)
    match_2 = re.search(pattern_2, obs_str)
    match_3 = re.search(pattern_3, obs_str)
    
    if match_1:
        subgoal_1 = 1
    if match_2:
        subgoal_2 = 1
    if match_3:
        subgoal_3 = 1
    return subgoal_1, subgoal_2, subgoal_3
    
from llfbench.envs.llf_env import Feedback
from decision_transformer.evaluation.gpt_prompts import *
import random

def _get_expert_action(infos):
    if "expert_plan" in infos and len(infos["expert_plan"]) == 1 and len(infos["expert_plan"][0]) == 1:
        return infos["expert_plan"][0][0]
    else:
        return None

def generate_gpt_feedback(env, action, reward, info, past_info, feedback_type=None, disturb_hind=0, disturb_fore=0):
    if feedback_type is None:
        feedback_type = env.feedback_type

    feedback = Feedback()

    if "r" in feedback_type:
        feedback.r = env.format(reward_descp, reward=reward)

    if "hn" in feedback_type:

        past_admissible_actions = past_info["admissible_commands"][0]
        past_opt_action = _get_expert_action(past_info)

        if env.already_won:
            feedback.hn = env.format(hn_no_op)
        elif past_opt_action is None:
            assert False, "Not Implemented Error"
        else:
            bad_actions = list(past_admissible_actions)
            bad_actions.remove(past_opt_action)

            avoid_action = random.choice(bad_actions)

            if action == avoid_action:
                if disturb_hind:
                    feedback.hn = env.format(correct_good_action_descp, past_opt_action=avoid_action)
                else:
                    feedback.hn = env.format(mistake_bad_action_descp, avoid_action=avoid_action)
            else:
                if disturb_hind:
                    feedback.hn = env.format(mistake_bad_action_descp, avoid_action=action)
                else:
                    feedback.hn = env.format(correct_bad_action_descp, avoid_action=avoid_action)

    if "hp" in feedback_type:

        past_opt_action = _get_expert_action(past_info)

        if env.already_won:
            feedback.hp = env.format(hp_no_op)
        elif past_opt_action is None:
            assert False, "Not Implemented Error"
        else:
            if past_opt_action == action.lower().strip():
                if disturb_hind:
                    feedback.hp = env.format(mistake_bad_action_descp, past_opt_action=past_opt_action)
                else:
                    feedback.hp = env.format(correct_good_action_descp, past_opt_action=past_opt_action)
            else:
                if disturb_hind:
                    feedback.hp = env.format(correct_bad_action_descp, past_opt_action=past_opt_action)
                else:
                    feedback.hp = env.format(mistake_good_action_descp, past_opt_action=past_opt_action)

    if "fn" in feedback_type:

        admissible_actions = info["admissible_commands"][0]
        opt_action = _get_expert_action(info)

        if env.already_won:
            feedback.fn = env.format(fn_no_op)
        elif opt_action is None:
            assert False, "Not Implemented Error"
        else:
            bad_actions = list(admissible_actions)
            bad_actions.remove(opt_action)

            avoid_action = random.choice(bad_actions)
            
            if disturb_fore:
                feedback.fn = env.format(follow_opt_action_descp, avoid_action=avoid_action)
            else:
                feedback.fn = env.format(avoid_bad_action_descp, avoid_action=avoid_action)

    if "fp" in feedback_type:

        opt_action = info['expert_action']

        if env.already_won:
            feedback.fp = env.format(fp_no_op)
        elif opt_action is None:
            assert False, "Not Implemented Error"
        else:
            if disturb_fore:
                feedback.fp = env.format(avoid_bad_action_descp, avoid_action=opt_action)
            else:
                feedback.fp = env.format(follow_opt_action_descp, opt_action=opt_action)

    return feedback

def update_task(yaml_path, rq, mode):
    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)
    write_flag = False
    task_type = None
    if rq == 1:
        task_type = [1, 3, 4]
        if data['env']['task_types'] != [1, 3, 4]:
            write_flag = True
    elif mode == "train" or mode == "val":
        task_type = [1, 3]
        if data['env']['task_types'] != [1, 3]:
            write_flag = True
    elif mode == "test":
        task_type = [4]
        if data['env']['task_types'] != [4]:
            write_flag = True
    if write_flag:
        print("Updating task type...")
        with open(yaml_path, "w") as f:
            data['env']['task_types'] = task_type
            yaml.dump(data, f)
