import json
import numpy as np

rooms = """......WWWWWWWW
......WLLLLLLW
......WLLLLLLW
......WLLLLLLW
......WLLLLLLW
WWWWWWWWLLLLLW
WKKKKKKWDDDDDW
WKKKKKKKDDDDDW
WKKKKKKKDDDDDW
WKKKKKKKDDDDDW
WKKKKKKWDDDDDW
WWWWWWWWWWWWWW
""".splitlines()

valid_position = """..............
..............
........xxxxx.
........xxxxx.
........xxxxx.
........xxxxx.
........xxxxx.
.xxxxxxxxxxxx.
.xxxxxxxx...x.
.xxxxxxxx...x.
........xxxxx.
..............
""".splitlines()

action_num2name = ['left', 'right', 'up', 'down', 'pickup',
                   'drop', 'get', 'pedal', 'grasp', 'lift']

RELATIVE_KEYS_FROM_OBJDICT_INITIAL = ['step', 'agent', 'objects', 'front_obj', 'success', 'action', 'action_status']
RELATIVE_KEYS_FROM_OBJDICT = [
    'step', 'agent', 'front_obj', 'truncated', 'success', 'action', 'action_status']
def generate_init_json():
    init_layout = {
        'rooms': rooms,
        'valid_poss': valid_position
    }

    action_space = action_num2name

    init_info = {
        'init_layout': init_layout,
        'action_space': action_space
    }
    info = {
        'init_layout_info': init_info,
        'init_state_info': None,
        'intermediate_info': []
    }
    return info

def extract_obj_name(task: str):
    tokens = task.split()
    if tokens[0] in ['find', 'get']:
        if tokens[-1] == 'bin':
            return [' '.join(tokens[-2:])]
        return [tokens[-1]]
    elif tokens[0] == 'put':
        return [tokens[2], ' '.join(tokens[-2:])]
    elif tokens[0] == 'move':
        return [tokens[2]]
    elif tokens[0] == 'open':
        return [' '.join(tokens[-2:])]
    else:
        raise ValueError(f"invalid task description: {task}")

def if_key_frame(cur_s: dict, init_s: dict):
    # agent performs non-trivial action
    if cur_s['truncated'] or cur_s['success']:
        return True
    if cur_s['action'] in ['pickup',
                           'drop', 'get', 'pedal', 'grasp', 'lift']:
        return True
    # agent failed to do some action
    if cur_s['action_status']['if_action_failed']:
        return True
    return if_significant_difference(cur_s, init_s)

def if_significant_difference(s1: dict, s2:dict):
    agent1, agent2 = s1['agent'], s2['agent']

    if agent1['room'] != agent2['room']:
        return True
    
    if agent1['carrying'] != agent2['carrying']:
        return True
    
    return False

def extract_related_obj_from_list(object_list: list, related_obj: list, all=False):
    compact_object_list = {}
    
    for obj_dict in object_list:
        if obj_dict['name'] in related_obj:
            if all:
                compact_object_list[obj_dict['name']] = obj_dict
            else:
                if obj_dict['type'] == 'Storage':
                    compact_object_list[obj_dict['name']] = {
                        key: obj_dict[key] for key in ['pos', 'state', 'contains', 'action']}
                elif obj_dict['type'] == "Pickable":
                    compact_object_list[obj_dict['name']] = {
                        key: obj_dict[key] for key in ['pos', 'room']}
                else:
                    raise ValueError(f"Unknown object type {obj_dict['type']}")
                
    return compact_object_list

def success_condition(task_description: str) -> str:
    # from language_wrappers.py
    tokens = task_description.split()
    related_obj = extract_obj_name(task_description)
    if tokens[0] == 'find':
        return f"front_obj == {related_obj[0]}"
    elif tokens[0] == 'get':
        return f"agent['carrying'] == {related_obj[0]}"
    elif tokens[0] == 'put':
        return f"'{related_obj[0]}' in objects['{related_obj[1]}']['contains']"
    elif tokens[0] == 'move':
        if tokens[-1] == 'kitchen':
            return f"objects['{related_obj[0]}']['room'] == '{tokens[-1]}'"
        elif tokens[-1] == 'room':
            return f"objects['{related_obj[0]}']['room'] == '{' '.join(tokens[-2:])}'"
        else:
            raise ValueError(f"Unknown task {task_description}")
    elif tokens[0] == 'open':
        return f"objects['{related_obj[0]}']['state'] == 'open'"
    else:
        raise ValueError(f"Unknown task {task_description}")
    
def compact_info_json(info):
    compact_info = {
        'task_description': None,        
        'success_condition': None,
        'init_state': None,        
        'trajectory': [],
    }
    
    task_name = info['init_state_info']['task_description']
    compact_info['task_description'] = task_name
    related_obj = extract_obj_name(task_name)
    init_state = {key: info['init_state_info'][key] for key in RELATIVE_KEYS_FROM_OBJDICT_INITIAL[:-3]}
    init_state['objects'] = extract_related_obj_from_list(init_state['objects'], related_obj, all=True)
    compact_info['init_state'] = init_state
    
    INTERVAL = 2
    
    last_key_frame_ind = 1
    cur_frame_ind = 0
    for state in info['intermediate_info']:
        cur_frame_ind += 1
        if if_key_frame(state, init_state) or cur_frame_ind - last_key_frame_ind >= INTERVAL:
            cur_state = {key: state[key]
                          for key in RELATIVE_KEYS_FROM_OBJDICT}
            compact_info['trajectory'].append(cur_state)
            init_state = state
            last_key_frame_ind = cur_frame_ind
        if state['task_description'] != task_name:
            break
        
    compact_info['success_condition'] = success_condition(task_name)
    return compact_info
