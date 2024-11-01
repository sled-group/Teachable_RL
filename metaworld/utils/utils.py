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
from metaworld.policies import *
from utils.languages import *
from metaworld.policies.action import Action
from metaworld.policies.policy import move
envList=['assembly-v2-goal-observable', 'basketball-v2-goal-observable', 'bin-picking-v2-goal-observable', 'box-close-v2-goal-observable', 'button-press-topdown-v2-goal-observable', 'button-press-topdown-wall-v2-goal-observable', 'button-press-v2-goal-observable', 'button-press-wall-v2-goal-observable', 'coffee-button-v2-goal-observable', 'coffee-pull-v2-goal-observable', 'coffee-push-v2-goal-observable', 'dial-turn-v2-goal-observable', 'disassemble-v2-goal-observable', 'door-close-v2-goal-observable', 'door-lock-v2-goal-observable', 'door-open-v2-goal-observable', 'door-unlock-v2-goal-observable', 'hand-insert-v2-goal-observable', 'drawer-close-v2-goal-observable', 'drawer-open-v2-goal-observable', 'faucet-open-v2-goal-observable', 'faucet-close-v2-goal-observable', 'hammer-v2-goal-observable', 'handle-press-side-v2-goal-observable', 'handle-press-v2-goal-observable', 'handle-pull-side-v2-goal-observable', 'handle-pull-v2-goal-observable', 'lever-pull-v2-goal-observable', 'peg-insert-side-v2-goal-observable', 'pick-place-wall-v2-goal-observable', 'pick-out-of-hole-v2-goal-observable', 'reach-v2-goal-observable', 'push-back-v2-goal-observable', 'push-v2-goal-observable', 'pick-place-v2-goal-observable', 'plate-slide-v2-goal-observable', 'plate-slide-side-v2-goal-observable', 'plate-slide-back-v2-goal-observable', 'plate-slide-back-side-v2-goal-observable', 'peg-unplug-side-v2-goal-observable', 'soccer-v2-goal-observable', 'stick-push-v2-goal-observable', 'stick-pull-v2-goal-observable', 'push-wall-v2-goal-observable', 'reach-wall-v2-goal-observable', 'shelf-place-v2-goal-observable', 'sweep-into-v2-goal-observable', 'sweep-v2-goal-observable', 'window-open-v2-goal-observable', 'window-close-v2-goal-observable']

P_GAINS = {
    SawyerAssemblyV1Policy: 10.0,
    SawyerAssemblyV2Policy: 10.0,
    SawyerBasketballV1Policy: 25.0,
    SawyerBasketballV2Policy: 25.0,
    SawyerBinPickingV2Policy: 25.0,
    SawyerBoxCloseV1Policy: 25.0,
    SawyerBoxCloseV2Policy: 25.0,
    SawyerButtonPressTopdownV1Policy: 25.0,
    SawyerButtonPressTopdownV2Policy: 25.0,
    SawyerButtonPressTopdownWallV1Policy: 25.0,
    SawyerButtonPressTopdownWallV2Policy: 25.0,
    SawyerButtonPressV1Policy: 4.0,
    SawyerButtonPressV2Policy: 25.0,
    SawyerButtonPressWallV1Policy: 15.0,
    SawyerButtonPressWallV2Policy: 15.0,
    SawyerCoffeeButtonV1Policy: 10.0,
    SawyerCoffeeButtonV2Policy: 10.0,
    SawyerCoffeePullV1Policy: 10.0,
    SawyerCoffeePullV2Policy: 10.0,
    SawyerCoffeePushV1Policy: 10.0,
    SawyerCoffeePushV2Policy: 10.0,
    SawyerDialTurnV1Policy: 5.0,
    SawyerDialTurnV2Policy: 10.0,
    SawyerDisassembleV1Policy: 10.0,
    SawyerDisassembleV2Policy: 10.0,
    SawyerDoorCloseV1Policy: 25.0,
    SawyerDoorCloseV2Policy: 25.0,
    SawyerDoorLockV1Policy: 25.0,
    SawyerDoorLockV2Policy: 25.0,
    SawyerDoorOpenV1Policy: 10.0,
    SawyerDoorOpenV2Policy: 25.0,
    SawyerDoorUnlockV1Policy: 25.0,
    SawyerDoorUnlockV2Policy: 25.0,
    SawyerDrawerCloseV1Policy: 10.0,
    SawyerDrawerCloseV2Policy: 25.0,
    SawyerDrawerOpenV1Policy: 4.0, # TODO (50.0) # NOTE this policy looks different from the others because it must modify its p constant part-way through the task
    SawyerDrawerOpenV2Policy: 4.0, # TODO (50.0) # NOTE this policy looks different from the others because it must modify its p constant part-way through the task
    SawyerFaucetCloseV1Policy: 25.0,
    SawyerFaucetCloseV2Policy: 25.0,
    SawyerFaucetOpenV1Policy: 25.0,
    SawyerFaucetOpenV2Policy: 25.0,
    SawyerHammerV1Policy: 10.0,
    SawyerHammerV2Policy: 10.0,
    SawyerHandInsertV1Policy: 10.0,
    SawyerHandInsertV2Policy: 10.0,
    SawyerHandlePressSideV2Policy: 25.0,
    SawyerHandlePressV1Policy: 25.0,
    SawyerHandlePressV2Policy: 25.0,
    SawyerHandlePullSideV1Policy: 25.0,
    SawyerHandlePullSideV2Policy: 25.0,
    SawyerHandlePullV1Policy: 25.0,
    SawyerHandlePullV2Policy: 25.0,
    SawyerLeverPullV2Policy: 25.0,
    SawyerPegInsertionSideV2Policy: 25.0,
    SawyerPegUnplugSideV1Policy: 25.0,
    SawyerPegUnplugSideV2Policy: 25.0,
    SawyerPickOutOfHoleV1Policy: 10.0,
    SawyerPickOutOfHoleV2Policy: 25.0,
    SawyerPickPlaceV2Policy: 10.0,
    SawyerPickPlaceWallV2Policy: 10.0,
    SawyerPlateSlideBackSideV2Policy: 10.0,
    SawyerPlateSlideBackV1Policy: 10.0,
    SawyerPlateSlideBackV2Policy: 10.0,
    SawyerPlateSlideSideV1Policy: 25.0,
    SawyerPlateSlideSideV2Policy: 25.0,
    SawyerPlateSlideV1Policy: 10.0,
    SawyerPlateSlideV2Policy: 10.0,
    SawyerPushBackV1Policy: 10.0,
    SawyerPushBackV2Policy: 10.0,
    SawyerPushV2Policy: 10.0,
    SawyerPushWallV2Policy: 10.0,
    SawyerReachV2Policy: 5.0,
    SawyerReachWallV2Policy: 5.0,
    SawyerShelfPlaceV1Policy: 25.0,
    SawyerShelfPlaceV2Policy: 25.0,
    SawyerSoccerV1Policy: 25.0,
    SawyerSoccerV2Policy: 25.0,
    SawyerStickPullV1Policy: 10.0,
    SawyerStickPullV2Policy: 25.0,
    SawyerStickPushV1Policy: 10.0,
    SawyerStickPushV2Policy: 10.0,
    SawyerSweepIntoV1Policy: 25.0,
    SawyerSweepIntoV2Policy: 25.0,
    SawyerSweepV1Policy: 25.0,
    SawyerSweepV2Policy: 25.0,
    SawyerWindowOpenV2Policy: 25.0,
    SawyerWindowCloseV2Policy: 25.0,
}

# P_CONTROL_TIME=15

def get_task_text(env_name):
    name = " ".join(env_name.split('-')[:-3])
    return name

def transform(strings:List[str])->List[str]:
    for i in range(len(strings)):
        strings[i]=strings[i][0].upper()+strings[i][1:]
    return strings        

def get_policy(env_name):
    name = "".join(transform(get_task_text(env_name).split(" ")))
    # print(name)
    name=name.replace("Insert","Insertion")
    policy_name = "Sawyer" + name + "V2Policy"   
    # print(policy_name)
    try:
        policy = getattr(policies, policy_name)()
    except:
        policy = None
    return policy

def get_action_space(env_name):
    if env_name=='assembly-v2-goal-observable':
        return (6,2)
    elif env_name=='hammer-v2-goal-observable':
        return (5,2)

def get_action(policy,action,obs,env_name):
    action_id,gripper_id=action
    o_d=policy._parse_obs(obs)
    if env_name=='assembly-v2-goal-observable':
        pos_curr = o_d['hand_pos']
        pos_wrench = o_d['wrench_pos'] + np.array([-.02, .0, .0])
        pos_peg = o_d['peg_pos'] + np.array([.12, .0, .14])
        action_space=[pos_curr+np.array([0., 0., 0.1]),pos_curr,pos_wrench + np.array([0., 0., 0.1]),pos_peg + np.array([(pos_curr[0]-pos_wrench[0]),0,0]) + np.array([.0, .0, -.2]),pos_wrench + np.array([0., 0., 0.03]),np.array([pos_curr[0], pos_curr[1], pos_peg[2]]),pos_peg+np.array([(pos_curr[0]-pos_wrench[0]),0,0])+np.array([0,-0.005,0])]
        action_id=action_id if action_id<=6 else 6
        action=action_space[action_id]
        pos_curr = o_d['hand_pos']
        pos_wrench = o_d['wrench_pos'] + np.array([-.02, .0, .0])
        pos_peg = o_d['peg_pos'] + np.array([.12, .0, .14])
        gripper_space=[-0.2,0,0.2]
        gripper_id=gripper_id if gripper_id <=2 else 2
        gripper=gripper_space[gripper_id]
        return np.concatenate([action,np.array([gripper])])
    elif env_name=='hammer-v2-goal-observable':
        pos_curr = o_d['hand_pos']
        pos_puck = o_d['hammer_pos'] + np.array([-.04, .0, -.01])
        pos_goal = np.array([0.24, 0.71, 0.11]) + np.array([-.19, .0, .05])
        action_space=[pos_curr+np.array([0., 0., 0.1]),pos_curr,pos_puck + np.array([0., 0., 0.1]),np.array([pos_goal[0], pos_curr[1], pos_goal[2]]),pos_puck + np.array([0., 0., 0.03]),pos_goal]
        action_id=action_id if action_id<=5 else 5
        action=action_space[action_id]
        pos_curr = o_d['hand_pos']
        pos_puck = o_d['hammer_pos'] + np.array([-.04, .0, -.01])
        gripper_space=[-0.2,0,0.8]
        gripper_id=gripper_id if gripper_id<=2 else 2
        gripper=gripper_space[gripper_id]
        return np.concatenate([action,np.array([gripper])])

def agent_step(policy,env,observation,action,env_name):
    def p_control(policy,action,observation):
        """ Compute the desired control based on a position target (action[:3])
        using P controller provided in Metaworld."""
        assert len(action)==4
        p_gain = P_GAINS[type(policy)]
        if type(policy) in [type(SawyerDrawerOpenV1Policy), type(SawyerDrawerOpenV2Policy)]:
            # This needs special cares. It's implemented differently.
            o_d = policy._parse_obs(observation)
            pos_curr = o_d["hand_pos"]
            pos_drwr = o_d["drwr_pos"]
            # align end effector's Z axis with drawer handle's Z axis
            if np.linalg.norm(pos_curr[:2] - pos_drwr[:2]) > 0.06:
                p_gain = 4.0
            # drop down to touch drawer handle
            elif abs(pos_curr[2] - pos_drwr[2]) > 0.04:
                p_gain = 4.0
            # push toward a point just behind the drawer handle
            # also increase p value to apply more force
            else:
                p_gain= 50.0
        current_pos=policy._parse_obs(observation)['hand_pos']
        control = Action({"delta_pos": np.arange(3), "grab_effort": 3})
        control["delta_pos"] = move(current_pos, to_xyz=action[:3], p=p_gain)
        control["grab_effort"] = action[3]
        return control.array
    P_CONTROL_TIME=4 if env_name=='hammer-v2-goal-observable' else 7
    # P_CONTROL_TIME=random.randint(4,8)
    for _ in range(P_CONTROL_TIME):
        control = p_control(policy,action,observation)
        observation, reward, terminated, truncated, info = env.step(control)
        desired_pos = action[:3]
        if np.abs(desired_pos - policy._parse_obs(observation)['hand_pos']).max() < 1e-4:
            break
    return observation,reward,terminated,truncated,info 

def get_state(observation,policy,env_name):
    parsed_obs=policy._parse_obs(observation)
    # Pick and Place
    if env_name=='hammer-v2-goal-observable':
        gripper_array = np.array([parsed_obs['gripper']])
        state = np.concatenate([parsed_obs['hand_pos'], gripper_array, parsed_obs['hammer_pos'], np.array([0,0,0])])
    elif env_name=='assembly-v2-goal-observable':
        gripper_array = np.array([parsed_obs['gripper']])
        state = np.concatenate([parsed_obs['hand_pos'], gripper_array, parsed_obs['wrench_pos'], parsed_obs['peg_pos']])
    elif env_name=='basketball-v2-goal-observable':
        gripper_array = np.array([parsed_obs['gripper']])
        x_pos=np.array([parsed_obs['hoop_x']])
        state = np.concatenate([parsed_obs['hand_pos'], gripper_array, parsed_obs['ball_pos'], x_pos, parsed_obs['hoop_yz']])
    elif env_name=='peg-insert-side-v2-goal-observable':
        gripper_array = np.array([parsed_obs['gripper_distance_apart']])
        state = np.concatenate([parsed_obs['hand_pos'], gripper_array, parsed_obs['peg_pos'], parsed_obs['goal_pos']])
    elif env_name=='shelf-place-v2-goal-observable':
        gripper_array = np.array([parsed_obs['unused_1']])
        goal_x_array = np.array([parsed_obs['shelf_x']])
        state = np.concatenate([parsed_obs['hand_pos'], gripper_array, parsed_obs['block_pos'], goal_x_array, parsed_obs['unused_3']])
    return state

def get_f_language(env_name,language_id,mode):
    if mode=="training":
        random_index=random.randint(0,143)
    elif mode=="validation":
        random_index=random.randint(144,161)
    else:
        random_index=random.randint(162,179)
    if env_name=='assembly-v2-goal-observable':
        languages=[raise_gripper,open_gripper,place_instructions,aim_instructions,grasp_tool,raise_tool,get_instructions]
        return (languages[language_id][random_index],languages[language_id][0])
    elif env_name=='hammer-v2-goal-observable':
        languages=[raise_gripper,open_gripper,place_instructions,aim_instructions,grasp_tool,get_instructions,get_instructions]
        return (languages[language_id][random_index],languages[language_id][0])

def get_h_language(env_name,language_id,mode,action_id, disturb=False):
    if mode=="training":
        random_index=random.randint(0,143)
    elif mode=="validation":
        random_index=random.randint(144,161)
    else:
        random_index=random.randint(162,179)
    # disturb=True
    if disturb:
        language_id=random.randint(0,1)
        action_id=random.randint(0,5)
    if env_name=='assembly-v2-goal-observable':
        do_pool=["raise the gripper","open the gripper", "place the gripper above the tool", "aim at the goal", "grasp the tool", "raise the tool to the goal", "get to the goal"]
        if language_id>0:
            return (hindsight_negative_do[random_index].format(do=do_pool[action_id]),hindsight_negative_do[0].format(do=do_pool[action_id]))
        else:
            return (hindsight_positive[random_index],hindsight_positive[0])
    elif env_name=='hammer-v2-goal-observable':
        do_pool=["raise the gripper","open the gripper", "place the gripper above the tool", "aim at the goal", "grasp the tool", "raise the tool to the goal","get to the goal"]
        if language_id>0:
            return (hindsight_negative_do[random_index].format(do=do_pool[action_id]),hindsight_negative_do[0].format(do=do_pool[action_id]))
        else:
            return (hindsight_positive[random_index],hindsight_positive[0])
        
            
        

if __name__=="__main__":
    env_name='hammer-v2-goal-observable'
    # env_name='shelf-place-v2-goal-observable'
    # env_name='basketball-v2-goal-observable'
    # env_name='assembly-v2-goal-observable'
    
    benchmark_env = env_dict[env_name]
    task_name=get_task_text(env_name)
    policy=get_policy(env_name)
    seed=21
    env = benchmark_env(seed)
    observation,info=env.reset()
    done=False
    get_state(observation,policy,env_name)
    for i in range(50):
        (action_id,gripper_id),language=policy.get_action(observation)
        print(i,(action_id,gripper_id),language)
        observation,reward,terminated,truncated,info =agent_step(policy,env,observation,get_action(policy,(action_id,gripper_id),observation,env_name))
        if terminated or truncated:
            done=True
            break
        if info['success']==1.0:
            break
    print(info)

        