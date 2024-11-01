'''
Script that allows users to play Messenger in the terminal.
'''
from PIL import Image, ImageDraw, ImageFont
import argparse
import numpy as np
import gym
import messenger
import os
from utils.observation_process import observationProcessor
from utils.deepCopy import copier
import torch
def numpy_formatter(i: int):
    ''' Format function passed to numpy print to make things pretty.
    '''
    id_map = {}
    for ent in messenger.envs.config.ALL_ENTITIES:
        id_map[ent.id] = ent.name[:2].upper()
    id_map[0] = '  '
    id_map[15] = 'A0'
    id_map[16] = 'AM'
    if i < 17:
        return id_map[i]
    else:
        return 'XX'

def print_instructions():
    ''' Print the Messenger instructions and header.
    '''
    print(f"\nMESSENGER\n")
    print("Read the manual to get the message and bring it to the goal.")
    print("A0 is you (agent) without the message, and AM is you with the message.")
    print("The following is the symbol legend (symbol : entity)\n")
    for ent in messenger.envs.config.ALL_ENTITIES[:12]:
        print(f"{ent.name[:2].upper()} : {ent.name}")    
    print("\nNote when entities overlap the symbol might not make sense. Good luck!\n")

def print_grid(obs):
    ''' Print the observation to terminal
    '''
    grid = np.concatenate((obs['entities'], obs['avatar']), axis=-1)
    print(np.sum(grid, axis=-1).astype('uint8'))

def print_manual(manual):
    ''' Print the manual to terminal
    '''
    man_str = f"Manual: {manual[0]}\n"
    for description in manual[1:]:
        man_str += f"        {description}\n"
    print(man_str)

def clear_terminal():
    ''' Special print that will clear terminal after each step.
    Replace with empty return if your terminal has issues with this.
    ''' 
    # print(chr(27) + "[2J")
    print("\033c\033[3J")

def _symbolic_to_multihot(obs):
    # (h, w, 2)
    layers = np.concatenate((obs["entities"], obs["avatar"]),
                            axis=-1).astype(int)
    new_ob = np.maximum.reduce([np.eye(17)[layers[..., i]] for i
                                in range(layers.shape[-1])])
    new_ob[:, :, 0] = 0
    # assert new_ob.shape == self.observation_space["image"].shape
    print("new ob")
    print(new_ob)
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
        print("here")
        letter = idx_to_letter[idxs[i][j]]
        print(letter)
        # x,y canvas reversed
        color = (247, 193, 119) if letter in ("a", "m") else (238, 108, 133)
        draw.text((int(j * scale), int(i * scale)), letter, fill=color)
    new_img = np.asarray(new_img)
    return new_img


if __name__ == "__main__":
    np.set_printoptions(threshold=np.inf)
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_id", type=str, default="msgr-train-v3", help='environment id for human play')
    args = parser.parse_args()
    np.set_printoptions(formatter={'int': numpy_formatter})
    tempenv = gym.make(args.env_id)
    obs, manual = tempenv.reset(seed=8)
    envCopier=copier(tempenv)
    env=envCopier.newTask(tempenv,True)
    
    # map from keyboard entry to gym action space
    action_map = {'w': 0, 's': 1, 'a': 2, 'd': 3, '': 4}
    
    keep_playing = "yes"
    total_games = 0
    total_wins = 0
    images=[]
    num=0
    observation_Processor=observationProcessor()
    while keep_playing.lower() not in ["no", "n"]:
        done = False
        eps_reward = 0
        eps_steps = 0
        reward = 0
        print_instructions()
        print_manual(manual)
        print_grid(obs)
        currentState=observation_Processor.generate_state(env)
        print(currentState)
        action = input('\nenter action [w,a,s,d,\'\']: ')

        while not done:
            num=num+1
            if(num>60):
                break
            if action.lower() in action_map:
                obs, reward, done, info = env.step(action_map[action])
                eps_steps += 1
                print_instructions()
                print_manual(manual)
                currentState=observation_Processor.generate_state(env)
                print(currentState)
                reward=reward*100+observation_Processor.process_reward(currentState)-0.5
                # print(observation_Processor.generateSubgoal(currentState))
                eps_reward += reward
                print_grid(obs)
                if reward != 0:
                    print(f"\ngot reward: {reward}\n")
            if done:
                break

            action = input('\nenter action [w,a,s,d,\'\']: ')

        print(f"\nFinished episode with reward {eps_reward} in {eps_steps} steps!\n")
        keep_playing = input("play again? [n/no] to quit: ")
        if keep_playing.lower() not in ["no", "n"]:
            break

    print(f"\nThanks for playing! You won {total_wins} / {total_games} games.\n")
    torch.save(images,'images')
