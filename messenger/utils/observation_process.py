import numpy as np
import messenger
import json
import math
import torch
character_list = [
    "airplane",
    "mage",
    "dog",
    "bird",
    "fish",
    "scientist",
    "thief",
    "ship",
    "ball",
    "robot",
    "queen",
    "sword",
    "wall",
    "Agent without message",
    "Agent carrying message",
]
import random
def calculate_distance(coord1, coord2):
    return abs(coord1[0] - coord2[0]) + abs(coord1[1] - coord2[1])



thresholds=torch.load("./data/thresholds.pth")
hp_templates,hn_target_templates,hn_enemy_templates,fp_oracle_direct,fp_oracle_indirect=torch.load("data/languages.pth")

def numpy_formatter(i: int):
    """Format function passed to numpy print to make things pretty."""
    id_map = {}
    for ent in messenger.envs.config.ALL_ENTITIES:
        id_map[ent.id] = ent.name[:2].upper()
    id_map[0] = "  "
    id_map[15] = "A0"
    id_map[16] = "AM"
    if i < 17:
        return id_map[i]
    else:
        return "XX"

def extract_characters(obs):
    ''' Extract characters and their coordinates from the observation, handling overlaps.
    
    Args:
        obs (dict): Observation containing 'entities' and 'avatar'.

    Returns:
        list: List of tuples, where each tuple contains character and its coordinates.
    '''
    
    # Combine the entities and avatar to create the full 3D grid
    grid = np.concatenate((obs['entities'], obs['avatar']), axis=-1)
    state=[]
    # print(obs)
    # Iterate through the layers
    for layer in range(grid.shape[2]):
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                if grid[i][j][layer] != 0:
                    state.append(int(grid[i][j][layer]))
                    state.append(i)
                    state.append(j)
    while(len(state)<18):
        state.append(0)
    return state

class observationProcessor():
    def generate_grid(self, obs):
        temp = np.concatenate((obs["entities"], obs["avatar"]), axis=-1)
        grid = np.sum(temp, axis=-1).astype("uint8")
        return grid

    def generate_state(self, env):
        state=env.msgrEnv.stateFrame
        if('agent' in state):
            agent_position=state['agent']['pos']
            if('message'in state):
                state['agent']['e']="without_Message"
            else:
                state['agent']['e']="with_Message"
            for character in state:
                state[character]['d']=calculate_distance(state[character]['pos'],agent_position)
                state[character]['D']=self.get_relative_direction(agent_position,state[character]['pos'])
                
        return state
    
    def simplifyState(self,state):
        newState={}
        if("agent" not in state):
            return {"dead":True}
        newState["agent"]=state["agent"]
        # print(state)
        if ('message' in state):
            newState["message"]=state["message"]
            newState['agent']['distance_to_message']=state["message"]['d']
        elif('goal' in state):
            newState["goal"]=state["goal"]
            newState['agent']['distance_to_goal']=state['goal']['d']
        else:
            newState['agent']['distance_to_goal']=0
        for character in state:
            if(state[character]['d']<=3):
                if('message' in state and character=="goal"):
                    newState["Enemy"]=state[character]
                else:
                    newState[character]=state[character]
        return newState
            
    def distance_to_enemy(self,state):
        if('agent' not in state):
            return 0
        distance=[]
        for character in state:
            if(character not in ['agent','message','goal']):
                distance.append(state[character]['d'])
        if ('message' in state):
            distance.append(state['goal']['d'])
        return min(distance)
       
    def distance_to_target(self,state):
        try:
            if('message' in state):
                return state['message']['d']
            else:
                return state['goal']['d']
        except:
            return 10000
    
    def generate_trajectory_state(self,obs):
        result=extract_characters(obs)
        return result

    def get_relative_direction(self, agent_pos, char_pos):
        """
        Determine the relative direction of a character with respect to an agent using angles.

        Args:
        - agent_pos (tuple): The (x, y) position of the agent.
        - char_pos (tuple): The (x, y) position of the character.

        Returns:
        - str: Relative direction.
        """
        dx = char_pos[0] - agent_pos[0]  # Change in x-coordinate, representing left-to-right
        dy = char_pos[1] - agent_pos[1]  # Change in y-coordinate, representing down-to-up

        if dx == 0 and dy == 0:
            return "Overlap"

        # Get angle in radians
        angle = math.atan2(dy, dx)

        # Convert to degrees for easier reasoning
        angle_deg = math.degrees(angle)

        # Determine direction based on angle
        if -22.5 <= angle_deg < 22.5:
            return 'right'
        elif 22.5 <= angle_deg < 67.5:
            return 'up-right'
        elif 67.5 <= angle_deg < 112.5:
            return 'up'
        elif 112.5 <= angle_deg < 157.5:
            return 'up-left'
        elif -67.5 <= angle_deg < -22.5:
            return 'down-right'
        elif -112.5 <= angle_deg < -67.5:
            return 'down'
        elif -157.5 <= angle_deg < -112.5:
            return 'down-left'
        else:
            return 'left'

    def generate_hindsight_language(self,state,newTask=False,mode="validation",diversity="augmented",moreInfo=False,expert_action=None,disturb=False):
        simplifyList=[]
        result={"hindsight positive":{"template":"","human":""},"hindsight negative":{"template":"","human":""}}
        count=0
        goodTargetComment=None
        badTargetComment=None
        goodEnemyComment=None
        badEnemyComment=None
        minDistance=np.inf
        for s in state:
            simplifyList.append(self.simplifyState(s))
        for i in range(len(simplifyList)-1):
            prev=simplifyList[i]
            curr=simplifyList[i+1]
            nearbyEnemy=[]
            if(not newTask):
                target="message" if "message" in prev else "goal"
                # Append enemies to the nearby enemy list.
            else:
                target="goal" if "goal" in prev else "message"
            for character in prev:
                    if (character!="agent" and character!="message" and (character!="goal")) or ("message" in prev and  character=="goal"):
                        nearbyEnemy.append((prev[character]["e"],prev[character]["D"],prev[character]["d"],character))
                        minDistance=min(minDistance,prev[character]["d"])
            # Judge Action Direction
            if("agent" in curr and "agent" in prev):
                x1,y1=prev["agent"]["pos"]
                x2,y2=curr["agent"]["pos"]
                deltax=x2-x1
                deltay=y2-y1
                action="done"
                if(deltax==-1):
                    action="left"
                elif(deltax==1):
                    action="right"
                elif(deltay==1):
                    action="up"
                elif (deltay==-1):
                    action="down"
                else:
                    action="noMotion"
            else:
                action="none"
            action=random.choice(["down","left","right","up","down","noMotion"]) if disturb else action
            
            # enemy comment
            goodEnemyComment,badEnemyComment=self.generate_enemy_comment(action,nearbyEnemy,mode=mode,diversity=diversity)
            if(not newTask):
                # Judge target identity
                if ("message" in simplifyList[0]):
                    target_name="message "+simplifyList[0]["message"]["e"]
                else:
                    target_name="goal "+simplifyList[0]["goal"]["e"]
            else:
                if ("message" in simplifyList[0]):
                    target_name="goal "+simplifyList[0]["message"]["e"]
                else:
                    target_name="message "+simplifyList[0]["goal"]["e"]
            # toward target comment
            flag,desc=self.generate_direction_comment(action,prev[target]["D"],target_name,mode=mode,diversity=diversity)
            if(not flag):
                count=count+1
                badTargetComment=desc
            else:
                goodTargetComment=desc
            hn=False
            # Judge comment about approaching target for the whole trajectory
            
            # if enemy around, good enemy comment -> good enemy comment
            expert_action=['up','down','left','right','noMotion'][expert_action]     
            if(expert_action==action):
                if(goodTargetComment!=None and goodTargetComment!={"template":"","human":""}):
                    result["hindsight positive"]={"template":goodTargetComment["template"],"human":goodTargetComment["human"]}
                elif(goodEnemyComment!=None and goodEnemyComment!={"template":"","human":""}):
                    result["hindsight positive"]={"template":goodEnemyComment["template"],"human":goodEnemyComment["human"]}
            else:
                if (minDistance < 3 and badEnemyComment!=None and badEnemyComment!={"template":"","human":""}):
                    result["hindsight negative"]={"template":badEnemyComment["template"],"human":badEnemyComment["human"]}
                elif(badTargetComment!=None and badTargetComment!={"template":"","human":""}):
                    result["hindsight negative"]={"template":badTargetComment["template"],"human":badTargetComment["human"]}
                elif(badEnemyComment!=None and badEnemyComment!={"template":"","human":""}):
                    result["hindsight negative"]={"template":badEnemyComment["template"],"human":badEnemyComment["human"]}
                elif(goodTargetComment!=None and goodTargetComment!={"template":"","human":""}):
                    result["hindsight positive"]={"template":goodTargetComment["template"],"human":goodTargetComment["human"]}
                elif(goodEnemyComment!=None and goodEnemyComment!={"template":"","human":""}):
                    result["hindsight positive"]={"template":goodEnemyComment["template"],"human":goodEnemyComment["human"]}
        if(moreInfo):
            return result,hn
        return result
    
    def generate_random_index(self,mode,language_type,diversity):
        if(mode=="training"):
                    random_index = random.randint(0,thresholds[language_type]["training_threshold"][str(diversity)])
        elif(mode=="validation"):
                    random_index = random.randint(thresholds[language_type]["validation_threshold"][0],thresholds[language_type]["validation_threshold"][1])
        elif(mode=="testing"):
                    random_index = random.randint(thresholds[language_type]["testing_threshold"][0],thresholds[language_type]["testing_threshold"][1])
        return random_index
    
    def generate_foresight_language(self,state,newTask=False,mode="validation",diversity=100,moreInfo=False,disturb=False):   
        simplifyList=[]
        result={}
        action_list=[]
        nearestEnemy={"e":'',"D":"","d":""}
        target_info={}
        distance_to_enemy=np.inf
        for s in state:
            simplifyList.append(self.simplifyState(s))
        first_state=simplifyList[0]
        if(not newTask):
            target="message" if "message" in first_state else "goal"
            if (target not in first_state):
                return ""
            result["target"]=target+" "+first_state[target]['e']+" at "+first_state[target]['D']
            enemy_list=[]
            for character in first_state:
                if (character!="agent" and character!="message" and (character!="goal")) or ("message" in first_state and character=="goal"):
                    enemy_list.append(character+" "+first_state[character]["e"] +" at "+first_state[character]["D"]+ " in distance "+str(first_state[character]["d"]))
                    if(distance_to_enemy>first_state[character]["d"]):
                        distance_to_enemy=min(distance_to_enemy,first_state[character]["d"])
                        nearestEnemy=first_state[character]
            target_info["name"]=target+" "+first_state[target]['e']
        if(newTask):
            target="message" if "message" in first_state else "goal"
            if (target not in first_state):
                return ""
            target_in_language="message" if target=="goal" else "goal"
            result["target"]=target_in_language+" "+first_state[target]['e']+" at "+first_state[target]['D']
            enemy_list=[]
            for character in first_state:
                if (character!="agent" and character!="message" and (character!="goal")) or ("message" in first_state and character=="goal"):
                    enemy_list.append(character+" "+first_state[character]["e"] +" at "+first_state[character]["D"]+ " in distance "+str(first_state[character]["d"]))
                    if(distance_to_enemy>first_state[character]["d"]):
                        distance_to_enemy=min(distance_to_enemy,first_state[character]["d"])
                        nearestEnemy=first_state[character]
            target_info["name"]=target_in_language+" "+first_state[target]['e']
        target_info["D"]=first_state[target]['D']
        distance_to_target=first_state[target]["d"]
        for i in range(len(simplifyList)-1):
            prev=simplifyList[i]
            curr=simplifyList[i+1]
            action="done"
            if("agent" in curr and "agent" in prev):
                x1,y1=prev["agent"]["pos"]
                x2,y2=curr["agent"]["pos"]
                # print(x1,y1,x2,y2)
                deltax=x2-x1
                deltay=y2-y1
                if(deltax==-1):
                    action="left"
                elif(deltax==1):
                    action="right"
                elif(deltay==1):
                    action="up"
                elif (deltay==-1):
                    action="down"
                else:
                    action="noMotion"
            action_list.append(action)
            result["nearby_enemies"]=enemy_list if (len(enemy_list)!=0) else ["No enemies around"]
        count=0
        actions=[]
        for act in action_list:
            if(count==0):
                actions.append(act)
                count=1
            else:
                if (act!=actions[-1]):
                    if count==2:
                        break
                    actions.append(act)
                    count=count+1
        result["optimal actions"]="Move towards the "+actions[0]
        realResult={}
        fn=False
        if disturb:
            actions[0]=random.choice(["right","left","up","down"])
        is_avoid_enemy=actions[0] in target_info["D"]
        if(is_avoid_enemy):
            random_index=self.generate_random_index(mode,"fp_oracle_direct",diversity)
            realResult["foresight positive"]={"template":f"Move {actions[0]} to approach the {target_info['name']}. ","human":fp_oracle_direct[random_index].format(target_name=target_info["name"],target_direction=target_info["D"],optimal_direction=actions[0])}
        else:
            random_index=self.generate_random_index(mode,"fp_oracle_indirect",diversity)
            realResult["foresight positive"]={"template":f"Move {actions[0]} to dodge the enemy {nearestEnemy['e']} . ","human":fp_oracle_indirect[random_index].format(target_name=target_info["name"],target_direction=target_info["D"],optimal_direction=actions[0])}
        if moreInfo:
            return realResult,distance_to_target,distance_to_enemy,fn
        return realResult
    
    def generate_direction_comment(self,action,target_direction,target,mode="validation",diversity="augmented"):
        if (action=="none"):
            return (False,"It's too bad, You die.")
        if (action in target_direction):
            random_index=self.generate_random_index(mode,"hp_templates",diversity)
            return (True,{"template":"You are making an excellent step. ","human":hp_templates[random_index].format(direction=action,target_direction=target_direction,target=target)})
        else:
            random_index=self.generate_random_index(mode,"hn_target_templates",diversity)
            return(False,{"template":f"It's a bad move not moving effectively towards {target}. ","human":hn_target_templates[random_index].format(action=action,target_direction=target_direction,target=target)})
    
    def generate_enemy_comment(self,action,enemy_list,mode="validation",diversity="augmented"):
        badresult={"template":"","human":""}
        goodresult={"template":"","human":""}
        if(action=="none"):
            return ({"template":"","human":""},{"template":"Too bad action! You touch the enemy and die. ","human":"Too bad action! You touch the enemy and die. "})
        if len(enemy_list)==0:
            return ({"template":"It's good that you keep no enemies around. ","human":"It's good that you keep no enemies around. "},{"template":"","human":""})
        min_distance=3
        for enemy in enemy_list:
            if(enemy[2]<min_distance):
                min_distance=enemy[2]
        for enemy in enemy_list:
            if(enemy[2]!=min_distance):
                break
            if(action in enemy[1] or action=='noMotion'):
                random_index=self.generate_random_index(mode,"hn_enemy_templates",diversity)
                badresult={"template":"You step "+action+" but not avoid "+enemy[3]+" "+enemy[0]+" at "+enemy[1]+". ","human":hn_enemy_templates[random_index].format(action_direction=action,enemy_name=enemy[3]+" "+enemy[0],direction=enemy[1])}
                break
            else:
                random_index=self.generate_random_index(mode,"hp_templates",diversity)
                goodresult={"template":"You are making an excellent step. ","human":hp_templates[random_index].format(action=action,enemy_name="enemy "+enemy[0],direction=enemy[1])}
                break
        return (goodresult,badresult)
                    
    def process_reward(self,state):
        dist_target=self.distance_to_target(state)
        dist_enemy=self.distance_to_enemy(state)
        if(dist_enemy!=0):
            reward=5/dist_target-1/dist_enemy
        else:
            reward=0
        return reward
    
    def compare(self,state):
        result=[]
        for i in state:
            result.append(self.simplifyState(i))
        return result
    
def generateSubgoal(self,state):
        subgoal=""
        if('agent' in state):
            if('message'in state):
                subgoal="Subgoal: Go to reach the message which is the {} at {}".format(state["message"]["e"],state["message"]["D"])
            elif('goal' in state):
                subgoal="Subgoal: Go to reach the goal which is the {} at {}".format(state["goal"]["e"],state["goal"]["D"])
        return subgoal