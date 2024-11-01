import heapq
import argparse
import json
import torch
import numpy as np
import gym
import messenger
from messenger.models.emma import EMMA
from messenger.models.utils import ObservationBuffer
from utils.observation_process import observationProcessor, numpy_formatter
# from state_container import ObservationContainer
import math
# from prompt import LLMPrompter
import copy
import random
class pathSolver:
    def __init__(self):
        self.agent_position = None
        self.target_position = None
        self.enemy_positions = []
        self.stepCount=0
        self.path = []

    def alterState(self,pos):
        x,y=pos
        return (9-y,x)


    def update(self,state):
        target=''
        self.enemy_positions=[]
        self.enemy_type={}
        self.agent_position=self.alterState(state['agent']['pos']  )
        target='message' if 'message' in state else 'goal'
        if('enemy'in state):
            self.enemy_positions.append(self.alterState(state['enemy']['pos']))
            self.enemy_type[str(self.alterState(state['enemy']['pos']))]=state['enemy']['type']
        if('decoy_message' in state):
            self.enemy_positions.append(self.alterState(state['decoy_message']['pos']))
            self.enemy_type[str(self.alterState(state['decoy_message']['pos']))]=state['decoy_message']['type']
        if('decoy_goal' in state):
            self.enemy_positions.append(self.alterState(state['decoy_goal']['pos']))
            self.enemy_type[str(self.alterState(state['decoy_goal']['pos']))]=state['decoy_goal']['type']
        if target=='message':
            self.enemy_positions.append(self.alterState(state['goal']['pos']))
            self.target_position=self.alterState(state['message']['pos'])
            self.enemy_type[str(self.alterState(state['goal']['pos']))]=state['goal']['type']
        elif target=='goal':
            self.target_position=self.alterState(state['goal']['pos'])
        else:
            exit(1)
            
    def a_star(self, start, goal):
        open_set = []
        heapq.heappush(open_set, (0, start))
        came_from = {}
        g_score = {(i, j): float("inf") for i in range(10) for j in range(10)}
        f_score = {(i, j): float("inf") for i in range(10) for j in range(10)}
        g_score[start] = 0
        f_score[start] = self.heuristic(start, goal)
        closed_set = set()  # Set to store nodes that have already been processed

        while open_set:
            _, current = heapq.heappop(open_set)
            if current in closed_set:
                continue
            if current == goal:
                return self.reconstruct_path(came_from, current)

            for neighbor in self.get_neighbors(current):
                tentative_g_score = (
                    g_score[current] + 1 + self.enemy_distance_cost(neighbor)
                )
                if tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = g_score[neighbor] + self.heuristic(
                        neighbor, goal
                    )
                    if neighbor not in [item[1] for item in open_set]:
                        heapq.heappush(open_set, (f_score[neighbor], neighbor))
        # Mark the current node as processed
        closed_set.add(current)
        return []

    def heuristic(self, cell, goal):
        return math.sqrt(abs(cell[0] - goal[0])**2 + abs(cell[1] - goal[1])**2)

    def enemy_distance_cost(self, cell):
        min_distance = min(
            [self.heuristic(cell, enemy) for enemy in self.enemy_positions]
        )

        # Gradient penalty based on distance to nearest enemy
        if min_distance == 0:
            return 1000000
        elif min_distance <=1.2:
            return 20000  # High penalty for distance 1
        elif min_distance < 2:
            return 3000  # Medium penalty for distance 2
        # elif min_distance <= 2.5:
        #     return 500  # Lower penalty for distance 3
        else:
            return 0  # No penalty beyond distance 3

    def get_neighbors(self, position):
        neighbors = [
            (position[0] - 1, position[1]),
            (position[0] + 1, position[1]),
            (position[0], position[1] - 1),
            (position[0], position[1] + 1),
        ]
        valid_neighbors = []
        for neighbor in neighbors:
            if 0 <= neighbor[0] < 10 and 0 <= neighbor[1] < 10:
                valid_neighbors.append(neighbor)
        return valid_neighbors

    def get_destination(self,tuple):
        x,y=tuple
        m,n=(0,0)
        m=9*random.randint(0,1)
        n=9*random.randint(0,1)
        return (m,n)

    def judge_next_step_valid(self,next_step):
        (x,y)=next_step
        if((x,y) in self.enemy_positions):
            return False
        if((x+1,y) in self.enemy_positions):
            return False
        if((x-1,y) in self.enemy_positions):
            return False
        if((x,y+1) in self.enemy_positions ):
            return False
        if((x,y-1) in self.enemy_positions):
            return False
        if((x+1,y+1) in self.enemy_positions):
            return False
        if((x-1,y-1) in self.enemy_positions):
            return False
        if((x-1,y+1) in self.enemy_positions):
            return False
        if((x+1,y-1) in self.enemy_positions):
            return False
        return True

    def judge_enemy_aside(self,next_step):
        (x,y)=next_step
        if((x,y) in self.enemy_positions):
            return False
        if((x+1,y) in self.enemy_positions):
            return False
        if((x-1,y) in self.enemy_positions):
            return False
        if((x,y+1) in self.enemy_positions):
            return False
        if((x,y-1) in self.enemy_positions):
            return False
        return True

    def get_action(self):
        self.destination=self.target_position
        self.path = self.a_star(self.agent_position, self.target_position)
        next_step = self.path[1]
        # trial=0
        # temp_step=next_step
        # while(not self.judge_next_step_valid(temp_step) and trial<5):
        #     self.destination=self.get_destination(self.agent_position)
        #     self.path=self.a_star(self.agent_position,self.get_destination(self.agent_position))
        #     if len(self.path)>=2:
        #         temp_step = self.path[1]
        #     trial=trial+1
        # if(not self.judge_next_step_valid(temp_step)):
        #     temp_step=next_step
        # next_step=temp_step   
        if next_step[0] > self.agent_position[0]:
            return 1 # down s
        elif next_step[0] < self.agent_position[0]:
            return 0 # up w
        elif next_step[1] > self.agent_position[1]:
            return 3 # right d
        elif next_step[1] < self.agent_position[1]:
            return 2 # left a
        else:
            return 4

    def reconstruct_path(self, came_from, current):
        path = []
        while current in came_from:
            path.append(current)
            current = came_from[current]
        path.append(current)
        return path[::-1]

    def print_map(self):
        # Create a map filled with dots
        map_representation = [["." for _ in range(12)] for _ in range(12)]

        # Add boundaries
        for i in range(12):
            map_representation[i][0] = map_representation[i][11] = "|"
            map_representation[0][i] = map_representation[11][i] = "-"
        map_representation[0][0] = map_representation[0][11] = map_representation[11][0] = map_representation[11][11] = "+"

        # Set the agent, target and enemies on the map
        map_representation[self.agent_position[0]+1][self.agent_position[1]+1] = "A"
        map_representation[self.destination[0]+1][self.destination[1]+1] = "T"
        for enemy in self.enemy_positions:
            map_representation[enemy[0]+1][enemy[1]+1] = "E"

        # Set the path on the map, excluding the start and end positions
        for position in self.path[1:-1]:
            map_representation[position[0]+1][position[1]+1] = "X"

        # Print the map
        for row in map_representation:
            print(" ".join(row))


if __name__ == "__main__":
    state=[{'ID': 2, 'name': 'airplane', 'Coordinates': [9, 5], 'DistanceToAgent': 8, 'Type': 'Goal'}, {'ID': 7, 'name': 'scientist', 'Coordinates': [3, 5], 'DistanceToAgent': 6, 'Type': 'Unknown'}, {'ID': 2, 'name': 'airplane', 'Coordinates': [1, 1], 'DistanceToAgent': 4, 'Type': 'Enemy'}, {'ID': 11, 'name': 'robot', 'Coordinates': [1, 8], 'DistanceToAgent': 11, 'Type': 'Enemy'}, {'ID': 7, 'name': 'scientist', 'Coordinates': [6, 8], 'DistanceToAgent': 8, 'Type': 'Message'}, {'ID': 15, 'name': 'Agent without message', 'Coordinates': [5, 1], 'DistanceToAgent': 0, 'Type': 'Agent'}]
    solver=pathSolver(state)