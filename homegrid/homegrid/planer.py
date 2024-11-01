
from collections import deque
from homegrid.utils import extract_obj_name, extract_related_obj_from_list

valid_position = """\
..............
..............
........xxxxx.
........x..xx.
........xxxxx.
........xxxxx.
........xxxxx.
.xxxxxxxxxxxx.
.xxxxxxxx...x.
.xxxxxxxx...x.
........xxxxx.
..............
""".splitlines()

height, width = len(valid_position), len(valid_position[0])


class Planer():
    # input: initial state after reset
    # output: list of action to be taken
    
    # left = 0
    # right = 1
    # up = 2
    # down = 3
    # # item actions
    # pickup = 4
    # drop = 5
    # # storage actions
    # get = 6
    # pedal = 7
    # grasp = 8
    # lift = 9
    def __init__(self, init_state_info):
        self.action_list = []
        self.init_state_info = init_state_info
        
        LIVINGROOM = [9, 4]
        DININGROOM = [9, 7]
        KITCHEN = [5, 8]
        
        for object in init_state_info["objects"]:
            x, y = object["pos"]
            row = list(valid_position[y])
            row[x] = '.'
            valid_position[y] = ''.join(row)
        
        agent_pos = init_state_info["agent"]["pos"]
        task = init_state_info["task_description"]
        task_tokens = task.split()
        
        related_obj = extract_obj_name(task=task)
        object_dict_list = extract_related_obj_from_list(init_state_info["objects"], related_obj)
        
        if task_tokens[0] == 'find':
            obj_pos = object_dict_list[related_obj[0]]["pos"] 
            action_list, agent_pos = Planer.move_and_face_object(agent_pos, obj_pos)
            self.action_list += action_list

        elif task_tokens[0] == 'get':
            obj_pos = object_dict_list[related_obj[0]]["pos"]
            self.action_list += Planer.move_and_face_object(agent_pos, obj_pos)[0]
        
            next_action = None
            if task_tokens[-1] == 'bin':
                next_action = 6
            else:
                next_action = 4
            self.action_list.append(next_action)
                
                
            
        elif task_tokens[0] == 'move':
            obj_pos = object_dict_list[related_obj[0]]["pos"]
            
            action_list, agent_pos = Planer.move_and_face_object(agent_pos, obj_pos)
            self.action_list += action_list
            
            self.action_list.append(4)
            
            next_room = None
            
            if task_tokens[-1] == 'kitchen':
                next_room = KITCHEN
                
            elif task_tokens[-2] == 'dining':
                next_room = DININGROOM
                    
            elif task_tokens[-2] == 'living':
                next_room = LIVINGROOM
            else:
                raise NotImplementedError
            
            self.action_list += Planer.move_and_face_object(agent_pos, next_room)[0]
            
            self.action_list.append(5)
                
        elif task_tokens[0] == 'open':
            obj_pos = object_dict_list[related_obj[0]]["pos"]
            available_action = object_dict_list[related_obj[0]]["action"]
            
            self.action_list += Planer.move_and_face_object(agent_pos, obj_pos)[0]
            
            next_action = None
            if available_action == "pedal":
                next_action = 7
            elif available_action == "grasp":
                next_action = 8
            elif available_action == "lift":
                next_action = 9
            else:
                raise NotImplementedError
            self.action_list.append(next_action)
                
        elif task_tokens[0] == 'put':
            obj_pos_1 = object_dict_list[related_obj[0]]["pos"]
            obj_pos_2 = object_dict_list[related_obj[1]]["pos"]
            
            storage_state = object_dict_list[related_obj[1]]["state"]
            
            action_list, agent_pos = Planer.move_and_face_object(agent_pos, obj_pos_1)
            self.action_list += action_list
            
            self.action_list.append(4)
            self.action_list += Planer.move_and_face_object(agent_pos, obj_pos_2)[0]
            
            if storage_state == "closed":
                available_action = object_dict_list[related_obj[1]]["action"]
                next_action = None
                if available_action == "pedal":
                    next_action = 7
                elif available_action == "grasp":
                    next_action = 8
                elif available_action == "lift":
                    next_action = 9
                else:
                    raise NotImplementedError
                self.action_list.append(next_action)
            elif storage_state != "open":
                raise NotImplementedError
            self.action_list.append(5)
        else:
            raise NotImplementedError

    def step(self):
        return self.action_list.pop(0)

    @staticmethod
    def bfs_actions_to_adjacent(start, end, valid_position=valid_position):
        start = tuple(start)
        end = tuple(end)
        height, width = len(valid_position), len(valid_position[0])
        directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]  # right, down, left, up

        queue = deque([start])
        came_from = {start: None}
        visited = set([start])
        found_adjacent = None

        while queue:
            current = queue.popleft()
            
            # Check if it's adjacent to the end
            if Planer.is_adjacent(current, end):
                found_adjacent = current
                break

            for dx, dy in directions:
                nx, ny = current[0] + dx, current[1] + dy
                neighbor = (nx, ny)

                if (0 <= nx < width and 0 <= ny < height and
                        valid_position[ny][nx] == 'x' and
                        neighbor not in visited):
                    
                    queue.append(neighbor)
                    visited.add(neighbor)
                    came_from[neighbor] = current

        # Reconstruct the actions
        if not found_adjacent:
            return None  # No path found

        path = [found_adjacent]
        actions = []
        while path[-1] != start:
            current = path[-1]
            previous = came_from[current]
            path.append(previous)
            actions.append(Planer.position_to_action(current, previous))
        
        actions.reverse()
        return actions
    
    @staticmethod
    def move_and_face_object(start_pos, obj_pos):
        action_list = Planer.bfs_actions_to_adjacent(start_pos, obj_pos)
        new_position = Planer.move_agent(start_pos, action_list)

        if len(action_list) > 0:
            action_list += Planer.faceObj(new_position, obj_pos, action_list[-1])
        else:
            action_list += Planer.faceObj(new_position, obj_pos, None)
        return action_list, new_position

    @staticmethod
    def faceObj(cur_pos, obj_pos, last_pos):
        # left = 0
        # right = 1
        # up = 2
        # down = 3
        cur_pos_x, cur_pos_y = cur_pos
        obj_pos_x, obj_pos_y = obj_pos
        next_action = None
        if cur_pos_y == obj_pos_y:
            if cur_pos_x - obj_pos_x == 1:
                next_action = 0
            elif cur_pos_x - obj_pos_x == -1:
                next_action = 1
            else:
                raise NotImplementedError
        elif cur_pos_x == obj_pos_x:
            if cur_pos_y - obj_pos_y == 1:
                next_action = 2
            elif cur_pos_y - obj_pos_y == -1:
                next_action = 3
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError
        
        if next_action != last_pos:
            return [next_action]
        else:
            return []

    @staticmethod
    def move_agent(start_position, actions):
        # Define the possible actions as:
        # go left = 0
        # go right = 1
        # go up = 2
        # go down = 3
        
        x, y = start_position
        new_x, new_y = x, y
        for action in actions:
            if action == 0:   # go left
                new_x = x - 1
            elif action == 1: # go right
                new_x = x + 1
            elif action == 2: # go up
                new_y = y - 1
            elif action == 3: # go down
                new_y = y + 1
            else:
                raise ValueError(f"Invalid action: {action}")
            if new_x in range(0, width) and new_y in range(0, height) and valid_position[new_y][new_x] == 'x':
                x, y = new_x, new_y
        return [x, y]
    
    @staticmethod   
    def position_to_action(current, previous):
        dx = current[0] - previous[0]
        dy = current[1] - previous[1]

        if dx == 1 and dy == 0:
            return 1  # right
        elif dx == -1 and dy == 0:
            return 0  # left
        elif dx == 0 and dy == 1:
            return 3  # down
        elif dx == 0 and dy == -1:
            return 2  # up
        else:
            raise NotImplementedError

    @staticmethod
    def is_adjacent(a, b):
        return abs(a[0]-b[0]) + abs(a[1]-b[1]) == 1