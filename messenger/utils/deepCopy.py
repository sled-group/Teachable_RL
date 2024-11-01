import gym
from utils.observation_process import observationProcessor

def generate_level_desc(state,newTask=False):
    # Initialize a 10x10 grid filled with dots
    grid = [['.' for _ in range(10)] for _ in range(10)]
    
    # Mapping for each entity to its character representation
    
    entity_map = {
        "enemy": "E",
        "message": "M",
        "goal": "G",
        "decoy_message": "N",
        "decoy_goal": "H",
    }
    
    if(newTask):
        entity_map={
        "enemy": "E",
        "message": "G",
        "goal": "M",
        "decoy_message": "N",
        "decoy_goal": "H",
        }
        
    # Fill in the grid based on the state
    for key, value in state.items():
        x, y = value["pos"]
        if(grid[9-y][x]!='.'):
            if(9-y<=8 and grid[9-y+1][x]=='.'):
                y=y-1
            elif (9-y>=1 and grid[9-y-1][x]=='.'):
                y=y+1
            elif (x<=8 and grid[9-y][x+1]=='.'):
                x=x+1
            elif (x>=1 and grid[9-y][x-1]=='.'):
                x=x-1
        if key == "agent":
            if value["e"] == "with_Message":
                grid[9-y][x] = "Y"
            elif value["e"] == "without_Message":
                grid[9-y][x] = "X"
        else:
            grid[9-y][x] = entity_map[key]
    
    # Convert the 2D grid to the required string format
    maze_str = '\n'.join([''.join(row) for row in grid])
    return maze_str

def generate_newTask_level_desc(state):
    # Initialize a 10x10 grid filled with dots
    grid = [['.' for _ in range(10)] for _ in range(10)]
    
    # Mapping for each entity to its character representation
    newTast_map={
        "enemy": "E",
        "message": "G",
        "goal": "M",
        "decoy_message": "N",
        "decoy_goal": "H",
    }
    # Fill in the grid based on the state
    for key, value in state.items():
        x, y = value["pos"]
        if(grid[9-y][x]!='.'):
            if(9-y<=8 and grid[9-y+1][x]=='.'):
                y=y-1
            elif (9-y>=1 and grid[9-y-1][x]=='.'):
                y=y+1
            elif (x<=8 and grid[9-y][x+1]=='.'):
                x=x+1
            elif (x>=1 and grid[9-y][x-1]=='.'):
                x=x-1
        if key == "agent":
            if value["e"] == "with_Message":
                grid[9-y][x] = "Y"
            elif value["e"] == "without_Message":
                grid[9-y][x] = "X"
        else:
            grid[9-y][x] = newTast_map[key]
    
    # Convert the 2D grid to the required string format
    maze_str = '\n'.join([''.join(row) for row in grid])
    return maze_str

def modify_game_config(game_config):
    """
    Modifies the InteractionSet and TerminationSet parts of the given game configuration string.

    Args:
    game_config (str): The original game configuration string.
    new_interaction_set (str): The new rules for the InteractionSet section.
    new_termination_set (str): The new rules for the TerminationSet section.

    Returns:
    str: Modified game configuration string.
    """
    # Split the configuration into sections
    parts = game_config.split('\t')
    truncated_config = game_config.split("InteractionSet")[0].rstrip()
    remaining="""
        InteractionSet
		root wall > stepBack
		root EOS > stepBack
		avatar enemy > killSprite scoreChange=-1
		avatar decoy_message > killSprite scoreChange=-1
		avatar decoy_goal > killSprite scoreChange=-1
		no_message message > killSprite scoreChange=-1
		no_message goal > transformTo stype=with_message scoreChange=0.5
		goal avatar > killSprite
		message with_message > killSprite scoreChange=1
	TerminationSet
		SpriteCounter stype=avatar limit=0 win=False
		SpriteCounter stype=message limit=0 win=True
	LevelMapping
		. > background
		E > background enemy
		M > background message
		G > background goal
		X > background no_message
		Y > background with_message
		W > background wall
		N > background decoy_message
		H > background decoy_goal
  """
    return truncated_config+remaining

class copier:
    def __init__(self, env):
        self.game_desc = env.msgrEnv.env.game_desc
        # print(self.game_desc)
        self.level_desc = None
        self.observation_processor = observationProcessor()

    def deep_copy(self, env,newTask=False):
        # print(env)
        new_env = gym.make("msgr-train-v3")
        new_env = env.deep_copy(new_env,newTask=newTask)
        new_env.reset()
        state = self.observation_processor.generate_state(env)
        self.level_desc=generate_level_desc(state,newTask)
        if(newTask):
            self.game_desc=modify_game_config(self.game_desc)
        new_env.msgrEnv.env.loadGame(self.game_desc,self.level_desc)
        new_env.msgrEnv.stateFrame=env.msgrEnv.stateFrame
        return new_env
    
    def newTask(self,env,newTask=True):
        if(not newTask):
            return env
        new_env = gym.make("msgr-train-v3")
        new_env = env.deep_copy(new_env,newTask=True)
        new_env.reset()
        state = self.observation_processor.generate_state(env)
        self.level_desc=generate_level_desc(state)
        self.game_desc=modify_game_config(self.game_desc)
        new_env.msgrEnv.env.loadGame(self.game_desc,self.level_desc)
        message=env.msgrEnv.stateFrame["message"].copy()
        goal=env.msgrEnv.stateFrame["goal"].copy()
        env.msgrEnv.stateFrame["message"]=goal
        env.msgrEnv.stateFrame["goal"]=message
        new_env.msgrEnv.stateFrame=env.msgrEnv.stateFrame
        return new_env
