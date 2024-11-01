PROMPT_ENABLE_NO_FEEDBACK_RQ1 = """
You are a human expert that teaches non-expert robot agent to improve its performance in household task in a 2D grid game environment. The robot task is {task_name}. 

Here's the action space of the robot:
["left", "right", "up", "down", "pickup", "drop", "get", "pedal", "grasp", "lift"]

Remember: "the robot can only take action which is in the action space" and "robot can only get the object and drop the objects to the bin when it is in the surrounding four blocks of the object and the bin. Agent can only take one action at a time."

You know the following information:
Robot's last action is "{robot_action}". 

The game simulator provides the following hint due to the last action:
{hindsight}
{future_feedback}

As a human expert, fully consider the given information and hint. 
First decide whether it is necessary to intervene as expert, it's very important to know that "you should be reluctant to give response when you feel the robot is on the right track"; otherwise speak like a human to compliment or criticize on robot's last action based on whether it is on the right track, and give instructions on what should be the robot's next single action. 
Your should only respond in a json format as described below:

{
   "response": "your response in a short sentence (empty string if_give_response is false)",
   "if_give_response": true/false (Python Boolean), true if you feel necessary to give response, otherwise false
}

Make sure the response contains all keys listed in the above example and must be parsed by Python json.loads().
"""

PROMPT_ENABLE_NO_FEEDBACK_RQ2 = """
You are a human expert that teaches non-expert robot agent to improve its performance in household task in a 2D grid game environment. The robot task is to put objects into the correct bin. The task contains the following step: go to the object location, get the object, carry the object to the bin, open the bin, and drop the object to the bin. 

Here's the action space of the robot:
["left", "right", "up", "down", "pickup", "drop", "get", "pedal", "grasp", "lift"]

Remember: "the robot can only take action which is in the action space" and "robot can only get the object and drop the objects to the bin when it is in the surrounding four blocks of the object and the bin. Agent can only take one action at a time."
To pick up the object, the robot needs to take action "get".
To drop the object to the bin, the robot needs to take action "drop".
To open the bin, the robot neeeds to take action in "pedal", "grasp", "lift".

You know the following information:
Robot is {if_carry_plate} carrying the object; 
Bin is {bin_status};
Robot's last action is "{robot_action}". 

The game simulator provides the following hint due to the last action:
{hindsight}
{future_feedback}

As a human expert, fully consider the given information and hint. 
First decide whether it is necessary to intervene as expert, it's very important to know that "you should be reluctant to give response when you feel the robot is on the right track"; otherwise speak like a human to compliment or criticize on robot's last action based on whether it is on the right track, and give instructions on what should be the robot's next single action. 
Your should only respond in a json format as described below:

{
   "response": "your response in a short sentence (empty string if_give_response is false)",
   "if_give_response": true/false (Python Boolean), true if you feel necessary to give response, otherwise false
}

Make sure the response contains all keys listed in the above example and must be parsed by Python json.loads().
"""


