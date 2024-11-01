import numpy as np

from metaworld.policies.action import Action
from metaworld.policies.policy import Policy, assert_fully_parsed, move


class SawyerHammerV2Policy(Policy):

    @staticmethod
    @assert_fully_parsed
    def _parse_obs(obs):
        return {
            'hand_pos': obs[:3],
            'gripper': obs[3],
            'hammer_pos': obs[4:7],
            'unused_info': obs[7:],
        }

    def get_action(self, obs):
        o_d = self._parse_obs(obs)

        action = Action({
            'delta_pos': np.arange(3),
            'grab_effort': 3
        })
        action_id,language=self._desired_pos(o_d)
        gripper_id = self._grab_effort(o_d)

        return (action_id,gripper_id),language

    @staticmethod
    def _desired_pos(o_d):
        pos_curr = o_d['hand_pos']
        pos_puck = o_d['hammer_pos'] + np.array([-.04, .0, -.01])
        pos_goal = np.array([0.24, 0.71, 0.11]) + np.array([-.19, .0, .05])
        # print(abs(pos_curr[2] - pos_puck[2]))
        # If error in the XY plane is greater than 0.02, place end effector above the puck
        if np.linalg.norm(pos_curr[:2] - pos_puck[:2]) > 0.04:
            if(np.linalg.norm(pos_curr[:2] - pos_puck[:2])<0.2 and pos_curr[2]<pos_puck[2]+0.1):
                if pos_curr[2]<0.1:
                    # return pos_curr+np.array([0., 0., 0.1]),"Raise the gripper."
                    return 0,0
                else:
                    return 1,1
                    # return pos_curr, "Open the gripper."
            return 2,2
            # return pos_puck + np.array([0., 0., 0.1]),"Place gripper above the tool."
        # Once XY error is low enough, drop end effector down on top of hammer
        elif (abs(pos_curr[2] - pos_puck[2]) > 0.05 and pos_puck[-1] < 0.03):
            if o_d['gripper']<0.8:
                return 1,1
                # return pos_curr, "Open the gripper."
            return 4,4
            # return pos_puck + np.array([0., 0., 0.03]), "Grasp the tool."
        # If not at the same X pos as the peg, move over to that plane
        elif np.linalg.norm(pos_curr[[0,2]] - pos_goal[[0,2]]) > 0.02:
            return 3,3
            # return np.array([pos_goal[0], pos_curr[1], pos_goal[2]]), "Aim at the goal."
        # Move to the peg
        else:
            return 6,6
            # return pos_goal, "Get to the goal."

    @staticmethod
    def _grab_effort(o_d):
        pos_curr = o_d['hand_pos']
        pos_puck = o_d['hammer_pos'] + np.array([-.04, .0, -.01])

        if np.linalg.norm(pos_curr[:2] - pos_puck[:2]) > 0.04 or abs(pos_curr[2] - pos_puck[2]) > 0.1:
            if(o_d['gripper']<0.9):
                # return -0.2
                return 0
            # return 0.
            return 1
        # While end effector is moving down toward the hammer, begin closing the grabber
        else:
            # return 0.8
            return 2