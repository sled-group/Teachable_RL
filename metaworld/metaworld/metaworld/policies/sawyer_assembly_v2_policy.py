import numpy as np

from metaworld.policies.action import Action
from metaworld.policies.policy import Policy, assert_fully_parsed, move


class SawyerAssemblyV2Policy(Policy):

    @staticmethod
    @assert_fully_parsed
    def _parse_obs(obs):
        return {
            'hand_pos': obs[:3],
            'gripper': obs[3],
            'wrench_pos': obs[4:7],
            'peg_pos': obs[-3:],
            'unused_info': obs[7:-3],
        }

    def get_action(self, obs):
        o_d = self._parse_obs(obs)
        action = Action({
            'delta_pos': np.arange(3),
            'grab_effort': 3
        })
        action_id,language_pos=self._desired_pos(o_d)
        # action['delta_pos'] = to_xyz
        # action['delta_pos'] = move(o_d['hand_pos'], to_xyz=to_xyz, p=10.)
        gripper_id,_ = self._grab_effort(o_d)
        # action['grab_effort'] = self._grab_effort(o_d)

        return (action_id,gripper_id),language_pos
        # return action.array,language

    @staticmethod
    def _desired_pos(o_d):
        pos_curr = o_d['hand_pos']
        pos_wrench = o_d['wrench_pos'] + np.array([-.02, .0, .0])
        pos_peg = o_d['peg_pos'] + np.array([.12, .0, .14])
        # If XY error is greater than 0.02, place end effector above the wrench
        if (np.linalg.norm(pos_curr[:2] - pos_wrench[:2]) > 0.02 and not (abs(pos_curr[2] - pos_wrench[2])<0.06)):
            if(np.linalg.norm(pos_curr[:2] - pos_wrench[:2])<0.2 and pos_curr[2] < pos_wrench[2]+0.1):
                if pos_curr[2] < 0.2:
                    # return 0,"Raise the gripper."
                    return 0,0
                    # return pos_curr+np.array([0., 0., 0.1]),"Place gripper above the tool."
                else:
                    # return 1,"Open the gripper."
                    return 1,1
                    # return pos_curr,"Open the gripper."
            # return 2, "Place gripper above the tool."
            return 2, 2
            # return pos_wrench + np.array([0., 0., 0.1]), "Place gripper above the tool."
        # (For later) if lined up with peg, drop down on top of it
        elif (np.linalg.norm(pos_wrench[:2]-pos_peg[:2]) <= 0.02):
            return 3,3
            # return 3,"Aim at the goal."
            # return pos_peg + np.array([(pos_curr[0]-pos_wrench[0]),0,0]) + np.array([.0, .0, -.2]),"Aim at the goal."
        # Once XY error is low enough, drop end effector down on top of wrench
        elif abs(pos_curr[2] - pos_wrench[2]) > 0.05 or  o_d['gripper'] > 0.6:
            return 4,4
            # return pos_wrench + np.array([0., 0., 0.03]),"Grasp the tool."
        # If not at the same Z height as the goal, move up to that plane
        elif abs(pos_curr[2] - pos_peg[2]) > 0.04:
            return 5,5
            # return 5,"Raise the tool."
            # return np.array([pos_curr[0], pos_curr[1], pos_peg[2]]),"Raise the gripper."
        # If XY error is greater than 0.02, place end effector above the peg
        else:
            return 6, 6
            # return pos_peg+np.array([(pos_curr[0]-pos_wrench[0]),0,0])+np.array([0,-0.005,0]), "Get to the goal."

    @staticmethod
    def _grab_effort(o_d):
        pos_curr = o_d['hand_pos']
        pos_wrench = o_d['wrench_pos'] + np.array([-.02, .0, .0])
        pos_peg = o_d['peg_pos'] + np.array([.12, .0, .14])
        # print("grasp debug: ",np.linalg.norm(pos_curr[:2] - pos_wrench[:2]),abs(pos_curr[2] - pos_wrench[2]) )
        if (np.linalg.norm(pos_curr[:2] - pos_wrench[:2]) > 0.02 or abs(pos_curr[2] - pos_wrench[2]) > 0.12):
            if(o_d['gripper']<0.9):
                # return -0.2
                return 0, "0"
            # return 0.
            return 1, "1"
        # Until hovering over peg, keep hold of wrench
        else:
            # return 0.2
            return 2, "2"