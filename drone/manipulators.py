import math

import numpy as np
from omni.isaac.lab.assets import Articulation
import carb.input

JOINT_NAMES = ["grip_1_joint", "grip_2_joint", "grip_3_joint"]

GRIPPER_CLOSE = 0.05
GRIPPER_OPEN = -0.95
from sim.dc_interface import dc


def lerp(a, b, t):
    return a + t * (b - a)


def step_towards(a, b, step_size=0.1):
    e = b - a
    if math.isclose(e, 0, abs_tol=1e-2):
        return b
    return a + step_size * e


class Grippers:
    def __init__(self, articulation: Articulation):
        def find_dof(joint_name):
            return dc.find_articulation_dof(articulation, joint_name)

        self.articulation_dof = list(map(find_dof, JOINT_NAMES))

    # action space (1,): [gripper_pos] (0.0, 1.0)
    # gripper_pos: 0.0 -> open, 1.0 -> close
    def move_grippers(self, desired_joint_pos, step_size=0.09):
        # internally, the gripper joint is in the range (0.05, -0.95)
        desired_joint_pos = lerp(GRIPPER_OPEN, GRIPPER_CLOSE, desired_joint_pos)
        for joint in self.articulation_dof:
            curr_pos = dc.get_dof_position(joint)
            target_pos = step_towards(curr_pos, desired_joint_pos, step_size)
            dc.set_dof_position_target(joint, target_pos)


#     arm_1_joint is just above the base joint of the manipulator, joint range (-2,1.37)
#     arm_2_joint is just above arm_1_joint, joint range (-3.78, .87)
class Arms:
    def __init__(self, drone):
        self.drone = drone
        self.joint_names = ["arm_1_joint", "arm_2_joint"]

        def find_dof(joint_name):
            dof = dc.find_articulation_dof(self.drone, joint_name)
            return dof

        self.articulation_dof = list(map(find_dof, self.joint_names))

    # action space (3,): [joint1_pos, joint2_pos]
    def move_arms(self, desired_joint_poses, step_size=0.1):
        for joint, desired_joint_pos in zip(self.articulation_dof, desired_joint_poses):
            curr_pos = dc.get_dof_position(joint)
            target_pos = step_towards(curr_pos, desired_joint_pos, step_size)
            dc.set_dof_position_target(joint, target_pos)

    def home_position(self):
        self.move_arms([0.0, 0.0])

    def pick_place_position(self):
        self.move_arms([-0.3, -1.3])


class Manipulators:
    def __init__(self, world, drone):
        self.world = world
        self.drone = drone

    #


class Manipulators:
    def __init__(self, world):
        self.world = world

    def post_init(self, articulation):
        articulation = articulation
        dc.wake_up_articulation(articulation)
        self.arms = Arms(articulation)
        self.grippers = Grippers(articulation)

    def step(self, action):
        self.arms.move_arms(action[:2])
        self.grippers.move_grippers(action[2])


class ManipulatorState:
    def __init__(self, joint1_pos=0,joint2_pos = 0, gripper=0.0):
        self.joint1_pos = joint1_pos
        self.joint2_pos = joint2_pos
        self.gripper = gripper

    def gamepad_callback(self, event):

        if event.input == carb.input.GamepadInput.X:
            if event.value > 0.5:
                self.gripper = 1 - self.gripper
        if event.input == carb.input.GamepadInput.DPAD_UP:
            self.joint1_pos += 0.1
        if event.input == carb.input.GamepadInput.DPAD_DOWN:
            self.joint1_pos -= 0.1
        if event.input == carb.input.GamepadInput.LEFT_SHOULDER:
            self.joint2_pos -= 0.05
        if event.input == carb.input.GamepadInput.RIGHT_SHOULDER:
            self.joint2_pos += 0.05
        return True

    def as_action(self):
        return np.array([self.joint1_pos, self.joint2_pos, self.gripper])
