import math

GRIPPER_CLOSE = 0.05
GRIPPER_OPEN = -0.95


def lerp(a, b, t):
    return a + t * (b - a)


def step_towards(a, b, step_size=0.1):
    e = b - a
    if math.isclose(e, 0, abs_tol=1e-2):
        return b
    return a + step_size * e


class Grippers:
    def __init__(self, world, drone):
        self.dc = world.dc_interface
        self.drone = drone
        self.joint_names = ["grip_1_joint", "grip_2_joint", "grip_3_joint"]

        def find_dof(joint_name):
            return self.dc.find_articulation_dof(self.drone, joint_name)

        self.articulation_dof = list(map(find_dof, self.joint_names))

    # action space (1,): [gripper_pos] (0.0, 1.0)
    # gripper_pos: 0.0 -> open, 1.0 -> close
    def move_grippers(self, desired_joint_pos, step_size=0.09):
        # internally, the gripper joint is in the range (0.05, -0.95)
        desired_joint_pos = lerp(GRIPPER_OPEN, GRIPPER_CLOSE, desired_joint_pos)
        for joint in self.articulation_dof:
            curr_pos = self.dc.get_dof_position(joint)
            target_pos = step_towards(curr_pos, desired_joint_pos, step_size)
            self.dc.set_dof_position_target(joint, target_pos)


#     arm_1_joint is just above the base joint of the manipulator, joint range (-2,1.37)
#     arm_2_joint is just above arm_1_joint, joint range (-3.78, .87)
class Arms:
    def __init__(self, world, drone):
        self.dc = world.dc_interface
        self.drone = drone
        self.joint_names = [ "arm_1_joint", "arm_2_joint"]

        def find_dof(joint_name):
            dof = self.dc.find_articulation_dof(self.drone, joint_name)
            print(joint_name, dof)
            return dof

        self.articulation_dof = list(map(find_dof, self.joint_names))

    # action space (3,): [joint1_pos, joint2_pos]
    def move_arms(self, desired_joint_poses, step_size=0.1):
        for joint, desired_joint_pos in zip(self.articulation_dof, desired_joint_poses):
            curr_pos = self.dc.get_dof_position(joint)
            target_pos = step_towards(curr_pos, desired_joint_pos, step_size)
            self.dc.set_dof_position_target(joint, target_pos)

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
    def __init__(self, world, drone):
        self.world = world
        self.drone = drone
        self.world.reset()
        dc = world.dc_interface
        drone_articulation = dc.get_articulation(self.drone._stage_prefix)
        dc.wake_up_articulation(drone_articulation)
        self.arms = Arms(world, drone_articulation)
        self.grippers = Grippers(world, drone_articulation)

    def step(self, action):
        self.arms.move_arms(action[:2])
        self.grippers.move_grippers(action[2])
