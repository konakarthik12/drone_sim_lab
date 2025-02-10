from omni.isaac.lab.assets import ArticulationCfg

from sim.controller import Controller


class ArtController(Controller):
    def __init__(self, parent_env, cfg: ArticulationCfg):
        from omni.isaac.lab.assets import Articulation
        self.env = parent_env
        self.sim = parent_env.sim
        self.robot = Articulation(cfg)

    def post_step(self, *args, **kwargs):
        self.robot.update(self.env.dt)

    def reset(self):
        root_state = self.robot.data.default_root_state.clone()
        self.robot.write_root_link_pose_to_sim(root_state[:, :7])
        self.robot.write_root_com_velocity_to_sim(root_state[:, 7:])
        # set joint positions with some noise
        joint_pos, joint_vel = self.robot.data.default_joint_pos.clone(), self.robot.data.default_joint_vel.clone()
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel)
        # clear internal buffers
        self.robot.reset()
