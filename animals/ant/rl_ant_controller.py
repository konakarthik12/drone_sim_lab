import math

import omni.isaac.core.utils.torch as torch_utils
import torch
from omni.isaac.core.utils.torch.rotations import compute_heading_and_up, compute_rot, quat_conjugate, normalize_angle
from omni.isaac.lab.envs.utils.spaces import sample_space, spec_to_gym_space
from omni.isaac.lab_tasks.direct.crab.crab_env import CrabEnvCfg
from omni.isaac.lab_tasks.direct.locomotion.locomotion_env import LocomotionEnv

from animals.ant.ant_env_cfg import AntEnvCfg
from animals.art_controller import ArtController
from sim.isaac_env import IsaacEnv

TASK_NAME = "Isaac-Ant-Direct-v0"
RESUME_PATH = "/home/kkona/Documents/research/drone_sim_lab/assets/animals/ant_direct_policy.pth"

# Almost all of the meaningful code is in the LocomotionEnv class from Isaac Lab.
# That is where the observation collection and applying actions is done
# The only issue with completely using that class is that the init function starts Isaac Sim, while we want
# to init Isaac Sim outside this controller class, which I why I override the init function
# Also, I chose not to reset the crabs in the reset_idx function, because it looks
# jarring when the crab teleports while the drone is trying to capture it

class RlAntController(ArtController, LocomotionEnv):
    @property
    def num_envs(self) -> int:
        """The number of instances of the environment that are running."""
        return 1

    # Mostly copied from LocomotionEnv init and it's parent DirectRLEnv init
    def __init__(self, parent_env: IsaacEnv, env_cfg: CrabEnvCfg):
        super().__init__(parent_env, env_cfg.robot)
        self.env = parent_env
        self.env_cfg = env_cfg
        self.cfg = env_cfg

        self.observation_space = spec_to_gym_space(self.env_cfg.observation_space)
        self.action_space = spec_to_gym_space(self.env_cfg.action_space)

        self.episode_length_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.reset_terminated = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.reset_time_outs = torch.zeros_like(self.reset_terminated)
        self.reset_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.sim.device)


        self.sim = parent_env.world
        self.action_scale = env_cfg.action_scale
        self.joint_gears = torch.tensor(env_cfg.joint_gears, dtype=torch.float32, device=self.sim.device)
        self.motor_effort_ratio = torch.ones_like(self.joint_gears, device=self.sim.device)

        self.potentials = torch.zeros(self.num_envs, dtype=torch.float32, device=self.sim.device)
        self.prev_potentials = torch.zeros_like(self.potentials)
        self.targets = torch.tensor([1000, 0, 0], dtype=torch.float32, device=self.sim.device).repeat(
            (self.num_envs, 1)
        )
        self.start_rotation = torch.tensor([1, 0, 0, 0], device=self.sim.device, dtype=torch.float32)
        self.up_vec = torch.tensor([0, 0, 1], dtype=torch.float32, device=self.sim.device).repeat((self.num_envs, 1))
        # heading_mark = [1, 0, 0]
        heading_mark = [0, 1, 0]
        self.heading_vec = torch.tensor(heading_mark, dtype=torch.float32, device=self.sim.device).repeat(
            (self.num_envs, 1)
        )
        self.inv_start_rot = quat_conjugate(self.start_rotation).repeat((self.num_envs, 1))
        self.basis_vec0 = self.heading_vec.clone()
        self.basis_vec1 = self.up_vec.clone()

        # instantiate actions (needed for tasks for which the observations computation is dependent on the actions)
        self.actions = sample_space(self.action_space, self.sim.device, batch_size=1, fill_value=0)

        self.episode_length_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        # self.max_episode_length = math.ceil(
        #     self.env_cfg.episode_length_s / (self.env_cfg.sim.dt * self.env_cfg.decimation))
        self._joint_dof_idx = None
        self.last_obs = None

        self.reset_counter = torch.zeros(self.num_envs, dtype=torch.int32, device=self.sim.device)
        self.died = torch.zeros(self.num_envs, dtype=torch.bool, device=self.sim.device)
        self.reached_destination = torch.zeros(self.num_envs, dtype=torch.bool, device=self.sim.device)
        self.to_target = torch.zeros_like(self.targets)

    def post_init(self):
        self._joint_dof_idx, _ = self.robot.find_joints(".*")

    def apply_action(self):
        self._apply_action()
        self.robot.write_data_to_sim()

    def update_obs(self):
        observations = self._get_observations()
        self.last_obs = observations["policy"]


    def get_dones(self):
        self._compute_intermediate_values()
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        died = self.torso_position[:, 2] < self.env_cfg.termination_height
        return died, time_out

    def reset_idx(self):
        self.episode_length_buf[:] = 0
        self.reset_counter[:] += 1
        self.sample_next_goal()
        self._compute_intermediate_values()
        self.robot.write_data_to_sim()
