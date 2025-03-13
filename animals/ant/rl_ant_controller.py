import math

import omni.isaac.core.utils.torch as torch_utils
import torch
from omni.isaac.core.utils.torch.rotations import compute_heading_and_up, compute_rot, quat_conjugate
from omni.isaac.lab.envs.utils.spaces import sample_space, spec_to_gym_space

from animals.ant.ant_env_cfg import AntEnvCfg
from animals.art_controller import ArtController
from sim.isaac_env import IsaacEnv

TASK_NAME = "Isaac-Ant-Direct-v0"
RESUME_PATH = "/home/kkona/Documents/research/drone_sim_lab/assets/animals/ant_direct_policy.pth"


class RlAntController(ArtController):
    def __init__(self, parent_env: IsaacEnv, env_cfg: AntEnvCfg):
        super().__init__(parent_env, env_cfg.robot)
        self.env = parent_env
        self.env_cfg = env_cfg

        self.observation_space = spec_to_gym_space(self.env_cfg.observation_space)
        self.action_space = spec_to_gym_space(self.env_cfg.action_space)

        # -- init buffers
        self.reset_terminated = False
        self.reset_time_outs = False
        self.reset_buf = False

        self.sim = parent_env.world
        self.action_scale = env_cfg.action_scale
        self.joint_gears = torch.tensor(env_cfg.joint_gears, dtype=torch.float32, device=self.sim.device)
        self.motor_effort_ratio = torch.ones_like(self.joint_gears, device=self.sim.device)

        self.potentials = torch.zeros(1, dtype=torch.float32, device=self.sim.device)
        self.prev_potentials = torch.zeros_like(self.potentials)
        self.targets = torch.tensor([1000, 0, 0], dtype=torch.float32, device=self.sim.device)
        self.start_rotation = torch.tensor([1, 0, 0, 0], device=self.sim.device, dtype=torch.float32)
        self.up_vec = torch.tensor([0, 0, 1], dtype=torch.float32, device=self.sim.device)
        # heading_mark = [1, 0, 0]
        heading_mark = [0, 1, 0]
        self.heading_vec = torch.tensor(heading_mark, dtype=torch.float32, device=self.sim.device)
        self.inv_start_rot = quat_conjugate(self.start_rotation)
        self.basis_vec0 = self.heading_vec.clone()
        self.basis_vec1 = self.up_vec.clone()

        # instantiate actions (needed for tasks for which the observations computation is dependent on the actions)
        self.actions = sample_space(self.action_space, self.sim.device, batch_size=1, fill_value=0)

        self.episode_length = 0
        self.max_episode_length = math.ceil(
            self.env_cfg.episode_length_s / (self.env_cfg.sim.dt * self.env_cfg.decimation))
        self.joint_dof_idx = None
        self.last_obs = None

    def post_init(self):
        self.joint_dof_idx, _ = self.robot.find_joints(".*")

    def apply_action(self):
        forces = self.action_scale * self.joint_gears * self.actions
        self.robot.set_joint_effort_target(forces, joint_ids=self.joint_dof_idx)
        self.robot.write_data_to_sim()

    def _compute_intermediate_values(self):
        self.torso_position, self.torso_rotation = self.robot.data.root_link_pos_w.squeeze(), self.robot.data.root_link_quat_w.squeeze()
        self.velocity, self.ang_velocity = self.robot.data.root_com_lin_vel_w.squeeze(), self.robot.data.root_com_ang_vel_w.squeeze()
        self.dof_pos, self.dof_vel = self.robot.data.joint_pos.squeeze(), self.robot.data.joint_vel.squeeze()

        (
            self.up_proj,
            self.heading_proj,
            self.up_vec,
            self.heading_vec,
            self.vel_loc,
            self.angvel_loc,
            self.roll,
            self.pitch,
            self.yaw,
            self.angle_to_target,
            self.dof_pos_scaled,
            self.prev_potentials,
            self.potentials,
        ) = compute_intermediate_values(
            self.targets,
            self.torso_position,
            self.torso_rotation,
            self.velocity,
            self.ang_velocity,
            self.dof_pos,
            self.robot.data.soft_joint_pos_limits[0, :, 0],
            self.robot.data.soft_joint_pos_limits[0, :, 1],
            self.inv_start_rot,
            self.basis_vec0,
            self.basis_vec1,
            self.potentials,
            self.prev_potentials,
            self.env_cfg.sim.dt,
        )

    def update_obs(self):
        obs = torch.cat(
            (
                self.torso_position[2].unsqueeze(0),
                self.vel_loc,
                (self.angvel_loc * self.env_cfg.angular_velocity_scale),
                normalize_angle(self.yaw),
                normalize_angle(self.roll),
                normalize_angle(self.angle_to_target),
                self.up_proj,
                self.heading_proj,
                self.dof_pos_scaled,
                self.dof_vel * self.env_cfg.dof_vel_scale,
                self.actions.squeeze(0),
            ),
            dim=-1,
        ).unsqueeze(0)
        self.last_obs = obs

    def get_rewards(self, reset_terminated):
        total_reward = compute_rewards(
            self.actions,
            reset_terminated,
            self.env_cfg.up_weight,
            self.env_cfg.heading_weight,
            self.heading_proj,
            self.up_proj,
            self.dof_vel,
            self.dof_pos_scaled,
            self.potentials,
            self.prev_potentials,
            self.env_cfg.actions_cost_scale,
            self.env_cfg.energy_cost_scale,
            self.env_cfg.dof_vel_scale,
            self.env_cfg.death_cost,
            self.env_cfg.alive_reward_scale,
            self.motor_effort_ratio,
        )
        return total_reward

    def get_dones(self):
        self._compute_intermediate_values()
        time_out = self.episode_length >= self.max_episode_length - 1
        died = self.torso_position[2] < self.env_cfg.termination_height
        return died, time_out

    def reset_idx(self):
        self.episode_length = 0
        self.robot.reset()
        joint_pos = self.robot.data.default_joint_pos
        joint_vel = self.robot.data.default_joint_vel
        default_root_state = self.robot.data.default_root_state
        self.robot.write_root_link_pose_to_sim(default_root_state[:, :7])
        self.robot.write_root_com_velocity_to_sim(default_root_state[:, 7:])
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None)

        to_target = self.targets - default_root_state[:, :3]
        to_target[:, 2] = 0.0
        self.potentials = -torch.norm(to_target, p=2, dim=-1) / self.env_cfg.sim.dt

        self._compute_intermediate_values()

        self.robot.write_data_to_sim()


@torch.jit.script
def compute_rewards(
        actions: torch.Tensor,
        reset_terminated: torch.Tensor,
        up_weight: float,
        heading_weight: float,
        heading_proj: torch.Tensor,
        up_proj: torch.Tensor,
        dof_vel: torch.Tensor,
        dof_pos_scaled: torch.Tensor,
        potentials: torch.Tensor,
        prev_potentials: torch.Tensor,
        actions_cost_scale: float,
        energy_cost_scale: float,
        dof_vel_scale: float,
        death_cost: float,
        alive_reward_scale: float,
        motor_effort_ratio: torch.Tensor,
):
    heading_weight_tensor = torch.ones_like(heading_proj) * heading_weight
    heading_reward = torch.where(heading_proj > 0.8, heading_weight_tensor, heading_weight * heading_proj / 0.8)

    # aligning up axis of robot and environment
    up_reward = torch.zeros_like(heading_reward)
    up_reward = torch.where(up_proj > 0.93, up_reward + up_weight, up_reward)

    # energy penalty for movement
    actions_cost = torch.sum(actions ** 2, dim=-1)
    electricity_cost = torch.sum(
        torch.abs(actions * dof_vel * dof_vel_scale) * motor_effort_ratio.unsqueeze(0),
        dim=-1,
    )

    # dof at limit cost
    dof_at_limit_cost = torch.sum(dof_pos_scaled > 0.98, dim=-1)

    # reward for duration of staying alive
    alive_reward = torch.ones_like(potentials) * alive_reward_scale
    progress_reward = potentials - prev_potentials
    progress_reward_scale = 2

    dof_at_limit_cost_scale = 0.05
    total_reward = (
            progress_reward_scale * progress_reward
            + alive_reward
            + up_reward
            + heading_reward
            - actions_cost_scale * actions_cost
            - energy_cost_scale * electricity_cost
            - dof_at_limit_cost_scale * dof_at_limit_cost
    )
    # adjust reward for fallen agents
    total_reward = torch.where(reset_terminated, torch.ones_like(total_reward) * death_cost, total_reward)
    return total_reward


@torch.jit.script
def compute_intermediate_values(
        targets: torch.Tensor,
        torso_position: torch.Tensor,
        torso_rotation: torch.Tensor,
        velocity: torch.Tensor,
        ang_velocity: torch.Tensor,
        dof_pos: torch.Tensor,
        dof_lower_limits: torch.Tensor,
        dof_upper_limits: torch.Tensor,
        inv_start_rot: torch.Tensor,
        basis_vec0: torch.Tensor,
        basis_vec1: torch.Tensor,
        potentials: torch.Tensor,
        prev_potentials: torch.Tensor,
        dt: float,
):
    to_target = targets - torso_position
    to_target[2] = 0.0

    torso_quat, up_proj, heading_proj, up_vec, heading_vec = compute_heading_and_up(
        torso_rotation.unsqueeze(0), inv_start_rot.unsqueeze(0), to_target.unsqueeze(0), basis_vec0.unsqueeze(0),
        basis_vec1.unsqueeze(0), 2
    )
    torso_quat = torso_quat.squeeze()
    vel_loc, angvel_loc, roll, pitch, yaw, angle_to_target = compute_rot(
        torso_quat.unsqueeze(0), velocity.unsqueeze(0), ang_velocity.unsqueeze(0), targets.unsqueeze(0),
        torso_position.unsqueeze(0)
    )

    dof_pos_scaled = torch_utils.maths.unscale(dof_pos, dof_lower_limits, dof_upper_limits)

    prev_potentials[:] = potentials
    potentials = -torch.norm(to_target, p=2, dim=-1) / dt

    return (
        up_proj,
        heading_proj,
        up_vec,
        heading_vec,
        vel_loc.squeeze(0),
        angvel_loc.squeeze(0),
        roll,
        pitch,
        yaw,
        angle_to_target,
        dof_pos_scaled,
        prev_potentials,
        potentials,
    )


def normalize_angle(x: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(x), torch.cos(x))
