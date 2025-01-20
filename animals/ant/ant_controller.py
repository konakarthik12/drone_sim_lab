import numpy as np


class AntController:
    def __init__(self, parent_env, init_pose=np.array([0, 0, 0.7]), crab_scale=0.1):
        from omni.isaac.core.articulations import Articulation

        self.world = parent_env.world

        crab_path = "/home/kkona/Documents/research/drone_sim_lab/assets/animals/ant.usd"
        prim_path = "/World/crab"
        import omni.isaac.core.utils.stage as stage_utils

        stage_utils.add_reference_to_stage(crab_path, prim_path)
        self.init_pose = [6, 0, 0.3]
        self.robot = Articulation(prim_path=prim_path,
                                  name="ant",
                                  position=self.init_pose,  # self.default_object_pose,
                                  scale=[0.045] * 3)
        self.world.scene.add(self.robot)
        # self.robot.initialize()
        num_dof = 5
        self.action_space = np.zeros((num_dof,), dtype=np.float32)


    def step(self, action):
        pass
    #
    # def apply_action(self, action):
    #     """
    #     action : np.array
    #         shape : (total_actions, )
    #         type : float
    #     """
    #     from omni.isaac.core.utils.types import ArticulationAction
    #     # assert -1 <= action.all() <= 1
    #     assert np.all((action > -1) & (action < 1))
    #     # action = np.clip(action, -1, 1) * 5
    #     action = action * 20
    #     # action = self.action_center + action * self.action_range
    #
    #     # action_ = torch.tensor(action, dtype=torch.float32)
    #     # self.robot.apply_action(action_)
    #     robot = self.robot
    #     # efforts = torch.randn_like(robot.data.joint_pos) * 12.0
    #     # print(efforts)
    #     # -- apply action to the robot
    #     robot.set_joint_effort_target(torch.Tensor(action))
    #     # -- write data to sim
    #     robot.write_data_to_sim()
    #     # Perform step
    #
    # def set_position(self, position=None):
    #     root_state = self.robot.data.default_root_state.clone()
    #     root_state[:, :3] = torch.tensor([0, 0, 0.7])
    #
    # def get_world_pose(self):
    #     return self.robot.data.default_root_state[:, :3].cpu().numpy().squeeze(), self.robot.data.default_root_state[:,
    #                                                                               3:].cpu().numpy().squeeze()
