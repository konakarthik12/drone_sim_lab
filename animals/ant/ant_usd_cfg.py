from __future__ import annotations

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.actuators import ImplicitActuatorCfg
from omni.isaac.lab.assets import ArticulationCfg


def get_ant_cfg(init_pos=(0.0, 0.0, 0.5), scale=None):
    if scale is not None and type(scale) is float:
        scale = [scale] * 3
    return ArticulationCfg(
        prim_path="/World/ant",
        spawn=sim_utils.UsdFileCfg(
            usd_path="/home/kkona/Documents/research/drone_sim_lab/assets/animals/ant_instanceable.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=10.0,
                enable_gyroscopic_forces=True,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=4,
                solver_velocity_iteration_count=0,
                sleep_threshold=0.005,
                stabilization_threshold=0.001,
            ),
            copy_from_source=False,
            scale=scale,

        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=init_pos,
            joint_pos={
                ".*_leg": 0.0,
                "front_left_foot": 0.785398,  # 45 degrees
                "front_right_foot": -0.785398,
                "left_back_foot": -0.785398,
                "right_back_foot": 0.785398,
            },
        ),
        actuators={
            "body": ImplicitActuatorCfg(
                joint_names_expr=[".*"],
                stiffness=0.0,
                damping=0.0,
            ),
        },
    )
