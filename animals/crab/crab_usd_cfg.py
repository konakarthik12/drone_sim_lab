from __future__ import annotations
from copy import deepcopy
from omni.isaac.lab_assets.crab import CRAB_CFG


def get_crab_cfg(init_pos=(0.0, 0.0, 0.5), scale=None):
    if scale is not None and type(scale) is float:
        scale = [scale] * 3

    config = deepcopy(CRAB_CFG)
    config.prim_path = "/World/robot"
    config.spawn.scale = scale
    config.init_state.pos = init_pos
    return config
