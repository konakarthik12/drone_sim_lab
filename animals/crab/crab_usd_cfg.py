from __future__ import annotations
from copy import deepcopy
from omni.isaac.lab_assets.crab import CRAB_CFG


def get_crab_cfg(init_pos=None, scale=None):
    if scale is not None and type(scale) is float:
        scale = [scale] * 3

    config = deepcopy(CRAB_CFG)
    config.prim_path = "/World/robot"
    if scale:
        config.spawn.scale = scale
    if init_pos:
        cur_pos = list(config.init_state.pos)
        for i in range(3):
            if init_pos[i] is not None:
                cur_pos[i] = init_pos[i]
        config.init_state.pos = cur_pos
    return config
