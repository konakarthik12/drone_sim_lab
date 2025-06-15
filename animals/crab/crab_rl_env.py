import os

from omni.isaac.lab_tasks.direct.crab.crab_env import CrabEnvCfg

from animals.crab.crab_usd_cfg import get_crab_cfg
from utils import get_project_path


CRAB_TASK_NAME = "Isaac-Crab-Direct-v0"
CRAB_RESUME_PATH = os.path.join(get_project_path(), "assets", "animals", "crab", "crab_direct.pth")  # "/data/assets/animals/crab/crab_direct.pth"


def crab_env_cfg():
    cfg = CrabEnvCfg()
    cfg.robot = get_crab_cfg(init_pos=(6.0, 0.0, None))  # Keep the z-position same as training environment

    return cfg


def crab_task_cfg():
    return CRAB_TASK_NAME, CRAB_RESUME_PATH

