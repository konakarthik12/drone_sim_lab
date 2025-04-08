from omni.isaac.lab_tasks.direct.crab.crab_env import CrabEnvCfg

from animals.crab.crab_usd_cfg import get_crab_cfg

CRAB_TASK_NAME = "Isaac-Crab-Direct-v0"
CRAB_RESUME_PATH = "/home/kkona/Documents/research/drone_sim_lab/assets/animals/crab/crab_direct.pth"


def crab_env_cfg():
    cfg = CrabEnvCfg()
    cfg.robot = get_crab_cfg(init_pos=(6.0, 0.0, None))  # Keep the z-position same as training environment

    return cfg


def crab_task_cfg():
    return CRAB_TASK_NAME, CRAB_RESUME_PATH

