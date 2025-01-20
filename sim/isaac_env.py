from typing import Any, SupportsFloat

import gymnasium
from gymnasium.core import ObsType, ActType, RenderFrame

from utils import enable_gpu_dynamics


class IsaacEnv(gymnasium.Env):
    def __init__(self, headless=True, layout_type="air", raw_init=False, **kwargs):

        # from isaacsim import SimulationApp
        #
        # DISP_FPS = 1 << 0
        # # DISP_AXIS = 1 << 1
        # DISP_RESOLUTION = 1 << 3
        # # DISP_SKELETON = 1 << 9
        # DISP_MESH = 1 << 10
        # # DISP_PROGRESS = 1 << 11
        # DISP_DEV_MEM = 1 << 13
        # DISP_HOST_MEM = 1 << 14
        self.headless = headless
        # self.app = SimulationApp({
        #     "headless": headless,
        #     "anti_aliasing": 0,
        #     "display_options": DISP_FPS | DISP_RESOLUTION | DISP_MESH | DISP_DEV_MEM | DISP_HOST_MEM,
        # })
        from omni.isaac.lab.app import AppLauncher
        from omni.isaac.kit import SimulationApp

        self.app:SimulationApp = AppLauncher(headless=headless).app
        if raw_init:
            return
        # from pegasus.simulator.logic import PegasusInterface
        # self.pg = PegasusInterface()
        from omni.isaac.lab.sim import SimulationCfg
        from sim.fake_world import FakeWorld

        sim_cfg = SimulationCfg(dt=1/250, render_interval=250/60,device="cpu")
        sim_cfg.use_fabric = False
        # sim_cfg.physx.enable_ccd = True
        # sim_cfg.physx.bounce_threshold_velocity = 0.0
        # sim_cfg.enable_scene_query_support = False
        # sim_cfg = SimulationCfg(dt=1/120, render_interval=1,device="cpu")
        from omni.isaac.lab.sim import SimulationContext
        # self.world = SimulationContext(sim_cfg)
        # self.world = World(**kwargs)
        from pegasus.simulator.logic import PegasusInterface
        self.pg = PegasusInterface()
        self.world= FakeWorld(sim_cfg)
        self.pg._world = self.world
        self.load_layout(layout_type)

    def get_world_settings(self):
        return {}
    def load_layout(self, layout_type):
        # from pegasus_custom.params import ENV_ASSETS
        import omni.isaac.lab.sim as sim_utils

        if layout_type == "air":
            # self.world.scene.add_default_ground_plane()
            from omni.isaac.lab.sim import GroundPlaneCfg
            from omni.isaac.lab.sim import DomeLightCfg

            cfg = GroundPlaneCfg()
            cfg.func("/World/defaultGroundPlane", cfg)
            cfg = DomeLightCfg(
                color=(0.1, 0.1, 0.1),
                enable_color_temperature=True,
                color_temperature=5500,
                intensity=10000,
            )
            # Dome light named specifically to avoid conflicts
            cfg.func(prim_path="/World/defaultDomeLight", cfg=cfg, translation=(0.0, 0.0, 10.0))

        elif layout_type == "water":
            pass
            # self.pg.load_environment(ENV_ASSETS + "/fluid_test_2.usd")
            # cfg = sim_utils.UsdFileCfg(usd_path=f"{ENV_ASSETS}/fluid_test_2.usd")
            # cfg.func("/World/layout", cfg)
            # enable_gpu_dynamics()
        elif layout_type == "grid":
            cfg = sim_utils.UsdFileCfg(
                usd_path=f"/home/kkona/Documents/research/drone_sim_lab/assets/worlds/grid_with_stand.usd")
            cfg.func("/World/layout", cfg)



    def reset(
            self,
            *,
            seed: int | None = None,
            options: dict[str, Any] | None = None,
    ) -> tuple[ObsType, dict[str, Any]]:
        super().reset(seed=seed, options=options)
        self.world.reset(soft=True)
        return {}, {}

    def step(
            self, action: ActType
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        self.world.step(render=not self.headless)
        return {}, 0.0, False, False, {}

    def render(self) -> RenderFrame | list[RenderFrame] | None:
        return self.world.render(mode="human")