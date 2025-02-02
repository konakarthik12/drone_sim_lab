from typing import Any, SupportsFloat

import gymnasium
from gymnasium.core import ObsType, ActType, RenderFrame

from utils import enable_gpu_dynamics


class IsaacEnv(gymnasium.Env):
    def __init__(self, layout_type="air"):
        # DISP_FPS = 1 << 0
        # # DISP_AXIS = 1 << 1
        # DISP_RESOLUTION = 1 << 3
        # # DISP_SKELETON = 1 << 9
        # DISP_MESH = 1 << 10
        # # DISP_PROGRESS = 1 << 11
        # DISP_DEV_MEM = 1 << 13
        # DISP_HOST_MEM = 1 << 14
        # self.headless = headless
        # self.app = SimulationApp({
        #     "headless": headless,
        #     "anti_aliasing": 0,
        #     "display_options": DISP_FPS | DISP_RESOLUTION | DISP_MESH | DISP_DEV_MEM | DISP_HOST_MEM,
        # })
        from omni.isaac.lab.app import AppLauncher
        from omni.isaac.kit import SimulationApp
        from sim.app import get_app
        self.app: SimulationApp = get_app()


        from omni.isaac.lab.sim import SimulationCfg
        from sim.fake_world import FakeWorld

        sim_cfg = SimulationCfg(dt=1 / 250, render_interval=250 / 60, device="cpu")
        sim_cfg.use_fabric = False

        self.world:FakeWorld = FakeWorld(sim_cfg)

        self.load_layout(layout_type)
        self.init_reset = False

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
            raise NotImplementedError("Water layout not implemented yet")
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
        if not self.init_reset:
            self.world.reset(soft=False)
            self.post_init()
        else:
            self.world.reset(soft=True)
        return {}, {}

    def step(
            self, action: ActType
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        self.world.step(render=True)
        return {}, 0.0, False, False, {}
    def post_init(self):
        pass
    # def render(self) -> RenderFrame | list[RenderFrame] | None:
    #     return self.world.render(mode="human")
