from typing import Any, SupportsFloat
import os

import gymnasium
from gymnasium.core import ObsType, ActType
from omni.isaac.lab.sim import GroundPlaneCfg
from omni.isaac.lab.sim import DomeLightCfg
import omni.isaac.core.utils.torch as torch_utils

from utils import get_project_path

SIM_DT = 1 / 250

class IsaacEnv(gymnasium.Env):
    def __init__(self, layout="air"):
        from omni.isaac.kit import SimulationApp
        from sim.app import get_app
        self.app: SimulationApp = get_app()

        from omni.isaac.lab.sim import SimulationCfg
        from sim.fake_world import FakeWorld

        self.sim_cfg = SimulationCfg(dt=SIM_DT, render_interval=1, device="cpu")
        self.sim_cfg.use_fabric = False
        self.world: FakeWorld = FakeWorld(self.sim_cfg)
        self.sim = self.world
        self.load_layout(layout)
        self.init_reset = False
        self.seed(42)
        self.dt = SIM_DT

    def get_world_settings(self):
        return {}

    def load_layout(self, layout):
        # from pegasus_custom.params import ENV_ASSETS
        import omni.isaac.lab.sim as sim_utils

        if layout == "air":
            cfg = GroundPlaneCfg(
                physics_material=sim_utils.RigidBodyMaterialCfg(
                    friction_combine_mode="average",
                    restitution_combine_mode="average",
                    static_friction=1.0,
                    dynamic_friction=1.0,
                    restitution=0.0,
                ),
            )
            cfg.func("/World/defaultGroundPlane", cfg)

            cfg = DomeLightCfg(
                color=(0.1, 0.1, 0.1),
                enable_color_temperature=True,
                color_temperature=5500,
                intensity=10000,
            )
            # Dome light named specifically to avoid conflicts
            cfg.func(prim_path="/World/defaultDomeLight", cfg=cfg, translation=(0.0, 0.0, 10.0))

        elif layout == "water":
            raise NotImplementedError("Water layout not implemented yet")
            # self.pg.load_environment(ENV_ASSETS + "/fluid_test_2.usd")
            # cfg = sim_utils.UsdFileCfg(usd_path=f"{ENV_ASSETS}/fluid_test_2.usd")
            # cfg.func("/World/layout", cfg)
            # enable_gpu_dynamics()
        elif layout == "grid":
            cfg = sim_utils.UsdFileCfg(
                usd_path= os.path.join(get_project_path(), "assets/worlds/grid_with_stand.usd"),
            )
            cfg.func("/World/layout", cfg)
        else:
            raise ValueError(f"Unknown layout type {layout}")

    def reset(
            self,
            seed: int | None = None,
            options: dict[str, Any] | None = None,
    ) -> tuple[ObsType, dict[str, Any]]:
        super().reset(seed=seed, options=options)
        if seed is not None:
            self.seed(seed)
        if not self.init_reset:
            self.world.reset(soft=False)
            self.post_init()
        else:
            self.world.reset(soft=True)
        return {}, {}

    def step(
            self, action: ActType
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        self.world.step(render=False)

        return {}, 0.0, False, False, {}

    def post_init(self):
        pass

    @staticmethod
    def seed(seed: int = -1) -> int:
        """Set the seed for the environment.

        Args:
            seed: The seed for random generator. Defaults to -1.

        Returns:
            The seed used for random generator.
        """
        # set seed for replicator
        try:
            import omni.replicator.core as rep

            rep.set_global_seed(seed)
        except ModuleNotFoundError:
            pass
        # set seed for torch and other libraries
        return torch_utils.set_seed(seed)

    def __del__(self):
        self.close()

    def close(self):
        self.sim.clear_all_callbacks()
        self.sim.clear_instance()

    @staticmethod
    def seed(seed: int = -1) -> int:
        """Set the seed for the environment.

        Args:
            seed: The seed for random generator. Defaults to -1.

        Returns:
            The seed used for random generator.
        """
        # set seed for replicator
        try:
            import omni.replicator.core as rep

            rep.set_global_seed(seed)
        except ModuleNotFoundError:
            pass
        # set seed for torch and other libraries
        return torch_utils.set_seed(seed)

    def __del__(self):
        self.close()

    def close(self):
        self.sim.clear_all_callbacks()
        self.sim.clear_instance()
