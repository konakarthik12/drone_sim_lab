# This class is based of the World class from omni.isaac.core.world

from omni.isaac.core.scenes.scene import Scene

from omni.isaac.lab.sim import SimulationContext, SimulationCfg



class FakeWorld(SimulationContext):

    _world_initialized = False

    def __init__(
        self,
        sim_config: SimulationCfg,
    ) -> None:
        SimulationContext.__init__(
            self,
            sim_config
        )
        if FakeWorld._world_initialized:
            return
        FakeWorld._world_initialized = True

        self._scene = Scene()

        from omni.isaac.dynamic_control import _dynamic_control
        self.dc_interface = _dynamic_control.acquire_dynamic_control_interface()

        return


    """
    Properties.
    """

    @property
    def scene(self) -> Scene:
        """
        Returns:
            Scene: Scene instance

        Example:

        .. code-block:: python

            >>> world.scene
            <omni.isaac.core.scenes.scene.Scene object at 0x>
        """
        return self._scene

