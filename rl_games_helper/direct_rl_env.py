from __future__ import annotations

import builtins
import math
from abc import abstractmethod
from dataclasses import MISSING
from typing import Any, ClassVar

import gymnasium as gym
import omni.isaac.core.utils.torch as torch_utils
import omni.kit.app
import omni.log
import torch
from omni.isaac.lab.envs.common import VecEnvObs, VecEnvStepReturn
from omni.isaac.lab.envs.direct_rl_env_cfg import DirectRLEnvCfg
from omni.isaac.lab.envs.utils.spaces import sample_space, spec_to_gym_space
from omni.isaac.lab.scene import InteractiveScene
from omni.isaac.lab.utils.timer import Timer
from omni.isaac.version import get_version

from sim.isaac_env import IsaacEnv


class DirectRLEnv(IsaacEnv, gym.Env):
    """The superclass for the direct workflow to design environments.

    This class implements the core functionality for reinforcement learning (RL)
    environments. It is designed to be used with any RL library. The class is designed
    to be used with vectorized environments, i.e., the environment is expected to be run
    in parallel with multiple sub-environments.

    While the environment itself is implemented as a vectorized environment, we do not
    inherit from :class:`gym.vector.VectorEnv`. This is mainly because the class adds
    various methods (for wait and asynchronous updates) which are not required.
    Additionally, each RL library typically has its own definition for a vectorized
    environment. Thus, to reduce complexity, we directly use the :class:`gym.Env` over
    here and leave it up to library-defined wrappers to take care of wrapping this
    environment for their agents.

    Note:
        For vectorized environments, it is recommended to **only** call the :meth:`reset`
        method once before the first call to :meth:`step`, i.e. after the environment is created.
        After that, the :meth:`step` function handles the reset of terminated sub-environments.
        This is because the simulator does not support resetting individual sub-environments
        in a vectorized environment.

    """

    """Whether the environment is a vectorized environment."""
    metadata: ClassVar[dict[str, Any]] = {
        "render_modes": [None, "human", "rgb_array"],
        "isaac_sim_version": get_version(),
    }
    """Metadata for the environment."""

    def __init__(self, cfg: DirectRLEnvCfg, **kwargs):
        """Initialize the environment.

        Args:
            cfg: The configuration object for the environment.
            render_mode: The render mode for the environment. Defaults to None, which
                is similar to ``"human"``.

        Raises:
            RuntimeError: If a simulation context already exists. The environment must always create one
                since it configures the simulation context and controls the simulation.
        """

        # store inputs to class
        self.cfg = cfg

        # initialize internal variables
        self._is_closed = False

        self.seed(42)
        self.cfg.sim.render_interval = 250 / 60
        assert self.cfg.sim.dt == 1 / 250
        self.sim = self.world

        # print useful information
        print("[INFO]: Base environment:")
        print(f"\tEnvironment device    : {self.device}")
        print(f"\tEnvironment seed      : {self.cfg.seed}")
        print(f"\tPhysics step-size     : {self.physics_dt}")
        print(f"\tRendering step-size   : {self.physics_dt * self.cfg.sim.render_interval}")
        print(f"\tEnvironment step-size : {self.step_dt}")

        assert self.cfg.sim.render_interval >= self.cfg.decimation, "Render interval should not be smaller than decimation, this will cause multiple render calls."

        # generate scene
        with Timer("[INFO]: Time taken for scene creation", "scene_creation"):
            self.scene = InteractiveScene(self.cfg.scene)
            self._setup_scene()
        print("[INFO]: Scene manager: ", self.scene)

        # play the simulator to activate physics handles
        # note: this activates the physics simulation view that exposes TensorAPIs
        # note: when started in extension mode, first call sim.reset_async() and then initialize the managers
        if builtins.ISAAC_LAUNCHED_FROM_TERMINAL is False:
            print("[INFO]: Starting the simulation. This may take a few seconds. Please wait...")
            with Timer("[INFO]: Time taken for simulation start", "simulation_start"):
                self.sim.reset()

        # make sure torch is running on the correct device
        # if "cuda" in self.device:
        #     torch.cuda.set_device(self.device)

        # allocate dictionary to store metrics
        self.extras = {}

        # initialize data and constants
        # -- counter for simulation steps
        self._sim_step_counter = 0
        # -- counter for curriculum
        self.common_step_counter = 0
        # -- init buffers
        self.episode_length = 0
        self.reset_terminated = False
        self.reset_time_outs = False
        self.reset_buf = False

        # setup the action and observation spaces for Gym
        self._configure_gym_env_spaces()

        # -- set the framerate of the gym video recorder wrapper so that the playback speed of the produced video matches the simulation
        self.metadata["render_fps"] = 1 / self.step_dt

        # print the environment information
        print("[INFO]: Completed setting up the environment...")

    def __del__(self):
        """Cleanup for the environment."""
        self.close()

    """
    Properties.
    """

    @property
    def num_envs(self) -> int:
        """The number of instances of the environment that are running."""
        return 1

    @property
    def physics_dt(self) -> float:
        """The physics time-step (in s).

        This is the lowest time-decimation at which the simulation is happening.
        """
        return self.cfg.sim.dt

    @property
    def step_dt(self) -> float:
        """The environment stepping time-step (in s).

        This is the time-step at which the environment steps forward.
        """
        return self.cfg.sim.dt * self.cfg.decimation

    @property
    def device(self):
        """The device on which the environment is running."""
        return self.sim.device

    @property
    def max_episode_length_s(self) -> float:
        """Maximum episode length in seconds."""
        return self.cfg.episode_length_s

    @property
    def max_episode_length(self):
        """The maximum episode length in steps adjusted from s."""
        return math.ceil(self.max_episode_length_s / (self.cfg.sim.dt * self.cfg.decimation))

    """
    Operations.
    """

    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[VecEnvObs, dict]:
        """Resets the environment and returns observations.

        This function calls the :meth:`_reset_idx` function to reset the environment.
        However, certain operations, such as procedural terrain generation, that happened during initialization
        are not repeated.

        Args:
            seed: The seed to use for randomization. Defaults to None, in which case the seed is not set.
            options: Additional information to specify how the environment is reset. Defaults to None.

                Note:
                    This argument is used for compatibility with Gymnasium environment definition.

        Returns:
            A tuple containing the observations and extras.
        """
        # set the seed
        if seed is not None:
            self.seed(seed)

        # reset state of scene
        self._reset_idx()

        # update articulation kinematics
        self.scene.write_data_to_sim()
        self.sim.forward()
        assert not self.sim.has_rtx_sensors() and not self.cfg.rerender_on_reset

        # return observations
        return self._get_observations(), self.extras

    def step(self, action: torch.Tensor) -> VecEnvStepReturn:
        """Execute one time-step of the environment's dynamics.

        The environment steps forward at a fixed time-step, while the physics simulation is decimated at a
        lower time-step. This is to ensure that the simulation is stable. These two time-steps can be configured
        independently using the :attr:`DirectRLEnvCfg.decimation` (number of simulation steps per environment step)
        and the :attr:`DirectRLEnvCfg.sim.physics_dt` (physics time-step). Based on these parameters, the environment
        time-step is computed as the product of the two.

        This function performs the following steps:

        1. Pre-process the actions before stepping through the physics.
        2. Apply the actions to the simulator and step through the physics in a decimated manner.
        3. Compute the reward and done signals.
        4. Reset the environment if it has terminated or reached the maximum episode length.
        5. Apply interval events if they are enabled.
        6. Compute observations.

        Args:
            action: The actions to apply on the environment. Shape is (action_dim,).

        Returns:
            A tuple containing the observations, rewards, resets (terminated and truncated) and extras.
        """
        action = action.to(self.device)
        assert not self.cfg.action_noise_model

        # process actions
        self._pre_physics_step(action)

        # check if we need to do rendering within the physics loop
        # note: checked here once to avoid multiple checks within the loop
        is_rendering = self.sim.has_gui() or self.sim.has_rtx_sensors()

        # perform physics stepping
        for _ in range(self.cfg.decimation):
            self._sim_step_counter += 1
            # set actions into buffers
            self._apply_action()
            # set actions into simulator
            self.scene.write_data_to_sim()
            # simulate
            self.sim.step(render=False)
            # render between steps only if the GUI or an RTX sensor needs it
            # note: we assume the render interval to be the shortest accepted rendering interval.
            #    If a camera needs rendering at a faster frequency, this will lead to unexpected behavior.
            if self._sim_step_counter % round(self.cfg.sim.render_interval) == 0 and is_rendering:
                self.sim.render()
            # update buffers at sim dt
            self.scene.update(dt=self.physics_dt)

        # post-step:
        # -- update env counters (used for curriculum generation)
        self.episode_length += 1  # step in current episode
        self.common_step_counter += 1  # total step

        self.reset_terminated, self.reset_time_outs = self._get_dones()
        self.reset_buf = self.reset_terminated or self.reset_time_outs
        self.reward_buf = self._get_rewards()

        # -- reset env if terminated/timed-out and log the episode information
        if self.reset_buf:
            self._reset_idx()
            # update articulation kinematics
            self.scene.write_data_to_sim()
            self.sim.forward()
            assert not self.sim.has_rtx_sensors() and not self.cfg.rerender_on_reset

        # update observations
        self.obs_buf = self._get_observations()

        # return observations, rewards, resets and extras
        return self.obs_buf, self.reward_buf, self.reset_terminated, self.reset_time_outs, self.extras

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

    def close(self):
        """Cleanup for the environment."""
        if not self._is_closed:
            # close entities related to the environment
            # note: this is order-sensitive to avoid any dangling references

            del self.scene

            # clear callbacks and instance
            self.sim.clear_all_callbacks()
            self.sim.clear_instance()

            # update closing status
            self._is_closed = True

    """
    Helper functions.
    """

    def _configure_gym_env_spaces(self):
        """Configure the action and observation spaces for the Gym environment."""

        # set up spaces
        self.single_observation_space = gym.spaces.Dict()
        self.single_observation_space["policy"] = spec_to_gym_space(self.cfg.observation_space)
        self.single_action_space = spec_to_gym_space(self.cfg.action_space)

        # batch the spaces for vectorized environments
        self.observation_space = self.single_observation_space["policy"]
        self.action_space = self.single_action_space

        # instantiate actions (needed for tasks for which the observations computation is dependent on the actions)
        self.actions = sample_space(self.single_action_space, self.sim.device, batch_size=1, fill_value=0)

    def _reset_idx(self):
        """Reset the environment."""
        self.scene.reset()

        self.episode_length = 0

    """
    Implementation-specific functions.
    """

    def _setup_scene(self):

        pass

    @abstractmethod
    def _pre_physics_step(self, actions: torch.Tensor):

        raise NotImplementedError(f"Please implement the '_pre_physics_step' method for {self.__class__.__name__}.")

    @abstractmethod
    def _apply_action(self):

        raise NotImplementedError(f"Please implement the '_apply_action' method for {self.__class__.__name__}.")

    @abstractmethod
    def _get_observations(self) -> VecEnvObs:

        raise NotImplementedError(f"Please implement the '_get_observations' method for {self.__class__.__name__}.")

    @abstractmethod
    def _get_rewards(self) -> torch.Tensor:

        raise NotImplementedError(f"Please implement the '_get_rewards' method for {self.__class__.__name__}.")

    @abstractmethod
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:

        raise NotImplementedError(f"Please implement the '_get_dones' method for {self.__class__.__name__}.")
