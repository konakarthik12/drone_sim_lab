# Copyright (c) 2021-2023, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#

import gc

# python
from typing import List, Optional

# from omni.isaac.core.loggers import DataLogger
from omni.isaac.core.scenes.scene import Scene

# isaac-core
from omni.isaac.lab.sim import SimulationContext, SimulationCfg
# from omni.isaac.lab.tasks import BaseTask

# omniverse
from pxr import Usd


class FakeWorld(SimulationContext):
    """This class inherits from SimulationContext which provides the following.

    SimulationContext provide functions that take care of many time-related events such as
    perform a physics or a render step for instance. Adding/ removing callback functions that
    gets triggered with certain events such as a physics step, timeline event
    (pause or play..etc), stage open/ close..etc.

    It also includes an instance of PhysicsContext which takes care of many physics related
    settings such as setting physics dt, solver type..etc.

    In addition to what is provided from SimulationContext, this class allows the user to add a
    task to the world and it contains a scene object.

    To control the default reset state of different objects easily, the object could be added to
    a Scene. Besides this, the object is bound to a short keyword that facilitates objects retrievals,
    like in a dict.

    Checkout the required tutorials at
    https://docs.omniverse.nvidia.com/app_isaacsim/app_isaacsim/overview.html

    Args:
        physics_dt (Optional[float], optional): dt between physics steps. Defaults to None.
        rendering_dt (Optional[float], optional): dt between rendering steps. Note: rendering means
                                                   rendering a frame of the current application and not
                                                   only rendering a frame to the viewports/ cameras. So UI
                                                   elements of Isaac Sim will be refreshed with this dt
                                                   as well if running non-headless.
                                                   Defaults to None.
        stage_units_in_meters (Optional[float], optional): The metric units of assets. This will affect gravity value..etc.
                                                   Defaults to None.
        physics_prim_path (Optional[str], optional): specifies the prim path to create a PhysicsScene at,
                                             only in the case where no PhysicsScene already defined.
                                             Defaults to "/physicsScene".
        set_defaults (bool, optional): set to True to use the defaults settings
                                        [physics_dt = 1.0/ 60.0,
                                        stage units in meters = 0.01 (i.e in cms),
                                        rendering_dt = 1.0 / 60.0,
                                        gravity = -9.81 m / s
                                        ccd_enabled,
                                        stabilization_enabled,
                                        gpu dynamics turned off,
                                        broadcast type is MBP,
                                        solver type is TGS]. Defaults to True.
        backend (str, optional): specifies the backend to be used (numpy or torch or warp). Defaults to numpy.
        device (Optional[str], optional): specifies the device to be used if running on the gpu with torch or warp backends.

    Example:

    .. code-block:: python

        >>> from omni.isaac.core import World
        >>>
        >>> world = World()
        >>> world
        <omni.isaac.core.world.world.World object at 0x...>
    """

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
        # self._task_scene_built = False
        # self._current_tasks = dict()
        self._scene = Scene()
        print("huh")

        from omni.isaac.dynamic_control import _dynamic_control
        self.dc = _dynamic_control.acquire_dynamic_control_interface()
        # if not builtins.ISAAC_LAUNCHED_FROM_TERMINAL:
        #     self.start_simulation()
        # self._data_logger = DataLogger()
        return

    # """
    # Instance handling.
    # """
    #
    # @classmethod
    # def clear_instance(cls):
    #     """Delete the world object, if it was instantiated before, and destroy any subscribed callback
    #
    #     Example:
    #
    #     .. code-block:: python
    #
    #         >>> World.clear_instance()
    #     """
    #     if World._world_initialized:
    #         if hasattr(SimulationContext._instance, "_scene"):
    #             del SimulationContext._instance._scene
    #             gc.collect()
    #         World._world_initialized = False
    #         SimulationContext.clear_instance()
    #     return

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



    # """
    # Operations - Tasks management.
    # """
    #
    # def add_task(self, task: BaseTask) -> None:
    #     """Add a task to the task registry
    #
    #     .. note::
    #
    #         Tasks should have a unique name
    #
    #     Args:
    #         task (BaseTask): task object
    #
    #     Example:
    #
    #     .. code-block:: python
    #
    #         >>> from omni.isaac.core.tasks import BaseTask
    #         >>>
    #         >>> class Task(BaseTask):
    #         ...    def get_observations(self):
    #         ...        return {'obs': [0]}
    #         ...
    #         ...    def calculate_metrics(self):
    #         ...        return {"reward": 1}
    #         ...
    #         ...    def is_done(self):
    #         ...        return False
    #         ...
    #         >>> task = Task(name="custom_task")
    #         >>> world.add_task(task)
    #     """
    #     if task.name in self._current_tasks:
    #         raise Exception("Task name should be unique in the world")
    #     self._current_tasks[task.name] = task
    #     return
    #
    # def is_tasks_scene_built(self) -> bool:
    #     """Check if the ``set_up_scene`` method was called for each registered task
    #
    #     Example:
    #
    #     .. code-block:: python
    #
    #         >>> # given a world instance that was rested at some point
    #         >>> world.is_tasks_scene_built()
    #         True
    #     """
    #     return self._task_scene_built
    #
    # def get_current_tasks(self) -> List[BaseTask]:
    #     """Get a dictionary of the registered tasks where keys are task names
    #
    #     Returns:
    #         List[BaseTask]: registered tasks
    #
    #     Example:
    #
    #     .. code-block:: python
    #
    #         >>> world.get_current_tasks()
    #         {'custom_task': <custom.task.scripts.extension.Task object at 0x...>}
    #     """
    #     return self._current_tasks
    #
    # def get_task(self, name: str) -> BaseTask:
    #     """Get a task by its name
    #
    #     Example:
    #
    #     .. code-block:: python
    #
    #         >>> world.get_task("custom_task")
    #         <custom.task.scripts.extension.Task object at 0x...>
    #     """
    #     if name not in self._current_tasks:
    #         raise Exception("task name {} doesn't exist in the current world tasks.".format(name))
    #     return self._current_tasks[name]
    #
    # """
    # Operations - Tasks state collection.
    # """
    #
    # def get_observations(self, task_name: Optional[str] = None) -> dict:
    #     """Get observations from all the tasks that were added
    #
    #     Args:
    #         task_name (Optional[str], optional): task name to ask for. Defaults to None, which means all the tasks
    #
    #     Returns:
    #         dict: the task (or all tasks) observations
    #
    #     Example:
    #
    #     .. code-block:: python
    #
    #         >>> world.get_observations("custom_task")
    #         {'obs': [0]}
    #     """
    #     if task_name is not None:
    #         return self._current_tasks[task_name].get_observations()
    #     else:
    #         observations = dict()
    #         for task in self._current_tasks.values():
    #             observations.update(task.get_observations())
    #         return observations
    #
    # def calculate_metrics(self, task_name: Optional[str] = None) -> None:
    #     """Get metrics from all the tasks that were added
    #
    #     Args:
    #         task_name (Optional[str], optional): task name to ask for. Defaults to None, which means all the tasks
    #
    #     Returns:
    #         dict: the computed task (or all tasks) metric
    #
    #     Example:
    #
    #     .. code-block:: python
    #
    #         >>> world.calculate_metrics("custom_task")
    #         {'reward': 1}
    #     """
    #     if task_name is not None:
    #         return self._current_tasks[task_name].calculate_metrics()
    #     else:
    #         metrics = dict()
    #         for task in self._current_tasks.values():
    #             metrics.update(task.calculate_metrics())
    #         return metrics
    #
    # def is_done(self, task_name: Optional[str] = None) -> bool:
    #     """Get done from all the tasks that were added
    #
    #     Args:
    #         task_name (Optional[str], optional): task name to ask for. Defaults to None, which means all the tasks
    #
    #     Returns:
    #         bool: whether the task (or all tasks) is done
    #
    #     Example:
    #
    #     .. code-block:: python
    #
    #         >>> world.is_done("custom_task")
    #         False
    #     """
    #     if task_name is not None:
    #         return self._current_tasks[task_name].is_done()
    #     else:
    #         result = [task.is_done() for task in self._current_tasks.values()]
    #         return all(result)
    #
    # """
    # Operations - Data logger.
    # """
    #
    # def get_data_logger(self) -> DataLogger:
    #     """Return the data logger of the world.
    #
    #     Returns:
    #         DataLogger: data logger instance
    #
    #     Example:
    #
    #     .. code-block:: python
    #
    #         >>> world.get_data_logger()
    #         <omni.isaac.core.loggers.data_logger.DataLogger object at 0x...>
    #     """
    #     return self._data_logger
    #
    # """
    # Operations.
    # """
    #
    # def initialize_physics(self) -> None:
    #     """Initialize the physics simulation view and each added object to the Scene
    #
    #     Example:
    #
    #     .. code-block:: python
    #
    #         >>> world.initialize_physics()
    #     """
    #     SimulationContext.initialize_physics(self)
    #     self._scene._finalize(self.physics_sim_view)
    #     return
    #
    # def reset(self, soft: bool = False) -> None:
    #     """Reset the stage to its initial state and each object included in the Scene to its default state
    #         as specified by the ``set_default_state`` and ``__init__`` methods
    #
    #     .. note::
    #
    #         - All tasks should be added before the first reset is called unless the ``clear`` method was called.
    #         - All articulations should be added before the first reset is called unless the ``clear`` method was called.
    #         - This method takes care of initializing articulation handles with the first reset called.
    #         - This will do one step internally regardless
    #         - Call ``post_reset`` on each object in the Scene
    #         - Call ``post_reset`` on each Task
    #
    #         Things like setting PD gains for instance should happen at a Task reset or a Robot reset since
    #         the defaults are restored after ``stop`` method is called.
    #
    #     .. warning::
    #
    #         This method is not intended to be used in the Isaac Sim's Extensions workflow since the Kit application
    #         has the control over the rendering steps. For the Extensions workflow use the ``reset_async`` method instead
    #
    #     Args:
    #         soft (bool, optional): if set to True simulation won't be stopped and start again. It only calls the reset on the scene objects.
    #
    #     Example:
    #
    #     .. code-block:: python
    #
    #         >>> world.reset()
    #     """
    #     if not self._task_scene_built:
    #         for task in self._current_tasks.values():
    #             task.set_up_scene(self.scene)
    #         self._task_scene_built = True
    #     if not soft:
    #         self.stop()
    #     for task in self._current_tasks.values():
    #         task.cleanup()
    #     SimulationContext.reset(self, soft=soft)
    #     self._scene._finalize(self.physics_sim_view)
    #     self.scene.post_reset()
    #     for task in self._current_tasks.values():
    #         task.post_reset()
    #
    # async def reset_async_set_up_scene(self, soft: bool = False) -> None:
    #     """Reset the stage to its initial state and each object included in the Scene to its default state
    #         as specified by the ``set_default_state`` and ``__init__`` methods
    #
    #     .. note::
    #
    #         - All tasks should be added before the first reset is called unless the ``clear`` method was called.
    #         - All articulations should be added before the first reset is called unless the ``clear`` method was called.
    #         - This method takes care of initializing articulation handles with the first reset called.
    #         - This will do one step internally regardless
    #         - Call ``post_reset`` on each object in the Scene
    #         - Call ``post_reset`` on each Task
    #
    #         Things like setting PD gains for instance should happen at a Task reset or a Robot reset since
    #         the defaults are restored after ``stop`` method is called.
    #
    #     Args:
    #         soft (bool, optional): if set to True simulation won't be stopped and start again. It only calls the reset on the scene objects.
    #
    #     Example:
    #
    #     .. code-block:: python
    #
    #         >>> from omni.kit.async_engine import run_coroutine
    #         >>>
    #         >>> async def task():
    #         >>>     await world.reset_async_set_up_scene()
    #         >>>
    #         >>> run_coroutine(task())
    #     """
    #     for task in self._current_tasks.values():
    #         task.set_up_scene(self.scene)
    #
    # async def reset_async_no_set_up_scene(self, soft: bool = False) -> None:
    #     """Reset the stage to its initial state and each object included in the Scene to its default state
    #         as specified by the ``set_default_state`` and ``__init__`` methods
    #
    #     .. note::
    #
    #         - All tasks should be added before the first reset is called unless the ``clear`` method was called.
    #         - All articulations should be added before the first reset is called unless the ``clear`` method was called.
    #         - This method takes care of initializing articulation handles with the first reset called.
    #         - This will do one step internally regardless
    #         - Call ``post_reset`` on each object in the Scene
    #         - Call ``post_reset`` on each Task
    #
    #         Things like setting PD gains for instance should happen at a Task reset or a Robot reset since
    #         the defaults are restored after ``stop`` method is called.
    #
    #     Args:
    #         soft (bool, optional): if set to True simulation won't be stopped and start again. It only calls the reset on the scene objects.
    #
    #     Example:
    #
    #     .. code-block:: python
    #
    #         >>> from omni.kit.async_engine import run_coroutine
    #         >>>
    #         >>> async def task():
    #         >>>     await world.reset_async_no_set_up_scene()
    #         >>>
    #         >>> run_coroutine(task())
    #     """
    #     if not soft:
    #         await self.stop_async()
    #     for task in self._current_tasks.values():
    #         task.cleanup()
    #     await SimulationContext.reset_async(self, soft=soft)
    #     self._scene._finalize(self.physics_sim_view)
    #     self._scene.post_reset()
    #     for task in self._current_tasks.values():
    #         task.post_reset()
    #     return
    #
    # async def reset_async(self, soft: bool = False) -> None:
    #     """Reset the stage to its initial state and each object included in the Scene to its default state
    #         as specified by the ``set_default_state`` and ``__init__`` methods
    #
    #     .. note::
    #
    #         - All tasks should be added before the first reset is called unless the ``clear`` method was called.
    #         - All articulations should be added before the first reset is called unless the ``clear`` method was called.
    #         - This method takes care of initializing articulation handles with the first reset called.
    #         - This will do one step internally regardless
    #         - Call ``post_reset`` on each object in the Scene
    #         - Call ``post_reset`` on each Task
    #
    #         Things like setting PD gains for instance should happen at a Task reset or a Robot reset since
    #         the defaults are restored after ``stop`` method is called.
    #
    #     Args:
    #         soft (bool, optional): if set to True simulation won't be stopped and start again. It only calls the reset on the scene objects.
    #
    #     Example:
    #
    #     .. code-block:: python
    #
    #         >>> from omni.kit.async_engine import run_coroutine
    #         >>>
    #         >>> async def task():
    #         ...     await world.reset_async()
    #         ...
    #         >>> run_coroutine(task())
    #     """
    #     if not self._task_scene_built:
    #         await self.reset_async_set_up_scene()
    #         self._task_scene_built = True
    #     await self.reset_async_no_set_up_scene(soft=soft)
    #     return
    #
    # def step(self, render: bool = True, step_sim: bool = True) -> None:
    #     """Step the physics simulation while rendering or without.
    #
    #     .. note::
    #
    #         The ``pre_step`` for each task is called before stepping. This method also update the Bounding Box Cache
    #         time for computing bounding box if enabled
    #
    #     .. warning::
    #
    #         Calling this method with the ``render`` parameter set to True (default value) is not intended to be used
    #         in the Isaac Sim's Extensions workflow since the Kit application has the control over the rendering steps
    #
    #     Args:
    #         render (bool, optional): Set to False to only do a physics simulation without rendering. Note:
    #                                  app UI will be frozen (since its not rendering) in this case.
    #                                  Defaults to True.
    #         step_sim (bool): True to step simulation (physics and/or rendering)
    #
    #     Example:
    #
    #     .. code-block:: python
    #
    #         >>> world.step()
    #     """
    #     if self._task_scene_built:
    #         for task in self._current_tasks.values():
    #             task.pre_step(self.current_time_step_index, self.current_time)
    #     if self.scene._enable_bounding_box_computations:
    #         self.scene._bbox_cache.SetTime(Usd.TimeCode(self._current_time))
    #
    #     if step_sim:
    #         SimulationContext.step(self, render=render)
    #     if self._data_logger.is_started():
    #         if self._data_logger._data_frame_logging_func is None:
    #             raise Exception("You need to add data logging function before starting the data logger")
    #         data = self._data_logger._data_frame_logging_func(tasks=self.get_current_tasks(), scene=self.scene)
    #         self._data_logger.add_data(
    #             data=data, current_time_step=self.current_time_step_index, current_time=self.current_time
    #         )
    #     return
    #
    # def step_async(self, step_size: Optional[float] = None) -> None:
    #     """Call all functions that should be called pre stepping the physics
    #
    #     .. note::
    #
    #         The ``pre_step`` for each task is called before stepping. This method also update the Bounding Box Cache
    #         time for computing bounding box if enabled
    #
    #     Args:
    #         step_size (float): step size
    #
    #     Example:
    #
    #     .. code-block:: python
    #
    #         >>> from omni.kit.async_engine import run_coroutine
    #         >>>
    #         >>> async def task():
    #         ...     await world.step_async()
    #         ...
    #         >>> run_coroutine(task())
    #     """
    #     if self._task_scene_built:
    #         for task in self._current_tasks.values():
    #             task.pre_step(self.current_time_step_index, self.current_time)
    #     if self.scene._enable_bounding_box_computations:
    #         self.scene._bbox_cache.SetTime(Usd.TimeCode(self._current_time))
    #     if self._data_logger.is_started():
    #         if self._data_logger._data_frame_logging_func is None:
    #             raise Exception("You need to add data logging function before starting the data logger")
    #         data = self._data_logger._data_frame_logging_func(tasks=self.get_current_tasks(), scene=self.scene)
    #         self._data_logger.add_data(
    #             data=data, current_time_step=self.current_time_step_index, current_time=self.current_time
    #         )
    #     return
    #
    # def clear(self) -> None:
    #     """Clear the current stage leaving the PhysicsScene and /World
    #
    #     Example:
    #
    #     .. code-block:: python
    #
    #         >>> world.clear()
    #     """
    #     self.scene.clear(registry_only=False)
    #     self._current_tasks = dict()
    #     self._task_scene_built = False
    #     self._data_logger = DataLogger()
    #     # clear all prims in the stage.
    #     SimulationContext.clear(self)
