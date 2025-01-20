#!/usr/bin/env python
"""
| File: 1_px4_single_vehicle.py
| Author: Marcelo Jacinto (marcelo.jacinto@tecnico.ulisboa.pt)
| License: BSD-3-Clause. Copyright (c) 2023, Marcelo Jacinto. All rights reserved.
| Description: This files serves as an example on how to build an app that makes use of the Pegasus API to run a simulation with a single vehicle, controlled using the MAVLink control backend.
"""

# Imports to start Isaac Sim from this script
from isaacsim import SimulationApp


# Start Isaac Sim's simulation environment
# Note: this simulation app must be instantiated right after the SimulationApp import, otherwise the simulator will crash
# as this is the object that will load all the extensions and load the actual simulator.
simulation_app = SimulationApp({"headless": False})
import carb
from omni.isaac.core.utils.extensions import enable_extension

enable_extension("pegasus.simulator")
# -----------------------------------
# The actual script should start here
# -----------------------------------
import omni.timeline
from omni.isaac.core.world import World

# Import the Pegasus API for simulating drones
from pegasus.simulator.params import ROBOTS, ASSET_PATH
from pegasus.simulator.logic.backends.px4_mavlink_backend import PX4MavlinkBackend, PX4MavlinkBackendConfig
from pegasus.simulator.logic.vehicles.multirotor import Multirotor, MultirotorConfig
from pegasus.simulator.logic.interface.pegasus_interface import PegasusInterface
# Auxiliary scipy and numpy modules
from scipy.spatial.transform import Rotation
from pxr import PhysxSchema
from pxr.PhysxSchema import PhysxSceneAPI, PhysxMaterialAPI
from omni.isaac.core.materials import PhysicsMaterial
from omni.isaac.lab.sim import bind_physics_material


class PegasusApp:
    """
    A Template class that serves as an example on how to build a simple Isaac Sim standalone App.
    """

    def __init__(self):
        """
        Method that initializes the PegasusApp and is used to setup the simulation environment.
        """

        # Acquire the timeline that will be used to start/stop the simulation
        # self.timeline = omni.timeline.get_timeline_interface()

        # Start the Pegasus Interface
        self.pg = PegasusInterface()

        self.close_gripper = False

        # Acquire the World, .i.e, the singleton that controls that is a one stop shop for setting up physics,
        # spawning asset primitives, etc.
        self.pg._world = World(**{"physics_dt": 1.0 / 250.0,
                                  "stage_units_in_meters": 1.0,
                                  "rendering_dt": 1.0 / 60.0,
                                  "device": "cpu"})
        self.world = self.pg.world
        # Launch one of the worlds provided by NVIDIA
        self.pg.load_environment(
            "/home/kkona/Documents/research/drone_sim_lab/assets/worlds/grid_with_stand.usd")
        # Create the vehicle
        # Try to spawn the selected robot in the world to the specified namespace
        config_multirotor = MultirotorConfig()
        # Create the multirotor configuration
        mavlink_config = PX4MavlinkBackendConfig({
            "vehicle_id": 0,
            "px4_autolaunch": True,
            "px4_dir": self.pg.px4_path,
            "px4_vehicle_model": self.pg.px4_default_airframe
            # CHANGE this line to 'iris' if using PX4 version bellow v1.14
        })
        config_multirotor.backends = [PX4MavlinkBackend(mavlink_config)]
        self.drone = Multirotor(
            "/World/quadrotor",
            "/home/kkona/Documents/research/PegasusSimulator/extensions/pegasus.simulator/pegasus/simulator/assets/Robots/Iris/iris_copy.usd",
            0,
            [0.0, 0.0, 1.02],
            Rotation.from_euler("XYZ", [0.0, 0.0, 0.0], degrees=True).as_quat(),
            config=config_multirotor,
        )

        from omni.isaac.core.articulations import Articulation

        # crab_path ="/home/kkona/Documents/research/drone_sim_lab/assets/animals/crab2.usd"
        # prim_path = "/World/crab"
        # import omni.isaac.core.utils.stage as stage_utils
        #
        # stage_utils.add_reference_to_stage(crab_path, prim_path)
        # self.init_pose = [6, 0, 0.3]
        # self.robot = Articulation(prim_path=prim_path,
        #                           name="crab",
        #                           position=self.init_pose,  # self.default_object_pose,
        #                           scale=[0.045] * 3)
        # self.world.scene.add(self.robot)

        self.joint1_pos = 0
        self.joint2_pos = 0
        carb_settings_iface = carb.settings.get_settings()
        carb_settings_iface.set_bool("/persistent/app/omniverse/gamepadCameraControl", False)
        self._input = carb.input.acquire_input_interface()
        self._appwindow = omni.appwindow.get_default_app_window()

        self._gamepad = self._appwindow.get_gamepad(0)

        self._gamepad_sub = self._input.subscribe_to_gamepad_events(
            self._gamepad,
            lambda event, *args,: self._on_gamepad_event(event, *args),
        )

        # from omni.physx import acquire_physx_interface, acquire_physx_simulation_interface()
        # physx_interface = acquire_physx_interface()

        #     float physxScene:frictionCorrelationDistance = 0.025
        #     float physxScene:frictionOffsetThreshold = 0.04
        #     uint physxScene:gpuCollisionStackSize = 67108864
        #     uint physxScene:gpuFoundLostAggregatePairsCapacity = 33554432
        #     uint physxScene:gpuFoundLostPairsCapacity = 2097152
        #     uint physxScene:gpuHeapCapacity = 67108864
        #     uint physxScene:gpuMaxNumPartitions = 8
        #     uint physxScene:gpuMaxParticleContacts = 1048576
        #     uint physxScene:gpuMaxRigidContactCount = 8388608
        #     uint physxScene:gpuMaxRigidPatchCount = 163840
        #     uint physxScene:gpuMaxSoftBodyContacts = 1048576
        #     uint64 physxScene:gpuTempBufferCapacity = 16777216
        #     uint physxScene:gpuTotalAggregatePairsCapacity = 2097152
        #     uniform uint physxScene:maxPositionIterationCount = 255
        #     uniform uint physxScene:maxVelocityIterationCount = 255
        #     uniform uint physxScene:minPositionIterationCount = 1
        #     uniform uint physxScene:minVelocityIterationCount = 0
        stage = self.world.stage

        physicsScene = stage.GetPrimAtPath("/physicsScene")
        physxSceneAPI: PhysxSceneAPI = PhysxSchema.PhysxSceneAPI.Apply(physicsScene)

        physxSceneAPI.CreateBounceThresholdAttr().Set(0.0)
        physxSceneAPI.CreateEnableSceneQuerySupportAttr().Set(True)

        physxSceneAPI.CreateFrictionCorrelationDistanceAttr().Set(0.025)
        physxSceneAPI.CreateFrictionOffsetThresholdAttr().Set(0.04)
        physxSceneAPI.CreateGpuCollisionStackSizeAttr().Set(67108864)
        physxSceneAPI.CreateGpuFoundLostAggregatePairsCapacityAttr().Set(33554432)
        physxSceneAPI.CreateGpuFoundLostPairsCapacityAttr().Set(2097152)
        physxSceneAPI.CreateGpuHeapCapacityAttr().Set(67108864)
        physxSceneAPI.CreateGpuMaxNumPartitionsAttr().Set(8)
        physxSceneAPI.CreateGpuMaxParticleContactsAttr().Set(1048576)
        physxSceneAPI.CreateGpuMaxRigidContactCountAttr().Set(8388608)
        physxSceneAPI.CreateGpuMaxRigidPatchCountAttr().Set(163840)
        physxSceneAPI.CreateGpuMaxSoftBodyContactsAttr().Set(1048576)
        physxSceneAPI.CreateGpuTempBufferCapacityAttr().Set(16777216)
        physxSceneAPI.CreateGpuTotalAggregatePairsCapacityAttr().Set(2097152)
        physxSceneAPI.CreateMaxPositionIterationCountAttr().Set(255)
        physxSceneAPI.CreateMaxVelocityIterationCountAttr().Set(255)
        physxSceneAPI.CreateMinPositionIterationCountAttr().Set(1)
        physxSceneAPI.CreateMinVelocityIterationCountAttr().Set(0)
        physxSceneAPI.CreateEnableEnhancedDeterminismAttr().Set(False)

        material = PhysicsMaterial("/physicsScene/defaultMaterial", static_friction=0.5, dynamic_friction=0.5,
                                   restitution=0.0)
        materialPrim = stage.GetPrimAtPath("/physicsScene/defaultMaterial")

        physxMaterialAPI: PhysxMaterialAPI = PhysxSchema.PhysxMaterialAPI.Apply(materialPrim)

        #    float physxMaterial:compliantContactDamping = 0
        #         float physxMaterial:compliantContactStiffness = 0
        #         uniform token physxMaterial:frictionCombineMode = "average"
        #         bool physxMaterial:improvePatchFriction = 1
        #         uniform token physxMaterial:restitutionCombineMode = "average"
        physxMaterialAPI.CreateCompliantContactDampingAttr().Set(0)
        physxMaterialAPI.CreateCompliantContactStiffnessAttr().Set(0)
        physxMaterialAPI.CreateFrictionCombineModeAttr().Set("average")
        physxMaterialAPI.CreateImprovePatchFrictionAttr().Set(1)
        physxMaterialAPI.CreateRestitutionCombineModeAttr().Set("average")
        bind_physics_material("/physicsScene", "/physicsScene/defaultMaterial")

        # Reset the simulation environment so that all articulations (aka robots) are initialized
        self.world.reset()
        from omni.isaac.dynamic_control import _dynamic_control
        self.world.dc = _dynamic_control.acquire_dynamic_control_interface()

        self.drone_articulation = self.world.dc.get_articulation(self.drone._stage_prefix)
        from drone.manipulators import Manipulators

        self.manipulators = Manipulators(self.world, self.drone)

        # Auxiliar variable for the timeline callback example
        self.stop_sim = False

    def _on_gamepad_event(self, event, *args, **kwargs):
        """Subscriber callback to when kit is updated.

        Reference:
            https://docs.omniverse.nvidia.com/dev-guide/latest/programmer_ref/input-devices/gamepad.html
        """
        # print(event.value)
        # # check if the event is a button press
        cur_val = event.value
        if abs(cur_val) < 0.03:
            cur_val = 0
        # # -- button
        if event.input == carb.input.GamepadInput.X:
            #     # toggle gripper based on the button pressed
            if cur_val > 0.5:
                self.close_gripper = not self.close_gripper

        if event.input == carb.input.GamepadInput.DPAD_UP:
            self.joint1_pos += 0.1
        if event.input == carb.input.GamepadInput.DPAD_DOWN:
            self.joint1_pos -= 0.1
        if event.input == carb.input.GamepadInput.LEFT_SHOULDER:
            self.joint2_pos += 0.1
        if event.input == carb.input.GamepadInput.RIGHT_SHOULDER:
            self.joint2_pos -= 0.1
        # # -- left and right stick
        # if event.input in self._INPUT_STICK_VALUE_MAPPING:
        #     direction, axis, value = self._INPUT_STICK_VALUE_MAPPING[event.input]
        #     # change the value only if the stick is moved (soft press)
        #     self._delta_pose_raw[direction, axis] = value * cur_val
        # # -- dpad (4 arrow buttons on the console)
        # if event.input in self._INPUT_DPAD_VALUE_MAPPING:
        #     direction, axis, value = self._INPUT_DPAD_VALUE_MAPPING[event.input]
        #     # change the value only if button is pressed on the DPAD
        #     if cur_val > 0.5:
        #         self._delta_pose_raw[direction, axis] = value
        #         self._delta_pose_raw[1 - direction, axis] = 0
        #     else:
        #         self._delta_pose_raw[:, axis] = 0
        # # additional callbacks
        # if event.input in self._additional_callbacks:
        #     self._additional_callbacks[event.input]()

        # since no error, we are fine :)
        return True

    def run(self):
        """
        Method that implements the application main loop, where the physics steps are executed.
        """

        # # Start the simulation
        # self.timeline.play()

        # The "infinite" loop
        while simulation_app.is_running() and not self.stop_sim:
            if omni.timeline.get_timeline_interface().is_playing():

                self.manipulators.step(1.5, self.joint1_pos, self.joint2_pos, self.close_gripper)

            # Update the UI of the app and perform the physics step
            self.world.step(render=True)

        # Cleanup and stop
        carb.log_warn("PegasusApp Simulation App is closing.")
        # self.timeline.stop()
        simulation_app.close()


def main():
    # Instantiate the template app
    pg_app = PegasusApp()

    # Run the application loop
    pg_app.run()


if __name__ == "__main__":
    main()
