## Drone Teleop Simulation Setup Guide

This guide will help you set up and run the drone teleoperation demo using Isaac Sim, Pegasus, and QGroundControl.

---

### Prerequisites

- **Git LFS:**  
  Install [Git LFS](https://git-lfs.com/) to handle large `.usd` files.

- **Isaac Sim & Isaac Lab:**  
  Follow the [Isaac Sim installation guide](https://isaac-sim.github.io/IsaacLab/v1.4.0/source/setup/installation/pip_installation.html#installing-isaac-sim) to install Isaac Sim and set up the IsaacLab conda/venv environment.

---

### Installation Steps

1. **Clone IsaacLab and Install RL Games:**
    ```bash
    git clone https://github.com/konakarthik12/IsaacLab.git -b crab-rl
    cd IsaacLab
    ./isaaclab.sh --install rl_games
    ```

2. **Clone and Install Pegasus (v4.2.0):**
    > **Note:** Clone Pegasus **outside** this project directory.
    ```bash
    git clone https://github.com/PegasusSimulator/PegasusSimulator.git
    cd PegasusSimulator
    git checkout v4.2.0
    pip install -e extensions/pegasus.simulator
    ```

3. **Install PX4 Autopilot (for Pegasus):**  
   Refer to the [Pegasus PX4 Installation Guide](https://pegasussimulator.github.io/PegasusSimulator/source/setup/installation.html#installing-px4-autopilot).

   **For Ubuntu 24.04:**  
   Before installing PX4, run:
    ```bash
    sudo apt update
    sudo apt install gcc-12 g++-12
    which gcc-12  # Verify installation
    export CC=/usr/bin/gcc-12
    export CXX=/usr/bin/g++-12
    ```

4. **Download and Set Up QGroundControl:**  
   Download QGroundControl from [here](https://docs.qgroundcontrol.com/master/en/getting_started/download_and_install.html).  
   You will use this software to control the drone (joystick recommended).

---

### Creating and Training New Animal Environment
* Details [here](docs/training_rl.md)

---

### Running the Drone Teleop Demo

1. **Start QGroundControl:**
    - Open QGroundControl.
    - Go to `Vehicle Configuration` and enable the joystick controller.

2. **Run the Drone Teleop Script:**
    - With your IsaacLab environment activated, run:
      ```bash
      python3 standalone/drone_teleop_qgroundcontrol.py
      ```
      If you encounter a "module not found" error, try:
      ```bash
      python -m standalone.drone_teleop_qgroundcontrol
      ```

3. **Operate the Drone:**
    - Once Isaac Sim launches, switch back to QGroundControl.
    - On the top left, check the status. If not "Ready", click it and select "Arm".
    - Click the "Takeoff" button on the left panel.
    - Wait 10–20 seconds for the drone to stabilize.
    - You can now use your joystick/controller to fly the drone and operate the manipulator.

---

**Tip:**  
For best results, keep IsaacLab, Pegasus, and QGroundControl in separate directories outside this project folder.