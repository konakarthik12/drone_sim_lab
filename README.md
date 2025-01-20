- You will need to install [Git LFS](https://git-lfs.com/) to handle large usd files. 
- Follow instruction at [Isaac Lab](https://isaac-sim.github.io/IsaacLab) to install Isaac Lab and setup a conda or venv environment. Clone Isaac Lab repository into a folder outside this project.

During the step of cloning the Isaac Lab repository, checkout version `1.4.0`. 

```bash
git clone https://github.com/isaac-sim/IsaacLab.git # Run this command in a folder outside this project
cd IsaacLab
git checkout v1.4.0 
```

- With your Isaac Lab conda/venv environment active, clone and install pegasus (v4.2.0) as an editable package
```bash
git clone https://github.com/PegasusSimulator/PegasusSimulator.git # Run this command in a folder outside this project
cd PegasusSimulator
git checkout v4.2.0
pip install -e extensions/pegasus.simulator
```
- To run the drone teleop demo, download and setup QGroundControl. You can control the drone inside that software with either keyboard or joystick (joystick is easier).
- Once QGroundControl is setup and running, you can run the drone teleop demo by running the following command:
```bash
python3 standalone/drone_teleop_qgroundcontrol.py
```