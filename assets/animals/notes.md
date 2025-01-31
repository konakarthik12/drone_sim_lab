### Ant
The ant.usd and ant_instanceable.usd asset is imported from Isaac Lab's ant environment which is hosted on Nvidia Omniverse assets

ant.usd: https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/4.0/Isaac/Robots/Ant/ant_instanceable.usd

ant_instanceable.usd: https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/4.0/Isaac/Robots/Ant/ant_instanceable.usd
The policy is trained on rl games
Use this command to train

```bash
./isaaclab.sh -p source/standalone/workflows/rl_games/train.py --task Isaac-Ant-Direct-v0 --headless --device cpu "agent.params.config.device=cpu" "agent.params.config.device_name=cpu"
```
### Crab2

- Crab2.usd Imported from blender URDF export
- I made the crab weight match the ant weight by setting a high density to all the crab parts


