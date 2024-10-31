## For RL_GAMES
1. install isaacgym -> https://developer.nvidia.com/isaac-gym/download
2. install isaacgymenv -> https://github.com/isaac-sim/IsaacGymEnvs

## For RSL_RL
1. install isaacgym -> https://developer.nvidia.com/isaac-gym/download
2. install isaacgymenv -> https://github.com/isaac-sim/IsaacGymEnvs
3. install rsl_rl
	3.1. cd rsl_rl && pip3 install -e .
4. install legged_gym
	3.1. cd legged_gym_custom && pip3 install -e .
	
## Run
```
python3 legged_gym/scripts/train.py --task=bolt6
```
or
```
python3 legged_gym/scripts/play.py --task=bolt6 --load_run=/home/${USER}/IssacGym/legged_gym_custom/logs/bolt6_history_length_10/${FILE DIR}
```
