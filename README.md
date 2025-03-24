# PyMARLzoo+
![Static Badge](https://img.shields.io/badge/Python-3.8-blue)
[![Tests](https://github.com/AILabDsUnipi/pymarlzooplus/actions/workflows/test.yml/badge.svg)](https://github.com/AILabDsUnipi/pymarlzooplus/actions/workflows/test.yml)
![GitHub License](https://img.shields.io/github/license/AILabDsUnipi/pymarlzooplus)


<img src="https://raw.githubusercontent.com/AILabDsUnipi/pymarlzooplus/main/images/logo.jpg" alt="logo" width=600/>

PyMARLzoo+ is an extension of [EPyMARL](https://github.com/uoe-agents/epymarl), and includes
- Additional (7) algorithms: 
  - HAPPO, 
  - MAT-DEC, 
  - QPLEX, 
  - EOI, 
  - EMC, 
  - MASER, 
  - CDS
- Support for [PettingZoo](https://github.com/Farama-Foundation/PettingZoo) environments
- Support for [Overcooked](https://github.com/HumanCompatibleAI/overcooked_ai) environments.
- Support for [Pressure plate](https://github.com/uoe-agents/pressureplate) environments.
- Support for [Capture Target](https://github.com/yuchen-x/MacDeepMARL/blob/master/src/rlmamr/my_env/capture_target.py) environment.
- Support for [Box Pushing](https://github.com/yuchen-x/MacDeepMARL/blob/master/src/rlmamr/my_env/box_pushing.py) environment.
- Support for [LBF](https://github.com/semitable/lb-foraging) environment.
- Support for [RWARE](https://github.com/semitable/robotic-warehouse) environment.

Algorithms (9) maintained from EPyMARL:
- COMA
- QMIX
- MAA2C
- MAPPO
- VDN
- MADDPG
- IQL
- IPPO
- IA2C

# Table of Contents
- [Installation](#installation)
- [Run training](#run-training)
- [Environment API](#environment-api)
- [Run a hyperparameter search](#run-a-hyperparameter-search)
- [Saving and loading learnt models](#saving-and-loading-learnt-models)
- [Citing PyMARLzoo+, PyMARL, and EPyMARL](#citing-pymarlzoo-epymarl-and-pymarl)
- [License](#license)


# Installation
Before installing PyMARLzoo+, we recommend creating a conda environment:

```sh
conda create -y -n pymarlzooplus python=3.8.18
conda activate pymarlzooplus
```

- To install and use PyMARLzoo+ as a package, run the following commands:
  ```sh
  pip install pymarlzooplus
  ``` 
  NOTE: We work on this!


- To build it from source, run the following commands:
  ```sh
  git clone https://github.com/AILabDsUnipi/pymarlzooplus.git
  cd pymarlzooplus 
  pip install -e .
  ``` 

**Known issues**:
- Before running an atari environment (from PettingZoo) for the first time, you have to run: 
  ```sh
  AutoROM -y
  ```
- Installation error for ``box2d-py``, run:
  ```sh
  sudo apt-get install swig
  ```
- In case you get `ImportError: Library "GLU" not found.`, run: 
  ```sh
  sudo apt-get install python3-opengl
  ```
- In case you get `IndexError: list index out of range` for pyglet in a non-headless machine using `render=True`, 
set `__GLX_VENDOR_LIBRARY_NAME=nvidia` and `__NV_PRIME_RENDER_OFFLOAD=1` as environment variables. 
  
  For example, if you want to run your own file `example.py` , run:
  ```sh
  __GLX_VENDOR_LIBRARY_NAME=nvidia __NV_PRIME_RENDER_OFFLOAD=1 python example.py
  ```
- In case you get `ERROR: Failed building wheel for multi-agent-ale-py`, probably the problem is with the compiler (if `cmake` is already installed), so run:
    ```sh
    sudo apt-get install g++-9 gcc-9
    export CC=gcc-9 CXX=g++-9
    ```
- In case you get `SystemError: ffi_prep_closure(): bad user_data (it seems that the version of the libffi library seen at runtime is different from the 'ffi.h' file seen at compile-time)`, probably for `"waterworld_v4"` of PettingZoo, run:
  ```sh
  pip install --upgrade --force-reinstall pymunk cffi
  ```

# Docker
You can also use the provided Docker files in [docker/](/docker/) to build and run a Docker container with the 
pymarlzooplus installed as a headless machine.

To build the image, run the container and get into the container, run the following commands after you clone the repository:
```sh
docker compose -f docker/docker-compose.yml up --build
docker exec -it $(docker ps | awk 'NR > 1 {print $1}') /bin/bash
```

# Run training
 
In the following instructions, ```<algo>``` (or ```"algo"```) can be replaced with one of the following algoriths:  
- "coma"
- "qmix"
- "maa2c"
- "mappo"
- "vdn"
- "maddpg"
- "iql"
- "ippo"
- "ia2c"
- "happo" 
- "mat-dec" 
- "qplex" 
- "eoi" 
- "emc" 
- "maser" 
- "cds"

You can specify the algorithm arguments:
- In case of running experiment **from source**, you can edit the corresponding configuration file in [pymarlzooplus/config/algs/](pymarlzooplus/config/algs) and the [pymarlzooplus/config/default.yaml](pymarlzooplus/config/default.yaml).
- In case of running experiment **as a package**, you can specify the algorithm arguments in the dictionary ```"algo_args"```.

## LBF

<img src="https://raw.githubusercontent.com/AILabDsUnipi/pymarlzooplus/main/images/lbf.gif" alt="lbf gif" align="center" width="40%"/>

- As a package (replace ```"algo"``` and ```"scenario"```):

```python
  from pymarlzooplus import pymarlzooplus
  
  params_dict = {
    "config": "algo",
    "algo_args": {},
    "env-config": "gymma",
    "env_args": {
        "key": "scenario",
        "time_limit": 50,
        "seed": 2024
    }
  }
  
  pymarlzooplus(params_dict)
```
 
  Find all the LBF arguments in the configuration file [pymarlzooplus/config/envs/gymma.yaml](pymarlzooplus/config/envs/gymma.yaml).

- From source (replace ```<algo>``` and ```<scenario>```):
  ```sh
  python pymarlzooplus/main.py --config=<algo> --env-config=gymma with env_args.time_limit=25 env_args.key=<scenario>
  ```
  Before running an experiment, edit the configuration file in [pymarlzooplus/config/envs/gymma.yaml](pymarlzooplus/config/envs/gymma.yaml).

Available scenarios (replace ```"scenario"``` or ```<scenario>```) we run experiments (our experiments were conducted with v2):
- "lbforaging:Foraging-4s-11x11-3p-2f-coop-v2" / "lbforaging:Foraging-4s-11x11-3p-2f-coop-v3",
- "lbforaging:Foraging-2s-11x11-3p-2f-coop-v2" / "lbforaging:Foraging-2s-11x11-3p-2f-coop-v3",
- "lbforaging:Foraging-2s-8x8-3p-2f-coop-v2" / "lbforaging:Foraging-2s-8x8-3p-2f-coop-v3",
- "lbforaging:Foraging-2s-9x9-3p-2f-coop-v2" / "lbforaging:Foraging-2s-9x9-3p-2f-coop-v3",
- "lbforaging:Foraging-7s-20x20-5p-3f-coop-v2" / "lbforaging:Foraging-7s-20x20-5p-3f-coop-v3",
- "lbforaging:Foraging-2s-12x12-2p-2f-coop-v2" / "lbforaging:Foraging-2s-12x12-2p-2f-coop-v3",
- "lbforaging:Foraging-8s-25x25-8p-5f-coop-v2" / "lbforaging:Foraging-8s-25x25-8p-5f-coop-v3",
- "lbforaging:Foraging-7s-30x30-7p-4f-coop-v2" / "lbforaging:Foraging-7s-30x30-7p-4f-coop-v3"

All the above scenarios are compatible both with our **training framework** and the **environment API**.


## RWARE
<img src="https://raw.githubusercontent.com/AILabDsUnipi/pymarlzooplus/main/images/rware.gif"  align="center" height="40%"/>

- As a package (replace ```"algo"``` and ```"scenario"```):
  ```python
  from pymarlzooplus import pymarlzooplus
  
  params_dict = {
    "config": "algo",
    "algo_args": {},
    "env-config": "gymma",
    "env_args": {
        "key": "scenario",
        "time_limit": 500,
        "seed": 2024
    }
  }
  
  pymarlzooplus(params_dict)
  ```
  
  Find all the RWARE arguments in the configuration file [pymarlzooplus/config/envs/gymma.yaml](pymarlzooplus/config/envs/gymma.yaml).

- From source (replace ```<algo>``` and ```<scenario>```):
  ```sh
  python pymarlzooplus/main.py --config=<algo> --env-config=gymma with env_args.time_limit=500 env_args.key=<scenario>
  ```
  Before running an experiment, edit the configuration file in [pymarlzooplus/config/envs/gymma.yaml](pymarlzooplus/config/envs/gymma.yaml).

Available scenarios we run experiments (our experiments were conducted with v1):
- "rware:rware-small-4ag-hard-v1" / "rware:rware-small-4ag-hard-v2"
- "rware:rware-tiny-4ag-hard-v1" / "rware:rware-tiny-4ag-hard-v2",
- "rware:rware-tiny-2ag-hard-v1" / "rware:rware-tiny-2ag-hard-v2"

All the above scenarios are compatible both with our **training framework** and the **environment API**.


## MPE
<img src="https://raw.githubusercontent.com/AILabDsUnipi/pymarlzooplus/main/images/mpe.gif" alt="logo" align="center" width="40%"/>

- As a package (replace ```"algo"``` and ```"scenario"```):
  ```python
  from pymarlzooplus import pymarlzooplus
  
  params_dict = {
    "config": "algo",
    "algo_args": {},
    "env-config": "gymma",
    "env_args": {
        "key": "scenario",
        "time_limit": 25,
        "seed": 2024
    }
  }
  
  pymarlzooplus(params_dict)
  ```
  Find all the MPE arguments in the configuration file [pymarlzooplus/config/envs/gymma.yaml](pymarlzooplus/config/envs/gymma.yaml).

- From source (replace ```<algo>``` and ```<scenario>```):
  ```sh
  python pymarlzooplus/main.py --config=<algo> --env-config=gymma with env_args.time_limit=25 env_args.key=<scenario>
  ```
  Before running an experiment, edit the configuration file in [pymarlzooplus/config/envs/gymma.yaml](pymarlzooplus/config/envs/gymma.yaml).

Available scenarios we run experiments:
- "mpe:SimpleSpeakerListener-v0", 
- "mpe:SimpleSpread-3-v0",
- "mpe:SimpleSpread-4-v0",
- "mpe:SimpleSpread-5-v0",
- "mpe:SimpleSpread-8-v0"

More available scenarios:
- "mpe:MultiSpeakerListener-v0",
- "mpe:SimpleAdversary-v0",
- "mpe:SimpleCrypto-v0",
- "mpe:SimplePush-v0",
- "mpe:SimpleReference-v0",
- "mpe:SimpleTag-v0",
- "mpe:SimpleWorldComm-v0",
- "mpe:ClimbingSpread-v0"

All the above scenarios are compatible both with our **training framework** and the **environment API**.


## PettingZoo
<img src="https://raw.githubusercontent.com/AILabDsUnipi/pymarlzooplus/main/images/pistonball.gif" alt="pistonball gif"  align="center" width="40%"/>

- As a package (replace ```"algo"```, ```"task"```, and ```0000```):
  ```python
  from pymarlzooplus import pymarlzooplus
  
  params_dict = {
    "config": "algo",
    "algo_args": {},
    "env-config": "pettingzoo",
    "env_args": {
        "key": "task",
        "time_limit": 0000,
        "render_mode": "rgb_array",  # Options: "human", "rgb_array
        "image_encoder": "ResNet18",  # Options: "ResNet18", "SlimSAM", "CLIP"
        "image_encoder_use_cuda": True,  # Whether to load image-encoder in GPU or not.
        "image_encoder_batch_size": 10,  # How many images to encode in a single pass.
        "partial_observation": False,  # Only for "Emtombed: Cooperative" and "Space Invaders"
        "trainable_cnn": False,  # Specifies whether to return image-observation or the encoded vector-observation
        "seed": 2024
    }
  }
  
  pymarlzooplus(params_dict)
  ```
  Find all the PettingZoo arguments in the configuration file [pymarlzooplus/config/envs/pettingzoo.yaml](pymarlzooplus/config/envs/pettingzoo.yaml). Also, there you can find useful comments for each task's arguments.

- From source (replace ```<algo>```, ```<time-limit>```, and ```<task>```):
  ```sh
  python pymarlzooplus/main.py --config=<algo> --env-config=pettingzoo with env_args.time_limit=<time-limit> env_args.key=<task>
  ```
  Before running an experiment, edit the configuration file in [pymarlzooplus/config/envs/pettingzoo.yaml](pymarlzooplus/config/envs/pettingzoo.yaml).


Available tasks we run experiments:
- Atari:
  - "entombed_cooperative_v3"
  - "space_invaders_v2"
- Butterfly:
  - "pistonball_v6"
  - "cooperative_pong_v5"

The above tasks are compatible both with our **training framework** and the **environment API**.
Below, we list more tasks which are compatible only with the **environment API**:
- Atari:
  - "basketball_pong_v3"
  - "boxing_v2"
  - "combat_plane_v2"
  - "combat_tank_v2"
  - "double_dunk_v3"
  - "entombed_competitive_v3"
  - "flag_capture_v2"
  - "foozpong_v3"
  - "ice_hockey_v2"
  - "joust_v3"
  - "mario_bros_v3"
  - "maze_craze_v3"
  - "othello_v3"
  - "pong_v3"
  - "quadrapong_v4"
  - "space_war_v2"
  - "surround_v2"
  - "tennis_v3"
  - "video_checkers_v4"
  - "volleyball_pong_v3"
  - "warlords_v3"
  - "wizard_of_wor_v3"
- Butterfly:
  - "knights_archers_zombies_v10"
- Classic:
  - "chess_v6"
  - "connect_four_v3"
  - "gin_rummy_v4"
  - "go_v5"
  - "hanabi_v5"
  - "leduc_holdem_v4"
  - "rps_v2"
  - "texas_holdem_no_limit_v6"
  - "texas_holdem_v4"
  - "tictactoe_v3"
- MPE
  - "simple_v3"
  - "simple_adversary_v3"
  - "simple_crypto_v3"
  - "simple_push_v3"
  - "simple_reference_v3"
  - "simple_speaker_listener_v4"
  - "simple_spread_v3"
  - "simple_tag_v3"
  - "simple_world_comm_v3"
- SISL
  - "multiwalker_v9"
  - "pursuit_v4"
  - "waterworld_v4"


## Overcooked
<img src="https://raw.githubusercontent.com/AILabDsUnipi/pymarlzooplus/main/images/overcooked.gif" alt="overcooked gif"  align="center" width="40%"/>

- As a package (replace ```"algo"```, ```"scenario"```, and ```"rewardType"```):
  ```python
  from pymarlzooplus import pymarlzooplus
  
  params_dict = {
    "config": "algo",
    "algo_args": {},
    "env-config": "overcooked",
    "env_args": {
        "time_limit": 500,
        "key": "scenario",
        "reward_type": "rewardType",
    }
  }
  
  pymarlzooplus(params_dict)
  ```
  Find all the Overcooked arguments in the configuration file [pymarlzooplus/config/envs/overcooked.yaml](pymarlzooplus/config/envs/overcooked.yaml).

- From source (replace ```<algo>```, ```<scenario>```, and ```<reward_type>```):
  ```sh
  python pymarlzooplus/main.py --config=<algo> --env-config=overcooked with env_args.time_limit=500 env_args.key=<scenario> env_args.reward_type=<reward_type>
  ```
  Before running an experiment, edit the configuration file in [pymarlzooplus/config/envs/overcooked.yaml](pymarlzooplus/config/envs/overcooked.yaml). 

Available scenarios we run experiments:
- "cramped_room"
- "asymmetric_advantages"
- "coordination_ring"

More available scenarios:
- "counter_circuit"
- "random3"
- "random0"
- "unident"
- "forced_coordination"
- "soup_coordination"
- "small_corridor"
- "simple_tomato"
- "simple_o_t"
- "simple_o"
- "schelling_s"
- "schelling"
- "m_shaped_s"
- "long_cook_time"
- "large_room"
- "forced_coordination_tomato"
- "cramped_room_tomato"
- "cramped_room_o_3orders"
- "asymmetric_advantages_tomato"
- "bottleneck"
- "cramped_corridor"
- "counter_circuit_o_1order"

Reward types available:
- "sparse" (we used it to run our experiments)
- "shaped"

All the above scenarios are compatible both with our **training framework** and the **environment API**.


## Pressure plate
<img src="https://raw.githubusercontent.com/AILabDsUnipi/pymarlzooplus/main/images/pressureplate.gif" alt="pressureplate gif"  align="center" width="40%"/>

- As a package (replace ```"algo"``` and ```"scenario"```):
  ```python
  from pymarlzooplus import pymarlzooplus
  
  params_dict = {
    "config": "algo",
    "algo_args": {},
    "env-config": "pressureplate",
    "env_args": {
        "key": "scenario",
        "time_limit": 500,
        "seed": 2024
    }
  }
  
  pymarlzooplus(params_dict)
  ```
  Find all the Pressure Plate arguments in the configuration file [pymarlzooplus/config/envs/pressureplate.yaml](pymarlzooplus/config/envs/pressureplate.yaml).

- From source (replace ```<algo>``` and ```<scenario>```):
  ```sh
  python pymarlzooplus/main.py --config=<algo> --env-config=pressureplate with env_args.key=<scenario> env_args.time_limit=500
  ```
  Before running an experiment, edit the configuration file in [pymarlzooplus/config/envs/pressureplate.yaml](pymarlzooplus/config/envs/pressureplate.yaml).

Available scenarios we run experiments:
- "pressureplate-linear-4p-v0"
- "pressureplate-linear-6p-v0"

More available scenarios:
- "pressureplate-linear-5p-v0"

All the above scenarios are compatible both with our **training framework** and the **environment API**.


## Capture Target
<img src="https://raw.githubusercontent.com/AILabDsUnipi/pymarlzooplus/main/images/capturetarget.gif" alt="capturetarget gif"  align="center" width="40%"/>

- As a package (replace ```"algo"``` and ```"scenario"```):
  ```python
  from pymarlzooplus import pymarlzooplus
  
  params_dict = {
    "config": "algo",
    "algo_args": {},
    "env-config": "capturetarget",
    "env_args": {
        "time_limit": 60,
        "key" : "scenario",
    }
  }
  
  pymarlzooplus(params_dict)
  ```
  Find all the Capture Target arguments in the configuration file [pymarlzooplus/config/envs/capturetarget.yaml](pymarlzooplus/config/envs/capturetarget.yaml).

- From source (replace ```<algo>``` and ```<scenario>```):
  ```sh
  python pymarlzooplus/main.py --config=<algo> --env-config=capturetarget with env_args.key=<scenario> env_args.time_limit=60
  ```
  Before running an experiment, edit the configuration file in [pymarlzooplus/config/envs/capturetarget.yaml](pymarlzooplus/config/envs/capturetarget.yaml). Also, there you can find useful comments for the rest of arguments.

Available scenario:
- "CaptureTarget-6x6-1t-2a-v0"

The above scenario is compatible both with our **training framework** and the **environment API**.


## Box Pushing

<img src="https://raw.githubusercontent.com/AILabDsUnipi/pymarlzooplus/main/images/boxpushing.gif" alt="boxpushing gif"  align="center" width="40%"/>

- As a package (replace ```"algo"``` and ```"scenario"```):
  ```python
  from pymarlzooplus import pymarlzooplus
  
  params_dict = {
    "config": "algo",
    "algo_args": {},
    "env-config": "boxpushing",
    "env_args": {
        "key": "scenario",
        "time_limit": 60,
        "seed": 2024
    }
  }
  
  pymarlzooplus(params_dict)
  ```
  Find all the Box Pushing arguments in the configuration file [pymarlzooplus/config/envs/boxpushing.yaml](pymarlzooplus/config/envs/boxpushing.yaml).

- From source (replace ```<algo>``` and ```<scenario>```):
  ```sh
  python pymarlzooplus/main.py --config=<algo> --env-config=boxpushing with env_args.key=<scenario> env_args.time_limit=60
  ```
  Before running an experiment, edit the configuration file in [pymarlzooplus/config/envs/boxpushing.yaml](pymarlzooplus/config/envs/boxpushing.yaml).

Available scenario:
- "BoxPushing-6x6-2a-v0"

The above scenario is compatible both with our **training framework** and the **environment API**.



# Environment API
Except from using our training framework, you can use PyMARLzoo+ in your custom training pipeline as follows:
```python
# Import package
from pymarlzooplus.envs import REGISTRY as env_REGISTRY

# Example of arguments for PettingZoo
args = {
  "env": "pettingzoo",
  "env_args": {
      "key": "pistonball_v6",
      "time_limit": 900,
      "render_mode": "rgb_array",
      "image_encoder": "ResNet18",
      "image_encoder_use_cuda": True,
      "image_encoder_batch_size": 10,
      "centralized_image_encoding": False,
      "partial_observation": False,
      "trainable_cnn": False,
      "kwargs": "('n_pistons',10),",
      "seed": 2024
  }
}

# Initialize environment
env = env_REGISTRY[args["env"]](**args["env_args"])
n_agns = env.get_n_agents()
n_acts = env.get_total_actions()
# Reset the environment
obs, state = env.reset()
done = False
# Run an episode
while not done:
    # Render the environment (optional)
    env.render()
    # Insert the policy's actions here
    actions = env.sample_actions()
    # Apply an environment step
    reward, done, extra_info = env.step(actions)
    obs = env.get_obs()
    state = env.get_state()
    info = env.get_info()
# Terminate the environment
env.close()
```
The code above can be used as is (changing the arguments accordingly) **with the fully cooperative tasks** (see section [Run training](#run-training) to find out which tasks are fully cooperative).

In the file [pymarlzooplus/scripts/api_script.py](pymarlzooplus/scripts/api_script.py), you can find a complete example for all the supported environments.

For a description of each environment's arguments, please see the corresponding config file in [pymarlzooplus/config/envs](pymarlzooplus/config/envs). 

# Run a hyperparameter search

We include a script named [search.py](pymarlzooplus/search.py) which reads a search configuration file (e.g. the included [search.config.example.yaml](pymarlzooplus/search.config.example.yaml)) and runs a hyperparameter search in one or more tasks. The script can be run using
```shell
python search.py run --config=search.config.example.yaml --seeds 5 locally
```
In a cluster environment where one run should go to a single process, it can also be called in a batch script like:
```shell
python search.py run --config=search.config.example.yaml --seeds 5 single 1
```
where the 1 is an index to the particular hyperparameter configuration and can take values from 1 to the number of different combinations.

# Saving and loading learnt models

## Saving models

You can save the learnt models to disk by setting `save_model = True`, which is set to `False` by default. The frequency of saving models can be adjusted using `save_model_interval` configuration. Models will be saved in the result directory, under the folder called *models*. The directory corresponding each run will contain models saved throughout the experiment, each within a folder corresponding to the number of timesteps passed since starting the learning process.

## Loading models

Learnt models can be loaded using the `checkpoint_path` parameter, after which the learning will proceed from the corresponding timestep. 

# Citing PyMARLzoo+, EPyMARL and PyMARL

If you use PyMARLzoo+ in your research, please cite the [An Extended Benchmarking of Multi-Agent Reinforcement Learning Algorithms in Complex Fully Cooperative Tasks
]([https://arxiv.org/abs/2006.07869](https://www.arxiv.org/pdf/2502.04773)).

*George Papadopoulos, Andreas Kontogiannis, Foteini Papadopoulou, Chaido Poulianou, Ioannis Koumentis, George Vouros. An Extended Benchmarking of Multi-Agent Reinforcement Learning Algorithms in Complex Fully Cooperative Tasks, AAMAS 2025.*

If you use PyMARLzoo+ in your research, please cite the following:

```tex
@article{papadopoulos2025extended,
  title={An Extended Benchmarking of Multi-Agent Reinforcement Learning Algorithms in Complex Fully Cooperative Tasks},
  author={Papadopoulos, George and Kontogiannis, Andreas and Papadopoulou, Foteini and Poulianou, Chaido and Koumentis, Ioannis and Vouros, George},
  journal={arXiv preprint arXiv:2502.04773},
  year={2025}
}
```

*Georgios Papoudakis, Filippos Christianos, Lukas Schäfer, & Stefano V. Albrecht. Benchmarking Multi-Agent Deep Reinforcement Learning Algorithms in Cooperative Tasks, Proceedings of the Neural Information Processing Systems Track on Datasets and Benchmarks (NeurIPS), 2021*

If you use EPyMARL in your research, please cite the following:

```tex
@inproceedings{papoudakis2021benchmarking,
   title={Benchmarking Multi-Agent Deep Reinforcement Learning Algorithms in Cooperative Tasks},
   author={Georgios Papoudakis and Filippos Christianos and Lukas Schäfer and Stefano V. Albrecht},
   booktitle = {Proceedings of the Neural Information Processing Systems Track on Datasets and Benchmarks (NeurIPS)},
   year={2021},
   url = {http://arxiv.org/abs/2006.07869},
   openreview = {https://openreview.net/forum?id=cIrPX-Sn5n},
   code = {https://github.com/uoe-agents/epymarl},
}
```

If you use the original PyMARL in your research, please cite the [SMAC paper](https://arxiv.org/abs/1902.04043).

*M. Samvelyan, T. Rashid, C. Schroeder de Witt, G. Farquhar, N. Nardelli, T.G.J. Rudner, C.-M. Hung, P.H.S. Torr, J. Foerster, S. Whiteson. The StarCraft Multi-Agent Challenge, CoRR abs/1902.04043, 2019.*

In BibTeX format:

```tex
@article{samvelyan19smac,
  title = {{The} {StarCraft} {Multi}-{Agent} {Challenge}},
  author = {Mikayel Samvelyan and Tabish Rashid and Christian Schroeder de Witt and Gregory Farquhar and Nantas Nardelli and Tim G. J. Rudner and Chia-Man Hung and Philiph H. S. Torr and Jakob Foerster and Shimon Whiteson},
  journal = {CoRR},
  volume = {abs/1902.04043},
  year = {2019},
}
```

# License
All the source code that has been taken from the EPyMARL and PyMARL repository was licensed (and remains so) under the Apache License v2.0 (included in `LICENSE` file).
Any new code is also licensed under the Apache License v2.0
