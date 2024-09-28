# Further Extended Python MARL framework - EPyMARL

FEPyMARL is  an extension of [EPyMARL](https://github.com/uoe-agents/epymarl), and includes
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
- Support for [Capture Target](https://github.com/yuchen-x/MacDeepMARL/blob/master/src/rlmamr/my_env/capture_target.py) environments.

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
- [Further Extended Python MARL framework - FEPyMARL](#further-extended-python-marl-framework---epymarl)
- [Table of Contents](#table-of-contents)
- [Installation & Run instructions](#installation--run-instructions)
  - [Base requirements installation](#base-requirements-installation)
  - [Torch installation](#torch-installation)
  - [Torch-scatter installation](#torch-scatter-installation)
  - [Installing LBF, RWARE, MPE, PettingZoo, Overcooked, and Pressure plate](#installing-lbf-rware-mpe-pettingzoo-overcooked-and-pressureplate)
  - [Using A Custom Gym Environment](#using-a-custom-gym-environment)
- [Run an experiment on a Gym environment](#run-an-experiment-on-a-gym-environment)
- [Run a hyperparameter search](#run-a-hyperparameter-search)
- [Saving and loading learnt models](#saving-and-loading-learnt-models)
  - [Saving models](#saving-models)
  - [Loading models](#loading-models)
- [Citing FEPyMARL, PyMARL, and EPyMARL](#citing-fepymarl-epymarl-and-pymarl)
- [License](#license)

# Installation & Run instructions

## Base requirements installation
To install the minimum requirements (without environments installation) run the following commands:
```sh
git clone ...
cd fepymarl/installation
conda create -n epymarl python=3.8.18 -y
conda activate epymarl
python3 -m pip install pip==24.0
pip install wheel==0.38.4 setuptools==65.5.0 einops
pip install -r requirements.txt
``` 
openCV:
```sh
pip install opencv-python==4.9.0.80
```
or, openCV for headless machines:
```sh
pip install opencv-python==4.9.0.80
```

## Torch installation
Then, run the following to install torch:
```sh
pip install protobuf==3.20.*
```
If you need torch with CUDA support:
```sh
pip3 install torch==2.1.2 torchvision==0.16.2--index-url https://download.pytorch.org/whl/cu121
```
otherwise:
```sh
pip3 install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cpu
```

## Torch-scatter installation
Finally, run the following for torch-scatter installation (which is needed for most of the Actor-Critic algorithms):
```sh
sh ./install_torch_scatter.sh
```

## Installing LBF, RWARE, MPE, PettingZoo, Overcooked, PressurePlate and Capture Target

### LBF
To install [Level Based Foraging](https://github.com/uoe-agents/lb-foraging), run:
```sh
pip install lbforaging==1.1.1
```

Example of using LBF (replace ```<algo>``` and ```<scenario>```):
```sh
python3 src/main.py --config=<algo> --env-config=gymma with env_args.time_limit=25 env_args.key=<scenario>
```

Available scenarios we run experiments:
- "lbforaging:Foraging-4s-11x11-3p-2f-coop-v2",
- "lbforaging:Foraging-2s-11x11-3p-2f-coop-v2",
- "lbforaging:Foraging-2s-8x8-3p-2f-coop-v2",
- "lbforaging:Foraging-2s-9x9-3p-2f-coop-v2",
- "lbforaging:Foraging-7s-20x20-5p-3f-coop-v2",
- "lbforaging:Foraging-2s-12x12-2p-2f-coop-v2",
- "lbforaging:Foraging-8s-25x25-8p-5f-coop-v2",
- "lbforaging:Foraging-7s-30x30-7p-4f-coop-v2"

### RWARE
To install [Multi-Robot Warehouse](https://github.com/uoe-agents/robotic-warehouse), run:
```sh
pip install rware==1.0.3
```

Example of using RWARE (replace ```<algo>``` and ```<scenario>```):
```sh
python3 src/main.py --config=<algo> --env-config=gymma with env_args.time_limit=500 env_args.key=<scenario>
```

Available scenarios we run experiments:
- "rware:rware-small-4ag-hard-v1",
- "rware:rware-tiny-4ag-hard-v1",
- "rware:rware-tiny-2ag-hard-v1"

### MPE
To install [Multi-agent Particle Environment](https://github.com/semitable/multiagent-particle-envs), being in ```fepymarl/``` directory, run:
```sh
cd src/envs/multiagent-particle-envs/
pip install -e .
pip install seaborn==0.13.2  # Required for colors of landmarks
```

Example of using MPE (replace ```<algo>``` and ```<scenario>```):
```sh
python3 src/main.py --config=<algo> --env-config=gymma with env_args.time_limit=25 env_args.key=<scenario>
```

Available scenarios we run experiments:
- "mpe:SimpleSpeakerListener-v0", 
- "mpe:SimpleSpread-3-v0",
- "mpe:SimpleSpread-4-v0",
- "mpe:SimpleSpread-5-v0",
- "mpe:SimpleSpread-8-v0"

More available scenarios:
- "mpe:MultiSpeakerListener-v0"
- "mpe:SimpleAdversary-v0",
- "mpe:SimpleCrypto-v0",
- "mpe:SimplePush-v0",
- "mpe:SimpleReference-v0",
- "mpe:SimpleTag-v0"
- "mpe:SimpleWorldComm-v0",
- "mpe:ClimbingSpread-v0"

### PettingZoo
To install PettingZoo run:
```sh
pip install opencv-python-headless==4.9.0.80
# or
pip install opencv-python==4.9.0.80 # only for rendering

pip install transformers==4.38.2 pettingzoo==1.24.3 'pettingzoo[atari]'==1.24.3 autorom==0.6.1 'pettingzoo[butterfly]'==1.24.3 
AutoROM -y
```
Example of using PettingZoo:
```sh
python3 src/main.py --config=qmix --env-config=pettingzoo with env_args.max_cycles=25 env_args.key="pistonball_v6" env_args.kwargs="('n_pistons',10),"
```

### Overcooked
To install Overcooked, being in ```fepymarl/``` directory, run:
```sh
cd src/envs/overcooked_ai/
pip install -e .
# Uninstall opencv because it installs newer version
pip uninstall opencv-python opencv-python-headless -y
pip install opencv-python-headless==4.9.0.80 # or pip install opencv-python==4.9.0.80 # only for rendering
```

Example of using Overcooked (replace ```<algo>```, ```<scenario>```, and ```<reward_type>```):
```sh
python3 src/main.py --config=<algo> --env-config=overcooked with env_args.horizon=500 env_args.key=<scenario> env_args.reward_type=<reward_type>
```

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

### Pressure plate
To install Pressure plate, being in ```fepymarl/``` directory, run:
```sh
cd src/envs/pressureplate_ai/
pip install -e .
pip install pyglet==1.5.29 # For rendering
```

Example of using Pressure plate (replace ```<algo>``` and ```<scenario>```):
```sh
python3 src/main.py --config=<algo> --env-config=pressureplate with env_args.key=<scenario> env_args.horizon=500
```

Available scenarios we run experiments:
- "pressureplate-linear-4p-v0"
- "pressureplate-linear-6p-v0"

More available scenarios:
- "pressureplate-linear-5p-v0"

### Capture Target
To install capture target, being in ```fepymarl/``` directory run:
```sh
cd src/envs/capture_target/
pip install -e .
pip install pyglet==1.5.27 # For rendering
pip install gym==0.21.0  # To avoid installing other gym version
```

Example of using Capture Target (replace ```<algo>```):
```sh
python3 src/main.py --config=<algo> --env-config=capture_target with env_args.terminate_step=60
```

## Using A Custom Gym Environment

EPyMARL supports environments that have been registered with Gym. 
The only difference with the Gym framework would be that the returned rewards should be a tuple (one reward for each agent). In this cooperative framework we sum these rewards together.

Environments that are supported out of the box are the ones that are registered in Gym automatically. Examples are: [LBF](https://github.com/semitable/lb-foraging) and [RWARE](https://github.com/semitable/robotic-warehouse). 

To register a custom environment with Gym, use the template below (taken from Level-Based Foraging).
```python
from gym.envs.registration import registry, register, make, spec
register(
  id="Foraging-8x8-2p-3f-v1",                     # Environment ID.
  entry_point="lbforaging.foraging:ForagingEnv",  # The entry point for the environment class
  kwargs={
            ...                                   # Arguments that go to ForagingEnv's __init__ function.
        },
    )
```

# Run an experiment on a Gym environment

```shell
python3 src/main.py --config=qmix --env-config=gymma with env_args.time_limit=50 env_args.key="lbforaging:Foraging-8x8-2p-3f-v1"
```
 In the above command `--env-config=gymma` (in constrast to `sc2` will use a Gym compatible wrapper). `env_args.time_limit=50` sets the maximum episode length to 50 and `env_args.key="..."` provides the Gym's environment ID. In the ID, the `lbforaging:` part is the module name (i.e. `import lbforaging` will run automatically).


The config files act as defaults for an algorithm or environment. 

They are all located in `src/config`.
`--config` refers to the config files in `src/config/algs`
`--env-config` refers to the config files in `src/config/envs`

All results will be stored in the `Results` folder.

# Run a hyperparameter search

We include a script named `search.py` which reads a search configuration file (e.g. the included `search.config.example.yaml`) and runs a hyperparameter search in one or more tasks. The script can be run using
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

# Citing FEPyMARL, EPyMARL and PyMARL

If you use FEPyMARL in your research, please cite the ...

In BibTeX format:
'''tex
'''

If you use the EPyMARL in your research, please cite the [Benchmarking Multi-Agent Deep Reinforcement Learning Algorithms in Cooperative Tasks](https://arxiv.org/abs/2006.07869).

*Georgios Papoudakis, Filippos Christianos, Lukas Schäfer, & Stefano V. Albrecht. Benchmarking Multi-Agent Deep Reinforcement Learning Algorithms in Cooperative Tasks, Proceedings of the Neural Information Processing Systems Track on Datasets and Benchmarks (NeurIPS), 2021*

In BibTeX format:

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
All the source code that has been taken from the and EPyMARL PyMARL repository was licensed (and remains so) under the Apache License v2.0 (included in `LICENSE` file).
Any new code is also licensed under the Apache License v2.0
