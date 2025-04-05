import time
import unittest
import traceback
import numpy as np

from pymarlzooplus.envs import REGISTRY as env_REGISTRY
from tests.config import (
    pettingzoo_fully_coop_env_keys,
    overcooked_env_keys,
    pressureplate_env_keys,
    lbf_env_keys,
    rware_env_keys,
    mpe_env_keys,
    capturetarget_env_keys,
    boxpushing_env_keys, pettingzoo_classic_keys, pettingzoo_fully_coop_partial_obs_env_keys,
    pettingzoo_non_fully_coop_env_butterfly_mpe_sisl_keys, pettingzoo_non_fully_coop_env_atari_keys,
)

class TestEnvironmentsAPI(unittest.TestCase):
    
    # noinspection PyUnresolvedReferences
    # noinspection PyDictCreation
    @classmethod
    def setUpClass(cls):
        # Set up parameters
        cls.env_api_fully_coop_params_dict = {}
        cls.env_api_non_fully_coop_params_dict = {}

        ############################################################
        # Arguments to test environment API with all fully cooperative environments

        for pettingzoo_fully_coop_env_key in pettingzoo_fully_coop_env_keys:
            cls.env_api_fully_coop_params_dict[pettingzoo_fully_coop_env_key] = {
                "env": "pettingzoo",
                "env_args": {
                    "key": pettingzoo_fully_coop_env_key,
                    "time_limit": 10,
                    "render_mode": "human",
                    "image_encoder": "ResNet18",
                    "image_encoder_use_cuda": False,
                    "image_encoder_batch_size": 2,
                    "trainable_cnn": False,
                    "kwargs": "",
                    "seed": 2024
                }
            }
            cls.env_api_fully_coop_params_dict[f"{pettingzoo_fully_coop_env_key}_raw_images"] = {
                "env": "pettingzoo",
                "env_args": {
                    "key": pettingzoo_fully_coop_env_key,
                    "time_limit": 10,
                    "render_mode": "human",
                    "image_encoder": "ResNet18",
                    "image_encoder_use_cuda": False,
                    "image_encoder_batch_size": 2,
                    "trainable_cnn": True,
                    "kwargs": "",
                    "seed": 2024
                }
            }

        for pettingzoo_fully_coop_partial_obs_env_key in pettingzoo_fully_coop_partial_obs_env_keys:
            cls.env_api_fully_coop_params_dict[f"{pettingzoo_fully_coop_partial_obs_env_key}_partial_observation"] = {
                "env": "pettingzoo",
                "env_args": {
                    "key": pettingzoo_fully_coop_partial_obs_env_key,
                    "time_limit": 10,
                    "render_mode": "human",
                    "image_encoder": "ResNet18",
                    "image_encoder_use_cuda": False,
                    "image_encoder_batch_size": 2,
                    "trainable_cnn": False,
                    "partial_observation": True,
                    "kwargs": "",
                    "seed": 2024
                }
            }
            cls.env_api_fully_coop_params_dict[
                f"{pettingzoo_fully_coop_partial_obs_env_key}_raw_images_partial_observation"
            ] = {
                "env": "pettingzoo",
                "env_args": {
                    "key": pettingzoo_fully_coop_partial_obs_env_key,
                    "time_limit": 10,
                    "render_mode": "human",
                    "image_encoder": "ResNet18",
                    "image_encoder_use_cuda": False,
                    "image_encoder_batch_size": 2,
                    "trainable_cnn": True,
                    "partial_observation": True,
                    "kwargs": "",
                    "seed": 2024
                }
            }

        for overcooked_env_key in overcooked_env_keys:
            cls.env_api_fully_coop_params_dict[f"{overcooked_env_key}_sparse"] = {
                "env": "overcooked",
                "env_args": {
                    "key": overcooked_env_key,
                    "time_limit": 10,
                    "reward_type": "sparse",
                    "seed": 2024
                }
            }
            cls.env_api_fully_coop_params_dict[f"{overcooked_env_key}_shaped"] = {
                "env": "overcooked",
                "env_args": {
                    "key": overcooked_env_key,
                    "time_limit": 10,
                    "reward_type": "shaped",
                    "seed": 2024
                }
            }

        for pressureplate_env_key in pressureplate_env_keys:
            cls.env_api_fully_coop_params_dict[pressureplate_env_key] = {
              "env": "pressureplate",
              "env_args": {
                  "key": pressureplate_env_key,
                  "time_limit": 10,
                  "seed": 2024
              }
            }

        for lbf_env_key in lbf_env_keys:
            cls.env_api_fully_coop_params_dict[lbf_env_key] = {
                "env": "gymma",
                "env_args": {
                    "key": lbf_env_key,
                    "time_limit": 10,
                    "seed": 2024,
                }
            }

        for rware_env_key in rware_env_keys:
            cls.env_api_fully_coop_params_dict[rware_env_key] = {
              "env": "gymma",
              "env_args": {
                  "key": rware_env_key,
                  "time_limit": 10,
                  "seed": 2024
              }
            }

        for mpe_env_key in mpe_env_keys:
            cls.env_api_fully_coop_params_dict[mpe_env_key] = {
                "env": "gymma",
                "env_args": {
                    "key": mpe_env_key,
                    "time_limit": 10,
                    "seed": 2024
                }
            }

        for capturetarget_env_key in capturetarget_env_keys:
            cls.env_api_fully_coop_params_dict[capturetarget_env_key] = {
                "env": "capturetarget",
                "env_args": {
                    "key": capturetarget_env_key,
                    "time_limit": 10,
                    "seed": 2024
                }
            }
            cls.env_api_fully_coop_params_dict[f"{capturetarget_env_key}_obs_one_hot"] = {
                "env": "capturetarget",
                "env_args": {
                    "key": capturetarget_env_key,
                    "time_limit": 10,
                    "seed": 2024,
                    "obs_one_hot": True
                }
            }
            cls.env_api_fully_coop_params_dict[f"{capturetarget_env_key}_wo_tgt_avoid_agent"] = {
                "env": "capturetarget",
                "env_args": {
                    "key": capturetarget_env_key,
                    "time_limit": 10,
                    "seed": 2024,
                    "tgt_avoid_agent": False
                }
            }

        for boxpushing_env_key in boxpushing_env_keys:
            cls.env_api_fully_coop_params_dict[boxpushing_env_key] = {
                "env": "boxpushing",
                "env_args": {
                    "key": boxpushing_env_key,
                    "time_limit": 10,
                    "seed": 2024
                }
            }

        ###########################################################
        # Arguments to test environment API with all NON-fully cooperative PettingZoo environments (except for Classic)

        for pettingzoo_non_fully_coop_env_atari_key in pettingzoo_non_fully_coop_env_atari_keys:
            cls.env_api_non_fully_coop_params_dict[pettingzoo_non_fully_coop_env_atari_key] = {
                "env": "pettingzoo",
                "env_args": {
                    "key": pettingzoo_non_fully_coop_env_atari_key,
                    "time_limit": 10,
                    "render_mode": "human",
                    "image_encoder": "ResNet18",
                    "image_encoder_use_cuda": False,
                    "image_encoder_batch_size": 2,
                    "trainable_cnn": False,
                    "kwargs": "",
                    "seed": 2024
                }
            }
            cls.env_api_non_fully_coop_params_dict[f"{pettingzoo_non_fully_coop_env_atari_key}_raw_images"] = {
                "env": "pettingzoo",
                "env_args": {
                    "key": pettingzoo_non_fully_coop_env_atari_key,
                    "time_limit": 10,
                    "render_mode": "human",
                    "image_encoder": "ResNet18",
                    "image_encoder_use_cuda": False,
                    "image_encoder_batch_size": 2,
                    "trainable_cnn": True,
                    "kwargs": "",
                    "seed": 2024
                }
            }

        for pettingzoo_non_fully_coop_env_butterfly_mpe_sisl_key in (
                pettingzoo_non_fully_coop_env_butterfly_mpe_sisl_keys
        ):
            cls.env_api_non_fully_coop_params_dict[pettingzoo_non_fully_coop_env_butterfly_mpe_sisl_key] = {
                "env": "pettingzoo",
                "env_args": {
                    "key": pettingzoo_non_fully_coop_env_butterfly_mpe_sisl_key,
                    "time_limit": 10,
                    "render_mode": "human",
                    "kwargs": "",
                    "seed": 2024
                }
            }

        ###########################################################
        # Arguments to test environment API with all PettingZoo Classic environments
        cls.env_api_pz_classic_params_dict = {}
        for pettingzoo_classic_key in pettingzoo_classic_keys:
            cls.env_api_pz_classic_params_dict[pettingzoo_classic_key] = {
                "env": "pettingzoo",
                "env_args": {
                    "key": pettingzoo_classic_key,
                    "render_mode": "human",
                    "kwargs": "",
                }
            }

    def test_env_api_fully_coop(self):
        completed = []
        failed = {}

        for name, params in self.env_api_fully_coop_params_dict.items():
            print(
                "\n\n###########################################"
                "\n###########################################"
                f"\nRunning test for: {name}\n"
            )
            with self.subTest(environment=name):
                try:

                    # Initialize environment
                    env = env_REGISTRY[params["env"]](**params["env_args"])

                    n_agns = env.get_n_agents()
                    assert isinstance(n_agns, int)

                    n_acts = env.get_total_actions()
                    assert isinstance(n_acts, int)

                    # Reset the environment
                    obs, state = env.reset()
                    assert isinstance(obs, tuple)
                    assert len(obs) == n_agns
                    obs_shape = obs[0].shape
                    for agent_obs in obs:
                        assert isinstance(agent_obs, np.ndarray)
                        assert agent_obs.shape == obs_shape
                    assert isinstance(state, np.ndarray)
                    if len(state.shape) == 1:
                        assert len(obs_shape) == 1
                        assert state.shape[0] == obs_shape[0] * n_agns
                    else:
                        assert len(state.shape) == 4
                        assert len(obs_shape) == 3
                        assert state.shape == (n_agns, *obs_shape)

                    done = False
                    # Run an episode
                    while not done:
                        # Render the environment (optional)
                        env.render()

                        # Insert the policy's actions here
                        actions = env.sample_actions()
                        assert isinstance(actions, list)
                        assert len(actions) == n_agns
                        for action in actions:
                            assert isinstance(action, int)

                        # Apply an environment step
                        reward, done, extra_info = env.step(actions)
                        assert isinstance(reward, float)
                        assert isinstance(done, bool)
                        assert isinstance(extra_info, dict)
                        assert len(extra_info) == 0

                        info = env.get_info()
                        assert 'TimeLimit.truncated' in list(info.keys())
                        assert isinstance(info['TimeLimit.truncated'], bool)

                        obs = env.get_obs()
                        state = env.get_state()
                        assert isinstance(obs, tuple)
                        assert len(obs) == n_agns
                        obs_shape = obs[0].shape
                        for agent_obs in obs:
                            assert isinstance(agent_obs, np.ndarray)
                            assert agent_obs.shape == obs_shape
                        assert isinstance(state, np.ndarray)
                        if len(state.shape) == 1:
                            assert len(obs_shape) == 1
                            assert state.shape[0] == obs_shape[0] * n_agns
                        else:
                            assert len(state.shape) == 4
                            assert len(obs_shape) == 3
                            assert state.shape == (n_agns, *obs_shape)

                    # Terminate the environment
                    env.close()

                    completed.append(name)
                except (Exception, SystemExit) as e:
                    # Convert the full traceback to a string
                    tb_str = traceback.format_exc()
                    # Store the traceback
                    failed[name] = tb_str
                    # Print just a simple message
                    self.fail(f"Test for '{name}' failed with exception: {e}")
                # Wait a short time to allow the environment to terminate.
                time.sleep(2)
        if failed:
            # Build a multiline message that unittest will print in full
            msg = "Some tests failed:\n"
            for name, tb_str in failed.items():
                msg += f"\n=== {name} ===\n{tb_str}\n"
            self.fail(msg)

    def test_env_api_non_fully_coop(self):
        completed = []
        failed = {}

        for name, params in self.env_api_non_fully_coop_params_dict.items():
            print(
                "\n\n###########################################"
                "\n###########################################"
                f"\nRunning test for: {name}\n"
            )
            with self.subTest(environment=name):
                try:

                    # Initialize environment
                    env = env_REGISTRY[params["env"]](**params["env_args"])

                    n_agns = env.get_n_agents()
                    assert isinstance(n_agns, int)

                    common_observation_space = env.common_observation_space()
                    assert isinstance(common_observation_space, bool)

                    is_image = env.is_image()
                    assert isinstance(is_image, bool)

                    agent_prefix = env.get_agent_prefix()
                    assert isinstance(agent_prefix, list)
                    assert len(agent_prefix) == n_agns
                    for _agent_prefix in agent_prefix:
                        assert isinstance(_agent_prefix, str)

                    # Reset the environment
                    _obs, _state = env.reset()

                    def check_obs_state(obs, state):
                        # Check the type of obs, its shape and
                        # (in case of common observation space) if all agents have the same observation space
                        assert isinstance(obs, dict)
                        assert len(obs) == n_agns
                        obs_shape = list(obs.values())[0].shape
                        for (agent_id, agent_obs), agent_prefix_id in zip(obs.items(), agent_prefix):
                            assert agent_id == agent_prefix_id
                            assert isinstance(agent_obs, np.ndarray)
                            if common_observation_space is True:
                                assert agent_obs.shape == obs_shape
                                if is_image is True:
                                    assert (
                                            (len(agent_obs.shape) == 3 and agent_obs.shape[0] == 3) or  # raw images
                                            len(agent_obs.shape) == 1  # encoded images
                                    )
                                else:
                                    assert len(agent_obs.shape) == 1, "agent_obs.shape: {}".format(agent_obs.shape)
                            else:
                                assert is_image is False
                                assert len(agent_obs.shape) == 1, "agent_obs.shape: {}".format(agent_obs.shape)

                        # Check the type of state and its shape
                        if common_observation_space is True:
                            assert isinstance(state, np.ndarray)
                            if is_image is True:
                                if len(state.shape) == 4:  # raw images
                                    assert state.shape == (n_agns, *obs_shape)
                                else:  # encoded images
                                    assert len(state.shape) == 1
                                    assert len(obs_shape) == 1
                                    assert state.shape == (n_agns * obs_shape[0],)
                            else:
                                assert len(state.shape) == 2
                                assert state.shape == (n_agns, obs_shape[0])
                        else:
                            assert isinstance(state, np.ndarray)
                            assert len(state.shape) == 1
                            state_shape = 0
                            for agent_obs in obs.values():
                                assert len(agent_obs.shape) == 1
                                state_shape += agent_obs.shape[0]
                            assert state.shape[0] == state_shape

                    check_obs_state(_obs, _state)

                    done = False
                    # Run an episode
                    while not done:

                        # Render the environment (optional)
                        env.render()

                        # Insert the policy's actions here
                        actions = env.sample_actions()
                        # Check the actions
                        assert isinstance(actions, list)
                        assert len(actions) == n_agns
                        for action in actions:
                            assert (
                                isinstance(action, int) or
                                (isinstance(action, np.ndarray) and env.key in ["waterworld_v4", "multiwalker_v9"])
                            )

                        # Apply an environment step
                        reward, done, info = env.step(actions)
                        # Check the rewards
                        assert isinstance(reward, dict)
                        assert len(reward) == n_agns
                        for (_agent_id, _reward), _agent_prefix_id in zip(reward.items(), agent_prefix):
                            assert _agent_id == _agent_prefix_id
                            assert isinstance(
                                _reward, (np.int64, np.int32, int, np.float64, float)
                            ), "type(_reward): {}".format(type(_reward))
                        # Check the dones
                        assert isinstance(done, dict)
                        assert len(done) == n_agns
                        for (_agent_id, _done), _agent_prefix_id in zip(done.items(), agent_prefix):
                            assert _agent_id == _agent_prefix_id
                            assert isinstance(_done, bool)
                        # Check the infos
                        assert isinstance(info, dict)
                        assert 'TimeLimit.truncated' in list(info.keys())
                        assert isinstance(info['TimeLimit.truncated'], bool)
                        assert 'infos' in list(info.keys())
                        assert isinstance(info['infos'], dict)
                        assert 'truncations' in list(info.keys())
                        assert isinstance(info['truncations'], dict)
                        for (_agent_id, _), _agent_prefix_id in zip(info['infos'].items(), agent_prefix):
                            assert _agent_id == _agent_prefix_id
                        for (_agent_id, _truncation), _agent_prefix_id in zip(
                                info['truncations'].items(), agent_prefix
                        ):
                            assert _agent_id == _agent_prefix_id
                            assert isinstance(_truncation, bool)
                        assert info['TimeLimit.truncated'] == any(
                            [_truncation for _truncation in info['truncations'].values()]
                        )

                        done = all([agent_done for agent_done in done.values()])
                        _obs = env.get_obs()
                        _state = env.get_state()
                        check_obs_state(_obs, _state)

                    # Terminate the environment
                    env.close()

                    completed.append(name)
                except (Exception, SystemExit) as e:
                    # Convert the full traceback to a string
                    tb_str = traceback.format_exc()
                    # Store the traceback
                    failed[name] = tb_str
                    # Print just a simple message
                    self.fail(f"Test for '{name}' failed with exception: {e}")
                # Wait a short time to allow the environment to terminate.
                time.sleep(2)
        if failed:
            # Build a multiline message that unittest will print in full
            msg = "Some tests failed:\n"
            for name, tb_str in failed.items():
                msg += f"\n=== {name} ===\n{tb_str}\n"
            self.fail(msg)

    def test_env_api_pz_classic(self):
        completed = []
        failed = {}

        for name, params in self.env_api_pz_classic_params_dict.items():
            print(
                "\n\n###########################################"
                "\n###########################################"
                f"\nRunning test for: {name}\n"
            )
            with self.subTest(environment=name):
                try:

                    # Initialize environment
                    env = env_REGISTRY[params["env"]](**params["env_args"]).original_env

                    # Get agents prefix
                    agents_prefix = [agent_prefix for agent_prefix in env.possible_agents]
                    n_agnts = len(agents_prefix)

                    # Reset environment
                    env.reset(seed=42)

                    # Run only for 10 steps, just for testing the functionality
                    count_steps = 0

                    # Run an episode
                    for agent in env.agent_iter():

                        # Render the environment (optional)
                        env.render()

                        # Get environment data
                        observation, reward, termination, truncation, info = env.last()
                        # Check obs
                        if name != "rps_v2":  # Except 'Rock Paper Scissors'
                            assert isinstance(observation, dict)
                            assert ['observation', 'action_mask'] == list(observation.keys())
                            assert isinstance(observation['observation'], np.ndarray)
                            assert isinstance(observation['action_mask'], np.ndarray)
                        else:
                            assert isinstance(observation, np.ndarray), f"type(observation): {type(observation)}"
                        # Check reward
                        assert isinstance(reward, (int, np.int64, float)), f"type(reward): {type(reward)}"
                        # Check termination
                        assert isinstance(termination, bool)
                        # Check truncation
                        assert isinstance(truncation, bool)
                        # Check info
                        assert isinstance(info, dict)
                        if name == "hanabi_v5":
                            assert ['action_mask'] == list(info.keys())
                            assert isinstance(info['action_mask'], np.ndarray)
                            assert np.all(info['action_mask'] == observation['action_mask'])
                        else:
                            assert len(info) == 0

                        # Get action
                        if termination or truncation:
                            action = None
                        else:
                            if name != "rps_v2":  # Except 'Rock Paper Scissors'
                                mask = observation["action_mask"]
                            # this is where you would insert your policy
                            if name != "rps_v2":  # Except 'Rock Paper Scissors'
                                action = env.action_space(agent).sample(mask)
                            else:
                                action = env.action_space(agent).sample()
                            # Check action
                            assert isinstance(action, np.int64)

                        # Apply an environment step
                        env.step(action)
                        # Stop after 10 steps
                        count_steps += 1
                        if count_steps >= 10:
                            break

                    # Terminate the environment
                    env.close()

                    completed.append(name)
                except (Exception, SystemExit) as e:
                    # Convert the full traceback to a string
                    tb_str = traceback.format_exc()
                    # Store the traceback
                    failed[name] = tb_str
                    # Print just a simple message
                    self.fail(f"Test for '{name}' failed with exception: {e}")
                # Wait a short time to allow the environment to terminate.
                time.sleep(2)
        if failed:
            # Build a multiline message that unittest will print in full
            msg = "Some tests failed:\n"
            for name, tb_str in failed.items():
                msg += f"\n=== {name} ===\n{tb_str}\n"
            self.fail(msg)

if __name__ == '__main__':
    unittest.main()
