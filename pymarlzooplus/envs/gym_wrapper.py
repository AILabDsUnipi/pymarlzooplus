import random
from typing import Any, Dict, Tuple
import os

import numpy as np
import gymnasium as gym
from gymnasium import ObservationWrapper, spaces
from gymnasium.spaces import flatdim
from gymnasium.wrappers import TimeLimit as GymTimeLimit
from gymnasium.utils.step_api_compatibility import step_api_compatibility

from pymarlzooplus.envs.multiagentenv import MultiAgentEnv


class TimeLimit(GymTimeLimit):
    def __init__(self, env, max_episode_steps=None):
        super().__init__(env, max_episode_steps=max_episode_steps)

        self._max_episode_steps = max_episode_steps
        self._elapsed_steps = None

    def timelimit_step(self, action) -> Tuple[Any, float, bool, Dict[str, Any]]:
        assert (self._elapsed_steps is not None), "Cannot call env.step() before calling reset()"
        observation, reward, done, info = step_api_compatibility(self.env.step(action), output_truncation_bool=False)

        # Add 'TimeLimit.truncated' in 'info' if it doesn't contain it
        if "TimeLimit.truncated" not in info:
            info["TimeLimit.truncated"] = False

        self._elapsed_steps += 1
        if self._elapsed_steps >= self._max_episode_steps:
            info["TimeLimit.truncated"] = not all(done) if type(done) is list else not done
            done = len(observation) * [True]

        return observation, reward, done, info


class FlattenObservation(ObservationWrapper):
    r"""Observation wrapper that flattens the observation of individual agents."""

    def __init__(self, env):
        super(FlattenObservation, self).__init__(env)

        # Keep the wrapped env to use its step method
        self.timelimit_env = env

        # Define the observation space
        if hasattr(env.observation_space, 'spaces'):
            env_observation_spaces = env.observation_space.spaces
        else:
            raise AttributeError(f"'spaces' attribute not found in observation space in the environment")
        ma_spaces = []
        for sa_obs in env_observation_spaces:
            flatted_dim = spaces.flatdim(sa_obs)
            ma_spaces += [
                spaces.Box(low=-float("inf"),
                           high=float("inf"),
                           shape=(flatted_dim,),
                           dtype=np.float32,)
                         ]
        self.observation_space = spaces.Tuple(tuple(ma_spaces))
        if hasattr(self.observation_space, 'spaces'):
            self.env_observation_spaces = self.observation_space.spaces
        else:
            raise AttributeError(f"'spaces' attribute not found in observation space in the environment")

    def observation(self, observation: tuple) -> tuple:

        return tuple(
            [spaces.flatten(obs_space, obs) for obs_space, obs in zip(self.env_observation_spaces, observation)]
                    )

    def step(self, action) -> Tuple[Any, float, bool, Dict[str, Any]]:

        if hasattr(self.timelimit_env, 'timelimit_step'):
            observation, reward, done, info = self.timelimit_env.timelimit_step(action)
            return self.observation(observation), reward, done, info
        else:
            raise AttributeError("The 'timelimit_step' method is not implemented in the 'FlattenObservation'")


class _GymmaWrapper(MultiAgentEnv):
    def __init__(self, key, time_limit=None, seed=1, render=False, **kwargs):

        super().__init__()

        # Check time_limit consistency
        if 'lbforaging' in key:
            if time_limit is None:
                time_limit = 50
            assert time_limit <= 50, 'LBF environments should have <=50 time_limit!'
        elif 'rware' in key:
            if time_limit is None:
                time_limit = 500
            assert time_limit <= 500, 'RWARE environments should have <=500 time_limit!'
        elif 'mpe' in key:
            if time_limit is None:
                time_limit = 25
        else:
            raise ValueError(f"key: {key}")

        # Fix rware v1 key
        if 'rware' in key and 'v1' in key:
            assert ':' in key, f"key: {key}"
            key = 'pymarlzooplus.envs.robotics_warehouse_v1.rware_v1:' + key.split(':')[1]

        # Fix lbforaging v2 key
        if 'lbforaging' in key and 'v2' in key:
            assert ':' in key, f"key: {key}"
            key = 'pymarlzooplus.envs.lb_foraging_v2.lbforaging_v2:' + key.split(':')[1]

        # Fix mpe key
        if 'mpe' in key:
            assert ':' in key, f"key: {key}"
            key = 'pymarlzooplus.envs.multiagent_particle_envs.mpe:' + key.split(':')[1]

        self.key = key
        self.episode_limit = time_limit
        self.render_bool = render
        self.internal_print_info = None
        # Check the consistency between the 'render_bool' and the display capabilities of the machine
        self.render_capable = True
        if self.render_bool is True and 'DISPLAY' not in os.environ:
            self.render_bool = False
            self.internal_print_info = (
                "\n\n###########################################################"
                "\nThe 'render' is set to 'False' due to the lack of display capabilities!"
                "\n###########################################################\n"
            )
            self.render_capable = False

        # Wrappers
        self.original_env = gym.make(f"{key}", **kwargs)
        self.timelimit_env = TimeLimit(self.original_env, max_episode_steps=time_limit)
        self._env = FlattenObservation(self.timelimit_env)

        # Define the number of agents
        if hasattr(self._env.unwrapped, 'n_agents'):
            self.n_agents = self._env.unwrapped.n_agents
        else:
            raise AttributeError(f"'n_agents' attribute not found in environment with key: {key}")

        # Placeholders
        self._obs = None
        self._info = None

        # Common observation and action spaces for different agent types
        if hasattr(self._env.action_space, 'spaces'):
            self.longest_action_space = max(self._env.action_space.spaces, key=lambda x: x.n)
        else:
            raise AttributeError(f"'spaces' attribute not found in action space in environment with key: {key}")
        if hasattr(self._env.observation_space, 'spaces'):
            self.longest_observation_space = max(self._env.observation_space.spaces, key=lambda x: x.shape)
        else:
            raise AttributeError(f"'spaces' attribute not found in observation space in environment with key: {key}")

        # Set seed
        self._seed = seed
        if hasattr(self._env.unwrapped, 'seed'):
            self._env.unwrapped.seed(self._seed)
        else:
            raise AttributeError(f"'seed' attribute not found in environment with key: {key}")

    def get_print_info(self):
        print_info = self.internal_print_info

        # Clear the internal print info
        self.internal_print_info = None

        return print_info

    def step(self, actions):
        """ Returns reward, terminated, info """

        if self.render_bool is True:
            self.render()

        actions = [int(a) for a in actions]
        actions = self.filter_actions(actions)

        self._obs, reward, done, self._info = self._env.step(actions)
        self._obs = tuple(
            np.pad(o, (0, self.longest_observation_space.shape[0] - len(o)), "constant", constant_values=0,)
            for o in self._obs
        )

        if isinstance(reward, (list, tuple)):
            reward = sum(reward)
        else:
            assert isinstance(reward, (int, float)), f"type(reward): {type(reward)}"
        if isinstance(done, (list, tuple)):
            done = all(done)
            assert isinstance(done, bool), f"type(done): {type(done)}"

        return float(reward), done, {}

    def filter_actions(self, actions):
        """
        Filter the actions of agents based on the available actions.
        If an invalid action is found, it will be replaced with the first available action.
        This allows the agents to learn that some actions have the same effects with others.
        Thus, we can have a shared NN policy for two or more agents which have different sets of available actions.
        """
        for agent_idx in range(self.n_agents):
            agent_avail_actions = self.get_avail_agent_actions(agent_idx)
            if not agent_avail_actions[actions[agent_idx]]:
                # Choose the first available action
                first_avail_action = agent_avail_actions.index(1)
                actions[agent_idx] = first_avail_action

        return actions

    def get_obs(self):
        """ Returns all agent observations in a list """
        return self._obs

    def get_obs_agent(self, agent_id):
        """ Returns observation for agent_id """
        raise self._obs[agent_id]

    def get_obs_size(self):
        """ Returns the shape of the observation """
        return flatdim(self.longest_observation_space)

    def get_state(self):
        return np.concatenate(self._obs, axis=0).astype(np.float32)

    def get_state_size(self):
        """ Returns the shape of the state"""
        if hasattr(self.original_env, 'state_size'):
            return self.original_env.state_size

        return self.n_agents * flatdim(self.longest_observation_space)

    def get_avail_actions(self):
        avail_actions = []
        for agent_id in range(self.n_agents):
            avail_agent = self.get_avail_agent_actions(agent_id)
            avail_actions.append(avail_agent)

        return avail_actions

    def get_avail_agent_actions(self, agent_id):
        """ Returns the available actions for agent_id """
        valid = flatdim(self._env.action_space[agent_id]) * [1]
        invalid = [0] * (self.longest_action_space.n - len(valid))

        return valid + invalid

    def get_total_actions(self):
        """ Returns the total number of actions an agent could ever take """
        return flatdim(self.longest_action_space)

    def sample_actions(self):
        return random.choices(range(0, self.get_total_actions()), k=self.n_agents)

    def reset(self, seed=None):
        """ Returns initial observations and states"""

        if seed is not None:
            self._seed = seed
            if hasattr(self._env.unwrapped, 'seed'):
                self._env.unwrapped.seed(self._seed)
            else:
                raise AttributeError(f"'seed' attribute not found in environment with key: {self.key}")

        self._obs, _ = self._env.reset()
        self._obs = tuple(
            np.pad(o, (0, self.longest_observation_space.shape[0] - len(o)), "constant", constant_values=0,)
            for o in self._obs
        )

        return self.get_obs(), self.get_state()

    def get_info(self):
        return self._info

    def get_n_agents(self):
        return self.n_agents

    def render(self):
        if self.render_capable is True:
            try:
                self._env.render()
            except (Exception, SystemExit) as e:
                self.internal_print_info = (
                    "\n\n###########################################################"
                    f"\nError during rendering: \n\n{e}"
                    f"\n\nRendering will be disabled to continue the training."
                    "\n###########################################################\n"
                )
                self.render_capable = False

    def close(self):
        self._env.close()

    def seed(self):
        if hasattr(self._env.unwrapped, 'seed'):
            return self._env.unwrapped.seed
        else:
            raise AttributeError(f"'seed' attribute not found in environment with key: {self.key}")

    def save_replay(self):
        pass

    @staticmethod
    def get_stats():
        return {}
