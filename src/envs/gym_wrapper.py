import random

import numpy as np
import gym
from gym import ObservationWrapper, spaces
from gym.spaces import flatdim
from gym.wrappers import TimeLimit as GymTimeLimit

from smac.env import MultiAgentEnv


class TimeLimit(GymTimeLimit):
    def __init__(self, env, max_episode_steps=None):
        super().__init__(env)
        if max_episode_steps is None and self.env.spec is not None:
            max_episode_steps = env.spec.max_episode_steps

        self._max_episode_steps = max_episode_steps
        self._elapsed_steps = None

    def step(self, action):
        assert (self._elapsed_steps is not None), "Cannot call env.step() before calling reset()"
        observation, reward, done, info = self.env.step(action)

        self._elapsed_steps += 1
        if self._elapsed_steps >= self._max_episode_steps:
            info["TimeLimit.truncated"] = not all(done) \
                if type(done) is list \
                else not done
            done = len(observation) * [True]
        return observation, reward, done, info


class FlattenObservation(ObservationWrapper):
    r"""Observation wrapper that flattens the observation of individual agents."""

    def __init__(self, env):
        super(FlattenObservation, self).__init__(env)

        ma_spaces = []

        for sa_obs in env.observation_space:
            flatdim_ = spaces.flatdim(sa_obs)
            ma_spaces += [
                spaces.Box(low=-float("inf"),
                           high=float("inf"),
                           shape=(flatdim_,),
                           dtype=np.float32,)
                         ]

        self.observation_space = spaces.Tuple(tuple(ma_spaces))

    def observation(self, observation):

        return tuple(
            [spaces.flatten(obs_space, obs) for obs_space, obs in zip(self.env.observation_space, observation)]
                    )


class _GymmaWrapper(MultiAgentEnv):
    def __init__(self, key, time_limit=None, seed=1, **kwargs):

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

        self.original_env = gym.make(f"{key}", **kwargs)
        self.episode_limit = time_limit
        self._env = TimeLimit(self.original_env, max_episode_steps=time_limit)
        self._env = FlattenObservation(self._env)

        self.n_agents = self._env.n_agents
        self._obs = None
        self._info = None

        self.longest_action_space = max(self._env.action_space, key=lambda x: x.n)
        self.longest_observation_space = max(self._env.observation_space, key=lambda x: x.shape)

        self._seed = seed
        self._env.seed(self._seed)

    def step(self, actions):
        """ Returns reward, terminated, info """

        actions = [int(a) for a in actions]
        actions = self.filter_actions(actions)

        self._obs, reward, done, self._info = self._env.step(actions)
        self._obs = [np.pad(o,
                            (0, self.longest_observation_space.shape[0] - len(o)),
                            "constant",
                            constant_values=0,)
                     for o in self._obs]

        if type(reward) is list:
            reward = sum(reward)
        if type(done) is list:
            done = all(done)

        return float(reward), done, {}

    def filter_actions(self, actions):
        """
        Filter the actions of agents based on the available actions.
        If an invalid action found, it will be replaced with the first available action.
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
            self._env.seed(self._seed)

        self._obs = self._env.reset()
        self._obs = [np.pad(o,
                            (0, self.longest_observation_space.shape[0] - len(o)),
                            "constant",
                            constant_values=0,)
                     for o in self._obs]

        return self.get_obs(), self.get_state()

    def render(self):
        self._env.render()

    def close(self):
        self._env.close()

    def seed(self):
        return self._env.seed

    def save_replay(self):
        pass

    def get_stats(self):
        return {}
