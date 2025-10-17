import datetime
from functools import partial
import numpy as np

from pymarlzooplus.envs import REGISTRY as env_REGISTRY
from pymarlzooplus.components.episode_buffer import EpisodeBatch
from pymarlzooplus.utils.env_utils import check_env_installation


class EpisodeRunner:

    def __init__(self, args, logger):

        # Check if the requirements for the selected environment are installed
        check_env_installation(args.env, env_REGISTRY, logger)

        self.batch = None
        self.new_batch = None
        self.mac = None
        self.explorer = None
        if args.explorer == 'maven':  # MAVEN uses a noise vector which augments the observation
            self.noise = None

        self.args = args
        self.logger = logger
        self.batch_size = self.args.batch_size_run
        assert self.batch_size == 1

        # Initialize environment
        assert not (self.args.env == 'pettingzoo' and self.args.env_args['centralized_image_encoding'] is True), (
            "In 'episode_runner', the argument 'centralized_image_encoding' of pettingzoo should be False "
            "since there is only one environment, and thus the encoding can be considered as centralized."
        )
        self.env = env_REGISTRY[self.args.env](**self.args.env_args)
        # Get info from environment to be printed
        print_info = self.env.get_print_info()
        if print_info != "None" and print_info is not None:
            # Simulate the message format of the logger defined in _logging.py
            current_time = datetime.datetime.now().strftime('%H:%M:%S')
            self.logger.console_logger.info(f"\n[INFO {current_time}] episode_runner {print_info}")

        self.episode_limit = self.env.episode_limit
        self.t = 0
        self.t_env = 0

        self.train_returns = []
        self.test_returns = []
        self.train_stats = {}
        self.test_stats = {}

        # Log the first run
        self.log_train_stats_t = -1000000

    def setup(self, scheme, groups, preprocess, mac, explorer):
        self.new_batch = partial(
            EpisodeBatch,
            scheme,
            groups,
            self.batch_size,
            self.episode_limit + 1,
            preprocess=preprocess,
            device=self.args.device
        )
        self.mac = mac
        self.explorer = explorer

    def get_env_info(self):
        return self.env.get_env_info()

    def save_replay(self):
        self.env.save_replay()

    def close_env(self):
        self.env.close()

    def reset(self):
        self.batch = self.new_batch()
        self.env.reset()
        self.t = 0

    def run(self, test_mode=False):
        self.reset()

        terminated = False
        episode_return = 0
        env_info = {}

        self.mac.init_hidden(batch_size=self.batch_size)

        while not terminated:

            pre_transition_data = {
                "state": [self.env.get_state()],
                "avail_actions": [self.env.get_avail_actions()],
                "obs": [self.env.get_obs()]
            }

            self.batch.update(pre_transition_data, ts=self.t)

            # In the case of MAVEN, at the beginning of each episode, sample the noise vector and add it to the batch.
            if self.args.explorer == 'maven' and self.t == 0:
                self.noise = self.explorer.sample(self.batch['state'][:, 0])
                self.batch.update({"noise": self.noise}, ts=0)

            # Pass the entire batch of experiences up till now to the agents
            # Receive the actions for each agent at this timestep in a batch of size 1
            actions, extra_returns = self.mac.select_actions(
                self.batch,
                t_ep=self.t,
                t_env=self.t_env,
                test_mode=test_mode
            )

            ## In the case of EOI, choose actions based on the explorer.
            if self.args.explorer == 'eoi':
                actions = self.explorer.select_actions(
                    actions,
                    self.t,
                    test_mode,
                    pre_transition_data
                )

            # Step
            reward, terminated, env_info = self.env.step(actions[0])

            # Render
            if test_mode and self.args.render:
                self.env.render()

            ## Print info
            # Get info from environment to be printed
            print_info = self.env.get_print_info()
            if print_info != "None" and print_info is not None:
                # Simulate the message format of the logger defined in _logging.py
                current_time = datetime.datetime.now().strftime('%H:%M:%S')
                self.logger.console_logger.info(f"\n[INFO {current_time}] episode_runner {print_info}")

            # Keep track of episode return
            episode_return += reward

            post_transition_data = {
                "actions": actions,
                "reward": [(reward,)],
                "terminated": [(terminated != env_info.get("episode_limit", False),)],
            }

            self.batch.update(post_transition_data, ts=self.t)

            self.t += 1

        last_data = {
            "state": [self.env.get_state()],
            "avail_actions": [self.env.get_avail_actions()],
            "obs": [self.env.get_obs()]
        }
        if test_mode and self.args.render:
            print(f"Episode return: {episode_return}")
        self.batch.update(last_data, ts=self.t)

        # Select actions in the last stored state
        actions, extra_returns = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode)
        self.batch.update({"actions": actions}, ts=self.t)

        cur_stats = self.test_stats if test_mode else self.train_stats
        cur_returns = self.test_returns if test_mode else self.train_returns
        log_prefix = "test_" if test_mode else ""
        cur_stats.update({k: cur_stats.get(k, 0) + env_info.get(k, 0) for k in set(cur_stats) | set(env_info)})
        cur_stats["n_episodes"] = 1 + cur_stats.get("n_episodes", 0)
        cur_stats["ep_length"] = self.t + cur_stats.get("ep_length", 0)

        if not test_mode:
            self.t_env += self.t

        cur_returns.append(episode_return)

        if test_mode and (len(self.test_returns) == self.args.test_nepisode):
            self._log(cur_returns, cur_stats, log_prefix)
        elif self.t_env - self.log_train_stats_t >= self.args.runner_log_interval:
            self._log(cur_returns, cur_stats, log_prefix)
            if hasattr(self.mac.action_selector, "epsilon"):
                self.logger.log_stat("epsilon", self.mac.action_selector.epsilon, self.t_env)
            self.log_train_stats_t = self.t_env

        return self.batch, [episode_return]

    def _log(self, returns, stats, prefix):
        self.logger.log_stat(prefix + "return_mean", np.mean(returns), self.t_env)
        self.logger.log_stat(prefix + "return_std", np.std(returns), self.t_env)
        returns.clear()

        for k, v in stats.items():
            if k != "n_episodes":
                self.logger.log_stat(prefix + k + "_mean", v / stats["n_episodes"], self.t_env)
        stats.clear()
