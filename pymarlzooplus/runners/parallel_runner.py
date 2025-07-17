import time
from functools import partial
import datetime
import numpy as np
import torch as th
from torch.multiprocessing import Pipe, Process

from pymarlzooplus.envs import REGISTRY as env_REGISTRY
from pymarlzooplus.components.episode_buffer import EpisodeBatch
from pymarlzooplus.utils.image_encoder import ImageEncoder
from pymarlzooplus.utils.env_utils import check_env_installation

from modules.bandits.uniform import Uniform
from modules.bandits.reinforce_hierarchial import EZ_agent as enza
from modules.bandits.returns_bandit import ReturnsBandit as RBandit


# Based (very) heavily on SubprocVecEnv from OpenAI Baselines
# https://github.com/openai/baselines/blob/master/baselines/common/vec_env/subproc_vec_env.py
class ParallelRunner:

    def __init__(self, args, logger):

        # Check if the requirements for the selected environment are installed
        check_env_installation(args.env, env_REGISTRY, logger)

        self.preprocess = None
        self.groups = None
        self.scheme = None
        self.mac = None
        self.explorer = None
        self.new_batch = None
        self.batch = None
        self.env_steps_this_run = None

        self.args = args
        self.logger = logger
        self.batch_size = self.args.batch_size_run

        # Enable multithreading access to GPU
        # Check if the start method is set to spawn, if not set it to spawn
        if th.multiprocessing.get_start_method(allow_none=True) is None:
            th.multiprocessing.set_start_method('spawn')

        # In case of pettingzoo and centralized image encoding, initialize image encoder here
        image_encoder = None
        if self.args.env == 'pettingzoo' and self.args.env_args['centralized_image_encoding'] is True:
            image_encoder_args = [
                "parallel_runner",
                self.args.env_args['centralized_image_encoding'],
                self.args.env_args['trainable_cnn'],
                self.args.env_args['image_encoder'],
                self.args.env_args['image_encoder_batch_size'],
                self.args.env_args['image_encoder_use_cuda']
            ]
            image_encoder = ImageEncoder(*image_encoder_args)
            image_encoder.share_memory()  # Make model parameters shareable across processes
            self.args.env_args['given_observation_space'] = image_encoder.observation_space
            self.logger.console_logger.info(image_encoder.print_info)

        # Make subprocesses for the envs
        self.parent_conns, self.worker_conns = zip(*[Pipe() for _ in range(self.batch_size)])
        env_fn = env_REGISTRY[self.args.env]
        env_args = [self.args.env_args.copy() for _ in range(self.batch_size)]
        for i in range(self.batch_size):
            env_args[i]["seed"] += i

        self.ps = [
            Process(
                target=env_worker,
                args=(
                    worker_conn,
                    CloudpickleWrapper(partial(env_fn, **env_arg)),
                    image_encoder
                )
            ) for env_arg, worker_conn in zip(env_args, self.worker_conns)
        ]

        for p in self.ps:
            p.daemon = True
            p.start()

        # Get info from environment to be printed
        self.parent_conns[0].send(("get_print_info", None))
        time.sleep(5)  # Wait a little to initialize the environment and get the print info
        print_info = self.parent_conns[0].recv()
        if print_info != "None" and print_info is not None:
            self.logger.console_logger.info(print_info)
        self.parent_conns[0].send(("get_env_info", None))
        self.env_info = self.parent_conns[0].recv()
        self.episode_limit = self.env_info["episode_limit"]

        self.t = 0

        self.t_env = 0

        self.train_returns = []
        self.test_returns = []
        self.train_stats = {}
        self.test_stats = {}

        self._in_test_phase = False
        
        self.log_train_stats_t = -100000

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
        self.scheme = scheme
        self.groups = groups
        self.preprocess = preprocess
        
        if hasattr(self.args, "noise_dim") and self.args.noise_dim is not None:
            if not hasattr(self, "noise_returns"): self.noise_returns = {}
            if not hasattr(self, "noise_test_won"): self.noise_test_won = {}
            if not hasattr(self, "noise_train_won"): self.noise_train_won = {}

            self.noise_distrib = enza(self.args, logger=self.logger)
               
    def get_env_info(self):
        return self.env_info

    def save_replay(self):
        self.parent_conns[0].send(("save_replay", None))

    def close_env(self):
        for parent_conn in self.parent_conns:
            parent_conn.send(("close", None))

    def reset(self, test_mode=False):
        self.batch = self.new_batch()

        # Reset the envs
        for parent_conn in self.parent_conns:
            parent_conn.send(("reset", None))

        # Reset hidden states
        self.mac.init_hidden(batch_size=self.batch_size)

        pre_transition_data = {
            "state": [],
            "avail_actions": [],
            "obs": []
        }

        # Get the obs, state and avail_actions back
        for parent_conn in self.parent_conns:
            data = parent_conn.recv()
            pre_transition_data["state"].append(data["state"])
            pre_transition_data["avail_actions"].append(data["avail_actions"])
            pre_transition_data["obs"].append(data["obs"])

        if "hidden_states" in self.args.extra_in_buffer or "hidden_states_critic" in self.args.extra_in_buffer:
            hidden_states_dict = self.mac.get_hidden_states()
            if "hidden_states" in self.args.extra_in_buffer:
                pre_transition_data["hidden_states"] = hidden_states_dict["hidden_states"]
            if "hidden_states_critic" in self.args.extra_in_buffer:
                pre_transition_data["hidden_states_critic"] = hidden_states_dict["hidden_states_critic"]

        self.batch.update(pre_transition_data, ts=0)
        
        if hasattr(self.args, "noise_dim") and self.args.noise_dim is not None:
            self.noise = self.noise_distrib.sample(self.batch['state'][:,0], test_mode)
            self.batch.update({"noise": self.noise}, ts=0)
        
        self.t = 0
        self.env_steps_this_run = 0

        return pre_transition_data

    def run(self, test_mode=False, test_uniform=False):
        
        if hasattr(self.args, "noise_dim") and self.args.noise_dim is not None:
            if test_mode and not self._in_test_phase:
                self.noise_returns = {}
                self.noise_test_won = {}
        
        self._in_test_phase = test_mode
        pre_transition_data = self.reset()

        episode_returns = [0 for _ in range(self.batch_size)]
        episode_lengths = [0 for _ in range(self.batch_size)]
        terminated = [False for _ in range(self.batch_size)]
        envs_not_terminated = [b_idx for b_idx, termed in enumerate(terminated) if not termed]
        final_env_infos = []  # may store extra stats like battle won. this is filled in ORDER OF TERMINATION

        while True:

            # Pass the entire batch of experiences up till now to the agents
            # Receive the actions for each agent at this timestep in a batch for each unterminated env
            actions, extra_returns = self.mac.select_actions(
                self.batch,
                t_ep=self.t,
                t_env=self.t_env,
                bs=envs_not_terminated,
                test_mode=test_mode
            )

            # Choose actions based on explorer, if applicable. This is for EOI.
            if self.explorer is not None:
                actions = self.explorer.select_actions(
                    actions,
                    self.t,
                    test_mode,
                    pre_transition_data
                )

            cpu_actions = actions.to("cpu").numpy()

            # Update the actions taken
            actions_chosen = {
                "actions": actions.unsqueeze(1)
            }
            if "log_probs" in self.args.extra_in_buffer:
                actions_chosen["log_probs"] = extra_returns["log_probs"].unsqueeze(1)
            if "values" in self.args.extra_in_buffer:
                actions_chosen["values"] = extra_returns["values"].unsqueeze(1)

            self.batch.update(actions_chosen, bs=envs_not_terminated, ts=self.t, mark_filled=False)

            # Send actions to each env
            action_idx = 0
            for idx, parent_conn in enumerate(self.parent_conns):
                if idx in envs_not_terminated:  # We produced actions for this env
                    if not terminated[idx]:  # Only send the actions to the env if it hasn't terminated
                        parent_conn.send(("step", cpu_actions[action_idx]))
                    action_idx += 1  # actions is not a list over every env
                    # Rendering
                    if idx == 0 and test_mode and self.args.render:
                        parent_conn.send(("render", None))

            # Update envs_not_terminated
            envs_not_terminated = [b_idx for b_idx, termed in enumerate(terminated) if not termed]
            all_terminated = all(terminated)
            if all_terminated:
                break

            # Post-step data we will insert for the current timestep
            post_transition_data = {
                "reward": [],
                "terminated": []
            }
            # Data for the next step we will insert in order to select an action
            pre_transition_data = {
                "state": [],
                "avail_actions": [],
                "obs": []
            }
            if "hidden_states" in self.args.extra_in_buffer:
                pre_transition_data["hidden_states"] = \
                    extra_returns["hidden_states"][
                        ~th.tensor(terminated).to(extra_returns["hidden_states"].device)
                                                  ]
            if "hidden_states_critic" in self.args.extra_in_buffer:
                pre_transition_data["hidden_states_critic"] = \
                    extra_returns["hidden_states_critic"][
                        ~th.tensor(terminated).to(extra_returns["hidden_states_critic"].device)
                                                         ]

            # Receive data back for each unterminated env
            for idx, parent_conn in enumerate(self.parent_conns):
                if not terminated[idx]:
                    data = parent_conn.recv()
                    # Remaining data for this current timestep
                    post_transition_data["reward"].append((data["reward"],))

                    episode_returns[idx] += data["reward"]
                    episode_lengths[idx] += 1
                    if not test_mode:
                        self.env_steps_this_run += 1

                    env_terminated = False
                    if data["terminated"]:
                        final_env_infos.append(data["info"])
                    if data["terminated"] and not data["info"].get("episode_limit", False):
                        env_terminated = True
                    terminated[idx] = data["terminated"]
                    post_transition_data["terminated"].append((env_terminated,))

                    # Data for the next timestep needed to select an action
                    pre_transition_data["state"].append(data["state"])
                    pre_transition_data["avail_actions"].append(data["avail_actions"])
                    pre_transition_data["obs"].append(data["obs"])

            # Get print info for all env
            for idx, parent_conn in enumerate(self.parent_conns):
                parent_conn.send(("get_print_info", None))
                print_info = parent_conn.recv()
                if print_info != "None" and print_info is not None:
                    self.logger.console_logger.info(print_info)

            # Add post-transition data into the batch
            self.batch.update(
                post_transition_data,
                bs=envs_not_terminated,
                ts=self.t,
                mark_filled=False
            )

            # Move onto the next timestep
            self.t += 1

            # Add the pre-transition data
            self.batch.update(pre_transition_data, bs=envs_not_terminated, ts=self.t, mark_filled=True)

        if not test_mode:
            self.t_env += self.env_steps_this_run

        # Get stats back for each env
        for parent_conn in self.parent_conns:
            parent_conn.send(("get_stats", None))

        env_stats = []
        for parent_conn in self.parent_conns:
            env_stat = parent_conn.recv()
            env_stats.append(env_stat)

        cur_stats = self.test_stats if test_mode else self.train_stats
        cur_returns = self.test_returns if test_mode else self.train_returns
        log_prefix = "test_" if test_mode else ""
        infos = [cur_stats] + final_env_infos
        cur_stats.update({k: sum(d.get(k, 0) for d in infos) for k in set.union(*[set(d) for d in infos])})
        cur_stats["n_episodes"] = self.batch_size + cur_stats.get("n_episodes", 0)
        cur_stats["ep_length"] = sum(episode_lengths) + cur_stats.get("ep_length", 0)
        
        cur_returns.extend(episode_returns)
       
        if hasattr(self.args, "noise_dim") and self.args.noise_dim is not None:
            self._update_noise_returns(episode_returns, self.noise, final_env_infos, test_mode)
            if not test_mode:
                self.noise_distrib.update_returns(self.batch['state'][:,0], self.noise, episode_returns, test_mode, self.t_env)
       
        n_test_runs = max(1, self.args.test_nepisode // self.batch_size) * self.batch_size
 
        if test_mode and (len(self.test_returns) >= n_test_runs):
            self._log(cur_returns, cur_stats, log_prefix)
            if hasattr(self.args, "noise_dim") and self.args.noise_dim is not None:
                self._log_noise_returns(test_mode, test_uniform)
       
        elif self.t_env - self.log_train_stats_t >= self.args.runner_log_interval:
           
            self._log(cur_returns, cur_stats, log_prefix)
            if hasattr(self.args, "noise_dim") and self.args.noise_dim is not None:
                self._log_noise_returns(test_mode, test_uniform)
 
            if hasattr(self.mac.action_selector, "epsilon"):
                self.logger.log_stat("epsilon", self.mac.action_selector.epsilon, self.t_env)
            self.log_train_stats_t = self.t_env
 
        return self.batch
    
    def _log(self, returns, stats, prefix):
        self.logger.log_stat(prefix + "return_mean", np.mean(returns), self.t_env)
        self.logger.log_stat(prefix + "return_std", np.std(returns), self.t_env)
        returns.clear()

        for k, v in stats.items():
            if k != "n_episodes":
                self.logger.log_stat(prefix + k + "_mean", v/stats["n_episodes"], self.t_env)
        stats.clear()

    def _update_noise_returns(self, returns, noise, stats, test_mode):
        for n, r in zip(noise, returns):
            n = int(np.argmax(n))
            if n in self.noise_returns:
                self.noise_returns[n].append(r)
            else:
                self.noise_returns[n] = [r]
        if test_mode:
            noise_won = self.noise_test_won
        else:
            noise_won = self.noise_train_won

        if stats != [] and "battle_won" in stats[0]:
            for n, info in zip(noise, stats):
                if "battle_won" not in info:
                    continue
                bw = info["battle_won"]
                n = int(np.argmax(n))
                if n in noise_won:
                    noise_won[n].append(bw)
                else:
                    noise_won[n] = [bw]

    def _log_noise_returns(self, test_mode, test_uniform):
        if test_mode:
            max_noise_return = -100000
            for n,rs in self.noise_returns.items():
                n_item = n
                r_mean = float(np.mean(rs))
                max_noise_return = max(r_mean, max_noise_return)
                self.logger.log_stat("{}_noise_test_ret_u_{:1}".format(n_item, test_uniform), r_mean, self.t_env)
            self.logger.log_stat("max_noise_test_ret_u_{:1}".format(test_uniform), max_noise_return, self.t_env)

        noise_won = self.noise_test_won
        prefix = "test"
        if test_uniform:
            prefix += "_uni"
        if not test_mode:
            noise_won = self.noise_train_won
            prefix = "train"
        if len(noise_won.keys()) > 0:
            max_test_won = 0
            for n, rs in noise_won.items():
                n_item = n #int(np.argmax(n))
                r_mean = float(np.mean(rs))
                max_test_won = max(r_mean, max_test_won)
                self.logger.log_stat("{}_noise_{}_won".format(n_item, prefix), r_mean, self.t_env)
            self.logger.log_stat("max_noise_{}_won".format(prefix), max_test_won, self.t_env)
        self.noise_returns = {}
        self.noise_test_won = {}
        self.noise_train_won = {}

def env_worker(remote, env_fn, image_encoder):
    # Make environment
    env = env_fn.x()

    while True:
        cmd, data = remote.recv()
        if cmd == "step":
            actions = data
            # Take a step in the environment
            reward, terminated, env_info = env.step(actions)
            # Return the observations, avail_actions and state to make the next action
            avail_actions = env.get_avail_actions()
            state = env.get_state()
            obs = env.get_obs()
            if image_encoder is not None:
                # 'obs' is tuple with a single element - a dictionary of observations, so we keep only this
                obs = image_encoder.observation(obs[0])
                state = np.concatenate(obs, axis=0).astype(np.float32)  # Concatenate the encoded observations (vectors)
            remote.send({
                # Data for the next timestep needed to pick an action
                "state": state,
                "avail_actions": avail_actions,
                "obs": obs,
                # Rest of the data for the current timestep
                "reward": reward,
                "terminated": terminated,
                "info": env_info
            })
        elif cmd == "reset":
            env.reset()
            avail_actions = env.get_avail_actions()
            state = env.get_state()
            obs = env.get_obs()
            if image_encoder is not None:
                # 'obs' is tuple with a single element - a dictionary of observations, so we let it as is since
                # the observations are in this format when coming from reset
                obs = image_encoder.observation(obs)
                # Concatenate the encoded observations (vectors)
                state = np.concatenate(obs, axis=0).astype(np.float32)
            remote.send({
                "state": state,
                "avail_actions": avail_actions,
                "obs": obs
            })
        elif cmd == "close":
            env.close()
            remote.close()
            break
        elif cmd == "get_env_info":
            remote.send(env.get_env_info())
        elif cmd == "get_stats":
            remote.send(env.get_stats())
        elif cmd == "render":
            env.render()
        elif cmd == "save_replay":
            env.save_replay()
        elif cmd == "get_print_info":
            print_info = env.get_print_info()
            if print_info is None:
                remote.send("None")
            else:
                # Simulate the message format of the logger defined in _logging.py
                current_time = datetime.datetime.now().strftime('%H:%M:%S')
                print_info = f"\n[INFO {current_time}] parallel_runner " + print_info
                remote.send(print_info)
        else:
            raise NotImplementedError


class CloudpickleWrapper:
    """
    Uses cloudpickle to serialize contents (otherwise multiprocessing tries to use pickle)
    """
    def __init__(self, x):
        self.x = x

    def __getstate__(self):
        import cloudpickle
        return cloudpickle.dumps(self.x)

    def __setstate__(self, ob):
        import pickle
        self.x = pickle.loads(ob)

