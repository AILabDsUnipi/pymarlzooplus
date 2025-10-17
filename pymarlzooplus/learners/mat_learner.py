import numpy as np
import torch as th
import torch.nn as nn
import math

from pymarlzooplus.components.standarize_stream import PopArt
from pymarlzooplus.modules.critics import REGISTRY as critic_registry
from pymarlzooplus.components.episode_buffer import EpisodeBatch


class MATLearner:

    def __init__(self, mac, scheme, logger, args):
        
        self.args = args
        self.logger = logger
        self.num_agents = args.n_agents

        self.training_steps = 0
        self.log_stats_t = -self.args.learner_log_interval - 1

        # Actor-Critic
        self.mac = mac
        self.critic = critic_registry[args.critic_type](scheme, args)
        self.mac.agent.critic = self.critic
        self.lr = float(args.lr)
        self.opti_eps = float(args.opti_eps)
        self.weight_decay = float(args.weight_decay)
        self.actor_critic_params = list(self.mac.parameters())  # It contains both actor and critic parameters
        self.optimizer = th.optim.Adam(params=self.actor_critic_params,
                                       lr=self.lr,
                                       eps=self.opti_eps,
                                       weight_decay=self.weight_decay)
        self.prep_rollout()

        # Hyperparameters
        self.clip_param = args.clip_param
        self.ppo_epoch = args.ppo_epoch
        self.num_mini_batch = args.num_mini_batch
        self.value_loss_coef = args.value_loss_coef
        self.entropy_coef = args.entropy_coef
        self.max_grad_norm = args.max_grad_norm
        self.huber_delta = args.huber_delta
        self.use_max_grad_norm = args.use_max_grad_norm
        self.use_clipped_value_loss = args.use_clipped_value_loss
        self.use_huber_loss = args.use_huber_loss
        self.use_popart = args.use_popart
        self.gamma = args.gamma
        self.gae_lambda = args.gae_lambda

        self.device = "cuda" if args.use_cuda else "cpu"
        self.mac.agent.device = self.device

        if self.use_popart:
            self.value_normalizer = PopArt(1, device=self.device)
        else:
            self.value_normalizer = None

        assert args.standardise_rewards is False, f"'args.standardise_rewards': {args.standardise_rewards}"
        assert args.standardise_returns is False, f"'args.standardise_rewards': {args.standardise_returns}"
        assert args.obs_last_action is False, f"'args.obs_last_action': {args.obs_last_action}"

    def cal_value_loss(self, values, value_preds_batch, return_batch):
        """
        Calculate value function loss.
        :param values: (torch.Tensor) value function predictions.
        :param value_preds_batch: (torch.Tensor) "old" value  predictions from data batch (used for value clip loss)
        :param return_batch: (torch.Tensor) reward to go returns.

        :return value_loss: (torch.Tensor) value function loss.
        """

        value_pred_clipped = value_preds_batch + (values - value_preds_batch).clamp(-self.clip_param, self.clip_param)

        if self.use_popart:
            self.value_normalizer.forward(return_batch, train=True)
            error_clipped = self.value_normalizer.forward(return_batch, train=False) - value_pred_clipped
            error_original = self.value_normalizer.forward(return_batch, train=False) - values
        else:
            error_clipped = return_batch - value_pred_clipped
            error_original = return_batch - values

        if self.use_huber_loss:
            value_loss_clipped = self.huber_loss(error_clipped, self.huber_delta)
            value_loss_original = self.huber_loss(error_original, self.huber_delta)
        else:
            value_loss_clipped = self.mse_loss(error_clipped)
            value_loss_original = self.mse_loss(error_original)

        if self.use_clipped_value_loss:
            value_loss = th.max(value_loss_original, value_loss_clipped)
        else:
            value_loss = value_loss_original

        value_loss = value_loss.mean()

        return value_loss

    def ppo_update(self, sample, train_stats):
        """
        Update actor and critic networks.
        :param sample: (Dictionary) contains data batch with which to update networks.
        :param train_stats: (Dictionary) contains the training results of all updates of the current training round

        :return train_stats: (Dictionary)
        """

        if train_stats is None:
            train_stats = {
                "value_loss": [],
                "grad_norm": [],
                "policy_loss": [],
                "entropy": [],
                "ratio": []
            }

        value_preds_batch = sample["values"].view(-1, 1)
        return_batch = sample["returns"].view(-1, 1)
        old_action_log_probs_batch = sample["log_probs"].view(-1, 1)
        adv_targ = sample["advantages"].view(-1, 1)

        values, action_log_probs, dist_entropy = self.mac.evaluate_actions(sample, t=0)

        # actor loss
        imp_weights = th.exp(action_log_probs - old_action_log_probs_batch)

        surr1 = imp_weights * adv_targ
        surr2 = th.clamp(imp_weights, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv_targ

        policy_loss = -th.sum(th.min(surr1, surr2), dim=-1, keepdim=True).mean()

        # critic loss
        value_loss = self.cal_value_loss(values, value_preds_batch, return_batch)

        # both losses
        loss = policy_loss - dist_entropy * self.entropy_coef + value_loss * self.value_loss_coef

        # update both actor and critic at once
        self.optimizer.zero_grad()
        loss.backward()

        if self.use_max_grad_norm:
            grad_norm = nn.utils.clip_grad_norm_(self.actor_critic_params, self.max_grad_norm)
        else:
            grad_norm = self.get_gard_norm(self.actor_critic_params)

        self.optimizer.step()

        train_stats["value_loss"].append(value_loss.item())
        train_stats["grad_norm"].append(grad_norm.item())
        train_stats["policy_loss"].append(policy_loss.item())
        train_stats["entropy"].append(dist_entropy.item())
        train_stats["ratio"].append(imp_weights.mean().item())

        return train_stats

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        """
        Perform a training update using minibatch GD.
        """

        # Calculate GAE returns
        returns = self.compute_returns(batch)

        # Calculate advantages based on the GAE returns
        advantages = self.compute_advantages(batch, returns)

        # Initialize stats
        train_stats = None

        # Prepare the networks for training
        self.prep_training()

        for _ in range(self.ppo_epoch):

            batch_size = batch["batch_size"] * (batch["max_seq_length"] - 1)
            mini_batch_size = batch_size // self.num_mini_batch
            rand = th.randperm(batch_size).numpy()

            sampler = [rand[i * mini_batch_size:(i + 1) * mini_batch_size] for i in range(self.num_mini_batch)]
            prepared_data = self.prepare_data(batch, returns, advantages)

            for indices in sampler:
                mini_batch = self.create_mini_batch(prepared_data, indices, mini_batch_size, 1, batch["device"])
                train_stats = self.ppo_update(mini_batch, train_stats)

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            ts_logged = len(train_stats["value_loss"])
            for key in train_stats.keys():
                self.logger.log_stat(key, sum(train_stats[key]) / ts_logged, t_env)

        # Prepare the networks for rollouts
        self.prep_rollout()

    def prep_training(self):
        self.mac.agent.train()

    def prep_rollout(self):
        self.mac.agent.eval()

    @staticmethod
    def get_gard_norm(it):
        sum_grad = 0
        for x in it:
            if x.grad is None:
                continue
            sum_grad += x.grad.norm() ** 2
        return math.sqrt(sum_grad)

    @staticmethod
    def huber_loss(e, d):
        a = (abs(e) <= d).float()
        b = (e > d).float()
        return a * e ** 2 / 2 + b * d * (abs(e) - d / 2)

    @staticmethod
    def mse_loss(e):
        return e ** 2 / 2

    def compute_returns(self, batch):
        bs = batch.batch_size
        max_t = batch.max_seq_length - 1
        returns = th.zeros((bs, max_t, self.num_agents, 1), dtype=th.float32, device=self.device)
        gae = 0

        rewards = batch["reward"][:, :-1, :, None].repeat(1, 1, self.num_agents, 1)
        mask = batch["filled"][:, :-1, :, None].float().repeat(1, 1, self.num_agents, 1)
        values = batch["values"]

        for step in reversed(range(rewards.shape[1])):
            if self.use_popart:
                next_value = self.value_normalizer.denormalize(values[:, step + 1])
                current_value = self.value_normalizer.denormalize(values[:, step])
            else:
                next_value = values[:, step + 1]
                current_value = values[:, step]

            delta = rewards[:, step] + self.gamma * next_value * mask[:, step] - current_value
            gae = delta + self.gamma * self.gae_lambda * mask[:, step] * gae
            returns[:, step] = gae + current_value

        return returns

    def compute_advantages(self, batch, returns):
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        mask = mask[..., None].repeat(1, 1, self.num_agents, 1)
        values = batch["values"]


        if self.use_popart:
            value_baseline = self.value_normalizer.denormalize(values[:, :-1])
        else:
            value_baseline = values[:, :-1]

        advantages = returns - value_baseline


        advantages_copy = advantages.clone().detach().cpu().numpy()
        advantages_copy[mask.detach().cpu().numpy() == 0.0] = np.nan
        mean_advantages = th.from_numpy(np.nanmean(advantages_copy, keepdims=True)).to(self.device)
        std_advantages = th.from_numpy(np.nanstd(advantages_copy, keepdims=True)).to(self.device)
        advantages = (advantages - mean_advantages) / (std_advantages + 1e-5)

        return advantages

    @staticmethod
    def prepare_data(batch, returns, advantages):
        max_ts = batch["max_seq_length"] - 1

        # [batch, timesteps, n_agents, dim] -->
        # [timesteps, batch, n_agents, dim] -->
        # [timesteps*batch, 1, n_agents, dim] (1 as a dummy timestep)
        obs = batch["obs"][:, :max_ts]
        obs = obs.transpose(1, 0).reshape(-1, 1, *obs.shape[2:])
        state = batch["state"][:, :max_ts]
        state = state.transpose(1, 0).reshape(-1, 1, *state.shape[2:])
        actions = batch["actions"][:, :max_ts]
        actions = actions.transpose(1, 0).reshape(-1, 1, *actions.shape[2:])
        actions_onehot = batch["actions_onehot"][:, :max_ts]
        actions_onehot = actions_onehot.transpose(1, 0).reshape(-1, 1, *actions_onehot.shape[2:])
        avail_actions = batch["avail_actions"][:, :max_ts]
        avail_actions = avail_actions.transpose(1, 0).reshape(-1, 1, *avail_actions.shape[2:])
        reward = batch["reward"][:, :max_ts]
        reward = reward.transpose(1, 0).reshape(-1, 1, *reward.shape[2:])
        terminated = batch["terminated"][:, :max_ts].float()
        terminated = terminated.transpose(1, 0).reshape(-1, 1, *terminated.shape[2:])
        filled = batch["filled"][:, :max_ts].float()
        filled = filled.transpose(1, 0).reshape(-1, 1, *filled.shape[2:])
        mask = batch["filled"][:, :max_ts].float()
        mask[:, 1:] = mask[:, 1:] * (1 - batch["terminated"][:, :max_ts].float()[:, :-1])
        mask = mask.transpose(1, 0).reshape(-1, 1, *mask.shape[2:])
        log_probs = batch["log_probs"][:, :max_ts]
        log_probs = log_probs.transpose(1, 0).reshape(-1, 1, *log_probs.shape[2:])
        values = batch["values"][:, :max_ts]
        values = values.transpose(1, 0).reshape(-1, 1, *values.shape[2:])
        returns = returns.transpose(1, 0).reshape(-1, 1, *returns.shape[2:])
        advantages = advantages.transpose(1, 0).reshape(-1, 1, *advantages.shape[2:])

        prepared_data = {
            "obs": obs,
            "state": state,
            "actions": actions,
            "actions_onehot": actions_onehot,
            "avail_actions": avail_actions,
            "reward": reward,
            "terminated": terminated,
            "filled": filled,
            "mask": mask,
            "log_probs": log_probs,
            "values": values,
            "returns": returns,
            "advantages": advantages,
        }
        return prepared_data

    @staticmethod
    def create_mini_batch(prepared_data, indices, mini_batch_size, max_seq_length, device):

        mini_batch = {
            "obs": prepared_data["obs"][indices],
            "state": prepared_data["state"][indices],
            "actions": prepared_data["actions"][indices],
            "actions_onehot": prepared_data["actions_onehot"][indices],
            "avail_actions": prepared_data["avail_actions"][indices],
            "reward": prepared_data["reward"][indices],
            "terminated": prepared_data["terminated"][indices],
            "filled": prepared_data['filled'][indices],
            "mask": prepared_data["mask"][indices],
            "log_probs": prepared_data["log_probs"][indices],
            "values": prepared_data["values"][indices],
            "returns": prepared_data["returns"][indices],
            "advantages": prepared_data["advantages"][indices],
            "max_seq_length": max_seq_length,
            "batch_size": mini_batch_size,
            "device": device,
        }
        return mini_batch

    def cuda(self):
        self.mac.agent.cuda()
