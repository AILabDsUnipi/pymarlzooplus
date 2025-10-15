import torch as th
import torch.nn as nn

from pymarlzooplus.components.episode_buffer import EpisodeBatch
from pymarlzooplus.learners.mat_learner import MATLearner


class CommFormerLearner(MATLearner):
    def __init__(self, mac, scheme, logger, args):

        super().__init__(mac, scheme, logger, args)

        self.edge_lr = float(getattr(args, "edge_lr", 1e-4))
        self.edge_params = self.edge_params = list(self.critic.edge_parameters())
        self.edge_optimizer = th.optim.Adam(self.edge_params, lr=self.edge_lr)

        # Hyperparameters
        self.use_bilevel = getattr(args, "use_bilevel", True)
        self.post_stable = args.post_stable
        self.post_ratio = args.post_ratio

    def ppo_update(self, sample, train_stats, steps=0, index=0, total_step=0):
        if train_stats is None:
            train_stats = {
                "value_loss": [],
                "grad_norm": [],
                "policy_loss": [],
                "entropy": [],
                "ratio": [],
            }

        value_preds_batch = sample["values"].view(-1, 1)
        return_batch = sample["returns"].view(-1, 1)
        old_action_log_probs_batch = sample["log_probs"].view(-1, 1)
        adv_targ = sample["advantages"].view(-1, 1)

        values, action_log_probs, dist_entropy = self.mac.evaluate_actions(
            sample, t=0, steps=steps, total_step=total_step
        )

        imp_weights = th.exp(action_log_probs - old_action_log_probs_batch)
        surr1 = imp_weights * adv_targ
        surr2 = th.clamp(imp_weights, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv_targ
        policy_loss = -th.sum(th.min(surr1, surr2), dim=-1, keepdim=True).mean()

        value_loss = self.cal_value_loss(values, value_preds_batch, return_batch)

        loss = policy_loss - dist_entropy * self.entropy_coef + value_loss * self.value_loss_coef

        if self.use_bilevel:
            if (index + 1) % 5 == 0 and (
                    (self.post_stable and steps <= int(self.post_ratio * total_step)) or not self.post_stable):
                self.edge_optimizer.zero_grad()
            else:
                self.optimizer.zero_grad()
        else:
            self.optimizer.zero_grad()
            if (index + 1) % 5 == 0:
                self.edge_optimizer.zero_grad()
        loss.backward()

        if self.use_max_grad_norm:
            grad_norm = nn.utils.clip_grad_norm_(self.actor_critic_params, self.max_grad_norm)
        else:
            grad_norm = self.get_grad_norm(self.actor_critic_params)

        if self.use_bilevel:
            if (index + 1) % 5 == 0 and (
                    (self.post_stable and steps <= int(self.post_ratio * total_step)) or not self.post_stable):
                self.edge_optimizer.step()
            else:
                self.optimizer.step()
        else:
            self.optimizer.step()

        train_stats["value_loss"].append(value_loss.item())
        train_stats["grad_norm"].append(grad_norm.item())
        train_stats["policy_loss"].append(policy_loss.item())
        train_stats["entropy"].append(dist_entropy.item())
        train_stats["ratio"].append(imp_weights.mean().item())

        return train_stats

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        returns = self.compute_returns(batch)
        advantages = self.compute_advantages(batch, returns)

        train_stats = None
        self.prep_training()

        step_idx = 0
        for _ in range(self.ppo_epoch):
            batch_size = batch["batch_size"] * (batch["max_seq_length"] - 1)
            mini_batch_size = batch_size // self.num_mini_batch
            rand = th.randperm(batch_size).numpy()
            sampler = [rand[i * mini_batch_size:(i + 1) * mini_batch_size] for i in range(self.num_mini_batch)]
            prepared_data = self.prepare_data(batch, returns, advantages)
            for indices in sampler:
                mini_batch = self.create_mini_batch(prepared_data, indices, mini_batch_size, 1, batch["device"])
                train_stats = self.ppo_update(
                    mini_batch, train_stats, steps=t_env, index=step_idx, total_step=self.args.t_max
                )
                step_idx += 1

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            ts_logged = len(train_stats["value_loss"])
            for key in train_stats.keys():
                self.logger.log_stat(key, sum(train_stats[key]) / ts_logged, t_env)

        self.prep_rollout()
