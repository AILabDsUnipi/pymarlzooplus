import torch as th
from torch.optim import RMSprop

from pymarlzooplus.components.episode_buffer import EpisodeBatch
from pymarlzooplus.components.standarize_stream import RunningMeanStd
from pymarlzooplus.modules.critics import REGISTRY as critic_registry


class MagicLearner:
    """Original-style MAGIC actor-critic loss adapted to EpisodeBatch data."""

    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.n_agents = args.n_agents
        self.n_actions = args.n_actions
        self.logger = logger

        self.mac = mac
        self.critic = critic_registry[args.critic_type](scheme, args)

        self.params = list(mac.parameters()) + list(self.critic.parameters())
        self.optimiser = RMSprop(
            params=self.params,
            lr=args.lr,
            alpha=args.magic_optim_alpha,
            eps=args.magic_optim_eps,
        )

        self.log_stats_t = -self.args.learner_log_interval - 1
        self.value_coeff = args.value_coeff
        self.entropy_coef = args.entropy_coef

        device = "cuda" if args.use_cuda else "cpu"
        if self.args.standardise_returns:
            self.ret_ms = RunningMeanStd(shape=(self.n_agents,), device=device)
        if self.args.standardise_rewards:
            self.rew_ms = RunningMeanStd(shape=(1,), device=device)

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        del episode_num

        # Get the relevant quantities
        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])

        if self.args.standardise_rewards:
            self.rew_ms.update(rewards)
            rewards = (rewards - self.rew_ms.mean) / th.sqrt(self.rew_ms.var)

        # No experiences to train on in this minibatch
        if mask.sum() == 0:
            self.logger.log_stat("Mask_Sum_Zero", 1, t_env)
            self.logger.console_logger.error("MagicLearner: mask.sum() == 0 at t_env {}".format(t_env))
            return

        self.mac.init_hidden(batch.batch_size)
        mac_out = []
        values = []
        for t in range(batch.max_seq_length - 1):
            agent_outs, critic_inputs = self.mac.forward(batch, t=t, return_critic_inputs=True)
            mac_out.append(agent_outs)
            values.append(self.critic(*critic_inputs))
        mac_out = th.stack(mac_out, dim=1)
        values = th.stack(values, dim=1).squeeze(3)

        returns = self._build_returns(rewards, terminated)
        if self.args.standardise_returns:
            self.ret_ms.update(returns)
            returns = (returns - self.ret_ms.mean) / th.sqrt(self.ret_ms.var)

        advantages = returns - values.detach()
        active_advantages = advantages[mask.expand_as(advantages) > 0]
        if active_advantages.numel() > 1:
            advantages = (advantages - active_advantages.mean()) / (active_advantages.std() + 1e-8)

        pi = mac_out
        agent_mask = mask.repeat(1, 1, self.n_agents)
        pi = pi.masked_fill(mask.unsqueeze(-1) == 0, 1.0)
        pi_taken = th.gather(pi, dim=3, index=actions).squeeze(3)
        log_pi_taken = th.log(pi_taken + 1e-10)
        entropy = -th.sum(pi * th.log(pi + 1e-10), dim=-1)

        # The original MAGIC implementation sums losses over agents and divides
        # gradients by environment steps, not by agent-time entries.
        normalizer = mask.sum().clamp_min(1.0)
        action_loss = -((advantages.detach() * log_pi_taken) * agent_mask).sum() / normalizer
        value_loss = (((values - returns.detach()) ** 2) * agent_mask).sum() / normalizer
        entropy_loss = (entropy * agent_mask).sum() / normalizer
        loss = action_loss + self.value_coeff * value_loss - self.entropy_coef * entropy_loss

        self.optimiser.zero_grad()
        loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.params, self.args.grad_norm_clip)
        self.optimiser.step()

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            active = agent_mask.sum().clamp_min(1.0)
            self.logger.log_stat("loss", loss.item(), t_env)
            self.logger.log_stat("action_loss", action_loss.item(), t_env)
            self.logger.log_stat("value_loss", value_loss.item(), t_env)
            self.logger.log_stat("entropy", entropy_loss.item(), t_env)
            self.logger.log_stat("agent_grad_norm", grad_norm.item(), t_env)
            self.logger.log_stat("advantage_mean", (advantages * agent_mask).sum().item() / active.item(), t_env)
            self.logger.log_stat("value_mean", (values * agent_mask).sum().item() / active.item(), t_env)
            self.logger.log_stat("magic_target_return_mean", (returns * agent_mask).sum().item() / active.item(), t_env)
            self.logger.log_stat("pi_max", (pi.max(dim=-1)[0] * agent_mask).sum().item() / active.item(), t_env)
            self.log_stats_t = t_env

    def _build_returns(self, rewards, terminated):
        batch_size, episode_len, _ = rewards.shape
        rewards = rewards.expand(batch_size, episode_len, self.n_agents)
        not_done = (1 - terminated).expand(batch_size, episode_len, self.n_agents)

        returns = th.zeros_like(rewards)
        prev_return = th.zeros(batch_size, self.n_agents, device=rewards.device)

        for t in reversed(range(episode_len)):
            returns[:, t] = rewards[:, t] + self.args.gamma * prev_return * not_done[:, t]
            prev_return = returns[:, t].clone()
        return returns

    def cuda(self):
        self.mac.cuda()
        self.critic.cuda()

    def save_models(self, path):
        self.mac.save_models(path)
        th.save(self.critic.state_dict(), "{}/critic.th".format(path))
        th.save(self.optimiser.state_dict(), "{}/opt.th".format(path))

    def load_models(self, path):
        self.mac.load_models(path)
        self.critic.load_state_dict(th.load("{}/critic.th".format(path), map_location=lambda storage, loc: storage))
        self.optimiser.load_state_dict(th.load("{}/opt.th".format(path), map_location=lambda storage, loc: storage))
