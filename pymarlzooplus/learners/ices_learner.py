import copy
from pymarlzooplus.components.episode_buffer import EpisodeBatch

# Mixers
from pymarlzooplus.modules.mixers.vdn import VDNMixer
from pymarlzooplus.modules.mixers.qmix import QMixer
from pymarlzooplus.modules.mixers.qatten import QattenMixer

# Utils
from pymarlzooplus.utils.rl_utils import build_td_lambda_targets, build_q_lambda_targets
from pymarlzooplus.utils.torch_utils import get_parameters_num

# Torch
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

# Misc
import numpy as np
import math
import torch
import pyro
from pyro.infer import SVI, Trace_ELBO

# Modules
from pymarlzooplus.modules.explorers.ICES import StatePredBL, StatePredCVAE

# Critics
from pymarlzooplus.modules.critics.ices_critic import ICESCritic


class ICESLearner:
    def __init__(self, mac, scheme, logger, args, env_info):
        self.args = args
        self.mac = mac
        self.logger = logger

        self.params = list(mac.parameters())

        self.last_target_update_episode = 0
        self.pred_s_len = args.pred_s_len
        self.n_agents = args.n_agents
        self.step_penalty = args.step_penalty

        self.device = th.device("cuda" if args.use_cuda else "cpu")

        self.mixer = QMixer(args)
        self.target_mixer = copy.deepcopy(self.mixer)
        self.params += list(self.mixer.parameters())
        self.int_params = list(mac.int_parameters())

        self.int_critic = ICESCritic(
            mac._get_input_shape(scheme) + int(np.prod(self.args.state_shape)), args
        )

        print("Mixer Size: ")
        print(get_parameters_num(self.mixer.parameters()))

        self.world_bl_global = StatePredBL(args)
        self.world_bl_local = StatePredBL(args)
        self.world_model = StatePredCVAE(
            args,
            baseline_net_global=self.world_bl_global,
            baseline_net_local=self.world_bl_local,
        )

        self.optimiser = Adam(
            params=self.params,
            lr=args.lr,
            weight_decay=getattr(args, "weight_decay", 0),
        )
        self.int_optimiser = Adam(
            params=self.int_params,
            lr=args.int_lr,
            weight_decay=getattr(args, "weight_decay", 0),
        )
        self.int_c_optimiser = Adam(
            params=self.int_critic.parameters(),
            lr=args.int_c_lr,
            weight_decay=getattr(args, "weight_decay", 0),
        )

        self.world_bl_optimizer_global = th.optim.Adam(
            self.world_model.baseline_net_global.parameters(),
            lr=args.world_bl_lr,
            eps=args.optim_eps,
            weight_decay=args.weight_decay,
        )
        self.world_bl_optimizer_local = th.optim.Adam(
            self.world_model.baseline_net_local.parameters(),
            lr=args.world_bl_lr,
            eps=args.optim_eps,
            weight_decay=args.weight_decay,
        )
        num_steps = args.t_max // env_info["episode_limit"] * self.pred_s_len
        lrd = args.world_gamma ** (1 / num_steps)

        self.world_optimizer_global = pyro.optim.ClippedAdam(
            {
                "lr": args.world_lr,
                "betas": (0.95, 0.999),
                "lrd": lrd,
                "clip_norm": args.world_clip_param,
            }
        )
        self.world_optimizer_local = pyro.optim.ClippedAdam(
            {
                "lr": args.world_lr,
                "betas": (0.95, 0.999),
                "lrd": lrd,
                "clip_norm": args.world_clip_param,
            }
        )

        self.world_svi_global = SVI(
            self.world_model.model_global,
            self.world_model.guide_global,
            self.world_optimizer_global,
            loss=Trace_ELBO(),
        )
        self.world_svi_local = SVI(
            self.world_model.model_local,
            self.world_model.guide_local,
            self.world_optimizer_local,
            loss=Trace_ELBO(),
        )

        # a little wasteful to deepcopy (e.g. duplicates action selector), but should work for any MAC
        self.target_mac = copy.deepcopy(mac)

        self.log_stats_t = -self.args.learner_log_interval - 1
        self.world_log_stats_t = -self.args.learner_log_interval - 1

        self.train_t = 0
        self.world_flag = True

        # priority replay
        self.use_per = getattr(self.args, "use_per", False)
        self.return_priority = getattr(self.args, "return_priority", False)
        if self.use_per:
            self.priority_max = float("-inf")
            self.priority_min = float("inf")

    def train_world(self, batch: EpisodeBatch, t_env: int):
        # update two model iteratively
        n_mini = self.pred_s_len
        idxes = np.arange(1, n_mini + 1)
        # ranom order of [1,2,...,n_mini]
        np.random.shuffle(idxes)

        # (bs, seq_len, s_dim)
        s = batch["state"][:, :-1, :]
        # (bs, seq_len, n_agents)
        a = batch["actions"][:, :-1, :, 0].long()
        n_agents = a.shape[-1]

        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])

        if self.world_flag:
            (
                pred_bl_loss_global_log,
                world_bl_grad_norm_global_log,
                pred_loss_global_log,
                world_grad_norm_log,
            ) = (0, 0, 0, 0)

            for idx in idxes:
                s_prime = batch["state"][:, idx:, :]
                bs, seq_len_cut, _ = s_prime.shape
                s_ori = s[:, :seq_len_cut, :]
                s_diff = (s_prime - s_ori).reshape(bs * seq_len_cut, -1)
                s_ori = s_ori.reshape(bs * seq_len_cut, -1)
                a_all = a[:, :seq_len_cut, :].reshape(bs * seq_len_cut, -1)
                mask_sample = mask[:, :seq_len_cut, :].reshape(bs * seq_len_cut, -1)

                s_diff_pred = self.world_model.baseline_net_global(s_ori, a_all)
                pred_bl_loss_global = F.mse_loss(
                    s_diff.detach() * mask_sample, s_diff_pred * mask_sample
                )
                self.world_bl_optimizer_global.zero_grad()
                pred_bl_loss_global.backward()
                world_bl_grad_norm_global = nn.utils.clip_grad_norm_(
                    self.world_model.baseline_net_global.parameters(),
                    self.args.grad_norm_clip,
                )
                self.world_bl_optimizer_global.step()

                # learn prediction network
                world_grad_norm = self.get_gard_norm(self.world_model.parameters())
                pred_loss_global = self.world_svi_global.step(
                    s_ori * mask_sample,
                    (a_all * mask_sample).long(),
                    s_prime=s_diff.detach() * mask_sample,
                )

                pred_bl_loss_global_log += pred_bl_loss_global.item()
                world_bl_grad_norm_global_log += world_bl_grad_norm_global.item()
                pred_loss_global_log += pred_loss_global
                world_grad_norm_log += world_grad_norm

            if t_env - self.world_log_stats_t >= self.args.learner_log_interval:
                pred_bl_loss_global_log /= n_mini
                world_bl_grad_norm_global_log /= n_mini
                pred_loss_global_log /= n_mini
                world_grad_norm_log /= n_mini
                self.logger.log_stat(
                    "pred_bl_loss_global", pred_bl_loss_global_log, t_env
                )
                self.logger.log_stat(
                    "world_bl_grad_norm_global", world_bl_grad_norm_global_log, t_env
                )
                self.logger.log_stat("pred_loss_global", pred_loss_global_log, t_env)
                self.logger.log_stat("world_grad_norm", world_grad_norm_log, t_env)
                self.world_log_stats_t = t_env
        else:
            (
                pred_bl_loss_local_log,
                world_bl_grad_norm_local_log,
                pred_loss_local_log,
                world_grad_norm_log,
            ) = (0, 0, 0, 0)

            for idx in idxes:
                s_prime = batch["state"][:, idx:, :]
                bs, seq_len_cut, _ = s_prime.shape
                s_ori = s[:, :seq_len_cut, :]
                s_diff = (s_prime - s_ori).reshape(bs * seq_len_cut, -1)
                s_ori = s_ori.reshape(bs * seq_len_cut, -1)
                a_all = a[:, :seq_len_cut, :].reshape(bs * seq_len_cut, -1)
                mask_sample = mask[:, :seq_len_cut, :].reshape(bs * seq_len_cut, -1)
                a_masked = a_all.clone()
                a_mask = th.ones_like(a_masked)
                one_pos = th.randint(n_agents, (bs * seq_len_cut, 1)).to(
                    self.args.device
                )
                a_mask.scatter_(1, index=one_pos, src=th.zeros_like(a_mask))
                a_masked = ((a_masked + 1) * a_mask - 1).long()

                s_diff_pred = self.world_model.baseline_net_local(s_ori, a_masked)
                pred_bl_loss_local = F.mse_loss(
                    s_diff.detach() * mask_sample, s_diff_pred * mask_sample
                )
                self.world_bl_optimizer_local.zero_grad()
                pred_bl_loss_local.backward()
                world_bl_grad_norm_local = nn.utils.clip_grad_norm_(
                    self.world_model.baseline_net_local.parameters(),
                    self.args.grad_norm_clip,
                )
                self.world_bl_optimizer_local.step()

                # learn prediction network
                world_grad_norm = self.get_gard_norm(self.world_model.parameters())
                pred_loss_local = self.world_svi_local.step(
                    s_ori * mask_sample,
                    (a_masked * mask_sample).long(),
                    s_prime=s_diff.detach() * mask_sample,
                )

                pred_bl_loss_local_log += pred_bl_loss_local.item()
                world_bl_grad_norm_local_log += world_bl_grad_norm_local.item()
                pred_loss_local_log += pred_loss_local
                world_grad_norm_log += world_grad_norm

            if t_env - self.world_log_stats_t >= self.args.learner_log_interval:
                pred_bl_loss_local_log /= n_mini
                world_bl_grad_norm_local_log /= n_mini
                pred_loss_local_log /= n_mini
                world_grad_norm_log /= n_mini
                self.logger.log_stat(
                    "pred_bl_loss_local", pred_bl_loss_local_log, t_env
                )
                self.logger.log_stat(
                    "world_bl_grad_norm_local", world_bl_grad_norm_local_log, t_env
                )
                self.logger.log_stat("pred_loss_local", pred_loss_local_log, t_env)
                self.logger.log_stat("world_grad_norm", world_grad_norm_log, t_env)
                self.world_log_stats_t = t_env

        self.world_flag = not self.world_flag

    def train(
        self,
        batch: EpisodeBatch,
        t_env: int,
        episode_num: int,
        per_weight=None,
    ):
        # Get the relevant quantities
        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"]

        # Calculate estimated Q-Values
        self.mac.agent.train()
        mac_out = []
        self.mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            agent_outs = self.mac.forward(batch, t=t)
            mac_out.append(agent_outs)
        mac_out = th.stack(mac_out, dim=1)  # Concat over time

        # Pick the Q-Values for the actions taken by each agent
        chosen_action_qvals = th.gather(mac_out[:, :-1], dim=3, index=actions).squeeze(
            3
        )  # Remove the last dim
        chosen_action_qvals_ = chosen_action_qvals

        # Calculate the Q-Values necessary for the target
        with th.no_grad():
            self.target_mac.agent.train()
            target_mac_out = []
            self.target_mac.init_hidden(batch.batch_size)
            for t in range(batch.max_seq_length):
                target_agent_outs = self.target_mac.forward(batch, t=t)
                target_mac_out.append(target_agent_outs)

            # We don't need the first timesteps Q-Value estimate for calculating targets
            target_mac_out = th.stack(target_mac_out, dim=1)  # Concat across time

            # Max over target Q-Values/ Double q learning
            mac_out_detach = mac_out.clone().detach()
            mac_out_detach[avail_actions == 0] = -9999999
            cur_max_actions = mac_out_detach.max(dim=3, keepdim=True)[1]
            target_max_qvals = th.gather(target_mac_out, 3, cur_max_actions).squeeze(3)

            # Calculate n-step Q-Learning targets
            target_max_qvals = self.target_mixer(target_max_qvals, batch["state"])

            rewards = rewards + self.step_penalty

            if getattr(self.args, "q_lambda", False):
                qvals = th.gather(target_mac_out, 3, batch["actions"]).squeeze(3)
                qvals = self.target_mixer(qvals, batch["state"])

                targets = build_q_lambda_targets(
                    rewards,
                    terminated,
                    mask,
                    target_max_qvals,
                    qvals,
                    self.args.gamma,
                    self.args.td_lambda,
                )
            else:
                targets = build_td_lambda_targets(
                    rewards,
                    terminated,
                    mask,
                    target_max_qvals,
                    self.args.n_agents,
                    self.args.gamma,
                    self.args.td_lambda,
                )

        # Mixer
        chosen_action_qvals = self.mixer(chosen_action_qvals, batch["state"][:, :-1])

        td_error = chosen_action_qvals - targets.detach()
        td_error2 = 0.5 * td_error.pow(2)

        mask = mask.expand_as(td_error2)
        masked_td_error = td_error2 * mask

        # important sampling for PER
        if self.use_per:
            per_weight = th.from_numpy(per_weight).unsqueeze(-1).to(device=self.device)
            masked_td_error = masked_td_error.sum(1) * per_weight

        loss = masked_td_error.sum() / mask.sum()

        # Optimise
        self.optimiser.zero_grad()
        loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.params, self.args.grad_norm_clip)
        self.optimiser.step()

        ########## intrinsic
        # Calculate estimated Q-Values
        self.mac.int_agent.train()
        int_c_hidden = (
            self.int_critic.init_hidden()
            .unsqueeze(0)
            .expand(batch.batch_size, self.n_agents, -1)
        )
        int_mac_out = []
        int_v = []
        for t in range(batch.max_seq_length):
            int_agent_outs = self.mac.int_forward(batch, t=t)
            input_here = self.mac._build_int_inputs(batch, t=t)
            int_agent_v, int_c_hidden = self.int_critic(input_here, int_c_hidden)
            int_mac_out.append(int_agent_outs)
            int_v.append(int_agent_v)
        int_mac_out = th.stack(int_mac_out, dim=1)  # Concat over time
        int_v = th.stack(int_v, dim=1)
        int_v = int_v[:, :-1, :, 0]

        # Pick the Q-Values for the actions taken by each agent
        int_mac_out_pi = F.log_softmax(int_mac_out, dim=-1)
        int_entropy = -th.sum(int_mac_out_pi * F.softmax(int_mac_out, dim=-1), dim=-1)[
            :, :-1
        ]
        int_mac_out_pi = th.gather(
            int_mac_out_pi[:, :-1], dim=3, index=actions
        ).squeeze(
            3
        )  # Remove the last dim

        bs, seq_len, n_agents, _ = actions.shape
        # (bs, seq_len, s_dim) -> (bs * seq_len, s_dim)
        s = batch["state"][:, :-1, :].reshape(bs * seq_len, -1)
        # (bs, seq_len, n_agents, 1) -> (bs * seq_len, n_agents)
        a_all = actions.reshape(bs * seq_len, n_agents).long()
        s_expand = (
            batch["state"][:, :-1, None, :]
            .expand(-1, -1, n_agents, -1)
            .reshape(bs * seq_len * n_agents, -1)
        )
        # (bs, seq_len, n_agents, n_agents)
        a_masked = actions[:, :, None, :, 0].expand(-1, -1, n_agents, -1)
        a_masked = (a_masked + 1) * (1.0 - th.eye(n_agents).to(self.args.device))[
            None, None, :, :
        ]
        a_masked = (a_masked - 1).long().reshape(bs * seq_len * n_agents, -1)

        with th.no_grad():
            global_mu, global_sigma, _, _ = self.world_model.model_global(s, a_all)
            local_mu, local_sigma, _, _ = self.world_model.model_local(
                s_expand, a_masked
            )
            global_mu = global_mu.reshape(bs * seq_len, 1, -1)
            global_sigma = global_sigma.reshape(bs * seq_len, 1, -1)
            local_mu = local_mu.reshape(bs * seq_len, n_agents, -1)
            local_sigma = local_sigma.reshape(bs * seq_len, n_agents, -1)
            div = self.KL_div(global_mu, global_sigma, local_mu, local_sigma)
            # normalize in an ugly way
            div = 1.0 - th.exp(-div)

            int_rewards = div.reshape(
                bs,
                seq_len,
                n_agents,
            )

            adv = int_rewards - int_v

        # Td-error
        mask = mask.expand_as(adv)
        masked_adv = adv * mask
        mask_entropy = int_entropy * mask
        int_loss = (
            -1.0
            * (
                (masked_adv * int_mac_out_pi).sum()
                + self.args.int_ent_coef * mask_entropy.sum()
            )
            / mask.sum()
        )

        # optimise

        self.int_optimiser.zero_grad()
        int_loss.backward()
        int_grad_norm = th.nn.utils.clip_grad_norm_(
            self.int_params, self.args.grad_norm_clip
        )
        self.int_optimiser.step()
        int_error = (int_v - int_rewards) * mask
        loss_int_c = (int_error**2).sum() / mask.sum()
        self.int_c_optimiser.zero_grad()
        loss_int_c.backward()
        int_c_grad_norm = th.nn.utils.clip_grad_norm_(
            self.int_critic.parameters(), self.args.grad_norm_clip
        )
        self.int_c_optimiser.step()

        if (
            episode_num - self.last_target_update_episode
        ) / self.args.target_update_interval >= 1.0:
            self._update_targets()
            self.last_target_update_episode = episode_num

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat("loss_td", loss.item(), t_env)
            self.logger.log_stat("loss", loss.item(), t_env)
            self.logger.log_stat("int_loss", int_loss.item(), t_env)
            self.logger.log_stat("loss_int_c", loss_int_c.item(), t_env)
            self.logger.log_stat("grad_norm", grad_norm, t_env)
            self.logger.log_stat("int_grad_norm", int_grad_norm, t_env)
            self.logger.log_stat("int_c_grad_norm", int_c_grad_norm, t_env)
            mask_elems = mask.sum().item()
            self.logger.log_stat(
                "td_error_abs", (masked_td_error.abs().sum().item() / mask_elems), t_env
            )
            self.logger.log_stat(
                "q_taken_mean",
                (chosen_action_qvals * mask).sum().item()
                / (mask_elems * self.args.n_agents),
                t_env,
            )
            self.logger.log_stat(
                "target_mean",
                (targets * mask).sum().item() / (mask_elems * self.args.n_agents),
                t_env,
            )
            self.logger.log_stat(
                "intrinsic_reward_mean", int_rewards.mean().item(), t_env
            )
            self.logger.log_stat(
                "intrinsic_reward_std", int_rewards.std().item(), t_env
            )
            self.log_stats_t = t_env

        # return info
        info = {}
        # calculate priority
        if self.use_per:
            if self.return_priority:
                info["td_errors_abs"] = rewards.sum(1).detach().to("cpu")
                # normalize to [0, 1]
                self.priority_max = max(
                    th.max(info["td_errors_abs"]).item(), self.priority_max
                )
                self.priority_min = min(
                    th.min(info["td_errors_abs"]).item(), self.priority_min
                )
                info["td_errors_abs"] = (info["td_errors_abs"] - self.priority_min) / (
                    self.priority_max - self.priority_min + 1e-5
                )
            else:
                info["td_errors_abs"] = (
                    ((td_error.abs() * mask).sum(1) / th.sqrt(mask.sum(1)))
                    .detach()
                    .to("cpu")
                )
        return info

    def _update_targets(self):
        self.target_mac.load_state(self.mac)
        if self.mixer is not None:
            self.target_mixer.load_state_dict(self.mixer.state_dict())
        self.logger.console_logger.info("Updated target network")

    def cuda(self):
        self.mac.cuda()
        self.target_mac.cuda()
        self.world_bl_global.cuda()
        self.world_bl_local.cuda()
        self.world_model.cuda()
        self.int_critic.cuda()
        if self.mixer is not None:
            self.mixer.cuda()
            self.target_mixer.cuda()

    def save_models(self, path):
        self.mac.save_models(path)
        if self.mixer is not None:
            th.save(self.mixer.state_dict(), "{}/mixer.th".format(path))
        th.save(self.optimiser.state_dict(), "{}/opt.th".format(path))

    def load_models(self, path):
        self.mac.load_models(path)
        # Not quite right but I don't want to save target networks
        self.target_mac.load_models(path)
        if self.mixer is not None:
            self.mixer.load_state_dict(
                th.load(
                    "{}/mixer.th".format(path),
                    map_location=lambda storage, loc: storage,
                )
            )
        self.optimiser.load_state_dict(
            th.load("{}/opt.th".format(path), map_location=lambda storage, loc: storage)
        )
    
    def KL_div(self, p_mu, p_sigma, q_mu, q_sigma):
        """_summary_
        Args:
            p_mu (bs, dist_dim): _description_
            p_sigma (bs, dist_dim): _description_
            q_mu (bs, dist_dim): _description_
            q_sigma (bs, dist_dim): _description_
        """

        div = (
            torch.log2(q_sigma)
            - torch.log2(p_sigma)
            + (p_sigma**2 + (p_mu - q_mu) ** 2) / (2 * q_sigma**2)
            - 0.5
        )
        div = div.mean(dim=-1)
        return div

    def get_gard_norm(self, it):
        sum_grad = 0
        for x in it:
            if x.grad is None:
                continue
            sum_grad += x.grad.norm() ** 2
        return math.sqrt(sum_grad)


