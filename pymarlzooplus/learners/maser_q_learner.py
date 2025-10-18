# Code based on the original implementation available at: https://github.com/Jiwonjeon9603/MASER

import copy

import torch as th
from torch.optim import RMSprop
import torch.nn.functional as F
import torch.nn as nn

from pymarlzooplus.components.episode_buffer import EpisodeBatch
from pymarlzooplus.modules.mixers.vdn import VDNMixer
from pymarlzooplus.modules.mixers.qmix import QMixer

class MASERQLearner:
    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.mac = mac

        assert self.mac.is_image is False, "MASER does not support image obs for the time being!"

        self.logger = logger
        self.n_agents = args.n_agents
        self.device = args.device
        self.params = list(mac.parameters())

        # MASER Hyperparameters
        self.last_target_update_episode = 0
        self.lam = args.lam
        self.alpha = args.alpha
        self.ind = args.ind
        self.mix = args.mix
        self.expl = args.expl
        self.dis = args.dis
        self.distance_embed_dim = args.distance_embed_dim

        assert args.mixer is not None
        if args.mixer == "vdn":
            self.mixer = VDNMixer()
        elif args.mixer == "qmix":
            self.mixer = QMixer(args)
        else:
            raise ValueError("Mixer {} not recognised.".format(args.mixer))
        self.params += list(self.mixer.parameters())
        self.target_mixer = copy.deepcopy(self.mixer)

        self.optimiser = RMSprop(params=self.params, lr=args.lr, alpha=args.optim_alpha, eps=args.optim_eps)

        # a little wasteful to deepcopy (e.g., duplicates action selector), but should work for any MAC
        self.target_mac = copy.deepcopy(mac)

        self.log_stats_t = -self.args.learner_log_interval - 1

        self.distance = nn.Sequential(
            nn.Linear(self.mac.scheme['obs']['vshape'], self.distance_embed_dim),
            nn.ReLU(),
            nn.Linear(self.distance_embed_dim, args.n_actions)
        ).to(device=self.device)

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        
        # Get the relevant quantities
        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :-1]
        observation = batch["obs"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"]

        # Calculate estimated Q-Values
        mac_out = []
        self.mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            agent_outs = self.mac.forward(batch, t=t)  # shape: (bs, n, n_actions)
            mac_out.append(agent_outs)  # shape: [t, (bs, n, n_actions)]
        mac_out = th.stack(mac_out, dim=1)  # Concat over time

        # Pick the Q-Values for the actions taken by each agent
        chosen_action_qvals = th.gather(mac_out[:, :-1], dim=3, index=actions).squeeze(3)  # Remove the last dim
        ind_qvals = th.gather(mac_out[:, :-1], dim=3, index=actions).squeeze(3)  # shape: (bs, t, n), Q value of an action

        # Calculate the Q-Values necessary for the target
        target_mac_out = []
        self.target_mac.init_hidden(batch.batch_size)  # shape: (bs, n, hidden_size)
        for t in range(batch.max_seq_length):
            target_agent_outs = self.target_mac.forward(batch, t=t)  # shape: (bs, n, n_actions)
            target_mac_out.append(target_agent_outs)  # shape: [t, (bs, n, n_actions)]

        # We don't need the first timesteps Q-Value estimate for calculating targets
        target_ind_q = th.stack(target_mac_out[:-1], dim=1)   # Q-values
        target_mac_out = th.stack(target_mac_out[1:], dim=1)  # Concat across time, dim=1 is time index, shape: (bs, t, n, n_actions)

        # Mask out unavailable actions
        target_mac_out[avail_actions[:, 1:] == 0] = -9999999  # Q values
        target_ind_q[avail_actions[:, :-1] == 0] = -9999999  # Q values

        ## Max over target Q-Values
        # Get actions that maximise live Q (for double q-learning)
        mac_out_detach = mac_out.clone().detach()  # return a new Tensor, detached from the current graph
        mac_out_detach[avail_actions == 0] = -9999999  # discard t=0, shape: (bs, t, n, n_actions)
        cur_max_actions = mac_out_detach[:, 1:].max(dim=3, keepdim=True)[1]  # indices instead of values
        cur_max_act = mac_out_detach[:, :-1].max(dim=3, keepdim=True)[1]  # indices instead of values, shape: (bs, t, n, 1)
        target_max_qvals = th.gather(target_mac_out, 3, cur_max_actions).squeeze(3)
        target_individual_qvals = th.gather(target_mac_out, 3, cur_max_actions).squeeze(3)
        target_ind_qvals = th.gather(target_ind_q, 3, cur_max_act).squeeze(3)  # max target-Q, shape: (bs, t, n, n_actions) ==> (bs, t, n, 1) ==> (bs, t, n)

        # Mix
        chosen_action_qvals = self.mixer(chosen_action_qvals, batch["state"][:, :-1])
        target_max_qvals = self.target_mixer(target_max_qvals, batch["state"][:, 1:])
        goal_target_max_qvals = self.target_mixer(target_ind_qvals, batch["state"][:, :-1])  # shape: (bs, t, 1)

        ###############################################################################
        # ################ Compute intrinsic reward and MASER losses ##################
        q_ind_tot_list = []
        for i in range(self.n_agents):
            target_qtot_per_agent = (goal_target_max_qvals / self.n_agents).squeeze()
            q_ind_tot_list.append(self.alpha * target_ind_qvals[:, :, i] + (1 - self.alpha) * target_qtot_per_agent)

        q_ind_tot = th.stack(q_ind_tot_list, dim=2)

        ddqn_qval_up_idx = th.max(q_ind_tot, dim=1)[1]  # Find out max Q value for t=1~T-1 (whole episode)

        explore_q_target = th.ones(target_ind_q.shape) / target_ind_q.shape[-1]
        explore_q_target = explore_q_target.to(device=self.device)

        ddqn_up_list = []
        distance_list = []
        explore_list = []
        for i in range(batch.batch_size):
            ddqn_up_list_subset = []
            distance_subset = []
            explore_loss_subset = []
            for j in range(self.n_agents):

                # For distillation (exploration)
                y = F.softmax(target_ind_q[i, ddqn_qval_up_idx[i][j]:, j, :], dim=-1) + 1e-6
                z = F.softmax(explore_q_target[i, ddqn_qval_up_idx[i][j]:, j, :], dim=-1)
                loss1 = y * ((y / z).log())
                explore_loss = th.sum(loss1, dim=-1)
                explore_loss_subset.append(th.mean(explore_loss))

                # For distance function
                cos = nn.CosineSimilarity(dim=-1, eps=1e-8)
                goal_q = target_ind_q[i, ddqn_qval_up_idx[i][j], j, :].repeat(target_ind_q.shape[1], 1)

                similarity = 1 - cos(target_ind_q[i, :, j, :], goal_q)
                dist_obs = self.distance(observation[i, :, j, :])
                dist_og = self.distance(observation[i, ddqn_qval_up_idx[i][j], j, :])

                dist_loss = th.norm(dist_obs - dist_og.repeat(dist_obs.shape[0], 1), dim=-1) - similarity
                distance_loss = th.mean(dist_loss ** 2)
                distance_subset.append(distance_loss)
                ddqn_up_list_subset.append(observation[i, ddqn_qval_up_idx[i][j], j, :])

            explore_loss1 = th.stack(explore_loss_subset)
            explore_list.append(explore_loss1)

            distance1 = th.stack(distance_subset)
            distance_list.append(distance1)

            ddqn_up1 = th.stack(ddqn_up_list_subset)
            ddqn_up_list.append(ddqn_up1)

        explore_losses = th.stack(explore_list)
        distance_losses = th.stack(distance_list)

        mix_explore_distance_losses = self.expl*explore_losses + self.dis*distance_losses

        ddqn_up = th.stack(ddqn_up_list)
        ddqn_up = ddqn_up.unsqueeze(dim=1)
        ddqn_up = ddqn_up.repeat(1, observation.shape[1], 1, 1)

        reward_ddqn_up = self.distance(observation) - self.distance(ddqn_up)

        intrinsic_reward_list = []
        for i in range(self.n_agents):
            intrinsic_reward_list.append(
                -th.norm(reward_ddqn_up[:, :, i, :], dim=2).reshape(batch.batch_size, observation.shape[1]))
        intrinsic_rewards_ind = th.stack(intrinsic_reward_list, dim=-1)

        intrinsic_rewards = th.zeros(rewards.shape).to(device=self.device)

        for i in range(self.n_agents):
            intrinsic_rewards += -th.norm(reward_ddqn_up[:, :, i, :], dim=2).reshape(
                batch.batch_size,
                observation.shape[1],
                1
            ) / self.n_agents
        rewards += self.lam * intrinsic_rewards
        ###############################################################################

        # Calculate 1-step Q-Learning targets
        targets = rewards + self.args.gamma * (1 - terminated) * target_max_qvals

        # Td-error
        td_error = (chosen_action_qvals - targets.detach())  # no gradient through target net, shape: (bs, t, 1)

        # distillation-error
        mask = mask.expand_as(td_error)

        # 0-out the targets that came from padded data
        masked_td_error = td_error * mask

        # Normal L2 loss, take mean over actual data
        loss = (masked_td_error ** 2).sum() / mask.sum()

        # MASER final loss
        y = F.softmax(target_individual_qvals, dim=-1)
        individual_targets = (
                y*rewards + self.lam*intrinsic_rewards_ind +
                self.args.gamma * (1 - terminated.repeat(1, 1, target_individual_qvals.shape[-1])) *
                target_individual_qvals
        )
        td_individual_error = (ind_qvals - individual_targets.detach())
        ind_mask = mask.expand_as(td_individual_error)
        masked_td_individual_error = td_individual_error * ind_mask
        individual_loss = th.sum(masked_td_individual_error ** 2).sum() / th.mean(ind_mask, dim=-1).sum()
        mix_explore_distance_loss = mix_explore_distance_losses.mean()
        loss += 0.001*(self.ind*individual_loss + self.mix*mix_explore_distance_loss)

        # Backpropagation
        self.optimiser.zero_grad()
        loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.params, self.args.grad_norm_clip)  # max_norm
        self.optimiser.step()

        if (episode_num - self.last_target_update_episode) / self.args.target_update_interval >= 1.0:
            self._update_targets()
            self.last_target_update_episode = episode_num

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat("loss", loss.item(), t_env)
            self.logger.log_stat("grad_norm", grad_norm.item(), t_env)
            mask_elems = mask.sum().item()
            self.logger.log_stat("td_error_abs", (masked_td_error.abs().sum().item()/mask_elems), t_env)
            self.logger.log_stat("q_taken_mean", (chosen_action_qvals * mask).sum().item()/(mask_elems * self.args.n_agents), t_env)
            self.logger.log_stat("target_mean", (targets * mask).sum().item()/(mask_elems * self.args.n_agents), t_env)
            self.log_stats_t = t_env

        if self.args.prioritized_buffer:
            return masked_td_error ** 2

    def _update_targets(self):
        self.target_mac.load_state(self.mac)
        if self.mixer is not None:
            self.target_mixer.load_state_dict(self.mixer.state_dict())
        self.logger.console_logger.info("Updated target network")

    def cuda(self):
        self.mac.cuda()
        self.target_mac.cuda()
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
        # Not quite right, but we don't want to save target networks
        self.target_mac.load_models(path)
        if self.mixer is not None:
            self.mixer.load_state_dict(th.load("{}/mixer.th".format(path), map_location=lambda storage, loc: storage))
        self.optimiser.load_state_dict(th.load("{}/opt.th".format(path), map_location=lambda storage, loc: storage))
