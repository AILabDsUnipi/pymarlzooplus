import copy

import numpy as np
import torch as th
from torch.optim import RMSprop
from torch.optim import Adam
import torch.nn.functional as func

from pymarlzooplus.components.episode_buffer import EpisodeBatch
from pymarlzooplus.modules.mixers.vdn import VDNMixer
from pymarlzooplus.modules.mixers.qmix import QMixer
from pymarlzooplus.utils.torch_utils import to_cuda
from pymarlzooplus.controllers import REGISTRY as mac_REGISTRY
from pymarlzooplus.components.standarize_stream import RunningMeanStd


class vdn_Learner:
    def __init__(self, mac, scheme, logger, args, groups=None):
        self.args = args
        self.mac = copy.deepcopy(mac)
        self.target_mac = copy.deepcopy(mac)
        self.soft_target_mac = copy.deepcopy(mac)
        self.logger = logger

        self.params = list(self.mac.parameters())

        self.last_target_update_episode = 0

        self.mixer = VDNMixer()
        self.target_mixer = copy.deepcopy(self.mixer)
        self.predict_mac = mac_REGISTRY[args.mac](scheme, groups, args)

        self.params += list(self.mixer.parameters())
        self.predict_params = list(self.predict_mac.parameters())

        self.decay_stats_t = 0
        self.decay_stats_t_2 = 0
        self.state_shape = scheme["state"]["vshape"]

        self.optimiser = RMSprop(params=self.params, lr=args.lr, alpha=args.optim_alpha, eps=args.optim_eps)
        self.predict_optimiser = Adam(params=self.predict_params, lr=args.lr)

        self.log_stats_t = -self.args.learner_log_interval - 1
        self._device = "cuda" if args.use_cuda and th.cuda.is_available() else "cpu"
        if self.args.standardise_rewards:
            self.rew_ms = RunningMeanStd(shape=(1,), device=self._device)

    def subtrain(self, batch: EpisodeBatch, t_env: int, episode_num: int, mac, imac=None, timac=None):

        # Get the relevant quantities
        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"]

        if self.args.standardise_rewards:
            self.rew_ms.update(rewards)
            rewards = (rewards - self.rew_ms.mean) / th.sqrt(self.rew_ms.var)

        # Calculate estimated Q-Values
        mac.init_hidden(batch.batch_size)
        self.predict_mac.init_hidden(batch.batch_size)
        self.target_mac.init_hidden(batch.batch_size)
        self.soft_target_mac.init_hidden(batch.batch_size)
        target_mac_out = self.target_mac.forward(batch, batch.max_seq_length, batch_inf=True)[:, 1:, ...]
        soft_target_mac_out = self.soft_target_mac.forward(batch, batch.max_seq_length, batch_inf=True)[:, 1:, ...]

        mac_out = mac.forward(batch, batch.max_seq_length, batch_inf=True)
        predict_mac_out = self.predict_mac.forward(batch, batch.max_seq_length, batch_inf=True)[:, 1:, ...]

        soft_target_mac_out_next = soft_target_mac_out.clone().detach()
        soft_target_mac_out_next = soft_target_mac_out_next.contiguous().view(-1, self.args.n_actions) * 10

        predict_mac_out = predict_mac_out.contiguous().view(-1, self.args.n_actions)

        prediction_error = func.pairwise_distance(predict_mac_out, soft_target_mac_out_next, p=2.0, keepdim=True)
        prediction_mask = mask.repeat(1, 1, self.args.n_agents)
        prediction_error = prediction_error.reshape(batch.batch_size, -1, self.args.n_agents) * prediction_mask

        if hasattr(self.args, 'mask_other_agents') and self.args.mask_other_agents:
            intrinsic_rewards = self.args.curiosity_scale * (prediction_error.detach()[:, :, 0:1])
        else:
            intrinsic_rewards = self.args.curiosity_scale * (prediction_error.mean(dim=-1, keepdim=True).detach())

        # Loss and optimization
        prediction_loss = prediction_error.sum() / prediction_mask.sum()
        self.predict_optimiser.zero_grad()
        prediction_loss.backward()
        predict_grad_norm = th.nn.utils.clip_grad_norm_(self.predict_params, self.args.grad_norm_clip)
        self.predict_optimiser.step()

        # Pick the Q-Values for the actions taken by each agent
        chosen_action_qvals = th.gather(mac_out[:, :-1], dim=3, index=actions).squeeze(3)  # Remove the last dim

        x_mac_out = mac_out.clone().detach()
        x_mac_out[avail_actions == 0] = -9999999
        max_action_qvals, max_action_index = x_mac_out[:, :-1].max(dim=3)

        max_action_index = max_action_index.detach().unsqueeze(3)
        is_max_action = (max_action_index == actions).int().float()

        # Calculate the Q-Values necessary for the target

        # Max over target Q-Values
        if self.args.double_q:
            # Get actions that maximise live Q (for double q-learning)
            mac_out_detach = mac_out.clone().detach()
            mac_out_detach[avail_actions == 0] = -9999999
            cur_max_actions = mac_out_detach[:, 1:].max(dim=3, keepdim=True)[1]
            target_max_qvals = th.gather(target_mac_out, 3, cur_max_actions).squeeze(3)
        else:
            target_max_qvals = target_mac_out.max(dim=3)[0]

        # Mix
        if self.mixer is not None:
            chosen_action_qvals = self.mixer(chosen_action_qvals, batch["state"][:, :-1])
            target_max_qvals = self.target_mixer(target_max_qvals, batch["state"][:, 1:])

        # Calculate 1-step Q-Learning targets
        targets = rewards + self.args.gamma * (1 - terminated) * target_max_qvals

        # Td-error
        td_error = (chosen_action_qvals - targets.detach())

        mask = mask.expand_as(td_error)

        # 0-out the targets that came from padded data
        masked_td_error = td_error * mask
        if hasattr(self.args, 'use_qtotal_td') and self.args.use_qtotal_td:
            intrinsic_rewards = self.args.curiosity_scale * th.abs(masked_td_error.clone().detach())

        # Normal L2 loss, take mean over actual data
        loss = (masked_td_error ** 2).sum() / mask.sum()

        masked_hit_prob = th.mean(is_max_action, dim=2) * mask
        hit_prob = masked_hit_prob.sum() / mask.sum()

        # Optimise
        self.optimiser.zero_grad()
        loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.params, self.args.grad_norm_clip)
        self.optimiser.step()

        if self.args.curiosity_decay:
            if t_env - self.decay_stats_t >= self.args.curiosity_decay_cycle:
                if self.args.curiosity_decay_rate <= 1.0:
                    if self.args.curiosity_scale > self.args.curiosity_decay_stop:
                        self.args.curiosity_scale = self.args.curiosity_scale * self.args.curiosity_decay_rate
                    else:
                        self.args.curiosity_scale = self.args.curiosity_decay_stop
                else:
                    if self.args.curiosity_scale < self.args.curiosity_decay_stop:
                        self.args.curiosity_scale = self.args.curiosity_scale * self.args.curiosity_decay_rate
                    else:
                        self.args.curiosity_scale = self.args.curiosity_decay_stop

                self.decay_stats_t = t_env

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat("vdn loss", loss.item(), t_env)
            self.logger.log_stat("curiosity_scale", self.args.curiosity_scale, t_env)
            self.logger.log_stat("curiosity_decay_rate", self.args.curiosity_decay_rate, t_env)
            self.logger.log_stat("curiosity_decay_cycle", self.args.curiosity_decay_cycle, t_env)
            self.logger.log_stat("curiosity_decay_stop", self.args.curiosity_decay_stop, t_env)
            self.logger.log_stat("vdn hit_prob", hit_prob.item(), t_env)
            self.logger.log_stat("vdn grad_norm", grad_norm.item(), t_env)
            mask_elems = mask.sum().item()
            self.logger.log_stat("vdn prediction loss", prediction_loss.item(), t_env)
            self.logger.log_stat("vdn predict grad_norm", predict_grad_norm.item(), t_env)
            self.logger.log_stat("vdn intrinsic rewards", intrinsic_rewards.sum().item() / mask_elems, t_env)
            self.logger.log_stat("vdn extrinsic rewards", rewards.sum().item() / mask_elems, t_env)
            self.logger.log_stat("vdn td_error_abs", (masked_td_error.abs().sum().item() / mask_elems), t_env)
            self.logger.log_stat(
                "vdn q_taken_mean",
                (chosen_action_qvals * mask).sum().item() / (mask_elems * self.args.n_agents),
                t_env
            )
            self.logger.log_stat(
                "vdn target_mean",
                (targets * mask).sum().item() / (mask_elems * self.args.n_agents),
                t_env
            )

            self.log_stats_t = t_env

        return intrinsic_rewards

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int, imac=None, timac=None):

        intrinsic_rewards = self.subtrain(batch, t_env, episode_num, self.mac, imac=imac, timac=timac)

        self._smooth_update_predict_targets()
        if (episode_num - self.last_target_update_episode) / self.args.target_update_interval >= 1.0:
            self._update_targets()
            self.last_target_update_episode = episode_num

        return intrinsic_rewards

    def _update_targets(self):
        self.target_mac.load_state(self.mac)
        if self.mixer is not None:
            self.target_mixer.load_state_dict(self.mixer.state_dict())
        self.logger.console_logger.info("Updated target network")

    def _smooth_update_predict_targets(self):
        self.soft_update(self.soft_target_mac, self.mac, self.args.soft_update_tau)

    @staticmethod
    def soft_update(target, source, tau):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

    def cuda(self):
        to_cuda(self.mac, self.args.device)
        to_cuda(self.target_mac, self.args.device)
        to_cuda(self.soft_target_mac, self.args.device)
        to_cuda(self.predict_mac, self.args.device)

        if self.mixer is not None:
            to_cuda(self.mixer, self.args.device)
            to_cuda(self.target_mixer, self.args.device)

    def save_models(self, path):
        self.mac.save_models(path)
        self.predict_mac.save_models('{}/predict_mac'.format(path))

        if self.mixer is not None:
            th.save(self.mixer.state_dict(), "{}/mixer.th".format(path))
            th.save(self.target_mixer.state_dict(), "{}/target_mixer.th".format(path))

        th.save(self.optimiser.state_dict(), "{}/opt.th".format(path))
        th.save(self.predict_optimiser.state_dict(), "{}/predict_opt.th".format(path))

    def load_models(self, path):
        self.mac.load_models(path)
        # Not quite right, but I don't want to save target networks
        self.target_mac.load_models(path)
        self.soft_target_mac.load_models(path)
        self.predict_mac.load_models('{}/predict_mac'.format(path))

        if self.mixer is not None:
            self.mixer.load_state_dict(th.load("{}/mixer.th".format(path), map_location=lambda storage, loc: storage))
            self.target_mixer.load_state_dict(th.load("{}/target_mixer.th".format(path),
                                                      map_location=lambda storage, loc: storage))

        self.optimiser.load_state_dict(th.load("{}/opt.th".format(path), map_location=lambda storage, loc: storage))
        self.predict_optimiser.load_state_dict(th.load("{}/predict_opt.th".format(path),
                                                       map_location=lambda storage, loc: storage))
