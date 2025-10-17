# Code adapted from: https://github.com/lich14/CDS

import copy
import torch.nn.functional as F
from pymarlzooplus.components.episode_buffer import EpisodeBatch
from pymarlzooplus.modules.mixers.dmaq_general import DMAQer
from pymarlzooplus.modules.mixers.dmaq_qatten import DMAQ_QattenMixer
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from torch.optim import RMSprop
from pymarlzooplus.modules.explorers.CDS import CDSExplorer
from pymarlzooplus.components.standarize_stream import RunningMeanStd
import torch as th


class DMAQ_qattenLearner:

    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.mac = mac
        self.logger = logger
        self.algo_name = args.name

        self.params = list(mac.parameters())
        self.last_target_update_episode = 0

        self.mixer = None
        if args.mixer is not None:
            if args.mixer == "dmaq":
                self.mixer = DMAQer(args)
            elif args.mixer == 'dmaq_qatten':
                self.mixer = DMAQ_QattenMixer(args)
            else:
                raise ValueError("Mixer {} not recognised.".format(args.mixer))
            self.params += list(self.mixer.parameters())
            self.target_mixer = copy.deepcopy(self.mixer)

        if self.algo_name == "cds":
            self.explorer = CDSExplorer(args, scheme)
            assert self.args.obs_agent_id is False
            assert self.args.obs_last_action is True

        self.optimiser = RMSprop(params=self.params, lr=args.lr, alpha=args.optim_alpha, eps=args.optim_eps)

        # a little wasteful to deepcopy (e.g., duplicates action selector), but should work for any MAC
        self.target_mac = copy.deepcopy(mac)

        self.log_stats_t = -self.args.learner_log_interval - 1
        self.n_actions = self.args.n_actions

        self._device = self.args.device
        if self.args.standardise_rewards:
            self.rew_ms = RunningMeanStd(shape=(1,), device=self._device)

    def sub_train(
            self, batch,
            t_env,
            episode_num,
            mac,
            mixer,
            optimiser,
            params
    ):

        # Get the relevant quantities
        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"]
        actions_onehot = batch["actions_onehot"][:, :-1]
        if self.algo_name == "cds":
            last_actions_onehot = th.cat(
                [th.zeros_like(actions_onehot[:, 0].unsqueeze(1)), actions_onehot], dim=1
            )
        if self.args.standardise_rewards:
            self.rew_ms.update(rewards)
            rewards = (rewards - self.rew_ms.mean) / th.sqrt(self.rew_ms.var)

        # Calculate estimated Q-Values
        mac_out = []
        mac.init_hidden(batch.batch_size)
        # will use obs and last actions to output local Q
        # while non cds calculates hidden states all at once per batch
        if self.algo_name == "cds":
            initial_hidden = mac.hidden_states.clone().detach()
            initial_hidden = initial_hidden.reshape(-1, initial_hidden.shape[-1]).to(self.args.device)
            input_here = th.cat(
                (batch["obs"], last_actions_onehot), dim=-1
            ).permute(0, 2, 1, 3).to(self.args.device)
            mac_out, hidden_store, local_qs = mac.agent.forward(
                input_here.clone().detach(), initial_hidden.clone().detach()
            )
            hidden_store = hidden_store.reshape(
                -1, input_here.shape[1], hidden_store.shape[-2], hidden_store.shape[-1]
            ).permute(0, 2, 1, 3)
        else:
            for t in range(batch.max_seq_length):
                agent_outs = mac.forward(batch, t=t)
                mac_out.append(agent_outs)
            mac_out = th.stack(mac_out, dim=1)  # Concat over time

        # Pick the Q-Values for the actions taken by each agent
        chosen_action_qvals = th.gather(mac_out[:, :-1], dim=3, index=actions).squeeze(3)  # Remove the last dim

        x_mac_out = mac_out.clone().detach()
        x_mac_out[avail_actions == 0] = -9999999
        max_action_qvals, max_action_index = x_mac_out[:, :-1].max(dim=3)

        max_action_index = max_action_index.detach().unsqueeze(3)
        is_max_action = (max_action_index == actions).int().float()

        # Calculate the Q-Values necessary for the target
        target_mac_out = []
        self.target_mac.init_hidden(batch.batch_size)
        if self.algo_name == "cds":
            initial_hidden_target = self.target_mac.hidden_states.clone().detach()
            initial_hidden_target = initial_hidden_target.reshape(
                -1, initial_hidden_target.shape[-1]
            ).to(self.args.device)
            target_mac_out, _, _ = self.target_mac.agent.forward(
                input_here.clone().detach(), initial_hidden_target.clone().detach()
            )
            target_mac_out = target_mac_out[:, 1:]
        else:
            for t in range(batch.max_seq_length):
                target_agent_outs = self.target_mac.forward(batch, t=t)
                target_mac_out.append(target_agent_outs)
            # We don't need the first timesteps Q-Value estimate for calculating targets
            target_mac_out = th.stack(target_mac_out[1:], dim=1)  # Concat across time

        # Mask out unavailable actions
        target_mac_out[avail_actions[:, 1:] == 0] = -9999999

        # Max over target Q-Values
        if self.args.double_q:
            # Get actions that maximise live Q (for double q-learning)
            mac_out_detach = mac_out.clone().detach()
            mac_out_detach[avail_actions == 0] = -9999999
            cur_max_actions = mac_out_detach[:, 1:].max(dim=3, keepdim=True)[1]
            target_chosen_qvals = th.gather(target_mac_out, 3, cur_max_actions).squeeze(3)
            target_max_qvals = target_mac_out.max(dim=3)[0]
            cur_max_actions_onehot = th.zeros(cur_max_actions.squeeze(3).shape + (self.n_actions,)).to(self._device)
            cur_max_actions_onehot = cur_max_actions_onehot.scatter_(3, cur_max_actions, 1)
        else:
            # Calculate the Q-Values necessary for the target
            target_mac_out = []
            self.target_mac.init_hidden(batch.batch_size)
            for t in range(batch.max_seq_length):
                target_agent_outs = self.target_mac.forward(batch, t=t)
                target_mac_out.append(target_agent_outs)
            # We don't need the first timesteps Q-Value estimate for calculating targets
            target_mac_out = th.stack(target_mac_out[1:], dim=1)  # Concat across time
            target_max_qvals = target_mac_out.max(dim=3)[0]

        if self.algo_name == "cds":
            # Intrinsic
            with th.no_grad():

                obs = batch["obs"][:, :-1]
                obs_next = batch["obs"][:, 1:]
                mask_clone = mask.detach().clone().unsqueeze(-2).expand(obs.shape[:-1] + mask.shape[-1:])
                mask_clone = mask_clone.permute(0, 2, 1, 3)
                mask_clone = mask_clone.reshape(-1, mask_clone.shape[-2], mask_clone.shape[-1])
                mask_clone = mask_clone.reshape(-1, mask_clone.shape[-1])

                obs_intrinsic = obs.clone().permute(0, 2, 1, 3)
                obs_intrinsic = obs_intrinsic.reshape(-1, obs_intrinsic.shape[-2], obs_intrinsic.shape[-1])
                eval_h_intrinsic = hidden_store.clone().permute(0, 2, 1, 3)
                eval_h_intrinsic = eval_h_intrinsic.reshape(-1, eval_h_intrinsic.shape[-2], eval_h_intrinsic.shape[-1])
                h_cat = th.cat(
                    [
                        initial_hidden.reshape(-1, initial_hidden.shape[-1]).unsqueeze(1),
                        eval_h_intrinsic[:, :-2]
                    ],
                    dim=1
                )
                add_id = th.eye(self.args.n_agents).to(obs.device).expand(
                    [obs.shape[0], obs.shape[1], self.args.n_agents, self.args.n_agents]
                ).permute(0, 2, 1, 3)

                actions_onehot_clone = actions_onehot.clone().permute(0, 2, 1, 3)

                intrinsic_input_1 = th.cat(
                    [
                        h_cat,
                        obs_intrinsic,
                        actions_onehot_clone.reshape(-1, actions_onehot_clone.shape[-2], actions_onehot_clone.shape[-1])
                    ],
                    dim=-1
                )

                intrinsic_input_2 = th.cat(
                    [
                        intrinsic_input_1,
                        add_id.reshape(-1, add_id.shape[-2], add_id.shape[-1])
                    ],
                    dim=-1
                )

                intrinsic_input_1 = intrinsic_input_1.reshape(-1, intrinsic_input_1.shape[-1])
                intrinsic_input_2 = intrinsic_input_2.reshape(-1, intrinsic_input_2.shape[-1])

                next_obs_intrinsic = obs_next.clone().permute(0, 2, 1, 3)
                next_obs_intrinsic = next_obs_intrinsic.reshape(
                    -1, next_obs_intrinsic.shape[-2], next_obs_intrinsic.shape[-1]
                )
                next_obs_intrinsic = next_obs_intrinsic.reshape(
                    -1, next_obs_intrinsic.shape[-1]
                )

                log_p_o = self.explorer.target_predict_without_id.get_log_pi(intrinsic_input_1, next_obs_intrinsic)
                log_q_o = self.explorer.target_predict_with_id.get_log_pi(
                    intrinsic_input_2, next_obs_intrinsic, add_id.reshape([-1, add_id.shape[-1]])
                )

                mean_p = th.softmax(mac_out[:, :-1], dim=-1).mean(dim=2)
                q_pi = th.softmax(self.args.beta1 * mac_out[:, :-1], dim=-1)

                pi_diverge = th.cat(
                    [
                        (q_pi[:, :, id] * th.log(q_pi[:, :, id] / mean_p)).sum(dim=-1, keepdim=True)
                        for id in range(self.args.n_agents)
                    ],
                    dim=-1
                ).permute(0, 2, 1).unsqueeze(-1)

                intrinsic_rewards = self.args.beta1 * log_q_o - log_p_o
                intrinsic_rewards = intrinsic_rewards.reshape(-1, obs_intrinsic.shape[1], intrinsic_rewards.shape[-1])
                intrinsic_rewards = intrinsic_rewards.reshape(
                    -1, obs.shape[2], obs_intrinsic.shape[1], intrinsic_rewards.shape[-1]
                )
                intrinsic_rewards = intrinsic_rewards + self.args.beta2 * pi_diverge

            # update predict network
            add_id = add_id.reshape([-1, add_id.shape[-1]])
            for index in BatchSampler(
                    SubsetRandomSampler(range(intrinsic_input_1.shape[0])), 256, False
            ):
                self.explorer.eval_predict_without_id.update(
                    intrinsic_input_1[index], next_obs_intrinsic[index], mask_clone[index]
                )
                self.explorer.eval_predict_with_id.update(
                    intrinsic_input_2[index], next_obs_intrinsic[index], add_id[index], mask_clone[index]
                )
        # Mix
        if mixer is not None:
            if self.args.mixer == "dmaq_qatten":
                ans_chosen, q_attend_regs, head_entropies = mixer(
                    chosen_action_qvals, batch["state"][:, :-1], is_v=True
                )
                ans_adv, _, _ = mixer(
                    chosen_action_qvals,
                    batch["state"][:, :-1],
                    actions=actions_onehot,
                    max_q_i=max_action_qvals,
                    is_v=False
                )
            else:
                ans_chosen = mixer(
                    chosen_action_qvals,
                    batch["state"][:, :-1],
                    is_v=True
                )
                ans_adv = mixer(
                    chosen_action_qvals,
                    batch["state"][:, :-1],
                    actions=actions_onehot,
                    max_q_i=max_action_qvals,
                    is_v=False
                )
            chosen_action_qvals = ans_chosen + ans_adv

            if self.args.double_q:
                if self.args.mixer == "dmaq_qatten":
                    target_chosen, _, _ = self.target_mixer(target_chosen_qvals, batch["state"][:, 1:], is_v=True)
                    target_adv, _, _ = self.target_mixer(
                        target_chosen_qvals,
                        batch["state"][:, 1:],
                        actions=cur_max_actions_onehot,
                        max_q_i=target_max_qvals,
                        is_v=False
                    )
                else:
                    target_chosen = self.target_mixer(target_chosen_qvals, batch["state"][:, 1:], is_v=True)
                    target_adv = self.target_mixer(
                        target_chosen_qvals,
                        batch["state"][:, 1:],
                        actions=cur_max_actions_onehot,
                        max_q_i=target_max_qvals,
                        is_v=False
                    )
                target_max_qvals = target_chosen + target_adv
            else:
                target_max_qvals = self.target_mixer(target_max_qvals, batch["state"][:, 1:], is_v=True)

        # Calculate 1-step Q-Learning targets
        if self.algo_name == "cds":
            targets = (
                    rewards + self.args.beta * intrinsic_rewards.mean(dim=1) +
                    self.args.gamma * (1 - terminated) * target_max_qvals
            )
        else:
            targets = rewards + self.args.gamma * (1 - terminated) * target_max_qvals

        # Td-error
        td_error = (chosen_action_qvals - targets.detach())

        # 0-out the targets that came from padded data
        mask = mask.expand_as(td_error)
        masked_td_error = td_error * mask

        # Normal L2 loss, take mean over actual data
        if self.args.mixer == "dmaq_qatten":
            loss = (masked_td_error ** 2).sum() / mask.sum() + q_attend_regs
        else:
            loss = (masked_td_error ** 2).sum() / mask.sum()
        if self.algo_name == "cds":
            norm_loss = F.l1_loss(local_qs, target=th.zeros_like(
                local_qs), reduction='mean')
            loss += self.args.lambda_ * norm_loss
        masked_hit_prob = th.mean(is_max_action, dim=2) * mask
        hit_prob = masked_hit_prob.sum() / mask.sum()

        # Optimise
        optimiser.zero_grad()
        loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(params, self.args.grad_norm_clip)
        optimiser.step()

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat("loss", loss.item(), t_env)
            self.logger.log_stat("hit_prob", hit_prob.item(), t_env)
            self.logger.log_stat("grad_norm", grad_norm.item(), t_env)
            mask_elems = mask.sum().item()
            self.logger.log_stat(
                "td_error_abs",
                (masked_td_error.abs().sum().item() / mask_elems),
                t_env
            )
            self.logger.log_stat(
                "q_taken_mean",
                (chosen_action_qvals * mask).sum().item() / (mask_elems * self.args.n_agents),
                t_env
            )
            self.logger.log_stat(
                "target_mean",
                (targets * mask).sum().item() / (mask_elems * self.args.n_agents),
                t_env
            )
            self.log_stats_t = t_env

        if self.args.prioritized_buffer:
            return masked_td_error ** 2

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        self.sub_train(batch, t_env, episode_num, self.mac, self.mixer, self.optimiser, self.params)
        if (episode_num - self.last_target_update_episode) / self.args.target_update_interval >= 1.0:
            self._update_targets()
            self.last_target_update_episode = episode_num

    def _update_targets(self):
        self.target_mac.load_state(self.mac)
        if self.mixer is not None:
            self.target_mixer.load_state_dict(self.mixer.state_dict())
        if self.algo_name == "cds":
            self.explorer.update_targets()
        self.logger.console_logger.info("Updated target network")

    def cuda(self):
        self.mac.cuda()
        self.target_mac.cuda()
        if self.mixer is not None:
            self.mixer.cuda()
            self.target_mixer.cuda()
        if self.algo_name == "cds":
            self.explorer.cuda()

    def save_models(self, path):
        self.mac.save_models(path)
        if self.mixer is not None:
            th.save(self.mixer.state_dict(), "{}/mixer.th".format(path))
        th.save(self.optimiser.state_dict(), "{}/opt.th".format(path))
        if self.algo_name == "cds":
            self.explorer.save_models(path)

    def load_models(self, path):
        self.mac.load_models(path)
        # Not quite right, but we don't need to save target networks
        self.target_mac.load_models(path)
        if self.mixer is not None:
            self.mixer.load_state_dict(
                th.load("{}/mixer.th".format(path), map_location=lambda storage, loc: storage)
            )
            self.target_mixer.load_state_dict(
                th.load("{}/mixer.th".format(path), map_location=lambda storage, loc: storage)
            )
        self.optimiser.load_state_dict(
            th.load("{}/opt.th".format(path), map_location=lambda storage, loc: storage)
        )
        if self.algo_name == "cds":
            self.explorer.load_models(path)
