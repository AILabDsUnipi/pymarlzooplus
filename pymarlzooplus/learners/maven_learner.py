import copy
from components.episode_buffer import EpisodeBatch
from modules.mixers.vdn import VDNMixer
from modules.mixers.qmix import QMixer
from modules.mixers.maven import MavenMixer as MavenMixer
import torch as th
from torch.optim import RMSprop
import numpy as np


class MavenLearner:
    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.mac = mac
        self.logger = logger

        self.params = list(mac.parameters())

        self.last_target_update_episode = 0

        self.mixer = None
        if args.mixer is not None:
            if args.mixer == "vdn":
                self.mixer = VDNMixer()
            elif args.mixer == "maven":
                self.mixer = MavenMixer(args)
            else:
                raise ValueError("Mixer {} not recognised.".format(args.mixer))
            self.params += list(self.mixer.parameters())
            self.target_mixer = copy.deepcopy(self.mixer)

        discrim_input = np.prod(self.args.state_shape) + self.args.n_agents * self.args.n_actions

        self.rnn_agg = RNNAggregator(discrim_input, args)
        self.discrim = Discrim(args.rnn_agg_size, self.args.noise_dim, args)
        self.params += list(self.discrim.parameters())
        self.params += list(self.rnn_agg.parameters())
        
        self.discrim_loss = th.nn.CrossEntropyLoss(reduction="none")

        self.optimiser = RMSprop(params=self.params, lr=args.lr, alpha=args.optim_alpha, eps=args.optim_eps)

        self.target_mac = copy.deepcopy(mac)

        self.log_stats_t = -self.args.learner_log_interval - 1

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        # Enable anomaly detection (optional but useful during debugging)
        # th.autograd.set_detect_anomaly(True)

        # Get the relevant quantities
        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float().clone() 
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"]
        noise = batch["noise"][:, 0].unsqueeze(1).repeat(1, rewards.shape[1], 1)

        # Calculate estimated Q-Values
        mac_out = []
        self.mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            agent_outs = self.mac.forward(batch, t=t)
            mac_out.append(agent_outs)
        mac_out = th.stack(mac_out, dim=1)  # Concat over time

        # Pick the Q-Values for the actions taken by each agent
        chosen_action_qvals = th.gather(mac_out[:, :-1], dim=3, index=actions).squeeze(3)

        # Calculate the Q-Values necessary for the target
        target_mac_out = []
        self.target_mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            target_agent_outs = self.target_mac.forward(batch, t=t)
            target_mac_out.append(target_agent_outs)
        target_mac_out = th.stack(target_mac_out[1:], dim=1)  # Skip first timestep

        # Mask out unavailable actions
        target_mac_out = target_mac_out.clone() 
        target_mac_out[avail_actions[:, 1:] == 0] = -9999999

        # Max over target Q-Values
        if self.args.double_q:
            mac_out = mac_out.clone() 
            mac_out[avail_actions == 0] = -9999999
            cur_max_actions = mac_out[:, 1:].max(dim=3, keepdim=True)[1]
            target_max_qvals = th.gather(target_mac_out, 3, cur_max_actions).squeeze(3)
        else:
            target_max_qvals = target_mac_out.max(dim=3)[0]

        # Mix
        if self.mixer is not None:
            chosen_action_qvals = self.mixer(chosen_action_qvals, batch["state"][:, :-1], noise)
            target_max_qvals = self.target_mixer(target_max_qvals, batch["state"][:, 1:], noise)

        # Discriminator
        mac_out = mac_out.clone() 
        mac_out[avail_actions == 0] = -9999999
        q_softmax_actions = th.nn.functional.softmax(mac_out[:, :-1], dim=3)

        q_softmax_agents = q_softmax_actions.reshape(q_softmax_actions.shape[0], q_softmax_actions.shape[1], -1)

        states = batch["state"][:, :-1]
        state_and_softactions = th.cat([q_softmax_agents, states], dim=2)

        h_to_use = th.zeros(size=(batch.batch_size, self.args.rnn_agg_size)).to(states.device)
        hs = th.ones_like(h_to_use)
        for t in range(batch.max_seq_length - 1):
            hs = self.rnn_agg(state_and_softactions[:, t], hs)
            for b in range(batch.batch_size):
                if t == batch.max_seq_length - 2 or (mask[b, t] == 1 and mask[b, t + 1] == 0):
                    h_to_use[b] = hs[b]
        s_and_softa_reshaped = h_to_use
            
        discrim_prediction = self.discrim(s_and_softa_reshaped)

        # Cross-Entropy
        target_repeats = 1
        discrim_target = batch["noise"][:, 0].long().detach().max(dim=1)[1].unsqueeze(1).repeat(1, target_repeats).reshape(-1)
        discrim_loss = self.discrim_loss(discrim_prediction, discrim_target)

        averaged_discrim_loss = discrim_loss.mean()
        self.logger.log_stat("discrim_loss", averaged_discrim_loss.item(), t_env)

        # Calculate 1-step Q-Learning targets
        targets = rewards + self.args.gamma * (1 - terminated) * target_max_qvals

        # Td-error
        td_error = (chosen_action_qvals - targets.detach())

        mask_expanded = mask.expand_as(td_error)
        masked_td_error = td_error * mask_expanded

        # Normal L2 loss, take mean over actual data
        loss = (masked_td_error ** 2).sum() / mask_expanded.sum()

        loss = loss + self.args.mi_loss * averaged_discrim_loss

        # Optimise
        self.optimiser.zero_grad()
        loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.params, self.args.grad_norm_clip)
        self.optimiser.step()

        if (episode_num - self.last_target_update_episode) / self.args.target_update_interval >= 1.0:
            self._update_targets()
            self.last_target_update_episode = episode_num

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat("loss", loss.item(), t_env)
            self.logger.log_stat("grad_norm", grad_norm.item(), t_env)
            mask_elems = mask_expanded.sum().item()
            self.logger.log_stat("td_error_abs", (masked_td_error.abs().sum().item() / mask_elems), t_env)
            self.logger.log_stat("q_taken_mean", (chosen_action_qvals * mask_expanded).sum().item() / (mask_elems * self.args.n_agents), t_env)
            self.logger.log_stat("target_mean", (targets * mask_expanded).sum().item() / (mask_elems * self.args.n_agents), t_env)
            self.log_stats_t = t_env


    def _update_targets(self):
        self.target_mac.load_state(self.mac)
        if self.mixer is not None:
            self.target_mixer.load_state_dict(self.mixer.state_dict())
        self.logger.console_logger.info("Updated target network")

    def cuda(self):
        self.mac.cuda()
        self.target_mac.cuda()
        self.discrim.cuda()
        self.rnn_agg.cuda()
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
        self.target_mac.load_models(path)
        if self.mixer is not None:
            self.mixer.load_state_dict(th.load("{}/mixer.th".format(path), map_location=lambda storage, loc: storage))
        self.optimiser.load_state_dict(th.load("{}/opt.th".format(path), map_location=lambda storage, loc: storage))


class Discrim(th.nn.Module):

    def __init__(self, input_size, output_size, args):
        super().__init__()
        self.args = args
        layers = [th.nn.Linear(input_size, self.args.discrim_size), th.nn.ReLU()]
        for _ in range(self.args.discrim_layers - 1):
            layers.append(th.nn.Linear(self.args.discrim_size, self.args.discrim_size))
            layers.append(th.nn.ReLU())
        layers.append(th.nn.Linear(self.args.discrim_size, output_size))
        self.model = th.nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class RNNAggregator(th.nn.Module):

    def __init__(self, input_size, args):
        super().__init__()
        self.args = args
        self.input_size = input_size
        output_size = args.rnn_agg_size
        self.rnn = th.nn.GRUCell(input_size, output_size)

    def forward(self, x, h):
        return self.rnn(x, h)
