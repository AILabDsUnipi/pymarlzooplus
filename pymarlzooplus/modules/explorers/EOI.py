"""
Code based on: https://github.com/jiechuanjiang/eoi_pymarl/tree/main
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np


class EOINet(nn.Module):
    def __init__(self, obs_len, n_agent):
        super(EOINet, self).__init__()
        self.fc1 = nn.Linear(obs_len, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, n_agent)

    def forward(self, x):
        y = F.relu(self.fc1(x))
        y = F.relu(self.fc2(y))
        y = F.softmax(self.fc3(y), dim=1)
        return y


class IVF(nn.Module):
    def __init__(self, obs_len, n_action):
        super(IVF, self).__init__()
        self.fc1 = nn.Linear(obs_len, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, n_action)

    def forward(self, x):
        y = F.relu(self.fc1(x))
        y = F.relu(self.fc2(y))
        y = self.fc3(y)
        return y


class EOITrainer(object):
    def __init__(
            self,
            eoi_net,
            ivf,
            ivf_tar,
            n_agent,
            n_feature,
            ivf_gamma,
            ivf_tau,
            ivf_lr,
            ivf_alpha_intrinsic_r,
            eoi_lr,
            eoi_b2_reg,
            device
    ):

        super(EOITrainer, self).__init__()

        # Parameters
        self.gamma = ivf_gamma
        self.tau = ivf_tau
        self.alpha_intrinsic_r = ivf_alpha_intrinsic_r
        self.b2_reg = eoi_b2_reg

        # Networks
        self.eoi_net = eoi_net
        self.ivf = ivf
        self.ivf_tar = ivf_tar

        # Optimizers
        self.optimizer_eoi = optim.Adam(self.eoi_net.parameters(), lr=eoi_lr)
        self.optimizer_ivf = optim.Adam(self.ivf.parameters(), lr=ivf_lr)

        # Specifications
        self.n_agent = n_agent
        self.n_feature = n_feature
        self.device = device

    def train(self, obs, obs_next, actions, episode_end):

        # numpy to tensor
        obs = torch.from_numpy(obs).to(torch.float32).to(self.device)
        obs_next = torch.from_numpy(obs_next).to(torch.float32).to(self.device)
        actions = torch.from_numpy(actions).to(torch.int32).to(self.device).long()
        episode_end = torch.from_numpy(episode_end).to(torch.float32).to(self.device)

        ## EOI net optimisation
        x = obs_next[:, 0: self.n_feature]
        y = obs_next[:, self.n_feature: self.n_feature + self.n_agent]
        # Get classifier probabilities
        p = self.eoi_net(x)
        # Compute EOI losses
        loss_ce_p_obs_onehot = - (y * (torch.log(p + 1e-8))).mean()
        loss_ce_p_obs_p_obs = - (p * (torch.log(p + 1e-8))).mean()
        # EOI total loss
        eoi_loss = loss_ce_p_obs_onehot + (self.b2_reg * loss_ce_p_obs_p_obs)
        self.optimizer_eoi.zero_grad()
        eoi_loss.backward()
        self.optimizer_eoi.step()

        ## IVF net optimisation
        agent_ids = obs[:, self.n_feature: self.n_feature + self.n_agent].argmax(axis=1, keepdim=True).long()
        intrinsic_r = self.eoi_net(obs[:, 0: self.n_feature]).gather(dim=-1, index=agent_ids)
        # Get intrinsic Q-values for obs
        q_intrinsic = self.ivf(obs)
        tar_q_intrinsic = q_intrinsic.clone().detach()
        # Get max intrinsic Q-values for next obs
        next_q_intrinsic = self.ivf_tar(obs_next).max(axis=1, keepdim=True)[0]
        next_q_intrinsic = (self.alpha_intrinsic_r * intrinsic_r) + (self.gamma * (1 - episode_end) * next_q_intrinsic)
        tar_q_intrinsic.scatter_(dim=-1, index=actions, src=next_q_intrinsic)
        # IVF loss
        ivf_loss = (q_intrinsic - tar_q_intrinsic).pow(2).mean()
        self.optimizer_ivf.zero_grad()
        ivf_loss.backward()
        self.optimizer_ivf.step()

        # Update target IVF net
        with torch.no_grad():
            for p, p_targ in zip(self.ivf.parameters(), self.ivf_tar.parameters()):
                p_targ.data.mul_(self.tau)
                p_targ.data.add_(self.tau * p.data)


class EOIBatchTrainer(object):
    def __init__(self, eoi_trainer, n_agent, n_feature, max_step, batch_size, eoi_batch_size):
        super(EOIBatchTrainer, self).__init__()

        self.batch_size = batch_size
        self.eoi_batch_size = eoi_batch_size
        self.n_agent = n_agent
        self.o_t = np.zeros((batch_size * n_agent * (max_step + 1), n_feature + n_agent))
        self.next_o_t = np.zeros((batch_size * n_agent * (max_step + 1), n_feature + n_agent))
        self.a_t = np.zeros((batch_size * n_agent * (max_step + 1), 1), dtype=np.int32)
        self.d_t = np.zeros((batch_size * n_agent * (max_step + 1), 1))
        self.eoi_trainer = eoi_trainer

    def train_batch(self, episode_sample):
        episode_obs = np.array(episode_sample["obs"])
        episode_actions = np.array(episode_sample["actions"])
        episode_terminated = np.array(episode_sample["terminated"])
        ind = 0

        # Add agent id
        for k in range(self.batch_size):
            for j in range(episode_obs.shape[1] - 2):
                for i in range(self.n_agent):
                    agent_id = np.zeros(self.n_agent)
                    agent_id[i] = 1
                    self.o_t[ind] = np.hstack((episode_obs[k][j][i], agent_id))
                    self.next_o_t[ind] = np.hstack((episode_obs[k][j + 1][i], agent_id))
                    self.a_t[ind] = episode_actions[k][j][i]
                    self.d_t[ind] = episode_terminated[k][j]
                    ind += 1
                if self.d_t[ind - 1] == 1:
                    break

        # Train in batches
        for k in range(int((ind - 1) / self.eoi_batch_size)):
            self.eoi_trainer.train(
                self.o_t[k * self.eoi_batch_size: (k + 1) * self.eoi_batch_size],
                self.next_o_t[k * self.eoi_batch_size: (k + 1) * self.eoi_batch_size],
                self.a_t[k * self.eoi_batch_size: (k + 1) * self.eoi_batch_size],
                self.d_t[k * self.eoi_batch_size: (k + 1) * self.eoi_batch_size]
            )


class Explorer(object):

    def __init__(self, scheme, groups, args, episode_limit, logger):

        self.episode_ratio = args.episode_ratio
        self.explore_ratio = args.explore_ratio
        self.n_agents = groups["agents"]
        self.device = args.device

        assert not isinstance(scheme["obs"]["vshape"], tuple), "EOI does not support image obs for the time being!"

        self.eoi_net = EOINet(
            scheme["obs"]["vshape"],
            self.n_agents
        ).to(self.device)

        self.ivf = IVF(
            scheme["obs"]["vshape"] + self.n_agents,
            scheme["avail_actions"]["vshape"][0]
        ).to(self.device)

        self.ivf_tar = IVF(
            scheme["obs"]["vshape"] + self.n_agents,
            scheme["avail_actions"]["vshape"][0]
        ).to(self.device)

        eoi_trainer = EOITrainer(
            self.eoi_net,
            self.ivf,
            self.ivf_tar,
            self.n_agents,
            scheme["obs"]["vshape"],
            args.ivf_gamma,
            args.ivf_tau,
            args.ivf_lr,
            args.ivf_alpha_intrinsic_r,
            args.eoi_lr,
            args.eoi_b2_reg,
            self.device
        )

        self.trainer = EOIBatchTrainer(
            eoi_trainer,
            self.n_agents,
            scheme["obs"]["vshape"],
            episode_limit,
            args.batch_size,
            args.eoi_batch_size
        )

        self.ivf_flag = [False]

    def train(self, episode_sample):
        self.trainer.train_batch(episode_sample)

    def build_obs(self, obs):
        for i in range(self.n_agents):
            index = np.zeros(self.n_agents)
            index[i] = 1
            obs[i] = np.hstack((obs[i], index))

        # List to numpy
        if isinstance(obs, list):
            obs = np.array(obs)
        assert isinstance(obs, np.ndarray), f"obs type: {type(obs)}"

        # Numpy to tensor
        obs = torch.from_numpy(obs).to(torch.float32).to(self.device)

        return obs

    def select_actions(self, actions, t, test_mode, data):
        """
        actions: torch tensor of shape [batch_size, n_agents]
        data["obs"]: list with "batch_size" elements, each with
                     "n_agents" elements, each with a numpy array (observation)
        """

        # batch_size
        bs = len(data["obs"])
        for batch_idx in range(bs):

            if t == 0:
                if len(self.ivf_flag) < batch_idx+1:
                    self.ivf_flag.append(False)
                self.ivf_flag[batch_idx] = (np.random.rand() < self.episode_ratio)
            else:
                assert len(self.ivf_flag) >= bs, f"len(self.ivf_flag): {len(self.ivf_flag)}, bs: {bs}"

            if (test_mode is False) & (self.ivf_flag[batch_idx] is True):
                if np.random.rand() < self.explore_ratio:

                    obs = data["obs"][batch_idx]

                    # tuple to list
                    if isinstance(obs, tuple):
                        obs = list(obs)

                    obs = self.build_obs(obs)
                    q_p = self.ivf(obs).detach().cpu().numpy()

                    # Change random actions based on Q-values from IVF net
                    j = np.random.randint(self.n_agents)
                    actions[batch_idx][j] = np.argmax(
                        q_p[j] - 9e15 * (1 - np.array(data["avail_actions"][batch_idx][j]))
                    )

        return actions

    def save_models(self, path):
        torch.save(self.eoi_net.state_dict(), f"{path}/eoi_net.th")
        torch.save(self.ivf.state_dict(), f"{path}/ivf.th")
        torch.save(self.ivf_tar.state_dict(), f"{path}/ivf_tar.th")
        torch.save(self.trainer.eoi_trainer.optimizer_eoi.state_dict(), f"{path}/eoi_optimizer.th")
        torch.save(self.trainer.eoi_trainer.optimizer_ivf.state_dict(), f"{path}/ivf_optimizer.th")

    def load_models(self, path):
        self.eoi_net.load_state_dict(
            torch.load(f"{path}/eoi_net.th", map_location=lambda storage, loc: storage)
        )
        self.ivf.load_state_dict(
            torch.load(f"{path}/ivf.th", map_location=lambda storage, loc: storage)
        )
        self.ivf_tar.load_state_dict(
            torch.load(f"{path}/ivf_tar.th", map_location=lambda storage, loc: storage)
        )
        self.trainer.eoi_trainer.optimizer_eoi.load_state_dict(
            torch.load(f"{path}/eoi_optimizer.th", map_location=lambda storage, loc: storage)
        )
        self.trainer.eoi_trainer.optimizer_ivf.load_state_dict(
            torch.load(f"{path}/ivf_optimizer.th", map_location=lambda storage, loc: storage)
        )

