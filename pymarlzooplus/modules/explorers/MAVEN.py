# Code based on:
# https://github.com/AnujMahajanOxf/MAVEN/blob/master/maven_code/src/modules/bandits/reinforce_hierarchial.py

from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np


# Categorical policy for discrete z
class Policy(nn.Module):
    def __init__(self, args):
        super(Policy, self).__init__()
        self.args = args
        self.affine1 = nn.Linear(args.state_shape, 128)
        self.affine2 = nn.Linear(128, args.noise_dim)

    def forward(self, x):
        x = x.view(-1, self.args.state_shape)
        x = self.affine1(x)
        x = F.relu(x)
        action_scores = self.affine2(x)
        return F.softmax(action_scores, dim=1)


# Max entropy Z agent
class EZExplorer:
    def __init__(self, scheme, groups, args, episode_limit, logger):
        self.args = args
        self.lr = args.lr

        self.policy = Policy(args)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=self.lr)

        # Scaling factor for entropy, it would roughly be similar to MI scaling
        self.entropy_scaling = args.entropy_scaling

        self.buffer = deque(maxlen=self.args.bandit_buffer)
        self.logger = logger

    def sample(self, state):
        probs = self.policy(state)
        m = torch.distributions.one_hot_categorical.OneHotCategorical(probs)
        action = m.sample().cpu()
        return action

    def train(self, states, actions, returns, t):

        for s, a, r in zip(states, actions, returns):
            self.buffer.append((s, a, torch.tensor(r, dtype=torch.float)))

        for _ in range(self.args.bandit_iters):
            idxs = np.random.randint(0, len(self.buffer), size=self.args.bandit_batch)
            batch_elems = [self.buffer[i] for i in idxs]
            states_ = torch.stack([x[0] for x in batch_elems]).to(states.device)
            actions_ = torch.stack([x[1] for x in batch_elems]).to(states.device)
            returns_ = torch.stack([x[2] for x in batch_elems]).to(states.device)

            probs = self.policy(states_)
            m = torch.distributions.one_hot_categorical.OneHotCategorical(probs)
            log_probs = m.log_prob(actions_.to(probs.device))
            self.optimizer.zero_grad()
            policy_loss = -torch.dot(
                log_probs, torch.tensor(returns_, device=log_probs.device).float()
            ) + self.entropy_scaling * log_probs.sum()
            policy_loss.backward()
            self.optimizer.step()

        mean_entropy = m.entropy().mean()
        self.logger.log_stat("explorer_entropy", mean_entropy.item(), t)

    def cuda(self):
        self.policy.cuda()

    def save_models(self, path):
        torch.save(self.policy.state_dict(), "{}/explorer_policy.th".format(path))
        torch.save(self.optimizer.state_dict(), "{}/explorer_optim.th".format(path))

    def load_models(self, path):
        self.policy.load_state_dict(
            torch.load("{}/explorer_policy.th".format(path), map_location=lambda storage, loc: storage)
        )
        self.optimizer.load_state_dict(
            torch.load("{}/explorer_optim.th".format(path), map_location=lambda storage, loc: storage)
        )
