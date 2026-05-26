from modules.agents import REGISTRY as agent_REGISTRY
from components.action_selectors import REGISTRY as action_REGISTRY
from .n_controller import NMAC
import torch as th
import torch.nn.functional as F
import numpy as np


# This multi-agent controller shares parameters between agents
class ICESNMAC(NMAC):
    def __init__(self, scheme, groups, args):
        super(ICESNMAC, self).__init__(scheme, groups, args)

        self.t_max = args.t_max
        self.int_ratio = args.int_ratio
        self.int_finish = args.int_finish

    def select_actions(self, ep_batch, t_ep, t_env, bs=slice(None), test_mode=False):
        # Only select actions for the selected batch elements in bs
        avail_actions = ep_batch["avail_actions"][:, t_ep]
        qvals = self.forward(ep_batch, t_ep, test_mode=test_mode)
        int_qvals = self.int_forward(ep_batch, t_ep, test_mode=test_mode)

        if test_mode:
            int_ratio = 0.0
            log_info = {}
        else:
            int_ratio = self.int_ratio * (1.0 - t_env / self.t_max)
            int_ratio = max(int_ratio, self.int_finish)

        chosen_actions, int_entropy = self.action_selector.select_action(
            qvals[bs],
            int_qvals[bs],
            avail_actions[bs],
            t_env,
            int_ratio,
            test_mode=test_mode,
        )
        if not test_mode:
            log_info = {
                "ext_value_estimation_m": qvals.detach().mean().item(),
                "ext_value_estimation_v": qvals.detach().var().item(),
                "int_entropy": int_entropy.detach().mean().item(),
                "int_ratio": int_ratio,
            }
        return chosen_actions, log_info

    def int_forward(self, ep_batch, t, test_mode=False):
        if test_mode:
            self.int_agent.eval()

        agent_inputs = self._build_int_inputs(ep_batch, t)
        avail_actions = ep_batch["avail_actions"][:, t]
        int_agent_outs, self.int_hidden_states = self.int_agent(
            agent_inputs, self.int_hidden_states
        )

        return int_agent_outs

    def init_hidden(self, batch_size):
        self.hidden_states = self.agent.init_hidden()
        self.int_hidden_states = self.int_agent.init_hidden()
        if self.hidden_states is not None:
            self.hidden_states = self.hidden_states.unsqueeze(0).expand(
                batch_size, self.n_agents, -1
            )  # bav
        if self.int_hidden_states is not None:
            self.int_hidden_states = self.int_hidden_states.unsqueeze(0).expand(
                batch_size, self.n_agents, -1
            )  # bav

    def int_parameters(self):
        return self.int_agent.parameters()

    def load_state(self, other_mac):
        self.agent.load_state_dict(other_mac.agent.state_dict())
        self.int_agent.load_state_dict(other_mac.int_agent.state_dict())

    def cuda(self):
        self.agent.cuda()
        self.int_agent.cuda()

    def save_models(self, path):
        th.save(self.agent.state_dict(), "{}/agent.th".format(path))
        th.save(self.int_agent.state_dict(), "{}/int_agent.th".format(path))

    def load_models(self, path):
        self.agent.load_state_dict(
            th.load(
                "{}/agent.th".format(path), map_location=lambda storage, loc: storage
            )
        )
        self.int_agent.load_state_dict(
            th.load(
                "{}/int_agent.th".format(path),
                map_location=lambda storage, loc: storage,
            )
        )

    def _build_agents(self, input_shape):
        self.agent = agent_REGISTRY[self.args.agent](input_shape, self.args)
        self.int_agent = agent_REGISTRY[self.args.agent](
            input_shape + int(np.prod(self.args.state_shape)), self.args
        )

    def _build_int_inputs(self, batch, t):
        # Assumes homogenous agents with flat observations.
        # Other MACs might want to e.g. delegate building inputs to each agent
        bs = batch.batch_size
        inputs = []
        inputs.append(batch["obs"][:, t])  # b1av
        inputs.append(batch["state"][:, t][:, None, :].expand(-1, self.n_agents, -1))
        if self.args.obs_last_action:
            if t == 0:
                inputs.append(th.zeros_like(batch["actions_onehot"][:, t]))
            else:
                inputs.append(batch["actions_onehot"][:, t - 1])
        if self.args.obs_agent_id:
            inputs.append(
                th.eye(self.n_agents, device=batch.device)
                .unsqueeze(0)
                .expand(bs, -1, -1)
            )

        inputs = th.cat([x.reshape(bs, self.n_agents, -1) for x in inputs], dim=-1)
        return inputs
