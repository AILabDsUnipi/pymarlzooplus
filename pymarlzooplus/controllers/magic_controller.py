import torch as th

from .basic_controller import BasicMAC


class MagicMAC(BasicMAC):
    """Multi-agent controller for MAGIC-style communication policies."""

    def __init__(self, scheme, groups, args):
        super().__init__(scheme, groups, args)

        assert self.agent_output_type == "pi_logits"
        assert args.action_selector in ["soft_policies", "multinomial"]

    def forward(self, ep_batch, t, test_mode=False, return_critic_inputs=False):
        del test_mode
        agent_inputs = self._build_inputs(ep_batch, t)
        avail_actions = ep_batch["avail_actions"][:, t]

        agent_outs, self.hidden_states, critic_inputs = self.agent(agent_inputs, self.hidden_states)

        if self.agent_output_type == "pi_logits":
            if self.mask_before_softmax:
                agent_outs = agent_outs.clone()
                reshaped_avail_actions = avail_actions.reshape(ep_batch.batch_size * self.n_agents, -1)
                agent_outs[reshaped_avail_actions == 0] = -1e10
            agent_outs = th.nn.functional.softmax(agent_outs, dim=-1)

        agent_outs = agent_outs.view(ep_batch.batch_size, self.n_agents, -1)
        if return_critic_inputs:
            return agent_outs, critic_inputs
        return agent_outs

    def init_hidden(self, batch_size):
        hidden_state, cell_state = self.agent.init_hidden()
        self.hidden_states = (
            hidden_state.unsqueeze(0).expand(batch_size, self.n_agents, -1),
            cell_state.unsqueeze(0).expand(batch_size, self.n_agents, -1)
        )
