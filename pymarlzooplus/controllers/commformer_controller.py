from .basic_controller import BasicMAC


class CommFormerMAC(BasicMAC):
    """Controller for CommFormer agent."""
    def __init__(self, scheme, groups, args):
        super().__init__(scheme, groups, args)
        self.n_agents = args.n_agents
        self.input_shape = self.agent.input_shape
        self.n_actions = args.n_actions
        self.n_embd = args.n_embd
        assert self.agent_output_type == "pi_logits"
        assert args.action_selector == "soft_policies"

    def select_actions(self, ep_batch, t_ep, t_env, bs=slice(None), test_mode=False):
        extra_returns = {}
        values, actions, action_log_probs = self.forward(ep_batch, t_ep, test_mode=test_mode)
        values, actions, action_log_probs = values[bs], actions[bs], action_log_probs[bs]
        extra_returns.update({'log_probs': action_log_probs.clone().detach(),
                              'values': values.clone().detach()})
        return actions, extra_returns

    def forward(self, ep_batch, t, test_mode=False):
        agent_inputs = self._build_inputs(ep_batch, t)
        avail_actions = ep_batch["avail_actions"][:, t]
        values, actions, action_log_probs = self.get_actions(ep_batch, t, agent_inputs, avail_actions, test_mode)
        return values, actions, action_log_probs

    def get_actions(self, ep_batch, t, obs, available_actions=None, deterministic=False):
        obs = obs.reshape(-1, self.n_agents, self.input_shape)
        if available_actions is not None:
            available_actions = available_actions.reshape(-1, self.n_agents, self.n_actions)
        actions, action_log_probs, values = self.agent.get_actions(ep_batch, t, obs, available_actions, deterministic)
        return values, actions, action_log_probs

    def get_values(self, obs, available_actions=None):
        obs = obs.reshape(-1, self.n_agents, self.input_shape)
        if available_actions is not None:
            available_actions = available_actions.reshape(-1, self.n_agents, self.n_actions)
        values = self.agent.get_values(obs)
        values = values.view(-1, 1)
        return values

    def evaluate_actions(self, ep_batch, t, steps=0, total_step=0):
        agent_inputs = self._build_inputs(ep_batch, t)
        actions = ep_batch["actions"][:, t]
        available_actions = ep_batch["avail_actions"][:, t]
        agent_inputs = agent_inputs.reshape(-1, self.n_agents, self.input_shape)
        actions = actions.reshape(-1, self.n_agents, 1)
        if available_actions is not None:
            available_actions = available_actions.reshape(-1, self.n_agents, self.n_actions)
        values, action_log_probs, entropy = self.agent.evaluate_actions(ep_batch,
                                                                        t,
                                                                        agent_inputs,
                                                                        actions,
                                                                        available_actions,
                                                                        steps,
                                                                        total_step)
        action_log_probs = action_log_probs.view(-1, 1)
        values = values.view(-1, 1)
        entropy = entropy.view(-1, 1)
        entropy = entropy.mean()
        return values, action_log_probs, entropy

    def init_hidden(self, batch_size):
        return
