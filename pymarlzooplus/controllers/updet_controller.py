import torch as th
from pymarlzooplus.controllers.basic_controller import BasicMAC


class UPDeTMAC(BasicMAC):

    def __init__(self, scheme, groups, args):
        assert not args.obs_agent_id, "UPDeTMAC does not support obs_agent_id"
        assert not args.obs_last_action, "UPDeTMAC does not support obs_last_action"
        assert not args.obs_individual_obs, "UPDeTMAC does not support obs_individual_obs"
        super().__init__(scheme, groups, args)

    def forward(self, ep_batch, t, test_mode=False):
        agent_inputs = self._build_inputs(ep_batch, t)
        agent_outs, self.hidden_states = self.agent(
            agent_inputs,
            self.hidden_states.reshape(-1, 1, self.args.emb)
        )
        return agent_outs.view(ep_batch.batch_size, self.n_agents, -1)

    def init_hidden(self, batch_size):
        # shape: (batch, n_agents, 1, emb)
        self.hidden_states = self.agent.init_hidden().unsqueeze(0).expand(
            batch_size, self.n_agents, 1, -1
        ).clone()

    def _build_inputs(self, batch, t):
        bs = batch.batch_size
        n = self.n_agents
        raw_obs = batch["obs"][:, t]  # (bs, n_agents, obs_dim)
        obs_dim = raw_obs.shape[-1]

        # Token = each agent's full observation. For agent i, own obs is placed at index 0
        # (the self-token); other agents fill indices 1..n-1 in circular order.
        agent_idx = th.arange(n, device=raw_obs.device)
        idx = (agent_idx.unsqueeze(1) + agent_idx.unsqueeze(0)) % n  # (n_agents, n_agents)
        return raw_obs[:, idx, :].reshape(bs * n, n, obs_dim)
