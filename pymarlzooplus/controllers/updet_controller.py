import torch as th
from pymarlzooplus.controllers.basic_controller import BasicMAC


class UPDeTMAC(BasicMAC):

    def forward(self, ep_batch, t, test_mode=False):
        agent_inputs = self._build_inputs_transformer(ep_batch, t)
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

    def _build_inputs_transformer(self, batch, t):
        bs = batch.batch_size
        raw_obs = batch["obs"][:, t]                              # (bs, n_agents, obs_dim)
        n_tokens = raw_obs.shape[-1] // self.args.token_dim
        return raw_obs.reshape(bs * self.n_agents, n_tokens, self.args.token_dim)
