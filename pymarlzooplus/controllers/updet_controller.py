import torch as th
import torch.nn.functional as F
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
        raw_obs = batch["obs"][:, t]                              # (bs, n_agents, obs_dim)

        # Pad obs_dim to the nearest multiple of token_dim so reshape is always valid.
        # This is needed for envs whose obs_dim is not divisible by token_dim (e.g. RWARE: 71).
        obs_dim = raw_obs.shape[-1]
        remainder = obs_dim % self.args.token_dim
        if remainder != 0:
            raw_obs = F.pad(raw_obs, (0, self.args.token_dim - remainder))

        n_tokens = raw_obs.shape[-1] // self.args.token_dim
        return raw_obs.reshape(bs * self.n_agents, n_tokens, self.args.token_dim)
