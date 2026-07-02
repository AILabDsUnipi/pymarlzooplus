import torch as th
import torch.nn as nn


class MagicCritic(nn.Module):
    """MAGIC value head over local recurrent state and communicated state."""

    def __init__(self, scheme, args):
        super(MagicCritic, self).__init__()

        self.args = args
        self.n_agents = args.n_agents
        self.input_shape = self._get_input_shape(scheme)
        self.output_type = "v"

        self.fc1 = nn.Linear(self.input_shape, 1)

    def forward(self, hidden_state, communicated_state):
        inputs = self._build_inputs(hidden_state, communicated_state)
        v = self.fc1(inputs)
        return v

    def _build_inputs(self, hidden_state, communicated_state):
        return th.cat([hidden_state, communicated_state], dim=-1)

    def _get_input_shape(self, scheme):
        del scheme
        return 2 * self.args.hidden_dim
