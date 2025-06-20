import torch as th
import torch.nn as nn
import torch.nn.functional as F
from pymarlzooplus.modules.critics.mlp import MLP


class ACCriticNS(nn.Module):
    def __init__(self, scheme, args):
        super(ACCriticNS, self).__init__()

        self.args = args
        self.n_actions = args.n_actions
        self.n_agents = args.n_agents
        self.is_image = False  # Image input
        input_shape = self._get_input_shape(scheme)
        self.output_type = "v"

        # Set up network layers
        self.critics = [MLP(input_shape, args, 1) for _ in range(self.n_agents)]

    def forward(self, batch, t=None):
        inputs, bs, max_t = self._build_inputs(batch, t=t)
        qs = []
        for i in range(self.n_agents):

            if self.is_image:
                agent_inputs = [inputs[:, :, i].view(bs, max_t, 1, *inputs.shape[3:])]
            else:
                agent_inputs = inputs[:, :, i]

            q = self.critics[i](agent_inputs)
            qs.append(q.view(bs, max_t, 1, -1))

        q = th.cat(qs, dim=2)

        return q

    def _build_inputs(self, batch, t=None):
        bs = batch.batch_size
        max_t = batch.max_seq_length if t is None else 1
        ts = slice(None) if t is None else slice(t, t+1)
        inputs = batch["obs"][:, ts]

        return inputs, bs, max_t

    def _get_input_shape(self, scheme):
        # observations
        input_shape = scheme["obs"]["vshape"]

        if isinstance(input_shape, tuple):  # image input
            # Change the number of agents to 1 for compatibility with the way that CNN infer the input shape
            input_shape = ([1, *input_shape],)
            self.is_image = True

        return input_shape

    def parameters(self):
        params = list(self.critics[0].parameters())
        for i in range(1, self.n_agents):
            params += list(self.critics[i].parameters())
        return params

    def state_dict(self):
        return [a.state_dict() for a in self.critics]

    def load_state_dict(self, state_dict):
        for i, a in enumerate(self.critics):
            a.load_state_dict(state_dict[i])

    def cuda(self):
        for c in self.critics:
            c.cuda()
