import torch as th
import torch.nn as nn
import torch.nn.functional as F


class MagicGraphAttention(nn.Module):
    """Graph-attention layer used by the MAGIC communication block."""

    def __init__(
            self,
            in_features,
            out_features,
            num_heads=1,
            self_loop_type=2,
            average=False,
            normalize=False,
            dropout=0.0,
            negative_slope=0.2,
            bias=True
    ):
        super(MagicGraphAttention, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_heads = num_heads
        self.self_loop_type = self_loop_type
        self.average = average
        self.normalize = normalize
        self.dropout = dropout

        self.W = nn.Parameter(th.zeros(in_features, num_heads * out_features))
        self.a_i = nn.Parameter(th.zeros(num_heads, out_features, 1))
        self.a_j = nn.Parameter(th.zeros(num_heads, out_features, 1))
        if bias:
            bias_dim = out_features if average else num_heads * out_features
            self.bias = nn.Parameter(th.zeros(bias_dim))
        else:
            self.register_parameter("bias", None)
        self.leakyrelu = nn.LeakyReLU(negative_slope)
        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain("relu")
        nn.init.xavier_normal_(self.W.data, gain=gain)
        nn.init.xavier_normal_(self.a_i.data, gain=gain)
        nn.init.xavier_normal_(self.a_j.data, gain=gain)
        if self.bias is not None:
            nn.init.zeros_(self.bias.data)

    def forward(self, inputs, adj):
        h = th.mm(inputs, self.W).view(-1, self.num_heads, self.out_features)
        n_agents = h.size(0)
        device = inputs.device
        dtype = inputs.dtype

        eye = th.eye(n_agents, device=device, dtype=dtype)
        ones = th.ones(n_agents, n_agents, device=device, dtype=dtype)
        adj = adj.to(device=device, dtype=dtype)
        if self.self_loop_type == 0:
            adj = adj * (ones - eye)
        elif self.self_loop_type == 1:
            adj = eye + adj * (ones - eye)

        coeffs = []
        for head in range(self.num_heads):
            coeff_i = th.mm(h[:, head, :], self.a_i[head])
            coeff_j = th.mm(h[:, head, :], self.a_j[head])
            coeffs.append((coeff_i.expand(n_agents, n_agents) +
                           coeff_j.transpose(0, 1).expand(n_agents, n_agents)).unsqueeze(-1))
        e = self.leakyrelu(th.cat(coeffs, dim=-1))

        adj = adj.unsqueeze(-1).expand(n_agents, n_agents, self.num_heads)
        attention = F.softmax(e * adj, dim=1) * adj
        if self.normalize:
            denom = attention.sum(dim=1, keepdim=True).clamp_min(1e-12)
            attention = attention / denom
            attention = attention * adj
        attention = F.dropout(attention, self.dropout, training=self.training)

        outputs = []
        for head in range(self.num_heads):
            outputs.append(th.matmul(attention[:, :, head], h[:, head, :]))
        if self.average:
            output = th.mean(th.stack(outputs, dim=-1), dim=-1)
        else:
            output = th.cat(outputs, dim=-1)

        if self.bias is not None:
            output = output + self.bias
        return output


class MagicAgent(nn.Module):
    """MAGIC communication module adapted to PyMARLZoo+'s MAC/Learner API."""

    def __init__(self, input_shape, args):
        super(MagicAgent, self).__init__()
        if not isinstance(input_shape, int):
            raise ValueError("MAGIC currently supports vector observations only in this integration.")

        self.args = args
        self.algo_name = args.name
        self.use_rnn = args.use_rnn
        self.n_agents = args.n_agents
        self.n_actions = args.n_actions
        self.hidden_dim = args.hidden_dim
        self.input_shape = input_shape
        self.is_image = False

        gat_hidden_dim = getattr(args, "magic_gat_hidden_dim", self.hidden_dim)
        gat_num_heads = getattr(args, "magic_gat_num_heads", 1)
        gat_num_heads_out = getattr(args, "magic_gat_num_heads_out", 1)

        self.directed = getattr(args, "magic_directed", False)
        self.first_graph_complete = getattr(args, "magic_first_graph_complete", False)
        self.second_graph_complete = getattr(args, "magic_second_graph_complete", False)
        self.learn_second_graph = getattr(args, "magic_learn_second_graph", True)
        self.use_gat_encoder = getattr(args, "magic_use_gat_encoder", False)
        self.message_encoder_enabled = getattr(args, "magic_message_encoder", False)
        self.message_decoder_enabled = getattr(args, "magic_message_decoder", False)
        self.comm_mask_zero = getattr(args, "magic_comm_mask_zero", False)

        self.obs_encoder = nn.Linear(input_shape, self.hidden_dim)
        self.lstm_cell = nn.LSTMCell(self.hidden_dim, self.hidden_dim)

        self.sub_processor1 = MagicGraphAttention(
            self.hidden_dim,
            gat_hidden_dim,
            num_heads=gat_num_heads,
            self_loop_type=getattr(args, "magic_self_loop_type1", 1),
            average=False,
            normalize=getattr(args, "magic_first_gat_normalize", False),
        )
        self.sub_processor2 = MagicGraphAttention(
            gat_hidden_dim * gat_num_heads,
            self.hidden_dim,
            num_heads=gat_num_heads_out,
            self_loop_type=getattr(args, "magic_self_loop_type2", 1),
            average=True,
            normalize=getattr(args, "magic_second_gat_normalize", False),
        )

        if self.use_gat_encoder:
            gat_encoder_out_dim = getattr(args, "magic_gat_encoder_out_dim", self.hidden_dim)
            self.gat_encoder = MagicGraphAttention(
                self.hidden_dim,
                gat_encoder_out_dim,
                num_heads=getattr(args, "magic_ge_num_heads", 4),
                self_loop_type=1,
                average=True,
                normalize=getattr(args, "magic_gat_encoder_normalize", False),
            )
            scheduler_input_dim = gat_encoder_out_dim
        else:
            scheduler_input_dim = self.hidden_dim

        if not self.first_graph_complete:
            self.sub_scheduler_mlp1 = self._build_scheduler(scheduler_input_dim)
        if self.learn_second_graph and not self.second_graph_complete:
            self.sub_scheduler_mlp2 = self._build_scheduler(scheduler_input_dim)

        if self.message_encoder_enabled:
            self.message_encoder = nn.Linear(self.hidden_dim, self.hidden_dim)
        if self.message_decoder_enabled:
            self.message_decoder = nn.Linear(self.hidden_dim, self.hidden_dim)

        self.action_head = nn.Linear(2 * self.hidden_dim, self.n_actions)

        if getattr(args, "magic_comm_init", "uniform") == "zeros":
            self._zero_init_communication()

    def _build_scheduler(self, input_dim):
        hidden_dim = max(input_dim // 2, 1)
        bottleneck_dim = max(input_dim // 8, 1)
        return nn.Sequential(
            nn.Linear(input_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, bottleneck_dim),
            nn.ReLU(),
            nn.Linear(bottleneck_dim, 2),
        )

    def _zero_init_communication(self):
        if self.message_encoder_enabled:
            nn.init.zeros_(self.message_encoder.weight)
            nn.init.zeros_(self.message_encoder.bias)
        if self.message_decoder_enabled:
            nn.init.zeros_(self.message_decoder.weight)
            nn.init.zeros_(self.message_decoder.bias)
        if hasattr(self, "sub_scheduler_mlp1"):
            self.sub_scheduler_mlp1.apply(self._zero_init_linear)
        if hasattr(self, "sub_scheduler_mlp2"):
            self.sub_scheduler_mlp2.apply(self._zero_init_linear)

    @staticmethod
    def _zero_init_linear(module):
        if isinstance(module, nn.Linear):
            nn.init.zeros_(module.weight)
            nn.init.zeros_(module.bias)

    def init_hidden(self):
        return (
            self.obs_encoder.weight.new(1, self.hidden_dim).zero_(),
            self.obs_encoder.weight.new(1, self.hidden_dim).zero_()
        )

    def forward(self, inputs, hidden_state):
        batch_size = hidden_state[0].shape[0]
        if inputs.dim() == 2:
            inputs = inputs.view(batch_size, self.n_agents, -1)
        _, n_agents, _ = inputs.shape
        assert n_agents == self.n_agents, f"'n_agents': {n_agents}"

        encoded_obs = self.obs_encoder(inputs).reshape(batch_size * n_agents, self.hidden_dim)
        h, c = hidden_state
        h = h.reshape(batch_size * n_agents, self.hidden_dim)
        c = c.reshape(batch_size * n_agents, self.hidden_dim)
        h, c = self.lstm_cell(encoded_obs, (h, c))

        h_batched = h.view(batch_size, n_agents, self.hidden_dim)
        comm_outputs = []
        for batch_idx in range(batch_size):
            comm_outputs.append(self._communicate(h_batched[batch_idx]))
        comm = th.stack(comm_outputs, dim=0)

        features = th.cat([h_batched, comm], dim=-1)
        logits = self.action_head(features)
        hidden_state = (
            h.view(batch_size, n_agents, self.hidden_dim),
            c.view(batch_size, n_agents, self.hidden_dim)
        )
        critic_inputs = (hidden_state[0], comm)

        return logits.view(batch_size * n_agents, -1), hidden_state, critic_inputs

    def _communicate(self, hidden_state):
        agent_mask = hidden_state.new_ones(self.n_agents, 1)
        if self.comm_mask_zero:
            agent_mask = agent_mask * 0.0

        comm = hidden_state
        if self.message_encoder_enabled:
            comm = self.message_encoder(comm)
        comm = comm * agent_mask
        comm_original = comm.clone()

        if self.first_graph_complete:
            adj1 = self._complete_graph(agent_mask)
            encoded_state1 = None
        elif self.use_gat_encoder:
            adj_complete = self._complete_graph(agent_mask)
            encoded_state1 = self.gat_encoder(comm, adj_complete)
            adj1 = self._sub_scheduler(self.sub_scheduler_mlp1, encoded_state1, agent_mask)
        else:
            encoded_state1 = None
            adj1 = self._sub_scheduler(self.sub_scheduler_mlp1, comm, agent_mask)

        comm = F.elu(self.sub_processor1(comm, adj1))

        if self.learn_second_graph and not self.second_graph_complete:
            if self.use_gat_encoder:
                if encoded_state1 is None:
                    encoded_state2 = self.gat_encoder(comm_original, self._complete_graph(agent_mask))
                else:
                    encoded_state2 = encoded_state1
                adj2 = self._sub_scheduler(self.sub_scheduler_mlp2, encoded_state2, agent_mask)
            else:
                adj2 = self._sub_scheduler(self.sub_scheduler_mlp2, comm_original, agent_mask)
        elif not self.learn_second_graph and not self.second_graph_complete:
            adj2 = adj1
        else:
            adj2 = self._complete_graph(agent_mask)

        comm = self.sub_processor2(comm, adj2)
        comm = comm * agent_mask
        if self.message_decoder_enabled:
            comm = self.message_decoder(comm)
        return comm

    def _sub_scheduler(self, scheduler, hidden_state, agent_mask):
        n_agents = self.n_agents
        hidden_dim = hidden_state.size(-1)
        pair_inputs = th.cat(
            [
                hidden_state.repeat(1, n_agents).view(n_agents * n_agents, -1),
                hidden_state.repeat(n_agents, 1),
            ],
            dim=1,
        ).view(n_agents, n_agents, 2 * hidden_dim)

        if self.directed:
            hard_attention = F.gumbel_softmax(scheduler(pair_inputs), hard=True, dim=-1)
        else:
            symmetric_logits = 0.5 * scheduler(pair_inputs) + 0.5 * scheduler(pair_inputs.permute(1, 0, 2))
            hard_attention = F.gumbel_softmax(symmetric_logits, hard=True, dim=-1)

        adj = hard_attention[:, :, 1]
        agent_mask = agent_mask.expand(n_agents, n_agents)
        return adj * agent_mask * agent_mask.transpose(0, 1)

    def _complete_graph(self, agent_mask):
        n_agents = self.n_agents
        adj = agent_mask.new_ones(n_agents, n_agents)
        agent_mask = agent_mask.expand(n_agents, n_agents)
        return adj * agent_mask * agent_mask.transpose(0, 1)
