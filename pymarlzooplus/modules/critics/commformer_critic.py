import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pymarlzooplus.modules.agents.commformer_agent import RelationMultiheadAttention, GraphTransformerLayer




class Encoder(nn.Module):

    def __init__(self, obs_dim, n_block, n_embd, n_head, n_agent, self_loop_add=True):
        super().__init__()

        self.obs_encoder = nn.Sequential(
            nn.LayerNorm(obs_dim),
            self.init_(nn.Linear(obs_dim, n_embd), activate=True),
            nn.GELU()
        )
        self.ln = nn.LayerNorm(n_embd)

        self.blocks = nn.ModuleList(
            [GraphTransformerLayer(n_embd, n_embd, n_head, n_agent, self_loop_add) for _ in range(n_block)])

        self.head = nn.Sequential(
            self.init_(nn.Linear(n_embd, n_embd), activate=True),
            nn.GELU(),
            nn.LayerNorm(n_embd),
            self.init_(nn.Linear(n_embd, 1))
        )

    def forward(self, obs, relation_embed, attn_mask, dec_agent):
        obs_embeddings = self.obs_encoder(obs)
        x = self.ln(obs_embeddings)
        x = x.permute(1, 0, 2).contiguous()
        relation = relation_embed.permute(1, 2, 0, 3).contiguous() if relation_embed is not None else relation_embed
        attn_mask = attn_mask.permute(1, 2, 0).contiguous()
        for layer in self.blocks:
            x = layer(x, relation, attn_mask=attn_mask, dec_agent=dec_agent)
        rep = x.permute(1, 0, 2).contiguous()
        v = self.head(rep)
        return v, rep

    def init(self,module, weight_init, bias_init, gain=1):
        weight_init(module.weight.data, gain=gain)
        if module.bias is not None:
            bias_init(module.bias.data)
        return module

    def init_(self,m, gain=0.01, activate=False):
        if activate:
            gain = nn.init.calculate_gain('relu')
        return self.init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), gain=gain)


def gumbel_softmax_topk(logits, topk=1, tau=1, hard=False, dim=-1):
    gumbels = -torch.empty_like(logits).exponential_().log()
    gumbels = (logits + gumbels) / tau
    y_soft = gumbels.softmax(dim)
    if hard:
        index = y_soft.topk(k=topk, dim=dim)[1]
        y_hard = torch.zeros_like(logits).scatter_(dim, index, 1.0)
        ret = y_hard - y_soft.detach() + y_soft
    else:
        ret = y_soft
    return ret


class CommFormerCritic(nn.Module):
    def __init__(self, scheme, args):
        super().__init__()
        self.args = args
        self.n_actions = args.n_actions
        self.n_agents = args.n_agents
        self.input_shape = self._get_input_shape(scheme)
        self.n_block = args.n_block
        self.n_embd = args.n_embd
        self.n_head = args.n_head
        self.warmup= args.warmup
        self.post_stable=args.post_stable
        self.post_ratio=args.post_ratio
        self.dec_agent = args.dec_agent
        self.encoder = Encoder(self.input_shape, self.n_block, self.n_embd, self.n_head, self.n_agents)
        self.edges = nn.Parameter(torch.ones(self.n_agents, self.n_agents), requires_grad=True)
        self.edges_embed = nn.Embedding(2, self.n_embd)


    def model_parameters(self):
        return [p for name, p in self.named_parameters() if name != "edges"]

    def edge_parameters(self):
        return [self.edges]

    def forward(self, batch, t=None):
        inputs, bs, max_t = self._build_inputs(batch, t)
        inputs = inputs.reshape(-1, self.n_agents, self.input_shape)
        relations = self.edge_return(exact=True)
        relations = relations.unsqueeze(0)
        relations_embed = self.edges_embed(relations.long())
        relations_embed = relations_embed.repeat(inputs.size(0), 1, 1, 1)
        v, obs_rep = self.encoder(inputs, relations_embed, relations, dec_agent=self.dec_agent)
        return v, obs_rep

    def edge_return(self, exact=False, topk=-1, steps=None, total_step=None):
        # Warmup and post-stable logic
        if steps is not None:
            if steps <= self.warmup:
                return self.edges
            if self.post_stable and total_step is not None and steps > int(self.post_ratio * total_step):
                exact = True

        edges = self.edges
        if not exact:
            relations = gumbel_softmax_topk(edges, topk=max(int(self.n_agents * 0.4), 1), hard=True, dim=-1)
        else:
            y_soft = edges.softmax(dim=-1)
            index = edges.topk(k=max(int(self.n_agents * 0.4), 1), dim=-1)[1]
            relations = torch.zeros_like(edges).scatter_(-1, index, 1.0)
            relations = relations - y_soft.detach() + y_soft

        if topk != -1:
            y_soft = edges.softmax(dim=-1)
            index = edges.topk(k=topk, dim=-1)[1]
            relations = torch.zeros_like(edges).scatter_(-1, index, 1.0)
            relations = relations - y_soft.detach() + y_soft

        return relations

    def _build_inputs(self, batch, t=None):
        bs = batch["batch_size"]
        max_t = batch["max_seq_length"] if t is None else 1
        ts = slice(None) if t is None else slice(t, t + 1)

        inputs = [batch["obs"][:, ts]]

        inputs = torch.cat([x.reshape(bs, max_t, self.n_agents, -1) for x in inputs], dim=-1)
        return inputs, bs, max_t

    def _get_input_shape(self, scheme):

        return scheme["obs"]["vshape"]