import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class RelationMultiheadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, n_agent, dropout=0., weights_dropout=False,
                 masked=False, self_loop_add=True):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim
        self.scaling = self.head_dim ** -0.5
        self.masked = masked
        self.self_loop_add = self_loop_add

        self.in_proj_weight = nn.Parameter(torch.Tensor(3 * embed_dim, embed_dim))
        self.in_proj_bias = nn.Parameter(torch.Tensor(3 * embed_dim))
        self.relation_in_proj = nn.Linear(embed_dim, 2 * embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.weights_dropout = weights_dropout
        self.register_buffer("mask", torch.tril(torch.ones(n_agent, n_agent)) == 0)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.in_proj_weight, std=0.02)
        nn.init.normal_(self.out_proj.weight, std=0.02)
        nn.init.normal_(self.relation_in_proj.weight, std=0.02)
        nn.init.constant_(self.in_proj_bias, 0.)
        nn.init.constant_(self.out_proj.bias, 0.)

    def forward(self, query, key, value, relation, attn_mask=None, need_weights=False, dec_agent=False):
        qkv_same = query.data_ptr() == key.data_ptr() == value.data_ptr()
        kv_same = key.data_ptr() == value.data_ptr()

        tgt_len, bsz, embed_dim = query.size()
        src_len = key.size(0)
        assert key.size() == value.size()

        if qkv_same:
            q, k, v = self.in_proj_qkv(query)
        elif kv_same:
            q = self.in_proj_q(query)
            k, v = self.in_proj_kv(key)
        else:
            q = self.in_proj_q(query)
            k = self.in_proj_k(key)
            v = self.in_proj_v(value)

        q = q.contiguous().view(tgt_len, bsz * self.num_heads, self.head_dim)
        k = k.contiguous().view(src_len, bsz * self.num_heads, self.head_dim)
        v = v.contiguous().view(src_len, bsz * self.num_heads, self.head_dim)

        if relation is None:
            attn_weights = torch.einsum('ibn,jbn->ijb', [q, k]) * (1.0 / math.sqrt(k.size(-1)))
        else:
            rel = self.relation_in_proj(relation)
            ra, rb = rel.chunk(2, dim=-1)
            ra = ra.contiguous().view(tgt_len, src_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
            rb = rb.contiguous().view(tgt_len, src_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
            q = q.unsqueeze(1) + ra
            k = k.unsqueeze(0) + rb
            q *= self.scaling
            attn_weights = torch.einsum('ijbn,ijbn->ijb', [q, k]) * (1.0 / math.sqrt(k.size(-1)))

        if self.masked:
            attn_weights.masked_fill_(self.mask.unsqueeze(-1), float('-inf'))

        if dec_agent:
            self_loop = torch.eye(tgt_len).unsqueeze(-1).long().to(attn_weights.device)
            if self.self_loop_add:
                attn_mask = attn_mask + self_loop
            else:
                attn_mask = attn_mask * (1 - self_loop) + self_loop
            attn_weights.masked_fill_(attn_mask == 0, float('-inf'))
            attn_weights = F.softmax(attn_weights, dim=1)
            attn_weights = attn_weights * attn_mask
        else:
            attn_weights = F.softmax(attn_weights, dim=1)

        if self.weights_dropout:
            attn_weights = F.dropout(attn_weights, p=self.dropout, training=self.training)

        attn = torch.einsum('ijb,jbn->bin', [attn_weights, v])
        if not self.weights_dropout:
            attn = F.dropout(attn, p=self.dropout, training=self.training)

        attn = attn.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        attn = self.out_proj(attn)

        if need_weights:
            attn_weights = attn_weights.view(tgt_len, src_len, bsz, self.num_heads)
        else:
            attn_weights = None
        return attn, attn_weights

    def in_proj_qkv(self, query):
        return self._in_proj(query).chunk(3, dim=-1)

    def in_proj_kv(self, key):
        return self._in_proj(key, start=self.embed_dim).chunk(2, dim=-1)

    def in_proj_q(self, query):
        return self._in_proj(query, end=self.embed_dim)

    def in_proj_k(self, key):
        return self._in_proj(key, start=self.embed_dim, end=2 * self.embed_dim)

    def in_proj_v(self, value):
        return self._in_proj(value, start=2 * self.embed_dim)

    def _in_proj(self, input, start=0, end=None):
        weight = self.in_proj_weight[start:end, :]
        bias = self.in_proj_bias[start:end]
        return F.linear(input, weight, bias)


class GraphTransformerLayer(nn.Module):
    def __init__(self, embed_dim, ff_embed_dim, num_heads, n_agent, self_loop_add, dropout=0.1,
                 weights_dropout=False, masked=False):
        super().__init__()
        self.self_attn = RelationMultiheadAttention(embed_dim, num_heads, n_agent, dropout,
                                                    weights_dropout, masked, self_loop_add)
        self.fc1 = nn.Linear(embed_dim, ff_embed_dim)
        self.fc2 = nn.Linear(ff_embed_dim, embed_dim)
        self.attn_layer_norm = nn.LayerNorm(embed_dim)
        self.ff_layer_norm = nn.LayerNorm(embed_dim)
        self.dropout = dropout

    def forward(self, x, relation, kv=None, attn_mask=None, need_weights=False, dec_agent=False):
        residual = x
        if kv is None:
            x, _ = self.self_attn(x, x, x, relation, attn_mask=attn_mask, need_weights=need_weights, dec_agent=dec_agent)
        else:
            x, _ = self.self_attn(x, kv, kv, relation, attn_mask=attn_mask, need_weights=need_weights, dec_agent=dec_agent)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.attn_layer_norm(residual + x)

        residual = x
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.ff_layer_norm(residual + x)
        return x


class EncodeBlock(nn.Module):
    def __init__(self, n_embd, n_head, n_agent):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.attn = RelationMultiheadAttention(n_embd, n_head, n_agent, masked=False)
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, n_embd),
            nn.GELU(),
            nn.Linear(n_embd, n_embd)
        )

    def forward(self, x, relation, attn_mask, dec_agent=False):
        x_back = x.permute(1, 0, 2)
        relation_back = relation.permute(1, 2, 0, 3)
        attn_mask_back = attn_mask.permute(1, 2, 0)
        y, _ = self.attn(x_back, x_back, x_back, relation_back, attn_mask_back, dec_agent=dec_agent)
        y = y.permute(1, 0, 2)
        x = self.ln1(x + y)
        x = self.ln2(x + self.mlp(x))
        return x


class Encoder(nn.Module):
    def __init__(self, obs_dim, n_block, n_embd, n_head, n_agent, self_loop_add=True):
        super().__init__()
        self.obs_encoder = nn.Sequential(nn.LayerNorm(obs_dim), nn.Linear(obs_dim, n_embd), nn.GELU())
        self.ln = nn.LayerNorm(n_embd)
        self.blocks = nn.ModuleList([GraphTransformerLayer(n_embd, n_embd, n_head, n_agent, self_loop_add) for _ in range(n_block)])
        self.head = nn.Sequential(nn.Linear(n_embd, n_embd), nn.GELU(), nn.LayerNorm(n_embd), nn.Linear(n_embd, 1))

    def forward(self, obs, relation_embed, attn_mask, dec_agent):
        obs_embeddings = self.obs_encoder(obs)
        x = self.ln(obs_embeddings)
        x = x.permute(1, 0, 2)
        relation = relation_embed.permute(1, 2, 0, 3) if relation_embed is not None else relation_embed
        attn_mask = attn_mask.permute(1, 2, 0)
        for layer in self.blocks:
            x = layer(x, relation, attn_mask=attn_mask, dec_agent=dec_agent)
        rep = x.permute(1, 0, 2)
        v = self.head(rep)
        return v, rep


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
        print("scheme is ", scheme)
        self.input_shape = self._get_input_shape(scheme)
        print("input shape is ", self.input_shape)
        self.n_block = args.n_block
        self.n_embd = args.n_embd
        self.n_head = args.n_head
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
        relations = relations.unsqueeze(0).repeat(inputs.size(0), 1, 1)
        relations_embed = self.edges_embed(relations.long())
        v, obs_rep = self.encoder(inputs, relations_embed, relations, dec_agent=False)
        return v, obs_rep

    def edge_return(self, exact=False, topk=-1):
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
        inputs = []
        # global state repeated for each agent
        inputs.append(batch["state"][:, ts].unsqueeze(2).repeat(1, 1, self.n_agents, 1))
        # individual observations
        inputs.append(batch["obs"][:, ts])
        actions = batch["actions_onehot"][ :, ts].view(bs, max_t, 1, -1).repeat(1, 1, self.n_agents, 1)
        agent_mask = (1 - torch.eye(self.n_agents, device=batch["device"]))
        agent_mask = agent_mask.view(-1, 1).repeat(1, self.n_actions).view(self.n_agents, -1)
        inputs.append(actions * agent_mask.unsqueeze(0).unsqueeze(0))
        if self.args.obs_last_action:
            if t == 0:
                inputs.append(torch.zeros_like(batch["actions_onehot"][ :, 0:1]).view(bs, max_t, 1, -1).repeat(1, 1, self.n_agents, 1))
            elif isinstance(t, int):
                inputs.append(batch["actions_onehot"][ :, slice(t-1, t)].view(bs, max_t, 1, -1).repeat(1, 1, self.n_agents, 1))
            else:
                last_actions = torch.cat([torch.zeros_like(batch["actions_onehot"][ :,0:1]), batch["actions_onehot"][ :, :-1]], dim=1)
                last_actions = last_actions.view(bs, max_t, 1, -1).repeat(1, 1, self.n_agents, 1)
                inputs.append(last_actions)
        if self.args.obs_agent_id:
            inputs.append(torch.eye(self.n_agents, device=batch["device"]).unsqueeze(0).unsqueeze(0).expand(bs, max_t, -1, -1))
        inputs = torch.cat([x.reshape(bs, max_t, self.n_agents, -1) for x in inputs], dim=-1)
        return inputs, bs, max_t

    def _get_input_shape(self, scheme):
        # This function calculates the size of the feature vector for a single agent.
        # It must match the concatenation order and dimensions in _build_inputs.

        # 1. Global State
        input_shape = scheme["state"]["vshape"]

        # 2. Individual Observation
        if self.args.obs_individual_obs:
            input_shape += scheme["obs"]["vshape"]

        # 3. Other Agents' Actions
        input_shape += scheme["actions_onehot"]["vshape"][0] * self.n_agents

        # 4. Last Action
        if self.args.obs_last_action:
            input_shape += scheme["actions_onehot"]["vshape"][0]

        # 5. Agent ID
        if self.args.obs_agent_id:
            input_shape += self.n_agents

        return input_shape