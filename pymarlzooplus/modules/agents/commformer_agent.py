import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


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
        nn.init.constant_(self.relation_in_proj.bias, 0.0)
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
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.fc1.weight, std=0.02)
        nn.init.normal_(self.fc2.weight, std=0.02)
        nn.init.constant_(self.fc1.bias, 0.0)
        nn.init.constant_(self.fc2.bias, 0.0)

    def forward(self, x, relation, kv=None, attn_mask=None, need_weights=False, dec_agent=False):
        residual = x
        if kv is None:
            x, _ = self.self_attn(x, x, x, relation, attn_mask=attn_mask, need_weights=need_weights,
                                  dec_agent=dec_agent)
        else:
            x, _ = self.self_attn(x, kv, kv, relation, attn_mask=attn_mask, need_weights=need_weights,
                                  dec_agent=dec_agent)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.attn_layer_norm(residual + x)

        residual = x
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.ff_layer_norm(residual + x)
        return x


class DecodeBlock(nn.Module):
    def __init__(self, n_embd, n_head, n_agent, self_loop_add):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.ln3 = nn.LayerNorm(n_embd)
        self.attn1 = RelationMultiheadAttention(n_embd, n_head, n_agent, masked=True, self_loop_add=self_loop_add)
        self.attn2 = RelationMultiheadAttention(n_embd, n_head, n_agent, masked=True, self_loop_add=self_loop_add)
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, n_embd),
            nn.GELU(),
            nn.Linear(n_embd, n_embd)
        )

    def forward(self, x, rep_enc, relation_embed, attn_mask, dec_agent):
        bs, n_agent, _ = x.shape
        x_back = x.permute(1, 0, 2).contiguous()
        if relation_embed is not None:
            relations_back = relation_embed.permute(1, 2, 0, 3).contiguous()
        else:
            relations_back = relation_embed
        attn_mask_back = attn_mask.permute(1, 2, 0).contiguous()
        y, _ = self.attn1(x_back, x_back, x_back, relations_back, attn_mask=attn_mask_back, dec_agent=dec_agent)
        y = y.permute(1, 0, 2).contiguous()
        x = self.ln1(x + y)

        rep_enc_back = rep_enc.permute(1, 0, 2).contiguous()
        x_back = x.permute(1, 0, 2).contiguous()
        y, _ = self.attn2(rep_enc_back, x_back, x_back, relations_back, attn_mask=attn_mask_back, dec_agent=dec_agent)
        y = y.permute(1, 0, 2).contiguous()
        x = self.ln2(rep_enc + y)
        x = self.ln3(x + self.mlp(x))
        return x


class Decoder(nn.Module):
    def __init__(self, obs_dim, action_dim, n_block, n_embd, n_head, n_agent, self_loop_add=True,dec_agent=True, share_actor=False):
        super().__init__()
        self.action_dim = action_dim
        self.n_embd = n_embd
        self.dec_agent=dec_agent
        self.share_actor=share_actor
        self.n_agent = n_agent

        self.action_encoder = nn.Sequential(self.init_(nn.Linear(action_dim + 1, n_embd, bias=False), activate=True),nn.GELU())

        self.obs_encoder = nn.Sequential( nn.LayerNorm(obs_dim),self.init_(nn.Linear(obs_dim, n_embd), activate=True),nn.GELU())

        self.ln = nn.LayerNorm(n_embd)

        self.blocks = nn.Sequential(*[DecodeBlock(n_embd, n_head, n_agent, self_loop_add) for _ in range(n_block)])

        if self.dec_agent:
            if self.share_actor:
                self.mlp = nn.Sequential(
                    nn.LayerNorm(n_embd),
                    self.init_(nn.Linear(n_embd, n_embd), activate=True),
                    nn.GELU(),
                    nn.LayerNorm(n_embd),
                    self.init_(nn.Linear(n_embd, n_embd), activate=True),
                    nn.GELU(),
                    nn.LayerNorm(n_embd),
                    self.init_(nn.Linear(n_embd, action_dim))
                )
            else:
                self.mlp = nn.ModuleList()
                for _ in range(n_agent):
                    self.mlp.append(nn.Sequential(
                        nn.LayerNorm(n_embd),
                        self.init_(nn.Linear(n_embd, n_embd), activate=True),
                        nn.GELU(),
                        nn.LayerNorm(n_embd),
                        self.init_(nn.Linear(n_embd, action_dim))
                    ))
        else:
            self.head = nn.Sequential(
                self.init_(nn.Linear(n_embd, n_embd), activate=True),
                nn.GELU(),
                nn.LayerNorm(n_embd),
                self.init_(nn.Linear(n_embd, action_dim))
            )

    def forward(self, action, obs_rep, obs, relation_embed, attn_mask):
        action_embeddings = self.action_encoder(action)
        obs_embeddings = self.obs_encoder(obs)
        x = self.ln(action_embeddings + obs_embeddings)

        for block in self.blocks:
            x = block(x, obs_rep, relation_embed, attn_mask, self.dec_agent)

        if self.dec_agent:
            if self.share_actor:
                logit = self.mlp(x)
            else:
                logit = torch.stack([self.mlp[i](x[:, i]) for i in range(self.n_agent)], dim=1)
        else:
            logit = self.head(x)

        return logit

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


class CommFormerAgent(nn.Module):
    def __init__(self, input_shape, args):
        super().__init__()
        self.args = args
        assert isinstance(input_shape, int), "CommFormer does not support image obs for the time being!"
        self.n_agent = args.n_agents
        self.input_shape = input_shape
        self.action_dim = args.n_actions
        self.n_embd = args.n_embd
        self.n_head = args.n_head
        self.n_block = args.n_block

        self.share_actor=args.share_actor
        self.dec_agent=args.dec_agent
        self.decoder = Decoder(input_shape, self.action_dim, self.n_block, self.n_embd, self.n_head, self.n_agent,dec_agent=self.dec_agent,share_actor=self.share_actor)

        self.sparsity = args.sparsity
        self.topk = max(int(self.n_agent * self.sparsity), 1)


        self.critic = None
        self.device = None

    def model_parameters(self):
        return list(self.parameters())


    def discrete_autoregreesive_act(self, obs_rep, obs, relations_embed, relations, batch_size, available_actions=None,
                                    deterministic=False):
        shifted_action = torch.zeros((batch_size, self.n_agent, self.action_dim + 1), device=obs_rep.device)
        shifted_action[:, 0, 0] = 1
        output_action = torch.zeros((batch_size, self.n_agent, 1), dtype=torch.long, device=obs_rep.device)
        output_action_log = torch.zeros_like(output_action, dtype=torch.float32)
        for i in range(self.n_agent):
            logit = self.decoder(shifted_action, obs_rep, obs, relations_embed, attn_mask=relations)[:, i, :]
            if available_actions is not None:
                logit[available_actions[:, i, :] == 0] = -1e10
            distri = Categorical(logits=logit)
            action = distri.probs.argmax(dim=-1) if deterministic else distri.sample()
            action_log = distri.log_prob(action)
            output_action[:, i, :] = action.unsqueeze(-1)
            output_action_log[:, i, :] = action_log.unsqueeze(-1)
            if i + 1 < self.n_agent:
                shifted_action[:, i + 1, 1:] = F.one_hot(action, num_classes=self.action_dim)
        return output_action, output_action_log

    def discrete_parallel_act(self, obs_rep, obs, action, relation_embed, relations, batch_size, available_actions=None,):
        one_hot_action = F.one_hot(action.squeeze(-1), num_classes=self.action_dim)
        shifted_action = torch.zeros((batch_size, self.n_agent, self.action_dim + 1), device=obs_rep.device)
        shifted_action[:, 0, 0] = 1
        shifted_action[:, 1:, 1:] = one_hot_action[:, :-1, :]
        logit = self.decoder(shifted_action, obs_rep, obs, relation_embed, attn_mask=relations)
        if available_actions is not None:
            logit[available_actions == 0] = -1e10
        distri = Categorical(logits=logit)
        action_log = distri.log_prob(action.squeeze(-1)).unsqueeze(-1)
        entropy = distri.entropy().unsqueeze(-1)
        return action_log, entropy

    def forward(self, obs_rep, obs, action, available_actions=None, steps=0, total_step=0):
        relations = self.critic.edge_return(steps=steps, total_step=total_step)
        relations = relations.unsqueeze(0)
        relations_embed = self.critic.edges_embed(relations.long()).repeat(obs.size(0), 1, 1, 1)

        action_log, entropy = self.discrete_parallel_act(
            obs_rep, obs, action, relations_embed, relations, obs.size(0), available_actions
        )
        return action_log, entropy

    def get_actions(self, ep_batch, t, obs, available_actions=None, deterministic=False):

        batch_size = np.shape(obs)[0]
        v_loc, obs_rep = self.critic(ep_batch, t)

        relations = self.critic.edge_return(exact=True)
        relations = relations.unsqueeze(0)
        relations_embed = self.critic.edges_embed(relations.long()).repeat(batch_size, 1, 1, 1)

        output_action, output_action_log = self.discrete_autoregreesive_act(
            obs_rep, obs, relations_embed, relations, batch_size, available_actions, deterministic
        )

        return output_action, output_action_log,v_loc

    def get_values(self, obs):
        v_tot = self.critic(obs)
        return v_tot

    def evaluate_actions(self, ep_batch, t, agent_inputs, actions, available_actions, steps=0, total_step=0):
        v_loc, obs_rep = self.critic(ep_batch, t)

        action_log, entropy = self.forward(
            obs_rep, agent_inputs, actions, available_actions, steps=steps, total_step=total_step
        )

        return action_log, v_loc, entropy