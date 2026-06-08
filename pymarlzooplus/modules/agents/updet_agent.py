import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttention(nn.Module):
    def __init__(self, emb, heads=8):
        super().__init__()
        self.emb = emb
        self.heads = heads
        self.tokeys = nn.Linear(emb, emb * heads, bias=False)
        self.toqueries = nn.Linear(emb, emb * heads, bias=False)
        self.tovalues = nn.Linear(emb, emb * heads, bias=False)
        self.unifyheads = nn.Linear(heads * emb, emb)

    def forward(self, x):
        b, t, e = x.size()
        h = self.heads
        keys = self.tokeys(x).view(b, t, h, e)
        queries = self.toqueries(x).view(b, t, h, e)
        values = self.tovalues(x).view(b, t, h, e)

        keys = keys.transpose(1, 2).contiguous().view(b * h, t, e)
        queries = queries.transpose(1, 2).contiguous().view(b * h, t, e)
        values = values.transpose(1, 2).contiguous().view(b * h, t, e)

        queries = queries / (e ** (1 / 4))
        keys = keys / (e ** (1 / 4))

        dot = torch.bmm(queries, keys.transpose(1, 2))
        assert dot.size() == (b * h, t, t)

        dot = F.softmax(dot, dim=2)
        out = torch.bmm(dot, values).view(b, h, t, e)
        out = out.transpose(1, 2).contiguous().view(b, t, h * e)
        return self.unifyheads(out)


class TransformerBlock(nn.Module):
    def __init__(self, emb, heads, ff_hidden_mult=4, dropout=0.0):
        super().__init__()
        self.attention = SelfAttention(emb, heads=heads)
        self.norm1 = nn.LayerNorm(emb)
        self.norm2 = nn.LayerNorm(emb)
        self.ff = nn.Sequential(
            nn.Linear(emb, ff_hidden_mult * emb),
            nn.ReLU(),
            nn.Linear(ff_hidden_mult * emb, emb)
        )
        self.do = nn.Dropout(dropout)

    def forward(self, x):
        attended = self.attention(x)
        x = self.norm1(attended + x)
        x = self.do(x)
        fedforward = self.ff(x)
        x = self.norm2(fedforward + x)
        x = self.do(x)
        return x


class Transformer(nn.Module):
    def __init__(self, input_dim, emb, heads, depth, output_dim):
        super().__init__()
        self.num_tokens = output_dim
        self.token_embedding = nn.Linear(input_dim, emb)
        tblocks = [TransformerBlock(emb=emb, heads=heads) for _ in range(depth)]
        self.tblocks = nn.Sequential(*tblocks)
        self.toprobs = nn.Linear(emb, output_dim)

    def forward(self, x, h):
        tokens = self.token_embedding(x)
        tokens = torch.cat((tokens, h), 1)
        b, t, e = tokens.size()
        x = self.tblocks(tokens)
        x = self.toprobs(x.view(b * t, e)).view(b, t, self.num_tokens)
        return x, tokens


class UPDeT(nn.Module):
    # Adapted from the original UPDeT (SMAC) for gymma environments (LBF, RWARE, MPE).
    #
    # What was removed vs. the original and why:
    #   - Policy decoupling (separate heads for ally/enemy/self actions): gymma envs have
    #     homogeneous agents with a shared, fixed action space and no opponent agents, so
    #     per-entity action heads are meaningless.
    #   - ally_num / enemy_num inputs: entity counts are not needed, the number of tokens
    #     is derived automatically as obs_dim // token_dim.
    #   - Aggregation (mean-pool) forward: the paper shows Aggregation Transformer < GRU,
    #     so we use the self-token output only.
    #   - PRESERVE and ABANDON (paper §4.2): these modes carry entity token representations
    #     across timesteps. Dropped because gymma envs use a flat obs vector with no explicit
    #     entity structure that benefits from per-entity recurrence — the single recurrent
    #     hidden-state token (index -1) is sufficient.
    #
    # What we keep:
    #   - Self-token output: after attention, token[0] (the agent's own token) aggregates
    #     context from all other tokens and is passed through a linear layer to produce Q-values.
    def __init__(self, input_shape, args):
        super(UPDeT, self).__init__()
        self.args = args
        self.transformer = Transformer(input_shape, args.emb, args.heads, args.depth, args.emb)
        self.q_linear = nn.Linear(args.emb, args.n_actions)

    def init_hidden(self):
        return self.transformer.token_embedding.weight.new(1, self.args.emb).zero_()

    def forward(self, inputs, hidden_state):
        outputs, _ = self.transformer.forward(inputs, hidden_state)

        # Self token (index 0) is enriched via attention over all entity tokens.
        q = self.q_linear(outputs[:, 0, :])   # (batch, n_actions)
        h = outputs[:, -1:, :]                 # (batch, 1, emb) — recurrent hidden state

        return q, h
