import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttention(nn.Module):
    def __init__(self, emb, heads=8, mask=False):
        super().__init__()
        self.emb = emb
        self.heads = heads
        self.mask = mask
        self.tokeys = nn.Linear(emb, emb * heads, bias=False)
        self.toqueries = nn.Linear(emb, emb * heads, bias=False)
        self.tovalues = nn.Linear(emb, emb * heads, bias=False)
        self.unifyheads = nn.Linear(heads * emb, emb)

    def forward(self, x, mask):
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

        if mask is not None:
            dot = dot.masked_fill(mask == 0, -1e9)

        dot = F.softmax(dot, dim=2)
        out = torch.bmm(dot, values).view(b, h, t, e)
        out = out.transpose(1, 2).contiguous().view(b, t, h * e)
        return self.unifyheads(out)


class TransformerBlock(nn.Module):
    def __init__(self, emb, heads, mask, ff_hidden_mult=4, dropout=0.0):
        super().__init__()
        self.attention = SelfAttention(emb, heads=heads, mask=mask)
        self.norm1 = nn.LayerNorm(emb)
        self.norm2 = nn.LayerNorm(emb)
        self.ff = nn.Sequential(
            nn.Linear(emb, ff_hidden_mult * emb),
            nn.ReLU(),
            nn.Linear(ff_hidden_mult * emb, emb)
        )
        self.do = nn.Dropout(dropout)

    def forward(self, x_mask):
        x, mask = x_mask
        attended = self.attention(x, mask)
        x = self.norm1(attended + x)
        x = self.do(x)
        fedforward = self.ff(x)
        x = self.norm2(fedforward + x)
        x = self.do(x)
        return x, mask


class Transformer(nn.Module):
    def __init__(self, input_dim, emb, heads, depth, output_dim):
        super().__init__()
        self.num_tokens = output_dim
        self.token_embedding = nn.Linear(input_dim, emb)
        tblocks = [TransformerBlock(emb=emb, heads=heads, mask=False) for _ in range(depth)]
        self.tblocks = nn.Sequential(*tblocks)
        self.toprobs = nn.Linear(emb, output_dim)

    def forward(self, x, h, mask):
        tokens = self.token_embedding(x)
        tokens = torch.cat((tokens, h), 1)
        b, t, e = tokens.size()
        x, mask = self.tblocks((tokens, mask))
        x = self.toprobs(x.view(b * t, e)).view(b, t, self.num_tokens)
        return x, tokens


class UPDeT(nn.Module):
    def __init__(self, input_shape, args):
        super(UPDeT, self).__init__()
        self.args = args
        self.transformer = Transformer(args.token_dim, args.emb, args.heads, args.depth, args.emb)
        self.q_linear = nn.Linear(args.emb, args.n_actions)

    def init_hidden(self):
        return self.transformer.token_embedding.weight.new(1, self.args.emb).zero_()

    def forward(self, inputs, hidden_state):
        outputs, _ = self.transformer.forward(inputs, hidden_state, None)

        # self token — enriched via attention over all entities
        q = self.q_linear(outputs[:, 0, :])   # (batch, n_actions)
        h = outputs[:, -1:, :]                 # (batch, 1, emb)

        return q, h
