import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        rms = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x / rms) * self.weight


def precompute_rope_cache(head_dim, max_seq_len, base=10000.0):
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
    t = torch.arange(max_seq_len).float()
    freqs = torch.outer(t, inv_freq)
    freqs = torch.cat((freqs, freqs), dim=-1)
    cos = freqs.cos()
    sin = freqs.sin()
    return cos, sin


def rotate_half(x):
    x1 = x[..., :x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(q, k, cos, sin):
    q = (q * cos) + (rotate_half(q) * sin)
    k = (k * cos) + (rotate_half(k) * sin)
    return q, k


class Attention(nn.Module):
    def __init__(self, heads=32, dim=4096, dropout=0.1):
        super().__init__()
        self.heads = heads
        self.head_dim = dim // heads
        self.Wq = nn.Linear(dim, dim, bias=False)
        self.Wk = nn.Linear(dim, dim, bias=False)
        self.Wv = nn.Linear(dim, dim, bias=False)
        self.Wout = nn.Linear(dim, dim, bias=False)
        self.attn_drop = nn.Dropout(dropout)

    def forward(self, x, cos, sin, mask=None):
        B, N, C = x.shape
        q = self.Wq(x).view(B, N, self.heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.Wk(x).view(B, N, self.heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.Wv(x).view(B, N, self.heads, self.head_dim).permute(0, 2, 1, 3)

        q, k = apply_rope(q, k, cos[:N], sin[:N])

        scores = q @ k.transpose(-2, -1) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores + mask[:N, :N]

        scores = self.attn_drop(scores.softmax(dim=-1))

        out = scores @ v
        out = out.transpose(1, 2).contiguous().reshape(B, N, C)

        return self.Wout(out)


class SwiGLU(nn.Module):
    def __init__(self, dim=4096, hidden=11008):
        super().__init__()
        self.gate = nn.Linear(dim, hidden, bias=False)
        self.up = nn.Linear(dim, hidden, bias=False)
        self.down = nn.Linear(hidden, dim, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


class DecoderLayer(nn.Module):
    def __init__(self, dim=4096, heads=32, hidden=11008, dropout=0.1):
        super().__init__()
        self.attn_norm = RMSNorm(dim)
        self.attn = Attention(heads=heads, dim=dim, dropout=dropout)
        self.ffn_norm = RMSNorm(dim)
        self.swi_glu = SwiGLU(dim=dim, hidden=hidden)

    def forward(self, x, cos, sin, mask):
        x = x + self.attn(self.attn_norm(x), cos, sin, mask)
        x = x + self.swi_glu(self.ffn_norm(x))
        return x


class LLaMA(nn.Module):
    def __init__(self, dim=4096, heads=32, hidden=11008, depth=4, voc_size=100, max_seq_len=1024, dropout=0.1):
        super().__init__()
        self.token_emb = nn.Embedding(voc_size, dim)
        self.layers = nn.ModuleList([
            DecoderLayer(dim=dim, heads=heads, hidden=hidden, dropout=dropout) for _ in range(depth)
        ])
        self.norm = RMSNorm(dim)
        self.llm_head = nn.Linear(dim, voc_size)

        cos, sin = precompute_rope_cache(dim // heads, max_seq_len)

        self.register_buffer('cos', cos)
        self.register_buffer('sin', sin)

    def forward(self, input_ids=None, input_embeds=None):
        if input_embeds is not None:
            x = input_embeds
        else:
            x = self.token_emb(input_ids)

        B, N, _ = x.shape
        mask = torch.triu(torch.full((N, N), float('-inf'), device=x.device), diagonal=1)

        for layer in self.layers:
            x = layer(x, cos=self.cos, sin=self.sin, mask=mask)

        x = self.norm(x)
        logit = self.llm_head(x)

        return logit


if __name__ == '__main__':
    m = LLaMA()
    ids = torch.randint(0, 100, (2, 64))
    logit = m(input_ids=ids)
    print(logit.shape)
    print("params = ", sum(p.numel() for p in m.parameters()))
