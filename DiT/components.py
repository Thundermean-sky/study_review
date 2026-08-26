import torch
import torch.nn as nn
from config import *
import torch.nn.functional as F
import math


def timestep_embedding(t, dim=FREQ_EMB_SIZE, max_period=10000):
    half = dim // 2

    freqs = torch.exp(-math.log(max_period) * torch.arange(0, half, dtype=torch.float32) / half).to(device=t.device)

    args = t[:, None].float() * freqs[None]

    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)

    return embedding


class TimeEmbed(nn.Module):
    def __init__(self, hidden_size=HIDDEN_SIZE, frequency_embedding_size=FREQ_EMB_SIZE):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )

    def forward(self, t):
        t_freq = timestep_embedding(t)
        return self.mlp(t_freq)


class LabelEmbedding(nn.Module):
    def __init__(self, num_class=NUM_CLASSES, hidden_size=HIDDEN_SIZE, dropout_prob=0.1):
        super().__init__()
        self.embedding = nn.Embedding(num_class + 1, hidden_size)
        self.num_classes = num_class
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
        drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob

        labels = torch.where(drop_ids, self.num_classes, labels)

        return labels

    def forward(self, labels, train=True):
        if train and self.dropout_prob > 0:
            labels = self.token_drop(labels)

        return self.embedding(labels)

class PatchEmbed(nn.Module):
    def __init__(self, in_channels=LATENT_CHANNELS, patch_size=PATCH_SIZE, embed_dim=HIDDEN_SIZE):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)


    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x


def unpatchify(x, patch_size=PATCH_SIZE, in_channels=LATENT_CHANNELS):
    h = w = int(x.shape[1] ** 0.5)
    x = x.reshape(x.shape[0], h, w, patch_size, patch_size, in_channels)
    x = torch.einsum('nhwpqc -> nchwpq', x)
    return x.reshape(x.shape[0], x.shape[1], patch_size*h, patch_size*w)


class Attention(nn.Module):
    def __init__(self, hidden_size=HIDDEN_SIZE, heads=NUM_HEADS):
        super().__init__()
        self.heads = heads
        self.head_dim = hidden_size // heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(hidden_size, hidden_size * 3)
        self.proj = nn.Linear(hidden_size, hidden_size)


    def forward(self, x, mask=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.heads, self.head_dim)  # 先按内存顺序切
        qkv = qkv.permute(2, 0, 3, 1, 4)  # 再把 QKV 轴挪到最前: (3, B, heads, N, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = q @ k.transpose(-2, -1) * self.scale
        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)

        attn = F.softmax(attn, dim=-1)

        out = attn @ v
        out = out.transpose(1, 2).reshape(B, N, C)
        out = self.proj(out)
        return out



def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class DiTBlock(nn.Module):
    def __init__(self, hidden_size=HIDDEN_SIZE, num_heads=NUM_HEADS, mlp_ratio=MLP_RATIO):
        super().__init__()
        self.norm_1 = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.norm_2 = nn.LayerNorm(hidden_size, elementwise_affine=False)

        self.attn = Attention(hidden_size, num_heads)

        mlp_hidden = hidden_size * mlp_ratio

        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden),
            nn.GELU(approximate='tanh'),
            nn.Linear(mlp_hidden, hidden_size)
        )

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size * 6)
        )

        nn.init.constant_(self.adaLN_modulation[1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[1].bias, 0)


    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)

        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm_1(x), shift_msa, scale_msa))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm_2(x), shift_mlp, scale_mlp))

        return x
if __name__ == '__main__':
    # t_emb = TimeEmbed()(torch.randint(0, 1000, (2,)))
    # y_emb = LabelEmbedding()(torch.randint(0, 10, (2,)), train=True)
    # c = t_emb + y_emb
    # print(c.shape)

    # attention = Attention()
    # x = torch.randn(2, NUM_PATCHES, HIDDEN_SIZE)
    # y = attention(x)
    # print(y.shape)
    #
    # import torch.nn.functional as F
    #
    # qkv_w = attention.qkv.weight
    # Q_part = F.linear(x, qkv_w[:256], attention.qkv.bias[:256])
    # print(Q_part.shape)
    # qkv = attention.qkv(x).reshape(2, 256, 3, 8, 32).permute(2, 0, 3, 1, 4)
    # print(qkv.shape)
    # print('q == Q_part :', torch.allclose(qkv[0].transpose(1, 2).reshape(2, 256, 256), Q_part))

    block = DiTBlock()

    x = torch.randn(2, NUM_PATCHES, HIDDEN_SIZE)
    c = torch.randn(2, HIDDEN_SIZE)
    y = block(x, c)
    print(y.shape)

    print(torch.allclose(x, y, atol=1e-6))