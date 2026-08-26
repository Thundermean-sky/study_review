import torch
import torch.nn as nn
import math


class PatchEmbed(nn.Module):
    def __init__(self, in_ch=3, dim=1024, patch=16):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, dim, kernel_size=patch, stride=patch)

    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class Attention(nn.Module):
    def __init__(self, dim=1024, heads=16):
        super().__init__()
        self.heads = heads
        self.head_dim = dim // heads
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(0.1)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        attn = self.attn_drop(attn.softmax(dim=-1))

        out = (attn @ v).transpose(1, 2).reshape(B, N, C)

        return self.proj(out)


class MLP(nn.Module):
    def __init__(self, dim=1024, hidden=4096):
        super().__init__()
        self.fc_1 = nn.Linear(dim, hidden)
        self.fc_2 = nn.Linear(hidden, dim)
        self.act = nn.GELU()
        self.mlp_drop = nn.Dropout(0.1)

    def forward(self, x):
        x = self.fc_1(x)
        x = self.act(x)
        x = self.fc_2(x)
        x = self.mlp_drop(x)
        return x


class Block(nn.Module):
    def __init__(self, dim=1024, heads=16, hidden=4096):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, heads=heads)
        self.ln2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, hidden)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class CLIPViT(nn.Module):
    def __init__(self, img_size=336, patch=14, dim=1024, hidden=4096, heads=16, depth=4):
        super().__init__()
        self.img_size = img_size
        self.patch = patch
        num_patches = (img_size // patch) ** 2
        self.patch_embed = PatchEmbed(dim=dim, patch=patch)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, dim))
        nn.init.normal_(self.cls_token, std=.02)
        nn.init.normal_(self.pos_embed, std=.02)
        self.block = nn.ModuleList(Block(dim=dim, heads=heads, hidden=hidden) for _ in range(depth))
        self.ln_out = nn.LayerNorm(dim)

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + self.pos_embed
        for blk in self.block:
            x = blk(x)

        x = self.ln_out(x)
        return x[:, 1:]



if __name__ == '__main__':
    m = CLIPViT()
    x = torch.randn(8, 3, 336, 336)
    y = m(x)

    print(y.shape)
    print("params = ", sum(p.numel() for p in m.parameters()))