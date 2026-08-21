from functools import partial
import torch
from torch import nn
from torchvision import transforms


class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, norm_layer=None):
        super().__init__()

        img_size = (img_size, img_size)

        patch_size = (patch_size, patch_size)

        self.img_size = img_size
        self.patch_size = patch_size

        self.grid_size = (self.img_size[0] // patch_size[0], self.img_size[1] // patch_size[1])

        self.num_patches = self.grid_size[0] * self.grid_size[1]

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

        self.norm = nn.LayerNorm(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size is {H}x{W}, Not match image size {self.img_size[0]}x{self.img_size[1]}"

        x = self.proj(x).flatten(2).transpose(1, 2)

        x = self.norm(x)

        return x


class Attention(nn.Module):
    def __init__(self, embed_dim, head, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()

        self.heads = head
        self.embed_dim = embed_dim
        self.head_dim = embed_dim // head
        self.scale = qk_scale or self.head_dim ** -0.5

        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

        self.proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x, mask=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale

        attn_weight = attn.softmax(dim=-1)

        x = (attn_weight @ v).transpose(1, 2).reshape(B, N, C)

        x = self.proj(x)

        x = self.proj_drop(x)

        return x


class FeedForward(nn.Module):
    def __init__(self, embed_dim, hidden_dim, output=None, dropout=0.):
        super().__init__()
        output = output or embed_dim

        self.fc_1 = nn.Linear(embed_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

        self.act = nn.GELU()

        self.fc_2 = nn.Linear(hidden_dim, output)

    def forward(self, x):
        return self.dropout(self.fc_2(self.dropout(self.act(self.fc_1(x)))))


class Block(nn.Module):
    def __init__(self, embed_dim, heads, ffn_ratio, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.,
                 mlp_dropout=0., norm=nn.LayerNorm):
        super(Block, self).__init__()

        self.norm_1 = norm(embed_dim)

        self.attn = Attention(embed_dim, heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop,
                              proj_drop=proj_drop)

        self.norm_2 = norm(embed_dim)

        self.mlp = FeedForward(embed_dim, embed_dim * ffn_ratio, output=embed_dim, dropout=mlp_dropout)

    def forward(self, x):
        x = self.attn(self.norm_1(x)) + x

        x = x + self.mlp(self.norm_2(x))

        return x


class EncoderBlock(nn.Module):
    def __init__(self, img_size, in_c, num_class, embed_dim, heads, patch, depth, ffn_ratio, qkv_bias=False,
                 qk_scale=None, attn_drop=0., proj_drop=0., norm=nn.LayerNorm, distilled=False, embed_layer=PatchEmbed):
        super(EncoderBlock, self).__init__()

        self.num_class = num_class
        self.num_features = self.embed_dim = embed_dim
        self.num_tokens = 2 if distilled else 1
        norm_layer = norm or partial(nn.LayerNorm, eps=1e-6)

        self.patch_embed = embed_layer(img_size, patch, in_c, embed_dim, norm_layer)

        num_patch = self.patch_embed.num_patches
        # 类别编码
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        # 位置编码
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patch + self.num_tokens, embed_dim))

        self.pos_drop = nn.Dropout(proj_drop)

        self.Block = nn.Sequential(*[
            Block(embed_dim, heads, ffn_ratio, qkv_bias, qk_scale, attn_drop, proj_drop, mlp_dropout=0.,
                  norm=norm_layer)
            for _ in range(depth)
        ])

        self.norm = norm_layer(embed_dim)
        self.pre_logits = nn.Identity()

        # 分类头
        self.head = nn.Linear(embed_dim, num_class) if num_class else nn.Identity()

        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        self.apply(_init_vit_weight)

    def forward_features(self, x):
        x = self.patch_embed(x)
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = self.pos_drop(x + self.pos_embed)

        x = self.Block(x)

        x = self.norm(x)

        return self.pre_logits(x[:, 0])

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x


def _init_vit_weight(m):
    if isinstance(m, nn.Linear):
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out")
    elif isinstance(m, nn.LayerNorm):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)


def vit_base_patch16_224(pretrained=False, num_classes=1000):
    model = EncoderBlock(
        img_size=224,
        in_c=3,
        num_class=num_classes,
        embed_dim=768,
        heads=12,
        patch=16,
        depth=12,
        ffn_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.,
        proj_drop=0.,
        norm=nn.LayerNorm,
        distilled=False,
        embed_layer=PatchEmbed,
    )
    return model
