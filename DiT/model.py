import torch
import torch.nn as nn
from config import *
from components import *


class FinalLayer(nn.Module):
    def __init__(self, hidden_size=HIDDEN_SIZE, patch_size=PATCH_SIZE, out_channels=LATENT_CHANNELS):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)

        nn.init.constant_(self.linear.weight, 0)
        nn.init.constant_(self.linear.bias, 0)

    def forward(self, x):
        x = self.norm_final(x)
        return self.linear(x)



class DiT(nn.Module):
    def __init__(self):
        super().__init__()
        self.t_embed = TimeEmbed()
        self.y_embed = LabelEmbedding()
        self.patch_embed = PatchEmbed()
        self.blocks = nn.ModuleList([
            DiTBlock() for _ in range(DEPTH)
        ])
        self.final_layer = FinalLayer()

    def forward(self, x, t, y):
        x = self.patch_embed(x)
        t = self.t_embed(t)
        y = self.y_embed(y, self.training)
        c = t + y
        for block in self.blocks:
            x = block(x, c)

        x = self.final_layer(x)
        x = unpatchify(x)

        return x


if __name__ == '__main__':
    model = DiT()
    latent = torch.randn(2, *LATENT_SHAPE)
    t = torch.randint(0, NUM_TIMESTEPS, (2, ))
    y = torch.randint(0, NUM_CLASSES, (2, ))
    out = model(latent, t, y)

    print(out.shape)
    print("params: ", sum(p.numel() for p in model.parameters()))
