import torch
import torch.nn as nn
import torch.nn.functional as F
from config import *


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.down = nn.Sequential(
            nn.Conv2d(IN_CHANNELS, 64, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
        )

        self.to_latent = nn.Conv2d(256, LATENT_CHANNELS, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.down(x)
        return self.to_latent(x)


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.from_latent = nn.Conv2d(LATENT_CHANNELS, 256, kernel_size=3, stride=1, padding=1)
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.SiLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.SiLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(64, IN_CHANNELS, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, z):
        x = self.from_latent(z)
        x = self.up(x)
        return x




if __name__ == '__main__':
    # enc = Encoder()
    # dec = Decoder()
    # img = torch.randn(2, 3, 256, 256)
    # out = enc(img)
    # print(out.shape)
    # out = dec(out)
    # print(out.shape)
    pe = PatchEmbed()
    z = torch.randn(2, LATENT_CHANNELS, LATENT_SIZE, LATENT_SIZE)
    token = pe(z)
    print(token.shape)

    patch_tokens = torch.randn(2, NUM_PATCHES, PATCH_DIM)

    back = unpatchify(patch_tokens)
    print(back.shape)