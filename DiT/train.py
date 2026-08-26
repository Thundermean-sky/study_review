import torch
import torch.nn as nn
import torch.nn.functional as F
from config import *
from model import DiT
from vae import Encoder


betas = torch.linspace(1e-4, 0.02, NUM_TIMESTEPS)
alphas = 1.0 - betas
alphas_cumprod = torch.cumprod(alphas, dim=0)

def q_sample(x0, t, noise):
    alphas_bar = alphas_cumprod.to(device=t.device)[t]
    sqrt_alpha_bar = torch.sqrt(alphas_bar)[:, None, None, None]
    sqrt_one_minus = torch.sqrt(1.0 - alphas_bar)[:, None, None, None]
    return sqrt_alpha_bar * x0 + sqrt_one_minus * noise

def make_fake_data(batch_size, encoder, device):
    img = torch.randn(batch_size, IN_CHANNELS, IMAGE_SIZE, IMAGE_SIZE, device=device)
    with torch.no_grad():
        return encoder(img)

def train(epochs=1000):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    encoder = Encoder().to(device)
    model = DiT().to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)

    for epoch in range(epochs):
        batch_size = 4
        x0 = make_fake_data(batch_size, encoder, device)
        t = torch.randint(0, NUM_TIMESTEPS, (batch_size,), device=device)
        noise = torch.randn_like(x0)
        z_t = q_sample(x0, t, noise)

        y = torch.randint(0, NUM_CLASSES, (batch_size, ), device=device)

        pred = model(z_t, t, y)

        loss = F.mse_loss(pred, noise)

        opt.zero_grad()
        loss.backward()
        opt.step()

        if epoch % 10 == 0:
            print(f'Epoch: {epoch}, Loss: {loss.item():.4f}')

if __name__ == '__main__':
    train()
