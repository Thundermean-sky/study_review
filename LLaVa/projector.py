import torch
import torch.nn as nn


class MLPProjector(nn.Module):
    def __init__(self, input_size=1024, output_size=4096, hidden_size=4096):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, x):
        return self.net(x)



if __name__ == '__main__':
    mlp = MLPProjector()

    x = torch.randn(8, 576, 1024)

    y = mlp(x)
    print(y.shape)
    print("params = ", sum(p.numel() for p in mlp.parameters()))