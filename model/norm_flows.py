import torch
import torch.nn as nn
from typing import Tuple

class RealNVP(nn.Module):
    def __init__(self, img_dim: Tuple[int, int], in_channels: int, hidden_channels: int, num_layers:int) -> None:
        super().__init__()

        self.affine_coupling: nn.ModuleList = nn.ModuleList()
        mask = torch.zeros(1, 1, img_dim[0], img_dim[1])
        mask[:, :, 0::2, 0::2] = 1
        mask[:, :, 1::2, 1::2] = 1
        
        for i in range(num_layers):
            current_mask = mask if (i % 2 == 0) else (1 - mask)
            self.affine_coupling.append(
                AffineCoupling(
                    in_channels=in_channels,
                    hidden_channels=hidden_channels,
                    mask=current_mask
                )
            )

    def forward(self, x: torch.Tensor):
        z = x
        total_log_det = 0
        for layer in self.affine_coupling:
            assert isinstance(layer, AffineCoupling) # To stop Linter from bitching.
            z, log_det = layer.inverse_mapping(z)
            total_log_det += log_det
        return z, total_log_det

    def generate(self, z: torch.Tensor):
        x = z
        for layer in reversed(self.affine_coupling):
            assert isinstance(layer, AffineCoupling)
            x, _ = layer.forward_mapping(x)
        return x


class AffineCoupling(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, mask: torch.Tensor):
        super().__init__()
        self.mask: torch.Tensor  # for linter
        self.register_buffer("mask", mask)

        self.nn = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=hidden_channels,
                kernel_size=3,
                padding="same",
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=hidden_channels,
                out_channels=hidden_channels,
                kernel_size=3,
                padding="same",
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=hidden_channels,
                out_channels=(in_channels) * 2,
                kernel_size=3,
                padding="same",
            ),
        )

        last_layer = self.nn[-1]
        assert isinstance(last_layer, nn.Conv2d)
        nn.init.zeros_(last_layer.weight)
        if last_layer.bias is not None:
            nn.init.zeros_(last_layer.bias)

    def inverse_mapping(self, x: torch.Tensor):

        x1, x2 = self.mask * x, (1 - self.mask) * x

        out = self.nn(x1)
        mu, alpha = out.chunk(2, dim=1)
        alpha = torch.tanh(alpha)

        mu, alpha = mu * (1 - self.mask), alpha * (1 - self.mask)

        z_transformed = (x2 - mu) * torch.exp(-alpha)
        log_det = -torch.sum(alpha, dim=tuple(range(1, alpha.ndim)))

        return x1 + z_transformed, log_det

    def forward_mapping(self, z: torch.Tensor):

        z1, z2 = self.mask * z, (1 - self.mask) * z

        out = self.nn(z1)
        mu, alpha = out.chunk(2, dim=1)
        alpha = torch.tanh(alpha)
        mu, alpha = mu * (1 - self.mask), alpha * (1 - self.mask)

        x_transformed = z2 * torch.exp(alpha) + mu
        log_det = torch.sum(alpha, dim=tuple(range(1, alpha.ndim)))

        return z1 + x_transformed, log_det
