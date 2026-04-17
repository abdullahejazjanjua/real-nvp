import torch
import torch.nn as nn
from torchvision.models import get_model


class RealNVP(nn.Module):
    def __init__(self, num_layers) -> None:
        super().__init__()

        model = get_model("resnet18", weights=None)
        self.backbone = nn.Sequential(*model.children())[:-1]
        self.num_layers = num_layers

        self.affine_coupling = [
            AffineCoupling(embed_dim=512, hidden_dim=64, flip=(i % 2 == 1))
            for i in range(num_layers)
        ]
    
    def forward(self, x: torch.Tensor):
        z = x.flatten(0)
        
        total_log_det = 0
        for layer in self.affine_coupling:
            z, log_det = layer(z)
            total_log_det += log_det
    
        return z, total_log_det
    
    def generate(self, z: torch.Tensor):
        x = z.flatten(0)

        total_log_det = 0
        for layer in self.affine_coupling:
            x, log_det = layer(x)
            total_log_det += log_det
    
        return x, total_log_det

class AffineCoupling(nn.Module):
    def __init__(self, embed_dim, hidden_dim, flip=False):
        super().__init__()
        self.flip = flip
        self.embed_dim = embed_dim

        self.mlp = nn.Sequential(
            nn.Linear(embed_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim),
        )

    def inverse_mapping(self, x: torch.Tensor):

        if not self.flip:
            x1, x2 = x[..., : self.embed_dim // 2], x[..., self.embed_dim // 2 :]
        else:
            x2, x1 = x[..., : self.embed_dim // 2], x[..., self.embed_dim // 2 :]

        out = self.mlp(x1)
        mu = out[..., : self.embed_dim // 2]
        sigma = out[..., self.embed_dim // 2 :]

        z_transformed = (x2 - mu) * torch.exp(-sigma)
        log_det = -torch.sum(sigma, dim=-1)

        if not self.flip:
            return torch.cat([x1, z_transformed], dim=-1), log_det
        else:
            return torch.cat([z_transformed, x1], dim=-1), log_det
        
    def forward_mapping(self, z: torch.Tensor):
        if not self.flip:
            z1, z2 = z[..., : self.embed_dim // 2], z[..., self.embed_dim // 2 :]
        else:
            z2, z1 = z[..., : self.embed_dim // 2], z[..., self.embed_dim // 2 :]

        out = self.mlp(z1)
        mu = out[..., : self.embed_dim // 2]
        sigma = out[..., self.embed_dim // 2 :]

        x_transformed = (z2 - mu) * torch.exp(-sigma)
        log_det = -torch.sum(sigma, dim=-1)

        if not self.flip:
            return torch.cat([z1, x_transformed], dim=-1), log_det
        else:
            return torch.cat([x_transformed, z1], dim=-1), log_det


