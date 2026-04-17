import os

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision.utils import make_grid


def train_one_epoch(model, dataloader, optimizer, args):
    model.train()
    total_len_dataset = len(dataloader)
    total_loss = 0
    for img_idx, imgs in enumerate(dataloader):
        optimizer.zero_grad()
        for i in range(args.grad_steps):
            start_idx = i * args.batch_size
            end_idx = start_idx + args.batch_size

            img = imgs[start_idx:end_idx, ...].to(args.device)
            z, log_dets = model(img)

            if torch.isnan(z).any():
                raise ValueError("NaN detected in latent representation (z).")
            if torch.isnan(log_dets).any():
                raise ValueError("NaN detected in log determinants.")
            
            p_z = torch.distributions.Normal(0, 1).log_prob(z).sum(dim=tuple(range(1, z.ndim)))
            loss = -(p_z + log_dets).mean()


            loss = loss / args.grad_steps
            total_loss += loss.item()
            loss.backward()

        nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
        optimizer.step()

        if img_idx % args.print_freq == 0 or img_idx == total_len_dataset - 1:
            print(
                f"   [{img_idx + 1}/{total_len_dataset}] loss: {total_loss / (img_idx + 1)}"
            )

    return total_loss


def evaluate_after_one_epoch(
    model,
    savepath: str,
    device: str,
    num_samples: int,
    channels: int,
    current_epoch: int,
):
    model.eval()
    with torch.no_grad():
        z = torch.randn(num_samples, channels, 1, 1).to(device)
        reconstructed_img = model.generate(z)

        grid = make_grid(reconstructed_img, nrow=num_samples, padding=2, normalize=True)
        grid_np = grid.cpu().numpy().transpose((1, 2, 0))

        plt.figure(figsize=(15, 3))
        plt.imshow(grid_np)
        plt.axis("off")
        plt.savefig(
            os.path.join(savepath, f"generated_img_{current_epoch}"),
            bbox_inches="tight",
        )
        plt.close()
