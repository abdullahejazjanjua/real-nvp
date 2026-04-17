import os
import time
import argparse
import numpy as np
from torch.utils.data import DataLoader, Subset
from torch.optim import AdamW

from utils.dataloader import CelebA
from model.norm_flows import RealNVP

from utils.engine import train_one_epoch, evaluate_after_one_epoch

def args_parser():
    parser = argparse.ArgumentParser(description="VAE parametres")
    
    parser.add_argument("--dataset_path", default="../variational-autoencoder/data/", type=str)
    
    # Training specifications
    parser.add_argument("--batch_size", default=4, type=int)
    parser.add_argument("--grad_steps", default=2, type=int, help="For simulating higher batch sizes")
    parser.add_argument("--epochs", default=3, type=int)
    parser.add_argument("--lr", default=0.0001, type=float, help="Base learning rate")
    parser.add_argument("--hidden_dim", default=256, type=int, help="latent dimension of z")
    parser.add_argument("--num_layers", default=4, type=int, help="number of layers of AffineCoupling")
    parser.add_argument("--grad_clip", default=3.0, type=float)

    # Additional parametres
    parser.add_argument("--print_freq", default=50, type=int)
    parser.add_argument("--num_workers", default=2, type=int)
    parser.add_argument("--device", default="mps", type=str)
    parser.add_argument("--savepath", default="logs/", type=str)
    parser.add_argument("--dataset_num_subset", default=2000, type=int, help="if you want to train on a subset of total images only. Set to -1 to use entire data")


    return parser


def main(args):
    model = RealNVP(img_dim=(128, 128), in_channels=3, hidden_channels=args.hidden_dim, num_layers=args.num_layers)
    optimizer = AdamW(params=model.parameters(), lr=args.lr)
    
    full_dataset = CelebA(imgs_path=args.dataset_path)
    if args.dataset_num_subset != -1:
        indices = np.arange(args.dataset_num_subset)
        dataset = Subset(full_dataset, indices)
    else:
        dataset = full_dataset
    dataloader = DataLoader(dataset, batch_size=(args.batch_size * args.grad_steps), shuffle=True, num_workers=args.num_workers, drop_last=True)
    
    print(f"Using device: {args.device}")
    print(f"Total Images: {len(dataset)}")
    print("\nUsing Arguments")
    print(args)
    
    model = model.to(args.device)

    print("\nStarting training")
    for epoch in range(args.epochs):
        print(f"Epoch [{epoch}]: ")
        start = time.time()
        loss = train_one_epoch(
            model=model,
            dataloader=dataloader,
            optimizer=optimizer,
            args=args,
        )
        end = time.time()
        print("Average stats:")
        print(f"    loss: {loss}, time: {(end-start):.4f}s")
        print("Starting Evaluating:")
        evaluate_after_one_epoch(model, savepath=args.savepath, device=args.device, num_samples=2, channels=3, img_dim=(128, 128), current_epoch=epoch)

if __name__ == "__main__":
    args = args_parser().parse_args()
    os.makedirs(args.savepath, exist_ok=True)
    main(args)