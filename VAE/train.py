import os
from pathlib import Path

import torch
import wandb
from dotenv import load_dotenv
from torchvision import datasets
from torchvision.transforms import v2
from torchvision.utils import make_grid
from torch.utils.data import DataLoader

import loss_fn
from models import vae


def kl_weight_for_epoch(epoch, epochs):
    stage = (epoch - 1) // 2 + 1
    stages = max(epochs // 2, 1)
    return min(0.5, stage / stages)
    # return 0 # testing that reconstruction works well


def train_loop(model, dataloader, loss_fn, optimizer, device, step, kl_weight, epoch):
    model.train()
    for batch, (x, _) in enumerate(dataloader):
        x = x.to(device)
        optimizer.zero_grad()
        x_reconstructed, z_mean, z_log_var = model(x)
        loss, mse, kl = loss_fn(x, x_reconstructed, z_mean, z_log_var, kl_weight)
        loss.backward()
        optimizer.step()
        step += 1
        if batch % 100 == 0:
            wandb.log(
                {
                    "train/loss": loss.item(),
                    "train/mse": mse.item(),
                    "train/kl": kl.item(),
                    "kl_weight": kl_weight,
                    "latent/mu_mean": z_mean.mean().item(),
                    "latent/std_mean": (0.5 * z_log_var).exp().mean().item(),
                    "epoch": epoch,
                },
                step=step,
            )
    return step


def validation_loop(model, dataloader, loss_fn, device, step, epoch, kl_weight):
    model.eval()
    total_loss = 0.0
    total_mse = 0.0
    total_kl = 0.0
    total_mu_mean = 0.0
    total_std_mean = 0.0
    num_batches = 0
    for x, _ in dataloader:
        x = x.to(device)
        with torch.no_grad():
            x_reconstructed, z_mean, z_log_var = model(x, sample=False)
            loss, mse, kl = loss_fn(
                x, x_reconstructed, z_mean, z_log_var, kl_weight
            )
            total_loss += loss.item()
            total_mse += mse.item()
            total_kl += kl.item()
            total_mu_mean += z_mean.mean().item()
            total_std_mean += (0.5 * z_log_var).exp().mean().item()
            num_batches += 1
    average_loss = total_loss / num_batches
    n = 8
    comparison = torch.cat([x[:n], x_reconstructed[:n]]).cpu().clamp(0, 1)
    wandb.log(
        {
            "val/loss": average_loss,
            "val/mse": total_mse / num_batches,
            "val/kl": total_kl / num_batches,
            "latent/val_mu_mean": total_mu_mean / num_batches,
            "latent/val_std_mean": total_std_mean / num_batches,
            "epoch": epoch,
            "kl_weight": kl_weight,
            "val/reconstructions": wandb.Image(make_grid(comparison, nrow=n) * 255),
        },
        step=step,
    )


def main():
    load_dotenv()
    if not os.environ.get("WANDB_API_KEY"):
        raise RuntimeError("Set WANDB_API_KEY in the repo .env file.")

    channels_dim = [1, 32, 64]
    latent_dim = 64
    batch_size = 32
    learning_rate = 1e-3
    epochs = 10
    seed = 0

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    model = vae.VAE(channels_dim, latent_dim=latent_dim)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    train_data = datasets.FashionMNIST(
        root="data",
        train=True,
        download=True,
        transform=v2.Compose(
            [
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.RandomHorizontalFlip(),
            ]
        ),
    )
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

    test_data = datasets.FashionMNIST(
        root="data",
        train=False,
        download=True,
        transform=v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]),
    )
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    criterion = loss_fn.VAELossMSE()

    wandb.init(
        project="vae2",
        config={
            "channels_dim": channels_dim,
            "latent_dim": latent_dim,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "epochs": epochs,
            "optimizer": "adam",
            "dataset": "FashionMNIST",
            "seed": seed,
        },
    )
    step = 0
    checkpoint_dir = Path("checkpoints")
    checkpoint_dir.mkdir(exist_ok=True)
    for epoch in range(1, epochs + 1):
        kl_weight = kl_weight_for_epoch(epoch, epochs)
        step = train_loop(
            model, train_loader, criterion, optimizer, device, step, kl_weight, epoch
        )
        validation_loop(
            model, test_loader, criterion, device, step, epoch, kl_weight
        )
        torch.save(model.state_dict(), checkpoint_dir / f"vae_epoch_{epoch}.pt")
    wandb.finish()


if __name__ == "__main__":
    main()
