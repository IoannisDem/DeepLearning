import os

import torch
import wandb
from dotenv import load_dotenv
from torchvision import datasets
from torchvision.transforms import v2
from torch.utils.data import DataLoader

import loss_fn
from models import vae


def train_loop(model, dataloader, loss_fn, optimizer, device, step):
    model.train()
    for batch, (x, _) in enumerate(dataloader):
        x = x.to(device)
        optimizer.zero_grad()
        x_reconstructed, z_mean, z_log_var = model(x)
        loss = loss_fn(x, x_reconstructed, z_mean, z_log_var)
        loss.backward()
        optimizer.step()
        step += 1
        if batch % 100 == 0:
            wandb.log({"train/loss": loss.item()}, step=step)
    return step


def validation_loop(model, dataloader, loss_fn, device, step, epoch):
    model.eval()
    total_loss = 0.0
    num_batches = 0
    for x, _ in dataloader:
        x = x.to(device)
        with torch.no_grad():
            x_reconstructed, z_mean, z_log_var = model(x)
            loss = loss_fn(x, x_reconstructed, z_mean, z_log_var)
            total_loss += loss.item()
            num_batches += 1
    average_loss = total_loss / num_batches
    wandb.log(
        {
            "val/loss": average_loss,
            "epoch": epoch,
            "val/reconstructions": [
                wandb.Image(image) for image in x_reconstructed[:8].detach().cpu()
            ],
        },
        step=step,
    )


def main():
    load_dotenv()
    if not os.environ.get("WANDB_API_KEY"):
        raise RuntimeError("Set WANDB_API_KEY in the repo .env file.")

    channels_dim = [1, 2, 4]
    latent_dim = 32
    batch_size = 32
    learning_rate = 0.001
    epochs = 1

    model = vae.VAE(channels_dim, latent_dim=latent_dim)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    train_data = datasets.FashionMNIST(
        root="data",
        train=True,
        download=True,
        transform=v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]),
    )
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

    test_data = datasets.FashionMNIST(
        root="data",
        train=False,
        download=True,
        transform=v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]),
    )
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    criterion = loss_fn.VAELoss()

    wandb.init(
        project="vae",
        config={
            "channels_dim": channels_dim,
            "latent_dim": latent_dim,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "epochs": epochs,
            "optimizer": "sgd",
            "dataset": "FashionMNIST",
        },
    )
    step = 0
    for epoch in range(1, epochs + 1):
        step = train_loop(model, train_loader, criterion, optimizer, device, step)
        validation_loop(model, test_loader, criterion, device, step, epoch)
    wandb.finish()


if __name__ == "__main__":
    main()
