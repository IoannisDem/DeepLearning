import torch
from models import vae


def main():
    model = vae.VAE([1, 2, 4])
    img = torch.randn(1, 1, 28, 28)
    model.eval()
    with torch.no_grad():
        x_reconstructed, z_mean, z_log_var = model(img)

    print(x_reconstructed.shape)
    print(z_mean.shape)
    print(z_log_var.shape)


if __name__ == "__main__":
    main()
