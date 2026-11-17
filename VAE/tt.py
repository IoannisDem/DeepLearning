import os

import torch
from models import vae
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import v2

def main():

    train_data = datasets.FashionMNIST(
        root="data",
        train=True,
        download=True,
        transform=v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]),
    )
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)

    x, _ = next(iter(train_loader))
    x = x[0]
    # print(x.shape)
    plt.imshow(x.squeeze(), cmap="gray")
    out = os.path.join(os.path.dirname(__file__), "sample.png")
    plt.savefig(out)
    plt.close()
    print(out)



if __name__ == "__main__":
    main()
