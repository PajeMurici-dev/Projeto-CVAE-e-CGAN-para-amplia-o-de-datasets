import torch
import torch.nn as nn

import config as cfg
from utils import onehot


class Generator(nn.Module):
    def __init__(self, z_dim=None, n_classes=None):
        super().__init__()
        z_dim = z_dim or cfg.Z_GAN
        n_classes = n_classes or cfg.N_CLASSES
        self.fc = nn.Sequential(
            nn.Linear(z_dim + n_classes, 128 * 7 * 7), nn.BatchNorm1d(128 * 7 * 7), nn.ReLU()
        )
        self.net = nn.Sequential(
            nn.Unflatten(1, (128, 7, 7)),
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.BatchNorm2d(64), nn.ReLU(),   # 14x14
            nn.ConvTranspose2d(64, 3, 4, 2, 1), nn.Tanh(),                          # 28x28
        )

    def forward(self, z, y1h):
        return self.net(self.fc(torch.cat([z, y1h], dim=1)))


class Discriminator(nn.Module):
    def __init__(self, n_classes=None):
        super().__init__()
        n_classes = n_classes or cfg.N_CLASSES
        self.net = nn.Sequential(
            nn.Conv2d(3 + n_classes, 64, 4, 2, 1), nn.LeakyReLU(0.2),                    # 14x14
            nn.Conv2d(64, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.LeakyReLU(0.2),         # 7x7
            nn.Flatten(), nn.Linear(128 * 7 * 7, 1),
        )

    def forward(self, x, y1h):
        y_map = y1h[:, :, None, None].expand(-1, -1, x.shape[2], x.shape[3])
        return self.net(torch.cat([x, y_map], dim=1)).squeeze(1)


def weights_init(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
        nn.init.normal_(m.weight, 0.0, 0.02)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


def train_cgan(X, y, epochs=None):
    epochs = epochs or cfg.EPOCHS_GAN
    dev, bs = cfg.DEVICE, cfg.BATCH_GEN
    G, D = Generator().to(dev), Discriminator().to(dev)
    G.apply(weights_init)
    D.apply(weights_init)
    optG = torch.optim.Adam(G.parameters(), lr=2e-4, betas=(0.5, 0.999))
    optD = torch.optim.Adam(D.parameters(), lr=2e-4, betas=(0.5, 0.999))
    bce = nn.BCEWithLogitsLoss()
    Xg = X * 2 - 1
    n = len(X)
    steps = max(1, n // bs)
    for ep in range(epochs):
        G.train()
        D.train()
        perm = torch.randperm(n, device=dev)
        lD = lG = 0.0
        for i in range(0, n - bs + 1, bs):   # descarta o último batch incompleto (BatchNorm)
            b = perm[i:i + bs]
            real, yb = Xg[b], onehot(y[b])
            z = torch.randn(len(b), cfg.Z_GAN, device=dev)
            fake = G(z, yb)

            # discriminador (rótulo "real" suavizado para 0.9)
            optD.zero_grad()
            loss_d = bce(D(real, yb), torch.full((len(b),), 0.9, device=dev)) + \
                     bce(D(fake.detach(), yb), torch.zeros(len(b), device=dev))
            loss_d.backward()
            optD.step()

            # gerador
            optG.zero_grad()
            loss_g = bce(D(fake, yb), torch.ones(len(b), device=dev))
            loss_g.backward()
            optG.step()
            lD += loss_d.item()
            lG += loss_g.item()
        if (ep + 1) % 25 == 0:
            print(f"   cGAN época {ep + 1}/{epochs} | perda D: {lD / steps:.3f} | perda G: {lG / steps:.3f}")
    return G


@torch.no_grad()
def generate_cgan(G, labels, bs=1024):
    G.eval()
    out = []
    for i in range(0, len(labels), bs):
        yb = onehot(labels[i:i + bs])
        z = torch.randn(len(yb), cfg.Z_GAN, device=cfg.DEVICE)
        out.append(((G(z, yb) + 1) / 2).clamp(0, 1))
    return torch.cat(out)
