import torch
import torch.nn as nn
import torch.nn.functional as F

import config as cfg
from utils import onehot


class CVAE(nn.Module):
    def __init__(self, latent=None, n_classes=None):
        super().__init__()
        latent = latent or cfg.LATENT_VAE
        n_classes = n_classes or cfg.N_CLASSES
        self.latent = latent
        self.enc = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1), nn.ReLU(),     # 14x14
            nn.Conv2d(32, 64, 4, 2, 1), nn.ReLU(),    # 7x7
            nn.Flatten(),
        )
        self.fc_mu = nn.Linear(64 * 7 * 7 + n_classes, latent)
        self.fc_logvar = nn.Linear(64 * 7 * 7 + n_classes, latent)
        self.fc_dec = nn.Linear(latent + n_classes, 64 * 7 * 7)
        self.dec = nn.Sequential(
            nn.ReLU(), nn.Unflatten(1, (64, 7, 7)),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.ReLU(),    # 14x14
            nn.ConvTranspose2d(32, 3, 4, 2, 1), nn.Sigmoid(),  # 28x28
        )

    def decode(self, z, y1h):
        return self.dec(self.fc_dec(torch.cat([z, y1h], dim=1)))

    def forward(self, x, y1h):
        h = torch.cat([self.enc(x), y1h], dim=1)
        mu, logvar = self.fc_mu(h), self.fc_logvar(h)
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)   # reparametrização
        return self.decode(z, y1h), mu, logvar


def train_cvae(X, y, epochs=None):
    epochs = epochs or cfg.EPOCHS_VAE
    model = CVAE().to(cfg.DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    n = len(X)
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n, device=cfg.DEVICE)
        total = 0.0
        for i in range(0, n, cfg.BATCH_GEN):
            b = perm[i:i + cfg.BATCH_GEN]
            xb, yb = X[b], onehot(y[b])
            recon, mu, logvar = model(xb, yb)
            rec = F.binary_cross_entropy(recon, xb, reduction="sum") / len(b)
            kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / len(b)
            loss = rec + kl
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(b)
        if (ep + 1) % 20 == 0:
            print(f"   CVAE época {ep + 1}/{epochs} | perda média: {total / n:.2f}")
    return model


@torch.no_grad()
def generate_cvae(model, labels, bs=1024):
    model.eval()
    out = []
    for i in range(0, len(labels), bs):
        yb = onehot(labels[i:i + bs])
        z = torch.randn(len(yb), model.latent, device=cfg.DEVICE)
        out.append(model.decode(z, yb).clamp(0, 1))
    return torch.cat(out)
