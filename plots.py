import matplotlib

matplotlib.use("Agg")   
import matplotlib.pyplot as plt
import torch

import config as cfg
from cgan import generate_cgan
from cvae import generate_cvae


@torch.no_grad()
def save_comparison_grid(cvae, G, X_real, y_real, path):
    n = cfg.N_CLASSES
    cls = torch.arange(n, device=cfg.DEVICE)
    real = torch.stack([X_real[(y_real == c).nonzero()[0, 0]] for c in range(n)])
    rows = [("Real", real), ("CVAE", generate_cvae(cvae, cls)), ("cGAN", generate_cgan(G, cls))]
    fig, axes = plt.subplots(3, n, figsize=(n * 1.1, 3.6))
    for r, (name, imgs) in enumerate(rows):
        for c in range(n):
            ax = axes[r, c]
            ax.imshow(imgs[c].permute(1, 2, 0).cpu().numpy())
            ax.set_xticks([])
            ax.set_yticks([])
            if c == 0:
                ax.set_ylabel(name, rotation=0, ha="right", va="center", fontsize=10)
            if r == 0:
                ax.set_title(f"classe {c}", fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"   figura salva em {path}")
