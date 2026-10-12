import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score

import config as cfg
from utils import set_seed


class SmallCNN(nn.Module):
    def __init__(self, n_classes=None):
        super().__init__()
        n_classes = n_classes or cfg.N_CLASSES
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),    # 14x14
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),   # 7x7
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  # 3x3
        )
        self.head = nn.Sequential(
            nn.Flatten(), nn.Dropout(0.3),
            nn.Linear(128 * 3 * 3, 128), nn.ReLU(),
            nn.Dropout(0.3), nn.Linear(128, n_classes),
        )

    def forward(self, x):
        return self.head(self.features(x))


@torch.no_grad()
def predict_logits(model, X, bs=2048):
    model.eval()
    return torch.cat([model(X[i:i + bs]) for i in range(0, len(X), bs)])


def _acc_f1(model, X, y):
    pred = predict_logits(model, X).argmax(1).cpu().numpy()
    yt = y.cpu().numpy()
    return float(accuracy_score(yt, pred)), float(f1_score(yt, pred, average="macro"))


def train_and_eval(X, y, seed, data):
    """Treina com early stopping na loss de validação e avalia em DOIS conjuntos reais:
    domínio do treino (data.X_ref, avaliação principal) e teste oficial (data.X_test, secundária).

    Devolve (métricas, modelo treinado).
    """
    set_seed(seed)
    model = SmallCNN().to(cfg.DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    n = len(X)
    best_loss, best_state, wait, epochs_run = float("inf"), None, 0, 0
    for ep in range(cfg.MAX_EPOCHS_CLF):
        model.train()
        perm = torch.randperm(n, device=cfg.DEVICE)
        for i in range(0, n, cfg.BATCH_CLF):
            b = perm[i:i + cfg.BATCH_CLF]
            opt.zero_grad()
            F.cross_entropy(model(X[b]), y[b]).backward()
            opt.step()
        val_loss = F.cross_entropy(predict_logits(model, data.X_val), data.y_val).item()
        epochs_run = ep + 1
        if val_loss < best_loss - 1e-4:
            best_loss, wait = val_loss, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= cfg.PATIENCE_CLF:
                break
    model.load_state_dict(best_state)
    acc_dom, f1_dom = _acc_f1(model, data.X_ref, data.y_ref)
    acc_test, f1_test = _acc_f1(model, data.X_test, data.y_test)
    metrics = {
        "acc_dom": acc_dom, "f1_dom": f1_dom,
        "acc_test": acc_test, "f1_test": f1_test,
        "n_train": int(n), "epochs": epochs_run,
    }
    return metrics, model
