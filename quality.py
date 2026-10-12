import numpy as np
import torch
import torch.nn.functional as F

import config as cfg
from classifier import predict_logits

_LAPLACIAN = torch.tensor([[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]]).view(1, 1, 3, 3)


@torch.no_grad()
def per_class_accuracy(model, X, y):
    pred = predict_logits(model, X).argmax(1)
    correct = (pred == y).float()
    per_class = []
    for c in range(cfg.N_CLASSES):
        m = y == c
        per_class.append(correct[m].mean().item() if m.any() else float("nan"))
    return {"acc": correct.mean().item(), "acc_macro": float(np.nanmean(per_class)), "per_class": per_class}


@torch.no_grad()
def _channel_cdfs(X):
    cdfs = []
    for ch in range(X.shape[1]):
        h = torch.histc(X[:, ch].flatten(), bins=cfg.COLOR_BINS, min=0.0, max=1.0)
        cdfs.append(torch.cumsum(h / h.sum(), dim=0))
    return torch.stack(cdfs)


@torch.no_grad()
def color_w1(Xa, Xb):
    ca, cb = _channel_cdfs(Xa), _channel_cdfs(Xb)
    return ((ca - cb).abs().sum(dim=1) / cfg.COLOR_BINS).mean().item()


@torch.no_grad()
def sharpness(X, bs=2048):
    k = _LAPLACIAN.to(X.device)
    vals = []
    for i in range(0, len(X), bs):
        gray = X[i:i + bs].mean(dim=1, keepdim=True)
        vals.append(F.conv2d(gray, k).flatten(1).var(dim=1))
    return torch.cat(vals).mean().item()


@torch.no_grad()
def diversity(X, y, seed=0):
    gen = torch.Generator().manual_seed(seed)
    vals = []
    for c in range(cfg.N_CLASSES):
        idx = (y == c).nonzero().squeeze(1)
        if len(idx) < 2:
            continue
        if len(idx) > cfg.DIVERSITY_MAX_PER_CLASS:
            sel = torch.randperm(len(idx), generator=gen)[:cfg.DIVERSITY_MAX_PER_CLASS].to(idx.device)
            idx = idx[sel]
        v = X[idx].flatten(1)
        d = torch.cdist(v, v)
        n = len(idx)
        vals.append((d.sum() / (n * (n - 1))).item())   # a diagonal é zero
    return float(np.mean(vals))


def evaluate_quality(judge, X_real, y_real, X_ref, y_ref, synth):
    ref_acc = per_class_accuracy(judge, X_ref, y_ref)
    out = {"referencia_real": {
        "judge_acc_ref": ref_acc["acc"],
        "judge_acc_ref_macro": ref_acc["acc_macro"],
        "judge_acc_ref_por_classe": ref_acc["per_class"],
        "color_w1_piso": color_w1(X_real, X_ref),
        "sharpness": sharpness(X_ref),
        "sharpness_treino_real": sharpness(X_real),
        "diversity": diversity(X_ref, y_ref),
        "diversity_treino_real": diversity(X_real, y_real),
    }}
    for name, (X_s, y_s) in synth.items():
        cc = per_class_accuracy(judge, X_s, y_s)
        out[name] = {
            "cc": cc["acc"], "cc_macro": cc["acc_macro"], "cc_por_classe": cc["per_class"],
            "color_w1": color_w1(X_s, X_ref),
            "sharpness": sharpness(X_s),
            "diversity": diversity(X_s, y_s),
        }
    return out


def print_quality(q):
    ref = q["referencia_real"]
    print(f"  [Qualidade | juiz: {q['juiz']}] acurácia do juiz em imagens reais de referência: "
          f"{100 * ref['judge_acc_ref']:.2f}%")
    for g in ("CVAE", "cGAN"):
        m = q[g]
        print(f"    {g}: CC={100 * m['cc']:.2f}% | CC macro={100 * m['cc_macro']:.2f}% | "
              f"cor W1={m['color_w1']:.4f} (piso {ref['color_w1_piso']:.4f}) | "
              f"nitidez={m['sharpness'] / ref['sharpness']:.2f}x | "
              f"diversidade={m['diversity'] / ref['diversity']:.2f}x")
