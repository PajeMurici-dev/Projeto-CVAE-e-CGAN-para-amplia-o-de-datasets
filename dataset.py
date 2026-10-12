from dataclasses import dataclass, replace

import numpy as np
import torch
from sklearn.model_selection import train_test_split

import config as cfg


@dataclass
class Data:
    X_full: torch.Tensor     
    y_full: torch.Tensor
    y_full_np: np.ndarray
    X_valall: torch.Tensor    
    y_valall: torch.Tensor
    y_valall_np: np.ndarray
    X_val: torch.Tensor      
    y_val: torch.Tensor
    X_ref: torch.Tensor       
    y_ref: torch.Tensor
    X_test: torch.Tensor      
    y_test: torch.Tensor


def _to_tensor(imgs):
    return torch.from_numpy(imgs).permute(0, 3, 1, 2).float().div(255.0).to(cfg.DEVICE)


def _split_val(X_all, y_all, y_all_np, frac):
    sub, _ = train_test_split(np.arange(len(y_all_np)), train_size=frac,
                              stratify=y_all_np, random_state=cfg.VAL_SEED)
    rest = np.setdiff1d(np.arange(len(y_all_np)), sub)
    sub_t = torch.from_numpy(sub).to(cfg.DEVICE)
    rest_t = torch.from_numpy(rest).to(cfg.DEVICE)
    return X_all[sub_t], y_all[sub_t], X_all[rest_t], y_all[rest_t]


def split_validation(data, frac):
    X_val, y_val, X_ref, y_ref = _split_val(data.X_valall, data.y_valall, data.y_valall_np, frac)
    return replace(data, X_val=X_val, y_val=y_val, X_ref=X_ref, y_ref=y_ref)


def load_data():
    from medmnist import PathMNIST

    cfg.DATA_DIR.mkdir(parents=True, exist_ok=True)
    kw = dict(download=True, root=str(cfg.DATA_DIR))
    tr = PathMNIST(split="train", **kw)
    va = PathMNIST(split="val", **kw)
    te = PathMNIST(split="test", **kw)

    y_tr = tr.labels.squeeze().astype(np.int64)   # (N,1) -> (N,)
    y_va = va.labels.squeeze().astype(np.int64)
    y_te = te.labels.squeeze().astype(np.int64)

    dev = cfg.DEVICE
    X_valall = _to_tensor(va.imgs)
    y_valall = torch.from_numpy(y_va).to(dev)
    X_val, y_val, X_ref, y_ref = _split_val(X_valall, y_valall, y_va, cfg.FRAC_REAL)

    data = Data(
        X_full=_to_tensor(tr.imgs), y_full=torch.from_numpy(y_tr).to(dev), y_full_np=y_tr,
        X_valall=X_valall, y_valall=y_valall, y_valall_np=y_va,
        X_val=X_val, y_val=y_val, X_ref=X_ref, y_ref=y_ref,
        X_test=_to_tensor(te.imgs), y_test=torch.from_numpy(y_te).to(dev),
    )
    print(f"treino completo: {tuple(data.X_full.shape)} | early stopping: {tuple(data.X_val.shape)} "
          f"| avaliação no domínio do treino: {tuple(data.X_ref.shape)} | teste oficial: {tuple(data.X_test.shape)}")
    print("imagens por classe (treino completo):", np.bincount(y_tr))
    return data


def sample_real_subset(data, seed, frac=None):
    frac = frac or cfg.FRAC_REAL
    idx, _ = train_test_split(
        np.arange(len(data.y_full_np)), train_size=frac,
        stratify=data.y_full_np, random_state=seed,
    )
    idx = torch.from_numpy(idx).to(cfg.DEVICE)
    return data.X_full[idx], data.y_full[idx]
