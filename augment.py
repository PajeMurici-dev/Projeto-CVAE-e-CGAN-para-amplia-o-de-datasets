
import torch


def classical_augment(X, ratio):

    N = len(X)
    combos = torch.rand(N, 7, device=X.device).argsort(dim=1)[:, :ratio] + 1   # valores 1..7
    outs = []
    for j in range(ratio):
        c = combos[:, j]
        k, f = c % 4, c // 4          # k = nº de rotações de 90°, f = flip horizontal
        Xa = X.clone()
        for kk in range(4):
            for ff in range(2):
                m = (k == kk) & (f == ff)
                if m.any():
                    t = X[m]
                    if ff:
                        t = torch.flip(t, dims=[3])
                    Xa[m] = torch.rot90(t, kk, dims=[2, 3])
        outs.append(Xa)
    return torch.cat(outs)
