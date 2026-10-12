import argparse
import json
from pathlib import Path

import pandas as pd
import torch

import config as cfg
from classifier import train_and_eval
from dataset import load_data, sample_real_subset


def fmt(s, scale=100.0, nd=2):
    sd = s.std() if len(s) > 1 else 0.0
    return f"{scale * s.mean():.{nd}f} ± {scale * sd:.{nd}f}"


def summarize(out, fracs):
    files = sorted(out.glob("piloto_seed_*.json"))
    if not files:
        raise SystemExit(f"Nenhum piloto_seed_*.json em {out.resolve()}")
    rows = []
    for f in files:
        seed = int(f.stem.split("_")[-1])
        d = json.loads(f.read_text())
        ref = d["referencia"]
        for fr in fracs:
            b = d.get(str(fr))
            if b is None:
                continue
            rows.append({
                "seed": seed, "frac": fr, "n_imagens": b["n_train"],
                "base_dom": b["acc_dom"], "ref_dom": ref["acc_dom"],
                "lacuna_dom": 100 * (ref["acc_dom"] - b["acc_dom"]),
                "base_test": b["acc_test"], "ref_test": ref["acc_test"],
                "lacuna_test": 100 * (ref["acc_test"] - b["acc_test"]),
            })
    df = pd.DataFrame(rows)
    df.to_csv(out / "piloto_por_semente.csv", index=False)
    g = df.groupby("frac", sort=True)
    table = pd.DataFrame({
        "N imagens": g["n_imagens"].first(),
        "Baseline dom. (%)": g["base_dom"].apply(fmt),
        "Referência dom. (%)": g["ref_dom"].apply(fmt),
        "Lacuna dom. (pp)": g["lacuna_dom"].apply(lambda s: fmt(s, 1.0)),
        "Baseline teste (%)": g["base_test"].apply(fmt),
        "Lacuna teste (pp)": g["lacuna_test"].apply(lambda s: fmt(s, 1.0)),
    }).reset_index().rename(columns={"frac": "Fração"})
    print(f"\nSementes: {sorted(df['seed'].unique().tolist())}\n")
    print(table.to_string(index=False))
    table.to_csv(out / "piloto_resumo.csv", index=False)
    print("\nComo ler: 'dom.' = imagens reais do mesmo domínio do treino (avaliação principal); "
          "'teste' = teste oficial (outro domínio).")
    print("Lacuna = referência (100% real) - baseline. Procure uma fração em que a lacuna seja bem "
          "maior que o desvio entre sementes, e registre essa regra de escolha no artigo.")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fracs", type=float, nargs="+", default=[0.01, 0.02, 0.05, 0.10])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--results-dir", help="pasta de resultados (padrão: ./resultados ou $RESULTS_DIR)")
    args = ap.parse_args()

    if args.results_dir:
        cfg.OUT_DIR = Path(args.results_dir)
    cfg.OUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.backends.cudnn.benchmark = True
    print("Dispositivo:", cfg.DEVICE, "| resultados em:", cfg.OUT_DIR.resolve())

    cfg.FRAC_REAL = 0.10   # divisão de validação fixa em todo o piloto
    data = load_data()

    for s in args.seeds:
        path = cfg.OUT_DIR / f"piloto_seed_{s}.json"
        if path.exists():
            print(f"Semente {s} já concluída, pulando.")
            continue
        res = {}
        r, _ = train_and_eval(data.X_full, data.y_full, s, data)
        res["referencia"] = r
        print(f"[semente {s}] referência (100% real): domínio={r['acc_dom']:.4f} | teste={r['acc_test']:.4f}")
        for fr in args.fracs:
            X, y = sample_real_subset(data, s, frac=fr)
            r, _ = train_and_eval(X, y, s, data)
            res[str(fr)] = r
            print(f"[semente {s}] baseline {fr:g} ({len(X)} imgs): domínio={r['acc_dom']:.4f} | teste={r['acc_test']:.4f}")
        path.write_text(json.dumps(res, indent=2))

    summarize(cfg.OUT_DIR, args.fracs)


if __name__ == "__main__":
    main()
