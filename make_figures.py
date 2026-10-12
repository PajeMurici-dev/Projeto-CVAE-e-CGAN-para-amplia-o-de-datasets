import argparse
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

plt.rcParams.update({
    "font.size": 7, "axes.titlesize": 7, "axes.labelsize": 7,
    "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "pdf.fonttype": 42,
})

PANELS = [("dom", "Domínio do treino"), ("test", "Teste oficial")]
COLORS = ["#7f7f7f", "#1f77b4", "#ff7f0e", "#2ca02c", "#d9d9d9"]


def load(out):
    files = sorted(out.glob("seed_*.json"), key=lambda p: int(p.stem.split("_")[1]))
    if not files:
        raise SystemExit(f"Nenhum seed_*.json em {out.resolve()}")
    rows = []
    for f in files:
        seed = int(f.stem.split("_")[1])
        d = json.loads(f.read_text())
        d.pop("_qualidade", None)
        for sc, r in d.items():
            rows.append({"seed": seed, "cenario": sc, **r})
    df = pd.DataFrame(rows)
    return df, list(dict.fromkeys(df["cenario"]))


def short(name):
    if name.startswith("Baseline"):
        return "Baseline"
    if name.startswith("100%"):
        return "100% real"
    m = re.match(r"^[\d.]+% real \+ (.*)$", name)
    return ("+" + m.group(1).replace("aumento clássico", "aum. clássico")) if m else name


def stat(df, scenario, metric):
    out = {}
    for key, _ in PANELS:
        s = 100 * df.loc[df.cenario == scenario, f"{metric}_{key}"]
        out[key] = (s.mean(), s.std() if len(s) > 1 else 0.0)
    return out


def draw(entries, stem, metric, width=3.5):
    n = len(entries)
    fig, axes = plt.subplots(1, 2, figsize=(width, 0.27 * n + 0.9), sharey=True)
    ypos = list(range(n))[::-1]
    xlabel = "Acurácia (%)" if metric == "acc" else "F1 macro (%)"
    for ax, (key, title) in zip(axes, PANELS):
        for e, y in zip(entries, ypos):
            m, s = e["stats"][key]
            ax.barh(y, m, xerr=s, height=0.7, color=e["color"], hatch=e.get("hatch"),
                    edgecolor="black", linewidth=0.5, error_kw={"elinewidth": 0.6, "capsize": 1.5})
            ax.text(m + s + 1.5, y, f"{m:.1f}", va="center", fontsize=6)
        ax.set_xlim(0, 112)
        ax.set_xticks([0, 25, 50, 75, 100])
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_yticks(ypos)
    axes[0].set_yticklabels([e["label"] for e in entries])
    plt.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(f"{stem}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"salvo: {stem}.pdf / .png")


def single_entries(df, order, metric):
    return [{"label": short(sc), "stats": stat(df, sc, metric), "color": COLORS[i % len(COLORS)]}
            for i, sc in enumerate(order)]


def comparison_entries(dfA, orderA, dfB, orderB, metric, labels):
    la, lb = labels
    ent = [{"label": short(orderA[0]), "stats": stat(dfA, orderA[0], metric), "color": COLORS[0]}]
    for i in (1, 2):
        ent.append({"label": f"{short(orderA[i])} ({la})", "stats": stat(dfA, orderA[i], metric), "color": COLORS[i]})
        ent.append({"label": f"{short(orderB[i])} ({lb})", "stats": stat(dfB, orderB[i], metric),
                    "color": COLORS[i], "hatch": "////"})
    for i in (3, 4):
        if i < len(orderA):
            ent.append({"label": short(orderA[i]), "stats": stat(dfA, orderA[i], metric), "color": COLORS[i]})
    return ent


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", required=True, help="pasta com os seed_*.json (configuração principal)")
    ap.add_argument("--results-dir2", help="segunda pasta (ex.: geradores com mais épocas) para a figura comparativa")
    ap.add_argument("--labels", nargs=2, default=["inicial", "estendida"], help="rótulos das duas configurações")
    ap.add_argument("--metric", choices=["acc", "f1"], default="acc")
    ap.add_argument("--out", help="pasta de saída (padrão: <results-dir>/figuras)")
    args = ap.parse_args()

    dirs = [Path(args.results_dir)] + ([Path(args.results_dir2)] if args.results_dir2 else [])
    out = Path(args.out) if args.out else dirs[0] / "figuras"
    out.mkdir(parents=True, exist_ok=True)

    loaded = [load(d) for d in dirs]
    for d, (df, order) in zip(dirs, loaded):
        draw(single_entries(df, order, args.metric), str(out / f"fig_cenarios_{d.name}"), args.metric)

    if len(dirs) == 2:
        (dfA, oA), (dfB, oB) = loaded
        if len(oA) < 3 or len(oB) < 3:
            raise SystemExit("As pastas precisam ter ao menos baseline, CVAE e cGAN.")
        draw(comparison_entries(dfA, oA, dfB, oB, args.metric, args.labels),
             str(out / "fig_cenarios_comparacao"), args.metric)
        print("Na comparação, baseline, aumento clássico e referência vêm da primeira pasta; "
              "CVAE e cGAN das duas (barras hachuradas = segunda pasta).")


if __name__ == "__main__":
    main()
