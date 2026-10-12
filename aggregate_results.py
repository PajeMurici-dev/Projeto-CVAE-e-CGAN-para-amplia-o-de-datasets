import argparse
import json
import os
from pathlib import Path

import pandas as pd
from scipy import stats

GENS = ["CVAE", "cGAN"]


def fmt(s, scale=1.0, nd=2):
    """média ± desvio-padrão amostral (n-1); com uma única semente o desvio é 0."""
    sd = s.std() if len(s) > 1 else 0.0
    return f"{scale * s.mean():.{nd}f} ± {scale * sd:.{nd}f}"


def to_latex(df):
    try:
        return df.to_latex(index=False)
    except Exception as e:   # to_latex pode exigir jinja2
        return f"to_latex indisponível: {e}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default=os.environ.get("RESULTS_DIR", "resultados"))
    args = ap.parse_args()
    out = Path(args.results_dir)

    files = sorted(out.glob("seed_*.json"), key=lambda p: int(p.stem.split("_")[1]))
    if not files:
        raise SystemExit(f"Nenhum seed_*.json em {out.resolve()}")

    rows, q_rows, cls_rows = [], [], []
    for f in files:
        seed = int(f.stem.split("_")[1])
        d = json.loads(f.read_text())
        q = d.pop("_qualidade", None)
        for sc, r in d.items():
            rows.append({"seed": seed, "cenario": sc, **r})
        if q:
            ref = q["referencia_real"]
            for g in GENS:
                m = q[g]
                q_rows.append({
                    "seed": seed, "gerador": g, "juiz": q.get("juiz", "?"),
                    "cc": m["cc"], "cc_macro": m["cc_macro"], "cc_rel": m["cc"] / ref["judge_acc_ref"],
                    "cor_w1": m["color_w1"], "cor_piso": ref["color_w1_piso"],
                    "nitidez_razao": m["sharpness"] / ref["sharpness"],
                    "diversidade_razao": m["diversity"] / ref["diversity"],
                    "juiz_acc_real": ref["judge_acc_ref"], "juiz_acc_real_macro": ref["judge_acc_ref_macro"],
                })
                for c, v in enumerate(m["cc_por_classe"]):
                    cls_rows.append({"seed": seed, "coluna": g, "classe": c, "valor": v})
            for c, v in enumerate(ref["judge_acc_ref_por_classe"]):
                cls_rows.append({"seed": seed, "coluna": "Real (juiz)", "classe": c, "valor": v})

    df = pd.DataFrame(rows)
    scenarios = list(dict.fromkeys(df["cenario"]))   # mantém a ordem de execução
    n_seeds = df["seed"].nunique()
    print(f"Sementes encontradas: {sorted(int(x) for x in df['seed'].unique())}\n")

    # ---------------------------------------------------- 1) cenários
    g = df.groupby("cenario", sort=False)
    summary = pd.DataFrame({
        "N treino": g["n_train"].first(),
        "Acc. domínio (%)": g["acc_dom"].apply(lambda s: fmt(s, 100)),
        "F1 domínio (%)": g["f1_dom"].apply(lambda s: fmt(s, 100)),
        "Acc. teste oficial (%)": g["acc_test"].apply(lambda s: fmt(s, 100)),
        "F1 teste oficial (%)": g["f1_test"].apply(lambda s: fmt(s, 100)),
    }).reset_index().rename(columns={"cenario": "Cenário"})
    print("=== Cenários (domínio = imagens reais do mesmo domínio do treino; teste oficial = outro domínio) ===")
    print(summary.to_string(index=False))
    summary.to_csv(out / "resumo.csv", index=False)
    df.to_csv(out / "resultados_por_semente.csv", index=False)
    print("\n--- LaTeX ---")
    print(to_latex(summary))

    # ---------------------------------------------------- 2) pareada vs baseline
    print(f"\n=== Diferença pareada vs. baseline (n = {n_seeds} sementes) ===")
    base = df[df.cenario == scenarios[0]].set_index("seed")
    for sc in scenarios[1:]:
        cur = df[df.cenario == sc].set_index("seed")
        common = base.index.intersection(cur.index)
        for met, nome in [("acc_dom", "acc. domínio"), ("f1_dom", "F1 domínio"),
                          ("acc_test", "acc. teste"), ("f1_test", "F1 teste")]:
            diff = 100 * (cur.loc[common, met] - base.loc[common, met])
            if len(common) > 1:
                p = stats.ttest_rel(cur.loc[common, met], base.loc[common, met]).pvalue
                print(f"{sc:38s} {nome:13s} Δ = {diff.mean():+6.2f} pp (dp {diff.std():.2f}) | p = {p:.3f}")
            else:
                print(f"{sc:38s} {nome:13s} Δ = {diff.mean():+6.2f} pp")
    print("Com poucas sementes, trate o valor de p como indicativo, não conclusivo.")

    # ---------------------------------------------------- 3) qualidade das sintéticas
    if q_rows:
        qdf = pd.DataFrame(q_rows)
        qdf.to_csv(out / "qualidade_por_semente.csv", index=False)
        gq = qdf.groupby("gerador", sort=False)
        qsum = pd.DataFrame({
            "CC (%)": gq["cc"].apply(lambda s: fmt(s, 100)),
            "CC macro (%)": gq["cc_macro"].apply(lambda s: fmt(s, 100)),
            "CC rel. ao real (%)": gq["cc_rel"].apply(lambda s: fmt(s, 100)),
            "Cor W1": gq["cor_w1"].apply(lambda s: fmt(s, 1, 4)),
            "Nitidez (x real)": gq["nitidez_razao"].apply(lambda s: fmt(s, 1, 2)),
            "Diversidade (x real)": gq["diversidade_razao"].apply(lambda s: fmt(s, 1, 2)),
        }).reset_index().rename(columns={"gerador": "Gerador"})

        one = qdf[qdf.gerador == GENS[0]]   # valores de referência são iguais entre geradores
        print(f"\n=== Qualidade das sintéticas (juiz: {one['juiz'].iloc[0]}) ===")
        print(qsum.to_string(index=False))
        print(f"\nReferência real: acurácia do juiz em imagens reais = {fmt(one['juiz_acc_real'], 100)}% "
              f"(macro {fmt(one['juiz_acc_real_macro'], 100)}%) | "
              f"piso de cor (real vs. real) W1 = {fmt(one['cor_piso'], 1, 4)} | nitidez e diversidade = 1.00x por definição")
        qsum.to_csv(out / "qualidade_resumo.csv", index=False)
        print("CC rel. ao real = CC dividida pela acurácia do juiz em imagens reais da mesma semente "
              "(compensa juízes de força diferente entre sementes e rodadas).")
        print("\n--- LaTeX ---")
        print(to_latex(qsum))

        # ------------------------------------------------ 4) CC por classe
        cdf = pd.DataFrame(cls_rows)
        per_class = (cdf.groupby(["classe", "coluna"])["valor"].mean().unstack() * 100).round(2)
        per_class = per_class[[c for c in GENS + ["Real (juiz)"] if c in per_class.columns]]
        print("\n=== Consistência condicional por classe (%, média entre sementes) ===")
        print(per_class.to_string())
        per_class.to_csv(out / "cc_por_classe.csv")
    else:
        print("\n(Sem métricas de qualidade nos JSONs: rode com a versão atual do run_experiment.py.)")


if __name__ == "__main__":
    main()