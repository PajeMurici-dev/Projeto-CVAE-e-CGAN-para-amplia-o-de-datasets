import argparse
import json
import time

import torch

import config as cfg
from augment import classical_augment
from cgan import generate_cgan, train_cgan
from classifier import train_and_eval
from cvae import generate_cvae, train_cvae
from dataset import load_data, sample_real_subset
from plots import save_comparison_grid
from quality import evaluate_quality, print_quality
from utils import set_seed


def run_seed(seed, data, first_seed):
    t0 = time.time()
    sc = cfg.scenario_names()
    set_seed(seed)
    X_real, y_real = sample_real_subset(data, seed)
    y_syn = y_real.repeat(cfg.SYN_RATIO)   # mantém a distribuição de classes do subconjunto real
    print(f"\n===== Semente {seed} | reais: {len(X_real)} | sintéticas por cenário: {len(y_syn)} =====")
    res = {}

    def run_scenario(name, X, y):
        r, model = train_and_eval(X, y, seed, data)
        res[name] = r
        print(f"  [{name}] n_treino={r['n_train']} | épocas={r['epochs']} | "
              f"domínio: acc={r['acc_dom']:.4f} F1={r['f1_dom']:.4f} | "
              f"teste oficial: acc={r['acc_test']:.4f} F1={r['f1_test']:.4f}")
        return model

    # 1) baseline
    base_model = run_scenario(sc[0], X_real, y_real)

    # 2) CVAE
    print("  Treinando CVAE...")
    cvae = train_cvae(X_real, y_real)
    X_vae = generate_cvae(cvae, y_syn)
    run_scenario(sc[1], torch.cat([X_real, X_vae]), torch.cat([y_real, y_syn]))

    # 3) cGAN
    print("  Treinando cGAN...")
    G = train_cgan(X_real, y_real)
    X_gan = generate_cgan(G, y_syn)
    run_scenario(sc[2], torch.cat([X_real, X_gan]), torch.cat([y_real, y_syn]))

    if first_seed:
        save_comparison_grid(cvae, G, X_real, y_real, cfg.OUT_DIR / "grade_real_cvae_cgan.png")

    # 4) aumento clássico
    X_aug = classical_augment(X_real, cfg.SYN_RATIO)
    run_scenario(sc[3], torch.cat([X_real, X_aug]), torch.cat([y_real, y_syn]))

    # 5) referência com 100% do treino real
    ref_model = run_scenario(sc[4], data.X_full, data.y_full) if cfg.RUN_REFERENCE else None

    # 6) qualidade das sintéticas (sem treino adicional)
    #    juiz = modelo treinado com 100% real; sem a referência, usa o do baseline (mais fraco)
    judge, judge_name = (ref_model, "100% real") if ref_model is not None else (base_model, "baseline")
    q = evaluate_quality(judge, X_real, y_real, data.X_ref, data.y_ref,
                         {"CVAE": (X_vae, y_syn), "cGAN": (X_gan, y_syn)})
    q["juiz"] = judge_name
    res["_qualidade"] = q
    print_quality(q)

    del cvae, G, X_vae, X_gan, X_aug, base_model, ref_model, judge
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f"  Tempo da semente {seed}: {(time.time() - t0) / 60:.1f} min")
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, nargs="+", help="sementes (padrão: config.SEEDS)")
    ap.add_argument("--frac", type=float, help="fração do treino real (padrão: config.FRAC_REAL)")
    ap.add_argument("--epochs-vae", type=int, help="épocas do CVAE")
    ap.add_argument("--epochs-gan", type=int, help="épocas da cGAN")
    ap.add_argument("--results-dir", help="pasta de resultados (padrão: ./resultados ou $RESULTS_DIR)")
    ap.add_argument("--no-reference", action="store_true",
                    help="pula o cenário com 100%% real (o juiz passa a ser o modelo do baseline)")
    args = ap.parse_args()

    if args.seeds:
        cfg.SEEDS = args.seeds
    if args.frac:
        cfg.FRAC_REAL = args.frac
    if args.epochs_vae:
        cfg.EPOCHS_VAE = args.epochs_vae
    if args.epochs_gan:
        cfg.EPOCHS_GAN = args.epochs_gan
    if args.results_dir:
        from pathlib import Path
        cfg.OUT_DIR = Path(args.results_dir)
    if args.no_reference:
        cfg.RUN_REFERENCE = False

    print("Dispositivo:", cfg.DEVICE)
    if cfg.DEVICE.type == "cpu":
        print("ATENÇÃO: sem GPU o treino fica muito lento.")
    torch.backends.cudnn.benchmark = True
    cfg.OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("Resultados em:", cfg.OUT_DIR.resolve())

    data = load_data()
    for s in cfg.SEEDS:
        path = cfg.OUT_DIR / f"seed_{s}.json"
        if path.exists():
            print(f"Semente {s} já concluída, pulando.")
            continue
        results = run_seed(s, data, first_seed=(s == cfg.SEEDS[0]))
        path.write_text(json.dumps(results, indent=2))

    print("\nConcluído. Rode `python aggregate_results.py` para gerar as tabelas finais.")


if __name__ == "__main__":
    main()
