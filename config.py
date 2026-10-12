import os
from pathlib import Path

import torch

# ------------------------- Experimento -------------------------
SEEDS = [0, 1, 2, 3, 4]       
FRAC_REAL = 0.10              
SYN_RATIO = 2                 
N_CLASSES = 9                 
VAL_SEED = 123                
RUN_REFERENCE = True          

# ------------------------- Geradores ---------------------------
LATENT_VAE = 64              
Z_GAN = 100                  
EPOCHS_VAE = 60
EPOCHS_GAN = 100
BATCH_GEN = 128

# ------------------------- Classificador -----------------------
BATCH_CLF = 128
MAX_EPOCHS_CLF = 40
PATIENCE_CLF = 5             

# ------------------------- Qualidade das sintéticas ------------
COLOR_BINS = 64              
DIVERSITY_MAX_PER_CLASS = 500 

# ------------------------- Caminhos ----------------------------
OUT_DIR = Path(os.environ.get("RESULTS_DIR", "resultados"))
DATA_DIR = Path(os.environ.get("DATA_DIR", "data"))

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def scenario_names():
    p = f"{FRAC_REAL * 100:g}"
    return [
        f"Baseline ({p}% real)",
        f"{p}% real + {SYN_RATIO}x CVAE",
        f"{p}% real + {SYN_RATIO}x cGAN",
        f"{p}% real + {SYN_RATIO}x aumento clássico",
        "100% real (referência)",
    ]
