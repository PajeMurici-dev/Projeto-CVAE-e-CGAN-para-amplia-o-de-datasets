import random

import numpy as np
import torch
import torch.nn.functional as F

import config as cfg


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def onehot(y):
    return F.one_hot(y, cfg.N_CLASSES).float()
