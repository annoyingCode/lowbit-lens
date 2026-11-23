"""Make runs repeatable.

Call `seed_everything` before any CUDA work: cuBLAS reads CUBLAS_WORKSPACE_CONFIG
only when it first initialises.
"""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def seed_everything(seed: int, deterministic: bool = True) -> torch.Generator:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # also seeds every CUDA device

    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        # A few CUDA kernels (e.g. adaptive pooling backward) have no deterministic
        # version; warn instead of failing. Runs are then repeatable up to those ops.
        torch.use_deterministic_algorithms(True, warn_only=True)

    g = torch.Generator()
    g.manual_seed(seed)
    return g
