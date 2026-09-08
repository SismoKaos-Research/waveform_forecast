"""Seeding, in one place.

Not a runnable script -- imported only.

The trainer draws several seeds per fold and reports the spread across them, so
which RNGs are seeded is part of what a reported number means: per-seed AUC
spread reaches 0.17 on this data. Leaving one unseeded would make part of that
spread an artefact of ordering rather than of initialisation.
"""
import random

import numpy as np
import torch


def seed_everything(seed: int):
    """Seeds Python's random, NumPy, and PyTorch RNGs.

    Args:
        seed: Seed value applied to all three RNGs.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
