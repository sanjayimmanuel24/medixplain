"""Shared paths, seeding and small helpers used by every module."""
from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
MODELS = ROOT / "models"
RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"

for _d in (MODELS, RESULTS, FIGURES):
    _d.mkdir(parents=True, exist_ok=True)

SEED = 42


def seed_everything(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def save_json(obj, path: Path) -> None:
    path.write_text(json.dumps(obj, indent=2, default=float))


def load_json(path: Path):
    return json.loads(path.read_text()) if path.exists() else {}


def top_k_overlap(a: np.ndarray, b: np.ndarray, k: int = 5) -> float:
    """Fraction of the k most important features (by |value|) that two explanations share."""
    ta = set(np.argsort(-np.abs(a))[:k])
    tb = set(np.argsort(-np.abs(b))[:k])
    return len(ta & tb) / k


def rank_corr(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman correlation of |attribution| ranks; 1 = identical ordering."""
    from scipy.stats import spearmanr

    r = spearmanr(np.abs(a), np.abs(b)).correlation
    return float(0.0 if np.isnan(r) else r)
