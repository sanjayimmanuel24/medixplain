"""Trust Lab: tests of the explanations themselves.

An explanation is only useful if it is
  * faithful  - removing what it calls important really changes the prediction,
  * stable    - it does not change when the input or the random seed barely changes,
  * sensitive to the model - it changes when the model's weights are destroyed
                (Adebayo et al. 2018 sanity check),
  * corroborated - independent methods point at the same evidence.
Each function returns plain numbers so the GUI and the report share one source.
"""
from __future__ import annotations

import itertools

import numpy as np
from scipy.stats import spearmanr

from .common import SEED, rank_corr, top_k_overlap


def _mean(xs):
    return float(np.mean(xs)) if len(xs) else float("nan")


# ================================================================ tabular
def tabular_glassbox_vector(tab, x):
    """Glass-box reference: logistic-regression coefficient x standardised feature value."""
    scaler, lr = tab.b["glass"][0], tab.b["glass"][-1]
    return lr.coef_[0] * scaler.transform(x.to_frame().T)[0]


def tabular_agreement(tab, n: int = 40, k: int = 5) -> dict:
    X = tab.b["X_test"].iloc[:n]
    rows = {"SHAP vs LIME": [], "SHAP vs glass-box": [], "LIME vs glass-box": []}
    corr = {key: [] for key in rows}
    for _, x in X.iterrows():
        s = tab.shap_vector(x)
        l = tab.lime_vector(x, seed=SEED)
        g = tabular_glassbox_vector(tab, x)
        for key, (a, b) in {"SHAP vs LIME": (s, l), "SHAP vs glass-box": (s, g), "LIME vs glass-box": (l, g)}.items():
            rows[key].append(top_k_overlap(a, b, k))
            corr[key].append(rank_corr(a, b))
    return {key: {"top5_overlap": _mean(rows[key]), "spearman": _mean(corr[key])} for key in rows}


def tabular_stability(tab, x, runs: int = 5, noise: float = 0.01) -> dict:
    """Seed stability (LIME is random, SHAP Tree is exact) and input-noise stability."""
    lime_runs = [tab.lime_vector(x, seed=SEED + r) for r in range(runs)]
    seed_lime = _mean([rank_corr(a, b) for a, b in itertools.combinations(lime_runs, 2)])
    rng = np.random.default_rng(SEED)
    sd = tab.b["X_train"].std().values
    s0, l0 = tab.shap_vector(x), lime_runs[0]
    s_noise, l_noise = [], []
    for r in range(runs):
        xn = x + rng.normal(0, noise * sd)
        s_noise.append(rank_corr(s0, tab.shap_vector(xn)))
        l_noise.append(rank_corr(l0, tab.lime_vector(xn, seed=SEED)))
    return {"LIME seed-to-seed spearman": seed_lime, "SHAP seed-to-seed spearman": 1.0,
            "SHAP under 1% input noise": _mean(s_noise), "LIME under 1% input noise": _mean(l_noise)}


def tabular_faithfulness(tab, n: int = 40, k: int = 5) -> dict:
    """Replace the top-k features of each method by the training mean; bigger drop = more faithful."""
    X = tab.b["X_test"].iloc[:n]
    mean = tab.b["X_train"].mean()
    rng = np.random.default_rng(SEED)
    drops = {"SHAP": [], "LIME": [], "Glass-box coef": [], "Random": []}
    for _, x in X.iterrows():
        p0 = tab.predict_proba(x.values)[0, 1]
        cls = int(p0 >= 0.5)
        p_cls = p0 if cls else 1 - p0
        vecs = {"SHAP": tab.shap_vector(x), "LIME": tab.lime_vector(x, seed=SEED),
                "Glass-box coef": tabular_glassbox_vector(tab, x),
                "Random": rng.normal(size=len(x))}
        for name, v in vecs.items():
            # features that support the predicted class, most important first
            support = v if cls else -v
            idx = np.argsort(-support)[:k] if name != "Random" else rng.choice(len(x), k, replace=False)
            xm = x.copy()
            xm.iloc[idx] = mean.iloc[idx]
            p = tab.predict_proba(xm.values)[0, 1]
            drops[name].append(p_cls - (p if cls else 1 - p))
    return {name: _mean(v) for name, v in drops.items()}


# ================================================================ image
def image_faithfulness(model, images, n: int = 30) -> dict:
    """Mean area under the deletion curve per method (lower = more faithful)."""
    from . import image as I

    rng = np.random.default_rng(SEED)
    aucs: dict[str, list] = {}
    for img in images[:n]:
        target = int(I.predict_proba(model, img[None])[0].argmax())
        maps = I.explain_all(model, img, target, include_lime=False)
        maps["Random"] = rng.random(img.shape)
        for name, m in maps.items():
            curve = I.deletion_curve(model, img, m, target)
            aucs.setdefault(name, []).append(float(np.trapezoid(curve, dx=1 / (len(curve) - 1))))
    return {k: _mean(v) for k, v in aucs.items()}


def image_sanity(model, images, n: int = 20) -> dict:
    """Spearman(map from trained model, map from randomised model). Near 0 = passes the test."""
    from . import image as I

    rnd = I.randomized_copy(model)
    out: dict[str, list] = {}
    for img in images[:n]:
        target = int(I.predict_proba(model, img[None])[0].argmax())
        for name, fn in I.METHODS.items():
            a, b = fn(model, img, target), fn(rnd, img, target)
            if np.ptp(a) == 0 or np.ptp(b) == 0:  # a blank map shares nothing with the original
                out.setdefault(name, []).append(0.0)
                continue
            r = spearmanr(np.abs(a).ravel(), np.abs(b).ravel()).correlation
            out.setdefault(name, []).append(0.0 if np.isnan(r) else float(r))
    return {k: _mean(v) for k, v in out.items()}


def image_method_agreement(model, img, target) -> tuple[list, np.ndarray]:
    from . import image as I

    maps = I.explain_all(model, img, target)
    names = list(maps)
    M = np.eye(len(names))
    for i, j in itertools.combinations(range(len(names)), 2):
        r = spearmanr(np.abs(maps[names[i]]).ravel(), np.abs(maps[names[j]]).ravel()).correlation
        M[i, j] = M[j, i] = 0.0 if np.isnan(r) else r
    return names, M


# ================================================================ text
def _text_maps(tx, text, cls, seed=SEED):
    return {
        "SHAP": tx.ngrams_to_words(tx.shap_ngrams(text, cls)),
        "LIME": tx.lime_words(text, cls, seed=seed),
        "Occlusion": tx.occlusion_words(text, cls),
        "Linear contribution": tx.ngrams_to_words(tx.linear_contributions(text, cls)),
    }


def _aligned(maps: dict, words: list) -> dict:
    return {k: np.array([v.get(w, 0.0) for w in words]) for k, v in maps.items()}


def text_agreement(tx, n: int = 40, k: int = 3) -> dict:
    pairs: dict[str, list] = {}
    for text in tx.b["X_test"][:n]:
        cls = tx.top_classes(text, 1)[0][0]
        words = list(dict.fromkeys(tx.words(text)))
        vecs = _aligned(_text_maps(tx, text, cls), words)
        for a, b in itertools.combinations(vecs, 2):
            pairs.setdefault(f"{a} vs {b}", []).append(top_k_overlap(vecs[a], vecs[b], k))
    return {key: _mean(v) for key, v in pairs.items()}


def text_comprehensiveness(tx, n: int = 40, k: int = 3) -> dict:
    """ERASER comprehensiveness (DeYoung et al. 2020): P(class) drop after deleting the top-k words."""
    import re

    rng = np.random.default_rng(SEED)
    out: dict[str, list] = {}
    for text in tx.b["X_test"][:n]:
        cls = tx.top_classes(text, 1)[0][0]
        c = tx.classes.index(cls)
        p0 = tx.predict_proba([text])[0, c]
        words = list(dict.fromkeys(tx.words(text)))
        maps = _text_maps(tx, text, cls)
        maps["Random"] = dict(zip(words, rng.random(len(words))))
        for name, m in maps.items():
            top = [w for w, s in sorted(m.items(), key=lambda kv: -kv[1]) if s > 0][:k] if name != "Random" \
                else list(rng.choice(words, min(k, len(words)), replace=False))
            t2 = text
            for w in top:
                t2 = re.sub(rf"\b{re.escape(w)}\b", " ", t2, flags=re.I)
            out.setdefault(name, []).append(float(p0 - tx.predict_proba([t2])[0, c]))
    return {key: _mean(v) for key, v in out.items()}


def text_lime_stability(tx, text, runs: int = 5) -> float:
    cls = tx.top_classes(text, 1)[0][0]
    words = list(dict.fromkeys(tx.words(text)))
    vecs = [np.array([tx.lime_words(text, cls, seed=SEED + r).get(w, 0.0) for w in words]) for r in range(runs)]
    return _mean([rank_corr(a, b) for a, b in itertools.combinations(vecs, 2)])
