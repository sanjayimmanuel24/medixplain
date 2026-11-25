"""Plain-English explanation cards for non-technical readers (clinicians, patients).

Every sentence is filled from computed numbers. Nothing is invented, so the text can
be checked against the plots next to it.
"""
from __future__ import annotations

import numpy as np


def _pretty(name: str) -> str:
    return name.replace("_", " ")


def _confidence_word(p: float) -> str:
    return "very confident" if p >= 0.95 else "confident" if p >= 0.8 else "leaning" if p >= 0.6 else "uncertain"


def tabular_card(features, x, shap_vec, prob_malignant, cf: dict | None, base_prob: float) -> str:
    label = "malignant" if prob_malignant >= 0.5 else "benign"
    p = prob_malignant if label == "malignant" else 1 - prob_malignant
    order = np.argsort(-np.abs(shap_vec))
    up = [i for i in order if shap_vec[i] > 0][:3]
    down = [i for i in order if shap_vec[i] < 0][:2]
    lines = [f"**Prediction:** {label} ({p:.1%}, the model is {_confidence_word(p)}). "
             f"The average case starts at a {base_prob:.0%} malignancy risk."]
    if up:
        lines.append("**Pushing toward malignant:** " + ", ".join(
            f"{_pretty(features[i])} = {x.iloc[i]:.3g} (+{shap_vec[i]:.2f} log-odds)" for i in up) + ".")
    if down:
        lines.append("**Pushing toward benign:** " + ", ".join(
            f"{_pretty(features[i])} = {x.iloc[i]:.3g} ({shap_vec[i]:.2f})" for i in down) + ".")
    if cf and cf.get("flipped"):
        ch = ", ".join(f"{_pretty(f)} {a:.3g} → {b:.3g}" for f, a, b, _ in cf["changes"][:3])
        lines.append(f"**What would change the call:** the prediction becomes *{cf['target']}* if {ch}"
                     f"{' (and smaller changes)' if len(cf['changes']) > 3 else ''}.")
    lines.append("_Decision support only. A pathologist makes the diagnosis._")
    return "\n\n".join(lines)


def image_card(prob: np.ndarray, class_names, cam: np.ndarray, sanity: dict | None = None) -> str:
    c = int(prob.argmax())
    h, w = cam.shape
    ys, xs = np.unravel_index(np.argsort(-cam.ravel())[: max(1, cam.size // 10)], cam.shape)
    vert = "upper" if ys.mean() < h / 3 else "lower" if ys.mean() > 2 * h / 3 else "middle"
    # image left = patient's right on a standard PA film
    side = "patient's right" if xs.mean() < w * 0.45 else "patient's left" if xs.mean() > w * 0.55 else "central"
    spread = float((cam > 0.5).mean())
    lines = [f"**Prediction:** {class_names[c]} ({prob[c]:.0%}, {_confidence_word(prob[c])}).",
             f"**Where the model looked:** mostly the {vert} {side} region. "
             f"{spread:.0%} of the image is strongly highlighted"
             f"{', so the evidence is focal' if spread < 0.15 else ', so the evidence is diffuse'}."]
    if sanity:
        best = min(sanity, key=sanity.get)
        lines.append(f"**Trust check:** on the test set, *{best}* changed most when the model's weights "
                     f"were randomised (Spearman {sanity[best]:.2f}), so it is the map most tied to what the model learned.")
    lines.append("_Heatmaps show what drove the model, not where the disease is. A radiologist must review._")
    return "\n\n".join(lines)


def text_card(top: list, word_scores: dict) -> str:
    (c1, p1), *rest = top
    pos = [w for w, s in sorted(word_scores.items(), key=lambda kv: -kv[1]) if s > 0][:4]
    neg = [w for w, s in sorted(word_scores.items(), key=lambda kv: kv[1]) if s < 0][:3]
    lines = [f"**Most likely:** {c1} ({p1:.0%}, {_confidence_word(p1)})."]
    if rest:
        lines.append("**Also considered:** " + ", ".join(f"{c} ({p:.0%})" for c, p in rest[:2]) + ".")
    if pos:
        lines.append("**Words that pointed to it:** " + ", ".join(f"“{w}”" for w in pos) + ".")
    if neg:
        lines.append("**Words that argued against it:** " + ", ".join(f"“{w}”" for w in neg) + ".")
    lines.append("_Triage aid only. It matches phrasing patterns and does not examine the patient._")
    return "\n\n".join(lines)
