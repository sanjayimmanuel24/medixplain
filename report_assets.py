"""Extra numbers and figures used only by the written report.

    python report_assets.py
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from xai import image as I
from xai import narrative
from xai import tabular as TB
from xai import text as TX
from xai.common import FIGURES, RESULTS, save_json


def axiom_checks() -> dict:
    """SHAP local accuracy (values + base = model log-odds) and IG completeness."""
    tab = TB.TabularExplainer(TB.load())
    X = tab.b["X_test"]
    sv = tab.shap_values(X)
    margin = tab.model.predict(X, output_margin=True)
    shap_err = np.abs(sv.values.sum(1) + sv.base_values - margin)

    model = I.load_model()
    Xte, _ = I.load_split("test")
    ig_err = []
    for img in Xte[:20]:
        t = int(I.predict_proba(model, img[None])[0].argmax())
        ig = I.integrated_gradients(model, img, t, steps=256)
        with torch.no_grad():
            f = model(torch.tensor(img[None, None], device=I.DEVICE))[0, t].item()
            f0 = model(torch.zeros(1, 1, I.SIZE, I.SIZE, device=I.DEVICE))[0, t].item()
        ig_err.append(abs(ig.sum() - (f - f0)) / (abs(f - f0) + 1e-8))
    return {"shap_additivity_max_abs_error": float(shap_err.max()), "shap_n": int(len(X)),
            "ig_completeness_mean_rel_error": float(np.mean(ig_err)), "ig_n": len(ig_err)}


def sample_cards() -> dict:
    tab = TB.TabularExplainer(TB.load())
    x = tab.b["X_test"].iloc[0]
    p = float(tab.predict_proba(x.values)[0, 1])
    card_tab = narrative.tabular_card(tab.features, x, tab.shap_vector(x), p, tab.counterfactual(x), tab.base_prob(x))

    model = I.load_model()
    Xte, _ = I.load_split("test")
    prob = I.predict_proba(model, Xte[:1])[0]
    card_img = narrative.image_card(prob, I.CLASS_NAMES, I.grad_cam(model, Xte[0], int(prob.argmax())))

    tx = TX.TextExplainer(TX.load())
    t = tx.b["X_test"][3]
    top = tx.top_classes(t, 5)
    card_txt = narrative.text_card(top, tx.ngrams_to_words(tx.shap_ngrams(t, top[0][0])))
    return {"tabular": card_tab, "image": card_img, "text": card_txt}


def box(ax, x, y, w, h, title, lines, color):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                fc=color, ec="#333333", lw=1.2))
    ax.text(x + w / 2, y + h - 0.17, title, ha="center", va="top", fontsize=10.5, weight="bold")
    ax.text(x + w / 2, y + h - 0.5, "\n".join(lines), ha="center", va="top", fontsize=8.3, linespacing=1.35)


def arrow(ax, a, b):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=14, lw=1.3, color="#333333"))


def architecture_figure():
    fig, ax = plt.subplots(figsize=(11, 7.2))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 7.4)
    ax.axis("off")
    cols = (0.2, 3.85, 7.5)
    # layer 1: data
    for x, (t, l) in zip(cols, [("Tabular data", ["Breast Cancer Wisconsin", "569 biopsies × 30 features"]),
                                ("Image data", ["PneumoniaMNIST 64×64", "5,856 paediatric X-rays"]),
                                ("Text data", ["symptom_to_diagnosis", "1,065 descriptions, 22 classes"])]):
        box(ax, x, 6.0, 3.3, 1.2, t, l, "#E3EEF9")
    # layer 2: models + explainers
    for x, (t, l) in zip(cols, [("Tabular module", ["XGBoost + glass-box LR", "TreeSHAP · LIME · PDP/ICE",
                                                    "permutation · counterfactual"]),
                                ("Image module", ["4-block CNN (PyTorch)", "Grad-CAM · IG · SmoothGrad",
                                                  "saliency · occlusion · LIME"]),
                                ("Text module", ["TF-IDF + logistic regression", "LIME · SHAP · occlusion",
                                                 "linear contribution · contrastive"])]):
        box(ax, x, 3.95, 3.3, 1.55, t, l, "#FDF0DC")
    # layers 3-4
    box(ax, 0.2, 2.15, 10.6, 1.2, "Trust Lab (trust.py)",
        ["faithfulness (deletion, comprehensiveness) · sanity check (weight randomisation) · agreement · stability"], "#E5F4E3")
    box(ax, 0.2, 0.2, 10.6, 1.35, "Presentation layer",
        ["narrative.py: plain-English cards built from computed numbers",
         "app.py: Streamlit GUI, 6 pages (Overview · Tumour · X-ray · Symptoms · Trust Lab · Concepts)"], "#EFE6F7")
    for x in (1.85, 5.5, 9.15):
        arrow(ax, (x, 6.0), (x, 5.5))
        arrow(ax, (x, 3.95), (x, 3.35))
    arrow(ax, (5.5, 2.15), (5.5, 1.55))
    ax.text(0.2, 5.75, "data", fontsize=8, color="#666666", style="italic")
    ax.text(0.2, 3.62, "models + explainers", fontsize=8, color="#666666", style="italic")
    fig.tight_layout()
    fig.savefig(FIGURES / "arch_overview.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def flow_figure():
    steps = ["User picks a case\n(or uploads / types)", "Model predicts\nclass + probability",
             "Explainers compute\nattributions", "Trust checks\n(deletion, sanity, seed)",
             "Card + plots\nrendered in GUI"]
    fig, ax = plt.subplots(figsize=(11, 1.9))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 1.9)
    ax.axis("off")
    for i, s in enumerate(steps):
        x = 0.15 + i * 2.2
        ax.add_patch(FancyBboxPatch((x, 0.35), 1.85, 1.2, boxstyle="round,pad=0.02,rounding_size=0.1",
                                    fc=["#E3EEF9", "#FDF0DC", "#FDF0DC", "#E5F4E3", "#EFE6F7"][i], ec="#333333"))
        ax.text(x + 0.925, 0.95, s, ha="center", va="center", fontsize=8.4)
        if i < len(steps) - 1:
            arrow(ax, (x + 1.85, 0.95), (x + 2.2, 0.95))
    fig.savefig(FIGURES / "arch_flow.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    architecture_figure()
    flow_figure()
    out = {"axioms": axiom_checks(), "cards": sample_cards()}
    save_json(out, RESULTS / "report_extras.json")
    print(out["axioms"])
    for k, v in out["cards"].items():
        print(f"--- {k}\n{v}")
