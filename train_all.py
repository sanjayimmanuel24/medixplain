"""Train the three models, run every Trust Lab test and save the figures used in the report.

    python train_all.py            # about 3-5 minutes on a laptop GPU
"""
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from xai import image as I
from xai import tabular as TB
from xai import text as TX
from xai import trust
from xai.common import FIGURES, RESULTS, save_json, seed_everything


def save(fig, name):
    fig.savefig(FIGURES / name, dpi=130, bbox_inches="tight")
    plt.close(fig)


def main():
    seed_everything()
    t0 = time.time()
    out = {}

    print("== Tabular: breast tumour malignancy")
    tb = TB.train()
    tab = TB.TabularExplainer(tb)
    out["tabular"] = {"metrics": tb["metrics"], "n_train": len(tb["X_train"]), "n_test": len(tb["X_test"])}
    x = tb["X_test"].iloc[0]
    save(tab.plot_global_importance(tb["X_test"]), "tab_global_importance.png")
    save(tab.plot_shap_beeswarm(tb["X_test"]), "tab_shap_beeswarm.png")
    save(tab.plot_shap_waterfall(x), "tab_shap_waterfall.png")
    save(tab.plot_lime(x)[0], "tab_lime.png")
    save(tab.plot_pdp("worst_concave_points"), "tab_pdp_ice.png")
    save(tab.plot_shap_dependence(tb["X_test"], "worst_area"), "tab_shap_dependence.png")
    cfs = [tab.counterfactual(r) for _, r in tb["X_test"].iloc[:40].iterrows()]
    out["tabular"]["counterfactual"] = {
        "flip_rate": float(np.mean([c["flipped"] for c in cfs])),
        "mean_features_changed": float(np.mean([len(c["changes"]) for c in cfs])),
        "mean_l1_std_distance": float(np.mean([c["l1_std_distance"] for c in cfs])),
        "example": tab.counterfactual(x),
    }
    out["tabular"]["agreement"] = trust.tabular_agreement(tab)
    out["tabular"]["stability"] = trust.tabular_stability(tab, x)
    out["tabular"]["faithfulness_top5_drop"] = trust.tabular_faithfulness(tab)
    print(out["tabular"]["metrics"], out["tabular"]["agreement"])

    print("== Image: pneumonia on chest X-ray")
    out["image"] = {"metrics": I.train()}
    model = I.load_model()
    Xte, yte = I.load_split("test")
    pos = int(np.where(yte == 1)[0][0])
    neg = int(np.where(yte == 0)[0][0])
    for idx, tag in [(pos, "pneumonia"), (neg, "normal")]:
        img = Xte[idx]
        tgt = int(I.predict_proba(model, img[None])[0].argmax())
        save(I.plot_maps(img, I.explain_all(model, img, tgt),
                         f"True: {I.CLASS_NAMES[yte[idx]]}, predicted: {I.CLASS_NAMES[tgt]}"), f"img_maps_{tag}.png")
    rnd = I.randomized_copy(model)
    img = Xte[pos]
    save(I.plot_maps(img, {f"{n} (random weights)": f(rnd, img, 1) for n, f in I.METHODS.items()},
                     "Sanity check: the same methods on a model with random weights"), "img_sanity.png")
    # balanced evaluation set for the explanation tests
    eval_idx = np.r_[np.where(yte == 1)[0][:15], np.where(yte == 0)[0][:15]]
    out["image"]["deletion_auc"] = trust.image_faithfulness(model, Xte[eval_idx])
    out["image"]["sanity_spearman"] = trust.image_sanity(model, Xte[eval_idx], n=20)
    names, M = trust.image_method_agreement(model, img, 1)
    out["image"]["method_agreement_example"] = {"methods": names, "matrix": M.round(3).tolist()}
    fig, ax = plt.subplots(figsize=(6, 4))
    for name, f in list(I.METHODS.items()) + [("Random", None)]:
        m = np.random.default_rng(0).random(img.shape) if f is None else f(model, img, 1)
        ax.plot(np.linspace(0, 100, 21), I.deletion_curve(model, img, m, 1), label=name,
                ls="--" if f is None else "-")
    ax.set_xlabel("% of pixels removed (most important first)")
    ax.set_ylabel("P(pneumonia)")
    ax.legend(fontsize=8)
    ax.set_title("Deletion curves (faster drop = more faithful)")
    save(fig, "img_deletion_curves.png")
    print(out["image"]["metrics"], out["image"]["deletion_auc"], out["image"]["sanity_spearman"])

    print("== Text: symptom description -> diagnosis")
    tx_b = TX.train()
    tx = TX.TextExplainer(tx_b)
    out["text"] = {"metrics": tx_b["metrics"], "n_classes": len(tx.classes),
                   "n_train": len(tx_b["X_train"]), "n_test": len(tx_b["X_test"])}
    out["text"]["agreement_top3"] = trust.text_agreement(tx)
    out["text"]["comprehensiveness_top3"] = trust.text_comprehensiveness(tx)
    out["text"]["lime_seed_stability"] = trust.text_lime_stability(tx, tx_b["X_test"][3])
    t = tx_b["X_test"][3]
    cls = tx.top_classes(t, 1)[0][0]
    save(TX.TextExplainer.plot_bars(tx.ngrams_to_words(tx.shap_ngrams(t, cls)), f"SHAP words for '{cls}'"),
         "txt_shap.png")
    save(TX.TextExplainer.plot_bars(tx.lime_words(t, cls, seed=0), f"LIME words for '{cls}'"), "txt_lime.png")
    print(out["text"])

    out["runtime_seconds"] = round(time.time() - t0, 1)
    save_json(out, RESULTS / "metrics.json")
    print(f"done in {out['runtime_seconds']} s -> results/metrics.json")


if __name__ == "__main__":
    main()
