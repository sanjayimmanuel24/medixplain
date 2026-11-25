"""MediXplain: a GUI workbench for explainable clinical AI on tabular, image and text data.

    streamlit run app.py
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from xai import image as I
from xai import narrative
from xai import tabular as TB
from xai import text as TX
from xai import trust
from xai.common import RESULTS, load_json

st.set_page_config(page_title="MediXplain: Explainable Clinical AI", page_icon="🩺", layout="wide")


def show(fig):
    st.pyplot(fig, clear_figure=True)
    plt.close(fig)


# ---------------------------------------------------------------- cached resources
@st.cache_resource(show_spinner="Loading tabular model…")
def get_tab():
    return TB.TabularExplainer(TB.load())


@st.cache_resource(show_spinner="Loading CNN…")
def get_cnn():
    return I.load_model()


@st.cache_resource(show_spinner="Loading text model…")
def get_text():
    return TX.TextExplainer(TX.load())


@st.cache_data(show_spinner=False)
def xray_test():
    return I.load_split("test")


@st.cache_data(show_spinner="Computing six attribution maps…")
def image_maps(img: np.ndarray, target: int, random_weights: bool = False):
    model = get_cnn()
    if random_weights:
        model = I.randomized_copy(model)
    return I.explain_all(model, img, target)


metrics = load_json(RESULTS / "metrics.json")

# ---------------------------------------------------------------- sidebar
st.sidebar.title("🩺 MediXplain")
st.sidebar.caption("Explainable AI for clinical decision support: tabular, image and text in one workbench.")
PAGES = ["Overview", "1 · Tumour (tabular)", "2 · Chest X-ray (image)", "3 · Symptoms (text)",
         "4 · Trust Lab", "Concepts & methods"]
_qp = st.query_params.get("page", "0")  # ?page=N deep-links to a page (used for demos and report screenshots)
page = st.sidebar.radio("Case study", PAGES, index=int(_qp) if _qp.isdigit() and int(_qp) < len(PAGES) else 0)


# ================================================================ overview
if page == "Overview":
    st.title("MediXplain: why did the model say that?")
    st.markdown(
        "A hospital is trialling three AI assistants. Clinicians will only trust them if every "
        "prediction comes with an explanation that they can **inspect, question and verify**. "
        "This workbench explains each model with several XAI methods. The **Trust Lab** then "
        "tests whether those explanations deserve trust.")
    c1, c2, c3 = st.columns(3)
    tm, im, xm = metrics.get("tabular", {}), metrics.get("image", {}), metrics.get("text", {})
    with c1:
        st.subheader("1 · Tumour biopsy")
        st.caption("Tabular · 569 patients · 30 features · XGBoost")
        if tm:
            st.metric("Test ROC-AUC", f"{tm['metrics']['xgboost']['roc_auc']:.3f}")
        st.markdown("SHAP · LIME · permutation importance · PDP/ICE · counterfactuals · glass-box model")
    with c2:
        st.subheader("2 · Chest X-ray")
        st.caption("Image · 5,856 paediatric X-rays · CNN")
        if im:
            st.metric("Test ROC-AUC", f"{im['metrics']['roc_auc']:.3f}")
        st.markdown("Grad-CAM · Integrated Gradients · SmoothGrad · saliency · occlusion · LIME superpixels")
    with c3:
        st.subheader("3 · Symptom triage")
        st.caption("Text · 1,065 descriptions · 22 diagnoses · TF-IDF + LR")
        if xm:
            st.metric("Test accuracy", f"{xm['metrics']['accuracy']:.3f}")
        st.markdown("LIME text · SHAP · word occlusion · linear contributions · contrastive *why A not B*")
    st.divider()
    st.subheader("What is new here")
    st.markdown(
        "- **Trust Lab**: every explanation is scored for *faithfulness* (deletion curves, ERASER "
        "comprehensiveness), *stability* (seed and input noise), *model sensitivity* (Adebayo "
        "weight-randomisation sanity check) and *cross-method agreement*.\n"
        "- **Counterfactuals**: a minimal-change search shows what would have to differ to flip a call.\n"
        "- **Plain-English cards**: every sentence is filled from computed numbers, so a clinician "
        "can check it against the plot.\n"
        "- **Glass-box baseline**: a logistic regression sits beside the black box. Is the black box worth its opacity?")
    if not metrics:
        st.error("No results yet. Run `python train_all.py` first.")


# ================================================================ tabular
elif page.startswith("1"):
    tab = get_tab()
    b = tab.b
    X, y = b["X_test"], b["y_test"]
    st.title("Case 1 · Is this breast tumour malignant?")
    st.caption("Wisconsin Diagnostic Breast Cancer data. 30 measurements of cell nuclei from a fine-needle "
               "aspirate. The model is XGBoost; a logistic regression is the glass-box reference.")

    left, right = st.columns([1, 2])
    with left:
        i = st.selectbox("Test patient", range(len(X)),
                         format_func=lambda k: f"#{k}  (true: {TB.CLASS_NAMES[int(y.iloc[k])]})")
        x = X.iloc[i].copy()
        with st.expander("✏️ What-if: edit the top features", expanded=False):
            sv0 = tab.shap_vector(x)
            for j in np.argsort(-np.abs(sv0))[:6]:
                f = tab.features[j]
                lo, hi = float(b["X_train"][f].min()), float(b["X_train"][f].max())
                x[f] = st.slider(f, lo, hi, float(x[f]), (hi - lo) / 200, key=f"wf_{i}_{f}")
        p = float(tab.predict_proba(x.values)[0, 1])
        st.metric("P(malignant)", f"{p:.1%}", delta="malignant" if p >= .5 else "benign",
                  delta_color="inverse" if p >= .5 else "normal")
        st.caption(f"Glass-box logistic regression says {b['glass'].predict_proba(x.to_frame().T)[0, 1]:.1%}")
        sv = tab.shap_vector(x)
        cf = tab.counterfactual(x)
        st.info(narrative.tabular_card(tab.features, x, sv, p, cf, tab.base_prob(x)))

    with right:
        t1, t2, t3, t4 = st.tabs(["Local: SHAP", "Local: LIME", "Counterfactual", "Global view"])
        with t1:
            st.markdown("**SHAP waterfall.** Starts from the average model output E[f(x)] in log-odds. "
                        "Each bar is one feature's Shapley value, and the bars add up exactly to this patient's output.")
            show(tab.plot_shap_waterfall(x))
        with t2:
            st.markdown("**LIME.** Samples perturbed patients around this one and fits a weighted linear "
                        "surrogate. The bars are the surrogate's weights on readable feature ranges.")
            seed = st.number_input("LIME random seed (change it to see how stable LIME is)", 0, 999, 42)
            fig, exp = tab.plot_lime(x, seed=int(seed))
            show(fig)
            st.caption(f"Local surrogate R² = {exp.score:.3f}. Low values mean the linear approximation is poor here.")
        with t3:
            st.markdown("**Counterfactual explanation.** What is the smallest change to the measurements "
                        "that flips the model's decision? Moves are in standard deviations and stay inside "
                        "the range seen in training.")
            if cf["flipped"]:
                st.success(f"Flipped to **{cf['target']}**: P(malignant) {cf['start_prob']:.1%} → {cf['end_prob']:.1%} "
                           f"by changing {len(cf['changes'])} feature(s) (total {cf['l1_std_distance']:.2f} SD).")
            else:
                st.warning("No counterfactual found within the search budget.")
            st.dataframe(pd.DataFrame(cf["changes"], columns=["feature", "current", "counterfactual", "change (SD)"])
                         .round(4), hide_index=True, width="stretch")
        with t4:
            st.markdown("**Three views of global importance.** SHAP (model-specific, exact for trees), "
                        "permutation importance (model-agnostic, drop in test AUC) and glass-box coefficients.")
            show(tab.plot_global_importance(X))
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**SHAP beeswarm.** One dot per patient. Colour is the feature value.")
                show(tab.plot_shap_beeswarm(X))
            with c2:
                feat = st.selectbox("Feature for dependence / PDP", tab.features,
                                    index=tab.features.index("worst_concave_points"))
                show(tab.plot_shap_dependence(X, feat))
                show(tab.plot_pdp(feat))


# ================================================================ image
elif page.startswith("2"):
    model = get_cnn()
    Xte, yte = xray_test()
    st.title("Case 2 · Does this chest X-ray show pneumonia?")
    st.caption("PneumoniaMNIST (MedMNIST v2, CC BY 4.0) at 64×64. The model is a 4-block CNN trained in this project.")
    left, right = st.columns([1, 2.2])
    with left:
        src = st.radio("Image source", ["Test set", "Upload your own"], horizontal=True)
        if src == "Test set":
            k = st.slider("Test image", 0, len(Xte) - 1, 0)
            img, true = Xte[k], I.CLASS_NAMES[yte[k]]
        else:
            up = st.file_uploader("Chest X-ray (PNG/JPG)", type=["png", "jpg", "jpeg"])
            if up is None:
                st.stop()
            img, true = I.preprocess_upload(Image.open(up)), "unknown"
        prob = I.predict_proba(model, img[None])[0]
        tgt = int(prob.argmax())
        st.image(img, caption=f"True label: {true}", width=220, clamp=True)
        st.metric("Prediction", I.CLASS_NAMES[tgt], f"{prob[tgt]:.1%} confidence", delta_color="off")
        explain_cls = st.radio("Explain the evidence for", I.CLASS_NAMES, index=tgt, horizontal=True)
        target = I.CLASS_NAMES.index(explain_cls)
        maps = image_maps(img, target)
        st.info(narrative.image_card(prob, I.CLASS_NAMES, maps["Grad-CAM"], metrics.get("image", {}).get("sanity_spearman")))
    with right:
        t1, t2, t3 = st.tabs(["Attribution maps", "Faithfulness on this image", "Sanity check"])
        with t1:
            chosen = st.multiselect("Methods", list(maps), default=list(maps))
            show(I.plot_maps(img, {m: maps[m] for m in chosen}))
            st.caption("Red/jet = evidence for the class. In the signed maps (IG, LIME), blue = evidence against it.")
        with t2:
            st.markdown("**Deletion test.** Pixels are removed in order of importance. "
                        "If a map is faithful, the probability collapses quickly. Lower AUC is better.")
            fig, ax = plt.subplots(figsize=(7, 4))
            rows = []
            rnd_map = np.random.default_rng(0).random(img.shape)
            for name, m in list(maps.items()) + [("Random", rnd_map)]:
                c = I.deletion_curve(model, img, m, target)
                ax.plot(np.linspace(0, 100, len(c)), c, label=name, ls="--" if name == "Random" else "-")
                rows.append((name, float(np.trapezoid(c, dx=1 / (len(c) - 1)))))
            ax.set_xlabel("% pixels removed")
            ax.set_ylabel(f"P({explain_cls})")
            ax.legend(fontsize=8)
            show(fig)
            st.dataframe(pd.DataFrame(rows, columns=["method", "deletion AUC"]).sort_values("deletion AUC").round(3),
                         hide_index=True)
        with t3:
            st.markdown("**Model-randomisation sanity check** (Adebayo et al., 2018). If a map looks the same "
                        "after the CNN's weights are scrambled, it explains the *image*, not the *model*.")
            rmaps = image_maps(img, target, random_weights=True)
            sel = st.selectbox("Method", list(I.METHODS))
            c1, c2 = st.columns(2)
            with c1:
                show(I.plot_maps(img, {f"{sel}: trained": maps[sel]}))
            with c2:
                show(I.plot_maps(img, {f"{sel}: random weights": rmaps[sel]}))
            from scipy.stats import spearmanr
            r = spearmanr(np.abs(maps[sel]).ravel(), np.abs(rmaps[sel]).ravel()).correlation
            st.metric("Spearman similarity (lower = method passes)", f"{0 if np.isnan(r) else r:.2f}")


# ================================================================ text
elif page.startswith("3"):
    tx = get_text()
    st.title("Case 3 · What might this patient have?")
    st.caption("A patient describes their symptoms in their own words, and the model suggests one of 22 diagnoses. "
               "Data: gretelai/symptom_to_diagnosis. Model: TF-IDF (1-2 grams) + logistic regression.")
    examples = {f"Test #{i}: {tx.b['y_test'][i]}": tx.b["X_test"][i] for i in (3, 10, 25, 60, 100, 150)}
    ex = st.selectbox("Start from an example (or type your own below)", list(examples))
    text = st.text_area("Symptom description", examples[ex], height=110)
    if not text.strip():
        st.stop()
    top = tx.top_classes(text, 5)
    method = st.radio("Highlight words by", ["SHAP", "LIME", "Occlusion (leave-one-word-out)", "Linear contribution"],
                      horizontal=True)
    left, right = st.columns([1, 2])
    with left:
        fig, ax = plt.subplots(figsize=(5, 2.6))
        ax.barh([c for c, _ in top][::-1], [p for _, p in top][::-1], color="#6a51a3")
        ax.set_xlim(0, 1)
        ax.set_xlabel("probability")
        show(fig)
        cls = st.selectbox("Explain class", tx.classes, index=tx.classes.index(top[0][0]))
    scores = {
        "SHAP": lambda: tx.ngrams_to_words(tx.shap_ngrams(text, cls)),
        "LIME": lambda: tx.lime_words(text, cls, seed=42),
        "Occlusion (leave-one-word-out)": lambda: tx.occlusion_words(text, cls),
        "Linear contribution": lambda: tx.ngrams_to_words(tx.linear_contributions(text, cls)),
    }[method]()
    with left:
        st.info(narrative.text_card(top, scores))
    with right:
        st.markdown(f"**{method}**: red words support *{cls}*, blue words argue against it. Hover a word to see its score.")
        st.markdown(tx.highlight_html(text, scores), unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            show(TX.TextExplainer.plot_bars(scores, f"{method} → {cls}"))
        with c2:
            others = [c for c in tx.classes if c != cls]
            runner_up = next((c for c, _ in top if c != cls), others[0])
            other = st.selectbox("Contrast: why this class and not…", others, index=others.index(runner_up))
            show(TX.TextExplainer.plot_bars(tx.ngrams_to_words(tx.contrastive(text, cls, other)),
                                            f"why '{cls}' (red) rather than '{other}' (blue)"))
        with st.expander(f"Global view: words the model associates with '{cls}'"):
            show(TX.TextExplainer.plot_bars(dict(tx.global_top_words(cls, 15)), f"top coefficients for {cls}", 15))


# ================================================================ trust lab
elif page.startswith("4"):
    st.title("Trust Lab: can we trust the explanations?")
    st.markdown("Explanations are model outputs too, and they can be wrong. Four properties are tested "
                "on held-out data. The numbers come from `train_all.py` and are saved in `results/metrics.json`.")
    if not metrics:
        st.error("Run `python train_all.py` first.")
        st.stop()
    tm, im, xm = metrics["tabular"], metrics["image"], metrics["text"]

    st.header("1. Faithfulness: does removing the 'important' evidence change the prediction?")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**Tabular**: drop in P(predicted class) after resetting the top-5 features to the mean (↑ better)")
        st.bar_chart(pd.Series(tm["faithfulness_top5_drop"], name="drop"), horizontal=True)
    with c2:
        st.markdown("**Image**: deletion-curve AUC on 30 test X-rays (↓ better)")
        st.bar_chart(pd.Series(im["deletion_auc"], name="AUC"), horizontal=True)
    with c3:
        st.markdown("**Text**: ERASER comprehensiveness, top-3 words deleted (↑ better)")
        st.bar_chart(pd.Series(xm["comprehensiveness_top3"], name="drop"), horizontal=True)

    st.header("2. Sanity check: does the explanation depend on the model?")
    st.markdown("Spearman similarity between maps from the trained CNN and from the same CNN with random weights. "
                "**Low is good.** A high score means the map mostly shows image edges.")
    st.bar_chart(pd.Series(im["sanity_spearman"], name="similarity"), horizontal=True)

    st.header("3. Agreement: do independent methods point at the same evidence?")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Tabular** (40 test patients)")
        st.dataframe(pd.DataFrame(tm["agreement"]).T.round(3), width="stretch")
    with c2:
        st.markdown("**Text**: top-3 word overlap (40 test descriptions)")
        st.dataframe(pd.Series(xm["agreement_top3"], name="top-3 overlap").round(3), width="stretch")
    ag = im["method_agreement_example"]
    fig, ax = plt.subplots(figsize=(6, 4.5))
    mat = np.array(ag["matrix"])
    ax.imshow(mat, cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(len(ag["methods"])), ag["methods"], rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(ag["methods"])), ag["methods"], fontsize=8)
    for a in range(len(mat)):
        for b_ in range(len(mat)):
            ax.text(b_, a, f"{mat[a, b_]:.2f}", ha="center", va="center", fontsize=7, color="w")
    ax.set_title("Image: Spearman agreement between maps (example X-ray)")
    c1, _ = st.columns([1, 1])
    with c1:
        show(fig)

    st.header("4. Stability: does the explanation survive tiny changes?")
    st.dataframe(pd.Series(tm["stability"], name="Spearman").round(3), width="stretch")
    st.caption(f"Text LIME seed-to-seed Spearman: {xm['lime_seed_stability']:.3f}")
    with st.expander("🔬 Run a live stability test on any tumour patient"):
        tab = get_tab()
        k = st.number_input("Test patient", 0, len(tab.b["X_test"]) - 1, 5)
        if st.button("Run 5 LIME seeds + 5 noisy copies"):
            with st.spinner("Explaining 15 times…"):
                st.json(trust.tabular_stability(tab, tab.b["X_test"].iloc[int(k)]))

    st.header("Takeaways")
    st.markdown(
        f"- **SHAP and LIME mostly agree on tabular data** (top-5 overlap {tm['agreement']['SHAP vs LIME']['top5_overlap']:.0%}). "
        "Both agree less with the glass-box coefficients, so the black box and the glass box use the features differently.\n"
        f"- **LIME depends on its random seed** (seed-to-seed Spearman {tm['stability']['LIME seed-to-seed spearman']:.2f}). "
        "TreeSHAP is exact, so it gives the same answer every time.\n"
        "- **Some popular saliency methods fail the sanity check.** Check the chart above before trusting a heatmap.\n"
        "- **On text, every method beats random deletion by a wide margin**, because the model is near-linear.")


# ================================================================ concepts
else:
    st.title("Concepts & methods")
    st.markdown(r"""
### Taxonomy used in this project
| | **Local** (one prediction) | **Global** (whole model) |
|---|---|---|
| **Model-agnostic** | LIME (tabular, image, text), occlusion, counterfactuals | Permutation importance, PDP/ICE |
| **Model-specific** | TreeSHAP, Grad-CAM, Integrated Gradients, SmoothGrad, saliency | Mean \|SHAP\|, glass-box coefficients |
| **Intrinsic** | Logistic-regression contributions $w_j x_j$ | Coefficients |

### Methods in one line each
- **SHAP**: Shapley values from cooperative game theory, $\phi_j=\sum_{S}\frac{|S|!(M-|S|-1)!}{M!}[f(S\cup j)-f(S)]$. They add up to the prediction minus the baseline (*local accuracy*). TreeSHAP computes them exactly for trees.
- **LIME**: fits a sparse linear model $g$ to minimise $\mathcal{L}(f,g,\pi_x)+\Omega(g)$ on perturbed samples near $x$.
- **Permutation importance**: drop in test AUC when one feature is shuffled.
- **PDP / ICE**: average (PDP) and per-patient (ICE) response as one feature sweeps its range.
- **Counterfactual**: the smallest $\|x'-x\|$ such that $f(x')$ changes class, kept within realistic ranges.
- **Grad-CAM**: $\mathrm{ReLU}(\sum_k \alpha_k A^k)$ where $\alpha_k$ is the mean gradient of the class score over feature map $A^k$.
- **Integrated Gradients**: $(x-x')\int_0^1 \partial f(x'+\alpha(x-x'))/\partial x\,d\alpha$. Satisfies *completeness*.
- **SmoothGrad**: the average gradient over noisy copies of the input, which reduces visual noise.
- **Occlusion**: the drop in probability when a patch (or word) is hidden.

### Evaluation of explanations
Deletion curves (Petsiuk et al., 2018), ERASER comprehensiveness (DeYoung et al., 2020), the model-randomisation sanity check (Adebayo et al., 2018), seed and input stability, and cross-method agreement.

### Limitations and ethics
- Attributions show **what the model used**, not what causes disease. A model can be right for the wrong reasons, for example by reading text markers burned into an X-ray.
- The X-rays are low resolution (64×64). The symptom data is synthetic patient phrasing. None of these models is clinically validated.
- Counterfactuals on biopsy measurements are *explanatory*, not *actionable*. A patient cannot change their cell nuclei.
""")
