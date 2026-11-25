"""Tabular case: breast-tumour malignancy from 30 cell-nucleus measurements.

Black-box model: XGBoost. Glass-box reference: standardised logistic regression.
Explanations: SHAP (TreeExplainer), LIME, permutation importance, PDP/ICE,
and a greedy counterfactual search ("what would have to change to flip the call?").
"""
from __future__ import annotations

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.inspection import PartialDependenceDisplay, permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from .common import MODELS, SEED

CLASS_NAMES = ["benign", "malignant"]
BUNDLE = MODELS / "tabular.joblib"


def load_data():
    raw = load_breast_cancer(as_frame=True)
    X = raw.data
    X.columns = [c.replace(" ", "_") for c in X.columns]
    y = (raw.target == 0).astype(int)  # sklearn uses 0 = malignant; we make malignant the positive class
    return X, y.rename("malignant")


def train() -> dict:
    X, y = load_data()
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, stratify=y, random_state=SEED)

    xgb = XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.9,
                        colsample_bytree=0.8, eval_metric="logloss", random_state=SEED)
    xgb.fit(X_tr, y_tr)
    glass = make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=5000))
    glass.fit(X_tr, y_tr)

    def scores(m):
        p = m.predict_proba(X_te)[:, 1]
        return {"accuracy": accuracy_score(y_te, p > 0.5), "f1": f1_score(y_te, p > 0.5),
                "roc_auc": roc_auc_score(y_te, p)}

    perm = permutation_importance(xgb, X_te, y_te, n_repeats=20, random_state=SEED, scoring="roc_auc")
    bundle = {
        "xgb": xgb, "glass": glass, "X_train": X_tr, "X_test": X_te, "y_train": y_tr, "y_test": y_te,
        "perm_mean": perm.importances_mean, "perm_std": perm.importances_std,
        "metrics": {"xgboost": scores(xgb), "logistic_regression": scores(glass)},
    }
    joblib.dump(bundle, BUNDLE)
    return bundle


def load() -> dict:
    return joblib.load(BUNDLE) if BUNDLE.exists() else train()


# ---------------------------------------------------------------- explainers
class TabularExplainer:
    def __init__(self, bundle: dict):
        import shap
        from lime.lime_tabular import LimeTabularExplainer

        self.b = bundle
        self.model = bundle["xgb"]
        self.features = list(bundle["X_train"].columns)
        self.shap = shap.TreeExplainer(self.model)
        self.lime = LimeTabularExplainer(
            bundle["X_train"].values, feature_names=self.features, class_names=CLASS_NAMES,
            discretize_continuous=True, mode="classification", random_state=SEED)

    def predict_proba(self, X) -> np.ndarray:
        X = pd.DataFrame(np.atleast_2d(X), columns=self.features)
        return self.model.predict_proba(X)

    # SHAP values are in log-odds of "malignant"
    def shap_values(self, X: pd.DataFrame):
        return self.shap(X)

    def shap_vector(self, x: pd.Series) -> np.ndarray:
        return self.shap(x.to_frame().T).values[0]

    def base_prob(self, x: pd.Series) -> float:
        """P(malignant) of the SHAP baseline E[f(X)]. Read from an explanation, because
        TreeExplainer.expected_value is only final after the first call."""
        return float(1 / (1 + np.exp(-np.ravel(self.shap(x.to_frame().T).base_values)[0])))

    def lime_explanation(self, x: pd.Series, num_features: int = 10, num_samples: int = 2000, seed=None):
        if seed is not None:
            self.lime.random_state = np.random.RandomState(seed)
        return self.lime.explain_instance(x.values, self.predict_proba, num_features=num_features,
                                          num_samples=num_samples, labels=(1,))

    def lime_vector(self, x: pd.Series, num_samples: int = 2000, seed=None) -> np.ndarray:
        """LIME weights for every feature, aligned to self.features (for agreement metrics)."""
        exp = self.lime_explanation(x, num_features=len(self.features), num_samples=num_samples, seed=seed)
        w = np.zeros(len(self.features))
        for idx, weight in exp.as_map()[1]:
            w[idx] = weight
        return w

    def counterfactual(self, x: pd.Series, max_steps: int = 80) -> dict:
        """Greedy minimal-change search toward the other class.

        At each step every feature may move 0.25, 0.5 or 1 standard deviation toward the
        median of the opposite class (never leaving the range seen in training). The move
        with the largest probability gain per unit of standardised distance is kept. Tree
        models have flat regions, so if no move helps we take the largest step on the
        feature with the biggest |SHAP| to cross the plateau. Stops when the class flips.
        """
        Xtr, ytr = self.b["X_train"], self.b["y_train"]
        sd = Xtr.std().replace(0, 1)
        lo, hi = Xtr.min(), Xtr.max()
        start_p = self.predict_proba(x.values)[0, 1]
        target = 0 if start_p >= 0.5 else 1
        goal = Xtr[ytr == target].median()
        cur = x.copy().astype(float)
        sign = -1 if target == 0 else 1  # direction we want P(malignant) to move
        shap_rank = np.abs(self.shap_vector(x))

        for _ in range(max_steps):
            p_now = self.predict_proba(cur.values)[0, 1]
            if (p_now >= 0.5) == bool(target):
                break
            cands = []
            for i, f in enumerate(self.features):
                gap = goal[f] - cur[f]
                if abs(gap) < 1e-9:
                    continue
                for step in (0.25, 0.5, 1.0):
                    trial = cur.copy()
                    move = np.sign(gap) * min(abs(gap), step * sd[f])
                    trial[f] = np.clip(cur[f] + move, lo[f], hi[f])
                    cands.append((i, abs(move) / sd[f], trial))
            if not cands:
                break
            probs = self.predict_proba(np.stack([t.values for *_, t in cands]))[:, 1]
            gains = sign * (probs - p_now) / np.array([d for _, d, _ in cands])
            best = int(np.argmax(gains))
            if gains[best] <= 0:  # plateau: push the most influential feature that can still move
                best = max(range(len(cands)), key=lambda j: (shap_rank[cands[j][0]], cands[j][1]))
            cur = cands[best][2]

        changed = [(f, float(x[f]), float(cur[f]), float((cur[f] - x[f]) / sd[f]))
                   for f in self.features if abs(cur[f] - x[f]) > 1e-9]
        changed.sort(key=lambda r: -abs(r[3]))
        end_p = self.predict_proba(cur.values)[0, 1]
        return {"start_prob": float(start_p), "end_prob": float(end_p),
                "flipped": bool((end_p >= 0.5) != (start_p >= 0.5)), "target": CLASS_NAMES[target],
                "changes": changed, "l1_std_distance": float(sum(abs(c[3]) for c in changed))}

    # ------------------------------------------------------------ plots
    def plot_shap_waterfall(self, x: pd.Series, max_display: int = 12):
        import shap

        exp = self.shap(x.to_frame().T)[0]
        plt.figure()
        shap.plots.waterfall(exp, max_display=max_display, show=False)
        fig = plt.gcf()
        fig.set_size_inches(8, 6)
        fig.tight_layout()
        return fig

    def plot_shap_beeswarm(self, X: pd.DataFrame, max_display: int = 15):
        import shap

        plt.figure()
        shap.plots.beeswarm(self.shap(X), max_display=max_display, show=False)
        fig = plt.gcf()
        fig.set_size_inches(8, 6.5)
        fig.tight_layout()
        return fig

    def plot_shap_dependence(self, X: pd.DataFrame, feature: str):
        import shap

        sv = self.shap(X)
        plt.figure()
        shap.plots.scatter(sv[:, feature], color=sv, show=False)
        fig = plt.gcf()
        fig.set_size_inches(7, 4.5)
        fig.tight_layout()
        return fig

    def plot_global_importance(self, X: pd.DataFrame, top: int = 12):
        sv = np.abs(self.shap(X).values).mean(0)
        perm = self.b["perm_mean"]
        coef = np.abs(self.b["glass"][-1].coef_[0])
        order = np.argsort(-sv)[:top][::-1]
        fig, axes = plt.subplots(1, 3, figsize=(13, 5), sharey=True)
        names = [self.features[i] for i in order]
        for ax, vals, title, col in [
            (axes[0], sv, "Mean |SHAP| (XGBoost)", "#1f77b4"),
            (axes[1], perm, "Permutation importance (drop in AUC)", "#ff7f0e"),
            (axes[2], coef, "|Coefficient| (glass-box logistic reg.)", "#2ca02c"),
        ]:
            ax.barh(names, vals[order], color=col)
            ax.set_title(title, fontsize=10)
        fig.tight_layout()
        return fig

    def plot_pdp(self, feature: str):
        fig, ax = plt.subplots(figsize=(7, 4.5))
        X = self.b["X_test"]
        PartialDependenceDisplay.from_estimator(
            self.model, X, [feature], kind="both", subsample=60, random_state=SEED, ax=ax,
            ice_lines_kw={"alpha": 0.25, "color": "#999999"}, pd_line_kw={"color": "#d62728", "lw": 3})
        ax.set_title(f"PDP (red) + ICE (grey) for {feature}")
        fig.tight_layout()
        return fig

    def plot_lime(self, x: pd.Series, num_features: int = 10, seed: int = SEED):
        exp = self.lime_explanation(x, num_features=num_features, seed=seed)
        items = exp.as_list(label=1)[::-1]
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.barh([k for k, _ in items], [v for _, v in items],
                color=["#d62728" if v > 0 else "#1f77b4" for _, v in items])
        ax.axvline(0, color="k", lw=0.8)
        ax.set_xlabel("LIME weight toward 'malignant'")
        ax.set_title("LIME local surrogate")
        fig.tight_layout()
        return fig, exp
