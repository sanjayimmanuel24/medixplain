"""Text case: predict one of 22 diagnoses from a patient's own symptom description.

Data: gretelai/symptom_to_diagnosis (Hugging Face, Apache-2.0), 853 train / 212 test.
Model: TF-IDF (1-2 grams) + multinomial logistic regression.
Explanations: LIME text, SHAP (LinearExplainer), leave-one-word-out occlusion,
the model's own linear contributions, and a contrastive "why A and not B?" view.
"""
from __future__ import annotations

import html
import json
import re

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import make_pipeline

from .common import DATA, MODELS, SEED

BUNDLE = MODELS / "text.joblib"
TOKEN = re.compile(r"(?u)\b\w\w+\b")


def load_split(split: str):
    rows = [json.loads(l) for l in (DATA / f"symptom_{split}.jsonl").read_text(encoding="utf-8").splitlines() if l]
    return [r["input_text"] for r in rows], [r["output_text"] for r in rows]


def train() -> dict:
    Xtr, ytr = load_split("train")
    Xte, yte = load_split("test")
    pipe = make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, stop_words="english"),
        LogisticRegression(C=10, max_iter=5000),
    )
    pipe.fit(Xtr, ytr)
    pred = pipe.predict(Xte)
    bundle = {"pipe": pipe, "X_train": Xtr, "y_train": ytr, "X_test": Xte, "y_test": yte,
              "metrics": {"accuracy": accuracy_score(yte, pred), "macro_f1": f1_score(yte, pred, average="macro")}}
    joblib.dump(bundle, BUNDLE)
    return bundle


def load() -> dict:
    return joblib.load(BUNDLE) if BUNDLE.exists() else train()


class TextExplainer:
    def __init__(self, bundle: dict):
        import shap
        from lime.lime_text import LimeTextExplainer

        self.b = bundle
        self.pipe = bundle["pipe"]
        self.vec, self.clf = self.pipe[0], self.pipe[-1]
        self.classes = [str(c) for c in self.clf.classes_]
        self.vocab = np.array(self.vec.get_feature_names_out())
        self.lime = LimeTextExplainer(class_names=self.classes, random_state=SEED)
        background = self.vec.transform(bundle["X_train"])
        self.shap = shap.LinearExplainer(self.clf, shap.maskers.Independent(background, max_samples=len(bundle["X_train"])))

    def predict_proba(self, texts) -> np.ndarray:
        return self.pipe.predict_proba(list(texts))

    def top_classes(self, text: str, k: int = 5):
        p = self.predict_proba([text])[0]
        idx = np.argsort(-p)[:k]
        return [(self.classes[i], float(p[i])) for i in idx]

    # --------------------------------------------------- word-level attributions
    def words(self, text: str):
        return TOKEN.findall(text.lower())

    def linear_contributions(self, text: str, cls: str) -> dict:
        """Exact contribution of every active n-gram: tfidf value x class coefficient."""
        v = self.vec.transform([text])
        c = self.classes.index(cls)
        coef = self.clf.coef_[c]
        return {self.vocab[j]: float(v[0, j] * coef[j]) for j in v.nonzero()[1]}

    def contrastive(self, text: str, cls_a: str, cls_b: str) -> dict:
        """Why A rather than B? Positive = pushes toward A, negative = toward B."""
        v = self.vec.transform([text])
        a, b = self.classes.index(cls_a), self.classes.index(cls_b)
        diff = self.clf.coef_[a] - self.clf.coef_[b]
        return {self.vocab[j]: float(v[0, j] * diff[j]) for j in v.nonzero()[1]}

    def shap_ngrams(self, text: str, cls: str) -> dict:
        v = self.vec.transform([text])
        sv = self.shap(v)
        c = self.classes.index(cls)
        vals = sv.values[0, :, c] if sv.values.ndim == 3 else sv.values[0]
        return {self.vocab[j]: float(vals[j]) for j in v.nonzero()[1]}

    def lime_words(self, text: str, cls: str, num_features: int = 12, num_samples: int = 1500, seed=None) -> dict:
        if seed is not None:
            self.lime.random_state = np.random.RandomState(seed)
        c = self.classes.index(cls)
        exp = self.lime.explain_instance(text, self.predict_proba, labels=(c,),
                                         num_features=num_features, num_samples=num_samples)
        out: dict[str, float] = {}
        for w, s in exp.as_list(label=c):
            out[str(w).lower()] = out.get(str(w).lower(), 0.0) + float(s)
        return out

    def occlusion_words(self, text: str, cls: str) -> dict:
        """Drop in P(cls) when each distinct word is deleted from the sentence."""
        c = self.classes.index(cls)
        base = self.predict_proba([text])[0, c]
        uniq = list(dict.fromkeys(self.words(text)))
        variants = [re.sub(rf"\b{re.escape(w)}\b", " ", text, flags=re.I) for w in uniq]
        probs = self.predict_proba(variants)[:, c] if variants else []
        return {w: float(base - p) for w, p in zip(uniq, probs)}

    @staticmethod
    def ngrams_to_words(scores: dict) -> dict:
        """Spread each bigram's score over its two words so maps can be compared word by word."""
        out: dict[str, float] = {}
        for ng, s in scores.items():
            parts = ng.split()
            for w in parts:
                out[w] = out.get(w, 0.0) + s / len(parts)
        return out

    def global_top_words(self, cls: str, k: int = 12):
        c = self.classes.index(cls)
        coef = self.clf.coef_[c]
        idx = np.argsort(-coef)[:k]
        return [(self.vocab[i], float(coef[i])) for i in idx]

    # --------------------------------------------------- rendering
    @staticmethod
    def highlight_html(text: str, word_scores: dict) -> str:
        """Colour each word: red supports the class, blue argues against it."""
        lim = max([abs(v) for v in word_scores.values()] + [1e-9])
        out = []
        for tok in re.split(r"(\W+)", text):
            s = word_scores.get(tok.lower(), 0.0)
            if not tok.strip() or abs(s) < 1e-12:
                out.append(html.escape(tok))
                continue
            a = min(abs(s) / lim, 1.0) * 0.85
            rgb = "214,39,40" if s > 0 else "31,119,180"
            out.append(f'<span title="{s:+.3f}" style="background:rgba({rgb},{a:.2f});'
                       f'border-radius:3px;padding:0 2px">{html.escape(tok)}</span>')
        return "<div style='line-height:2;font-size:1.05rem'>" + "".join(out) + "</div>"

    @staticmethod
    def plot_bars(scores: dict, title: str, k: int = 12):
        items = sorted(scores.items(), key=lambda kv: -abs(kv[1]))[:k][::-1]
        fig, ax = plt.subplots(figsize=(7, 0.38 * len(items) + 1.2))
        ax.barh([w for w, _ in items], [v for _, v in items],
                color=["#d62728" if v > 0 else "#1f77b4" for _, v in items])
        ax.axvline(0, color="k", lw=0.8)
        ax.set_title(title, fontsize=10)
        fig.tight_layout()
        return fig
