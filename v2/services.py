"""Model wrappers for the v2 API.

Every number the page shows comes from the existing `xai/` package (the same models and explainers
as v1); this module only shapes those results into JSON-friendly dicts, as CONTRACT.md §5 specifies.
"""
from __future__ import annotations

import base64
import io
import json
import math
import re
import sys
import threading
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent  # the medixplain folder
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from xai import image as I  # noqa: E402
from xai import narrative  # noqa: E402
from xai import tabular as TB  # noqa: E402
from xai import text as TX  # noqa: E402
from xai.common import RESULTS, SEED, load_json, rank_corr  # noqa: E402

# torch, LIME's random state and the SHAP explainers are not thread-safe: one model call at a time
LOCK = threading.Lock()


# ---------------------------------------------------------------- JSON helpers
def r4(v) -> float | None:
    """Round to 4 significant digits; NaN/inf become None so the JSON stays valid."""
    v = float(v)
    if not math.isfinite(v):
        return None
    return float(f"{v:.4g}")


def grid(a, digits: int = 3) -> list:
    return np.round(np.nan_to_num(np.asarray(a, dtype=float)), digits).tolist()


def clean(obj):
    """Numpy scalars/arrays → plain Python, NaN → None (Starlette's JSON refuses NaN)."""
    if isinstance(obj, dict):
        return {str(k): clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [clean(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return clean(obj.tolist())
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        return None if not math.isfinite(float(obj)) else float(obj)
    if isinstance(obj, np.str_):
        return str(obj)
    return obj


def label_of(name: str) -> str:
    return name.replace("_", " ")


# ---------------------------------------------------------------- metrics
def metrics() -> dict:
    out = load_json(RESULTS / "metrics.json")
    out["axioms"] = load_json(RESULTS / "report_extras.json").get("axioms", {})
    return clean(out)


# ---------------------------------------------------------------- tabular
class Tabular:
    def __init__(self):
        self.tab = TB.TabularExplainer(TB.load())
        b = self.tab.b
        self.b = b
        self.features = self.tab.features
        self.X_test, self.y_test = b["X_test"], b["y_test"]
        sv = self.tab.shap_values(self.X_test)
        self.test_shap = np.asarray(sv.values)
        self.base_value = float(np.ravel(sv.base_values)[0])
        self.base_prob = 1 / (1 + math.exp(-self.base_value))
        self.mean_abs = np.abs(self.test_shap).mean(0)
        self.order = np.argsort(-self.mean_abs)
        self.test_prob = self.tab.predict_proba(self.X_test.values)[:, 1]

    def meta(self) -> dict:
        Xtr = self.b["X_train"]
        feats = []
        for f in self.features:
            lo, hi = float(Xtr[f].min()), float(Xtr[f].max())
            feats.append({"name": f, "label": label_of(f), "min": r4(lo), "max": r4(hi),
                          "median": r4(Xtr[f].median()), "std": r4(Xtr[f].std()), "step": r4((hi - lo) / 200)})
        patients = [{"index": i, "label": TB.CLASS_NAMES[int(self.y_test.iloc[i])], "prob": r4(self.test_prob[i])}
                    for i in range(len(self.X_test))]
        return {"class_names": TB.CLASS_NAMES, "base_prob": r4(self.base_prob), "features": feats,
                "global_order": [self.features[i] for i in self.order], "patients": patients}

    def _row(self, index: int, values: dict | None):
        if not 0 <= index < len(self.X_test):
            raise ValueError(f"patient index must be 0..{len(self.X_test) - 1}")
        x = self.X_test.iloc[index].copy().astype(float)
        for k, v in (values or {}).items():
            if k not in self.features:
                raise ValueError(f"unknown feature: {k}")
            x[k] = float(v)
        return x

    def explain(self, index: int, values: dict | None = None, lime_seed: int = 42) -> dict:
        x = self._row(index, values)
        tab = self.tab
        p = float(tab.predict_proba(x.values)[0, 1])
        glass = float(self.b["glass"].predict_proba(x.to_frame().T)[0, 1])
        exp = tab.shap(x.to_frame().T)
        sv = np.asarray(exp.values[0])
        base = float(np.ravel(exp.base_values)[0])
        base_prob = 1 / (1 + math.exp(-base))
        order = np.argsort(-np.abs(sv))
        shap_items = [{"feature": self.features[i], "label": label_of(self.features[i]),
                       "value": r4(x.iloc[i]), "shap": r4(sv[i])} for i in order]
        le = tab.lime_explanation(x, num_features=10, seed=lime_seed)
        weights = [{"rule": str(rule), "feature": self.features[idx], "weight": r4(w)}
                   for (rule, w), (idx, _) in zip(le.as_list(label=1), le.as_map()[1])]
        cf = tab.counterfactual(x)
        card = narrative.tabular_card(self.features, x, sv, p, cf, base_prob)
        return {
            "index": index, "label": TB.CLASS_NAMES[int(p >= 0.5)], "prob": r4(p), "glass_prob": r4(glass),
            "base_value": r4(base), "base_prob": r4(base_prob), "output_value": r4(base + sv.sum()),
            "values": {f: r4(x[f]) for f in self.features},
            "shap": shap_items,
            "lime": {"seed": lime_seed, "score": r4(le.score), "weights": weights},
            "counterfactual": {
                "flipped": bool(cf["flipped"]), "target": cf["target"], "start_prob": r4(cf["start_prob"]),
                "end_prob": r4(cf["end_prob"]), "l1_std_distance": r4(cf["l1_std_distance"]),
                "changes": [{"feature": f, "label": label_of(f), "from": r4(a), "to": r4(b), "delta_sd": r4(d)}
                            for f, a, b, d in cf["changes"]],
            },
            "card": card,
        }

    def global_(self) -> dict:
        perm_m, perm_s = self.b["perm_mean"], self.b["perm_std"]
        coef = np.abs(self.b["glass"][-1].coef_[0])
        importance = [{"feature": self.features[i], "label": label_of(self.features[i]),
                       "mean_abs_shap": r4(self.mean_abs[i]), "permutation": r4(perm_m[i]),
                       "permutation_std": r4(perm_s[i]), "glass_coef": r4(coef[i])} for i in self.order]
        top = [self.features[i] for i in self.order[:12]]
        rows = []
        for f in top:
            j = self.features.index(f)
            col = self.X_test[f].values.astype(float)
            span = col.max() - col.min() or 1.0
            rows.append({"feature": f, "label": label_of(f), "shap": grid(self.test_shap[:, j], 4),
                         "value_norm": grid((col - col.min()) / span, 3)})
        return {"importance": importance, "beeswarm": {"features": top, "rows": rows}}

    def pdp(self, feature: str) -> dict:
        if feature not in self.features:
            raise ValueError(f"unknown feature: {feature}")
        Xtr = self.b["X_train"]
        g = np.linspace(float(Xtr[feature].min()), float(Xtr[feature].max()), 30)
        curves = []
        for v in g:
            Xm = self.X_test.copy()
            Xm[feature] = v
            curves.append(self.tab.predict_proba(Xm.values)[:, 1])
        P = np.stack(curves, axis=1)  # patients x grid
        return {"feature": feature, "label": label_of(feature), "grid": grid(g, 5), "pdp": grid(P.mean(0), 4),
                "ice": grid(P[:40], 4), "rug": grid(self.X_test[feature].values, 5)}

    def lime_stability(self, index: int, runs: int = 5) -> dict:
        x = self._row(index, None)
        runs = max(2, min(int(runs), 8))
        vecs = [self.tab.lime_vector(x, seed=SEED + k) for k in range(runs)]
        out = []
        for v in vecs:
            top = np.argsort(-np.abs(v))[:8]
            out.append([{"feature": self.features[i], "label": label_of(self.features[i]), "weight": r4(v[i])} for i in top])
        pairs = [rank_corr(vecs[a], vecs[b]) for a in range(runs) for b in range(a + 1, runs)]
        s1, s2 = self.tab.shap_vector(x), self.tab.shap_vector(x)
        return {"runs": out, "spearman": r4(np.mean(pairs)), "shap_spearman": r4(rank_corr(s1, s2))}


# ---------------------------------------------------------------- image
METHODS = [
    # key, display name, plain-English line, signed, attribution function
    ("gradcam", "Grad-CAM", "Where the AI's last layer focused", False, I.grad_cam),
    ("ig", "Integrated Gradients", "How much each pixel added, from a black image to this one", True, I.integrated_gradients),
    ("smoothgrad", "SmoothGrad", "Pixel sensitivity, averaged over noisy copies", False, I.smoothgrad),
    ("saliency", "Vanilla gradient", "Which pixels change the answer most if nudged", False, I.vanilla_gradient),
    ("occlusion", "Occlusion", "How much the answer drops when a patch is covered", True, I.occlusion),
    ("lime", "LIME superpixels", "Which image regions a simple local model relies on", True,
     lambda model, img, t: I.lime_image(model, img, t)[0]),
]
METHOD_FN = {k: (fn, signed) for k, _, _, signed, fn in METHODS}


def scale_map(m: np.ndarray, signed: bool) -> np.ndarray:
    m = np.nan_to_num(np.asarray(m, dtype=float))
    if signed:
        lim = np.abs(m).max()
        return m / lim if lim > 0 else m
    m = np.clip(m, 0, None)
    lim = m.max()
    return m / lim if lim > 0 else m


def png_data_url(img: np.ndarray) -> str:
    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray((np.clip(img, 0, 1) * 255).astype("uint8"), mode="L").save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def decode_upload(data_url: str) -> np.ndarray:
    from PIL import Image

    if "," in data_url:
        data_url = data_url.split(",", 1)[1]
    try:
        raw = base64.b64decode(data_url)
        return I.preprocess_upload(Image.open(io.BytesIO(raw)))
    except Exception as e:  # noqa: BLE001
        raise ValueError(f"could not read the uploaded image: {e}") from e


class Images:
    def __init__(self):
        self.model = I.load_model()
        self.X, self.y = I.load_split("test")
        self._rnd = None
        self._cache: dict = {}
        m = load_json(RESULTS / "metrics.json")
        self.sanity_metrics = m.get("image", {}).get("sanity_spearman")

    @property
    def rnd(self):
        if self._rnd is None:
            self._rnd = I.randomized_copy(self.model)
        return self._rnd

    def samples(self) -> dict:
        idx = list(np.where(self.y == 1)[0][:12]) + list(np.where(self.y == 0)[0][:12])
        probs = I.predict_proba(self.model, self.X[idx])[:, 1]
        return {"samples": [{"index": int(i), "label": I.CLASS_NAMES[int(self.y[i])], "prob": r4(p),
                             "png": png_data_url(self.X[i])} for i, p in zip(idx, probs)]}

    def _image(self, index, image):
        if image:
            return None, decode_upload(image)
        if index is None or not 0 <= int(index) < len(self.X):
            raise ValueError(f"image index must be 0..{len(self.X) - 1}")
        return int(index), self.X[int(index)]

    def explain(self, index=None, image=None, target=None) -> dict:
        index, img = self._image(index, image)
        prob = I.predict_proba(self.model, img[None])[0]
        t = int(prob.argmax()) if target is None else int(target)
        if t not in (0, 1):
            raise ValueError("target must be 0 (normal) or 1 (pneumonia)")
        key = (index, t)
        if index is not None and key in self._cache:
            return self._cache[key]
        raw = {k: fn(self.model, img, t) for k, (fn, _) in METHOD_FN.items()}
        maps = {k: grid(scale_map(raw[k], METHOD_FN[k][1])) for k in raw}
        rng = np.random.default_rng(0)
        curves = {k: I.deletion_curve(self.model, img, raw[k], t) for k in raw}
        curves["random"] = I.deletion_curve(self.model, img, rng.random(img.shape), t)
        auc = {k: r4(np.trapezoid(c, dx=1 / (len(c) - 1))) for k, c in curves.items()}
        cam = scale_map(raw["gradcam"], False)
        out = {
            "index": index, "image": grid(img), "prob": {"normal": r4(prob[0]), "pneumonia": r4(prob[1])},
            "predicted": I.CLASS_NAMES[int(prob.argmax())], "target": I.CLASS_NAMES[t],
            "methods": [{"key": k, "name": n, "plain": pl, "signed": s} for k, n, pl, s, _ in METHODS],
            "maps": maps,
            "deletion": {"x": [round(v, 1) for v in np.linspace(0, 100, 21)],
                         "curves": {k: grid(c, 4) for k, c in curves.items()}, "auc": auc},
            "card": narrative.image_card(prob, I.CLASS_NAMES, cam, self.sanity_metrics),
        }
        if index is not None:
            self._cache[key] = out
        return out

    def sanity(self, index, method: str, target=None) -> dict:
        if method not in METHOD_FN:
            raise ValueError(f"method must be one of {list(METHOD_FN)}")
        index, img = self._image(index, None)
        prob = I.predict_proba(self.model, img[None])[0]
        t = int(prob.argmax()) if target is None else int(target)
        fn, signed = METHOD_FN[method]
        a, b = fn(self.model, img, t), fn(self.rnd, img, t)
        if np.ptp(a) == 0 or np.ptp(b) == 0:
            r = 0.0
        else:
            from scipy.stats import spearmanr

            r = spearmanr(np.abs(a).ravel(), np.abs(b).ravel()).correlation
        return {"method": method, "trained": grid(scale_map(a, signed)), "random": grid(scale_map(b, signed)),
                "spearman": r4(0.0 if r is None or not math.isfinite(r) else r)}


# ---------------------------------------------------------------- text
WORD = re.compile(r"(\w+)")


class Texts:
    EXAMPLES = (3, 10, 25, 60, 100, 150)

    def __init__(self):
        self.tx = TX.TextExplainer(TX.load())

    def meta(self) -> dict:
        b = self.tx.b
        return {"classes": self.tx.classes,
                "examples": [{"id": i, "label": str(b["y_test"][i]), "text": b["X_test"][i]} for i in self.EXAMPLES]}

    def explain(self, text: str, cls: str | None = None, contrast: str | None = None) -> dict:
        text = (text or "").strip()
        if not text:
            raise ValueError("type a symptom description first")
        if len(text) > 2000:
            raise ValueError("please keep the description under 2,000 characters")
        tx = self.tx
        top = [(str(c), float(p)) for c, p in tx.top_classes(text, 5)]
        cls = cls or top[0][0]
        if cls not in tx.classes:
            raise ValueError(f"unknown class: {cls}")
        if not contrast or contrast == cls or contrast not in tx.classes:
            contrast = next(c for c, _ in top if c != cls)
        tokens = []
        for part in WORD.split(text):
            if not part:
                continue
            is_word = bool(re.fullmatch(r"\w+", part))
            tokens.append({"t": part, "w": part.lower() if is_word else None})
        rnd = lambda d: {str(k): r4(v) for k, v in d.items()}  # noqa: E731
        shap_words = tx.ngrams_to_words(tx.shap_ngrams(text, cls))
        return {
            "top": [{"cls": c, "prob": r4(p)} for c, p in top],
            "cls": cls, "contrast": contrast, "tokens": tokens,
            "scores": {
                "shap": rnd(shap_words),
                "lime": rnd(tx.lime_words(text, cls, seed=42)),
                "occlusion": rnd(tx.occlusion_words(text, cls)),
                "linear": rnd(tx.ngrams_to_words(tx.linear_contributions(text, cls))),
            },
            "contrastive": rnd(tx.ngrams_to_words(tx.contrastive(text, cls, contrast))),
            "global": [{"word": str(w), "coef": r4(c)} for w, c in tx.global_top_words(cls, 12)],
            "card": narrative.text_card(top, shap_words),
        }


# ---------------------------------------------------------------- registry
class Services:
    """Loads every model once; the server calls these methods inside LOCK."""

    def __init__(self):
        self.tabular = Tabular()
        self.images = Images()
        self.texts = Texts()
        self._tab_global = None
        self._image_samples = None

    def tab_global(self):
        if self._tab_global is None:
            self._tab_global = self.tabular.global_()
        return self._tab_global

    def image_samples(self):
        if self._image_samples is None:
            self._image_samples = self.images.samples()
        return self._image_samples


if __name__ == "__main__":  # quick smoke test: python v2/services.py
    import time

    t0 = time.time()
    s = Services()
    print("loaded in", round(time.time() - t0, 1), "s")
    for name, fn in [
        ("metrics", metrics),
        ("tab meta", s.tabular.meta),
        ("tab explain", lambda: s.tabular.explain(0)),
        ("tab what-if", lambda: s.tabular.explain(0, {"worst_area": 700.0})),
        ("tab global", s.tab_global),
        ("tab pdp", lambda: s.tabular.pdp("worst_concave_points")),
        ("tab lime-stability", lambda: s.tabular.lime_stability(5)),
        ("img samples", s.image_samples),
        ("img explain", lambda: s.images.explain(index=0)),
        ("img explain cached", lambda: s.images.explain(index=0)),
        ("img sanity", lambda: s.images.sanity(0, "smoothgrad")),
        ("txt meta", s.texts.meta),
        ("txt explain", lambda: s.texts.explain(s.texts.meta()["examples"][0]["text"])),
    ]:
        t = time.time()
        out = clean(fn())
        size = len(json.dumps(out))
        print(f"{name:20s} {time.time() - t:6.2f}s {size / 1024:8.1f} KB")
