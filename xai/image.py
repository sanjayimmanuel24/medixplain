"""Image case: pneumonia vs normal on paediatric chest X-rays (PneumoniaMNIST, 64x64).

Model: a small CNN trained here. Explanations, all written out in PyTorch so each one
can be read and explained: vanilla gradients, SmoothGrad, Integrated Gradients,
Grad-CAM, occlusion sensitivity, plus LIME on superpixels.
"""
from __future__ import annotations

import copy

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from .common import DATA, MODELS, SEED, seed_everything

CLASS_NAMES = ["normal", "pneumonia"]
WEIGHTS = MODELS / "cnn_pneumonia.pt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SIZE = 64


def load_split(split: str):
    d = np.load(DATA / "pneumoniamnist_64.npz")
    return d[f"{split}_images"].astype(np.float32) / 255.0, d[f"{split}_labels"].ravel().astype(int)


class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()

        def block(i, o):
            return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(inplace=True))

        self.features = nn.Sequential(
            block(1, 16), nn.MaxPool2d(2),       # 32x32
            block(16, 32), nn.MaxPool2d(2),      # 16x16
            block(32, 64), nn.MaxPool2d(2),      # 8x8
            block(64, 128),                      # 8x8  <- Grad-CAM target
        )
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.3), nn.Linear(128, 2))

    def forward(self, x):
        return self.head(self.features(x))

    @property
    def cam_layer(self):
        return self.features[-1]


def train(epochs: int = 15, log=print) -> dict:
    seed_everything()
    Xtr, ytr = load_split("train")
    Xva, yva = load_split("val")
    Xte, yte = load_split("test")
    model = SmallCNN().to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-3, total_steps=epochs * (len(Xtr) // 64 + 1))
    w = torch.tensor(len(ytr) / (2 * np.bincount(ytr)), dtype=torch.float32, device=DEVICE)
    xt = torch.tensor(Xtr)[:, None]
    yt = torch.tensor(ytr)
    best, best_state = -1, None
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(xt))
        for i in range(0, len(xt), 64):
            idx = perm[i:i + 64]
            xb, yb = xt[idx].to(DEVICE), yt[idx].to(DEVICE)
            # light augmentation: small random shifts and brightness jitter
            shift = np.random.randint(-3, 4, size=2)
            xb = torch.roll(xb, shifts=tuple(shift), dims=(2, 3))
            xb = (xb * (0.9 + 0.2 * torch.rand(len(xb), 1, 1, 1, device=DEVICE))).clamp(0, 1)
            loss = F.cross_entropy(model(xb), yb, weight=w)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
        auc = roc_auc_score(yva, predict_proba(model, Xva)[:, 1])
        log(f"epoch {ep + 1:2d}  loss {loss.item():.3f}  val AUC {auc:.4f}")
        if auc > best:
            best, best_state = auc, copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    torch.save(model.state_dict(), WEIGHTS)
    p = predict_proba(model, Xte)[:, 1]
    return {"accuracy": accuracy_score(yte, p > 0.5), "f1": f1_score(yte, p > 0.5),
            "roc_auc": roc_auc_score(yte, p), "val_auc": best}


def load_model() -> SmallCNN:
    model = SmallCNN().to(DEVICE)
    model.load_state_dict(torch.load(WEIGHTS, map_location=DEVICE))
    return model.eval()


@torch.no_grad()
def predict_proba(model: nn.Module, images: np.ndarray, batch: int = 256) -> np.ndarray:
    """images: (N, 64, 64) floats in [0, 1]."""
    was_training = model.training
    model.eval()
    out = []
    for i in range(0, len(images), batch):
        xb = torch.tensor(images[i:i + batch], dtype=torch.float32, device=DEVICE)[:, None]
        out.append(F.softmax(model(xb), 1).cpu().numpy())
    model.train(was_training)
    return np.concatenate(out)


def preprocess_upload(pil_image) -> np.ndarray:
    """Any uploaded picture -> centre-cropped 64x64 grayscale in [0, 1]."""
    img = pil_image.convert("L")
    w, h = img.size
    s = min(w, h)
    img = img.crop(((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2)).resize((SIZE, SIZE))
    return np.asarray(img, dtype=np.float32) / 255.0


# ---------------------------------------------------------------- attribution methods
def _tensor(img: np.ndarray) -> torch.Tensor:
    return torch.tensor(img, dtype=torch.float32, device=DEVICE)[None, None]


def vanilla_gradient(model, img, target: int) -> np.ndarray:
    x = _tensor(img).requires_grad_(True)
    model(x)[0, target].backward()
    return x.grad[0, 0].abs().cpu().numpy()


def smoothgrad(model, img, target: int, n: int = 30, sigma: float = 0.1) -> np.ndarray:
    total = np.zeros_like(img)
    rng = np.random.default_rng(SEED)
    for _ in range(n):
        noisy = np.clip(img + rng.normal(0, sigma, img.shape).astype(np.float32), 0, 1)
        total += vanilla_gradient(model, noisy, target)
    return total / n


def integrated_gradients(model, img, target: int, steps: int = 64, baseline=None) -> np.ndarray:
    """Sundararajan et al. 2017, Riemann (midpoint) approximation from a black baseline."""
    base = np.zeros_like(img) if baseline is None else baseline
    alphas = (torch.arange(steps, device=DEVICE, dtype=torch.float32) + 0.5) / steps
    x, b = _tensor(img), _tensor(base)
    path = (b + alphas[:, None, None, None] * (x - b)).requires_grad_(True)
    model(path)[:, target].sum().backward()
    ig = (x - b)[0, 0] * path.grad.mean(0)[0]
    return ig.detach().cpu().numpy()


def grad_cam(model: SmallCNN, img, target: int) -> np.ndarray:
    """Selvaraju et al. 2017: ReLU(sum_k alpha_k A^k), alpha_k = mean gradient of channel k."""
    store = {}
    h1 = model.cam_layer.register_forward_hook(lambda m, i, o: store.__setitem__("act", o))
    h2 = model.cam_layer.register_full_backward_hook(lambda m, gi, go: store.__setitem__("grad", go[0]))
    try:
        x = _tensor(img).requires_grad_(True)
        model.zero_grad()
        model(x)[0, target].backward()
    finally:
        h1.remove()
        h2.remove()
    alpha = store["grad"].mean(dim=(2, 3), keepdim=True)
    cam = F.relu((alpha * store["act"]).sum(1, keepdim=True))
    cam = F.interpolate(cam, size=img.shape, mode="bilinear", align_corners=False)[0, 0]
    cam = cam.detach().cpu().numpy()
    return cam / (cam.max() + 1e-8)


def occlusion(model, img, target: int, patch: int = 8, stride: int = 4) -> np.ndarray:
    """Drop in target probability when a grey patch hides each region."""
    base_p = predict_proba(model, img[None])[0, target]
    fill = float(img.mean())
    batch, coords = [], []
    for y in range(0, SIZE - patch + 1, stride):
        for x in range(0, SIZE - patch + 1, stride):
            o = img.copy()
            o[y:y + patch, x:x + patch] = fill
            batch.append(o)
            coords.append((y, x))
    probs = predict_proba(model, np.stack(batch))[:, target]
    heat, count = np.zeros_like(img), np.zeros_like(img)
    for (y, x), p in zip(coords, probs):
        heat[y:y + patch, x:x + patch] += base_p - p
        count[y:y + patch, x:x + patch] += 1
    return heat / np.maximum(count, 1)


def lime_image(model, img, target: int, num_samples: int = 800):
    from lime import lime_image as li
    from skimage.segmentation import slic

    rgb = np.repeat(img[..., None], 3, axis=2)

    def classifier(batch_rgb):
        return predict_proba(model, batch_rgb[..., 0].astype(np.float32))

    explainer = li.LimeImageExplainer(random_state=SEED)
    exp = explainer.explain_instance(
        rgb, classifier, labels=(target,), top_labels=None, hide_color=float(img.mean()),
        num_samples=num_samples, segmentation_fn=lambda im: slic(im, n_segments=40, compactness=8, start_label=0),
        random_seed=SEED)
    weights = dict(exp.local_exp[target])
    heat = np.vectorize(lambda s: weights.get(s, 0.0))(exp.segments).astype(np.float32)
    return heat, exp.segments


METHODS = {
    "Grad-CAM": grad_cam,
    "Integrated Gradients": integrated_gradients,
    "SmoothGrad": smoothgrad,
    "Vanilla gradient": vanilla_gradient,
    "Occlusion": occlusion,
}


def explain_all(model, img, target: int, include_lime: bool = True) -> dict:
    out = {name: fn(model, img, target) for name, fn in METHODS.items()}
    if include_lime:
        out["LIME (superpixels)"] = lime_image(model, img, target)[0]
    return out


# ---------------------------------------------------------------- evaluation of explanations
def deletion_curve(model, img, attribution: np.ndarray, target: int, steps: int = 20) -> np.ndarray:
    """Remove the most important pixels first and track P(target).

    "Removed" pixels take the value of a heavily blurred copy of the image. A flat grey
    fill leaves scattered-pixel artefacts that the CNN reacts to, which wrongly makes
    random maps look faithful. A faithful map makes the probability fall fast, so a
    lower area under the curve is better.
    """
    from scipy.ndimage import gaussian_filter

    order = np.argsort(-attribution.ravel())
    fill = gaussian_filter(img, 8).ravel()
    frames = []
    for k in np.linspace(0, len(order), steps + 1).astype(int):
        o = img.copy().ravel()
        o[order[:k]] = fill[order[:k]]
        frames.append(o.reshape(img.shape))
    return predict_proba(model, np.stack(frames))[:, target]


def randomized_copy(model: SmallCNN) -> SmallCNN:
    """Model-parameter randomisation test (Adebayo et al. 2018): same architecture, random weights."""
    rnd = copy.deepcopy(model)
    torch.manual_seed(SEED + 1)
    for m in rnd.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            m.reset_parameters()
    return rnd.eval()


# ---------------------------------------------------------------- plots
def plot_maps(img: np.ndarray, maps: dict, title: str = ""):
    n = len(maps) + 1
    cols = min(n, 4)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.3 * cols, 3.4 * rows))
    axes = np.atleast_1d(axes).ravel()
    axes[0].imshow(img, cmap="gray")
    axes[0].set_title("Input X-ray")
    for ax, (name, m) in zip(axes[1:], maps.items()):
        ax.imshow(img, cmap="gray")
        signed = m.min() < 0
        lim = np.abs(m).max() + 1e-8
        ax.imshow(m, cmap="bwr" if signed else "jet", alpha=0.5, vmin=-lim if signed else 0, vmax=lim)
        ax.set_title(name, fontsize=10)
    for ax in axes:
        ax.axis("off")
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    return fig
