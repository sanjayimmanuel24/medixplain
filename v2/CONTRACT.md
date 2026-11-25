# MediXplain v2 — engineering contract

Every builder works to this file. The look (tokens, type, motion, section concepts) lives in
`DESIGN.md`; this file fixes the folder layout, the API, the module interface and the shared
libraries so parts written in parallel fit together. If something here and `DESIGN.md` disagree
about *look*, `DESIGN.md` wins; about *structure or data*, this file wins.

v1 (`medixplain/app.py`, Streamlit, port 8502) is never edited. v2 lives entirely in `medixplain/v2/`.

## 1. Run

```
python v2/server.py                # http://127.0.0.1:8503  (serves the page and the API)
python v2/server.py --port 8511    # any other port, for parallel testing
python v2/make_hero.py             # regenerates the hero frames and particle data
python v2/tools/shot.py --url http://127.0.0.1:8503/ --out v2/shots/x.png --cdp-port 9341
```

The server is Python **Starlette** (already installed; do NOT install FastAPI — it would change
the Starlette version v1 depends on) on uvicorn. It imports the existing models from
`medixplain/xai/` (add the `medixplain` folder to `sys.path`). Models load once at start-up.
All model calls go through one `threading.Lock` (torch and LIME explainers are not thread-safe).
Heavy calls run in the thread pool (`starlette.concurrency.run_in_threadpool`).

## 2. Folder layout and ownership

```
v2/
  DESIGN.md                 design language (owner: design)
  CONTRACT.md               this file
  README.md                 how to run v2
  server.py                 Starlette app: static files at /, JSON API at /api   (owner: backend)
  services.py               model wrappers used by server.py                     (owner: backend)
  make_hero.py              renders hero frames + particle data from real model  (owner: hero)
  tools/shot.py             headless screenshot + console-error tool
  shots/                    screenshots (scratch, git-ignorable)
  web/
    index.html              shell: head, fonts, vendor scripts, nav mount, one empty <section> per chapter (owner: foundation)
    vendor/                 gsap.min.js ScrollTrigger SplitText ScrambleTextPlugin DrawSVGPlugin Flip CustomEase Observer lenis.min.js lenis.css (present)
    fonts/                  self-hosted woff2 + fonts.css                         (owner: foundation)
    css/tokens.css          design tokens, copied from DESIGN.md §10              (owner: foundation)
    css/base.css            reset, typography, layout primitives (.l-*), utilities (.u-*), grain   (owner: foundation)
    css/components.css      shared components (.c-*)                              (owner: foundation)
    css/sections/<id>.css   one file per section, every selector starts with .s-<id>  (owner: that section)
    js/main.js              boot: plugins, Lenis, nav, mounts sections            (owner: foundation)
    js/lib/api.js           fetch wrapper                                         (owner: foundation)
    js/lib/dom.js           h(), qs, esc, fmt, debounce                           (owner: foundation)
    js/lib/color.js         attribution + heat colour scales from tokens          (owner: foundation)
    js/lib/motion.js        shared motion helpers                                 (owner: foundation)
    js/charts/*.js          shared SVG/canvas charts + charts/index.js            (owner: charts)
    js/sections/<id>.js     one module per section                                (owner: that section)
    assets/hero/            frame-1.png frame-2.png frame-3.png hero.json         (owner: hero)
    dev/                    test pages with mock data (charts.html etc.)          (owner: whoever)
```

Sections, in scroll order (ids are fixed; `DESIGN.md` gives each one's concept):

| id | content | owner |
|---|---|---|
| `hero` | particle X-ray hero, headline, key numbers | hero |
| `problem` | AI gives answers, not reasons; why that matters | story |
| `models` | the three AI helpers at a glance | story |
| `tumour` | tabular lab | tumour |
| `xray` | image lab | xray |
| `symptoms` | text lab | symptoms |
| `trust` | Trust Lab: tests of the explanations + findings | trust |
| `outro` | takeaway, tools, credits | story |

Never edit a file you do not own. Need something shared changed? Note it in your final report.

## 3. Page shell (owner: foundation)

`index.html` loads, in order: `fonts/fonts.css`, `vendor/lenis.css`, `css/tokens.css`, `css/base.css`,
`css/components.css`, every `css/sections/<id>.css`; then the vendor scripts as classic `<script defer>`
(globals: `gsap`, `ScrollTrigger`, `SplitText`, `ScrambleTextPlugin`, `DrawSVGPlugin`, `Flip`,
`CustomEase`, `Observer`, `Lenis`), then `<script type="module" src="js/main.js">`.

Body: `<header class="c-nav" id="nav"></header>` then `<main id="main">` holding, in the order above,
`<section id="<id>" class="sec s-<id>" data-section="<id>" aria-label="<human name>"></section>`.
Sections are empty in the HTML; each module renders its own markup.

`main.js`:
1. `gsap.registerPlugin(ScrollTrigger, SplitText, ScrambleTextPlugin, DrawSVGPlugin, Flip, CustomEase, Observer)`.
2. `reduced = matchMedia('(prefers-reduced-motion: reduce)').matches`. Not reduced → Lenis:
   `const lenis = new Lenis({ autoRaf: false, lerp: 0.1 }); lenis.on('scroll', ScrollTrigger.update);
   gsap.ticker.add(t => lenis.raf(t * 1000)); gsap.ticker.lagSmoothing(0); window.__lenis = lenis;`
3. Fetch `/api/metrics` once → `ctx.metrics` (if it fails, `ctx.metrics = null` and sections show their error state).
4. Build the nav (logo, chapter links with smooth scroll via Lenis, a scroll-progress indicator, current-chapter highlight).
5. For each `section[data-section]` in order: `import('./sections/<id>.js')` then `await mod.mount(el, ctx)`
   inside try/catch; a failing section renders a small inline error card and never breaks the others.
   Mount the hero first and without waiting for the others; mount the rest in order.
6. After all mounts: `ScrollTrigger.refresh()`.

## 4. Section module interface

```js
// js/sections/<id>.js
export async function mount(root, ctx) { /* render into root, wire interactions, set up motion */ }
```

`ctx`:

```js
{
  api,                 // js/lib/api.js: { get(path, {signal}), post(path, body, {signal}) } -> parsed JSON, throws ApiError
  gsap, ScrollTrigger, SplitText, Flip, lenis,   // lenis is null when reduced
  reduced,             // boolean: prefers-reduced-motion
  motion,              // js/lib/motion.js
  charts,              // js/charts/index.js namespace
  color,               // js/lib/color.js
  dom,                 // js/lib/dom.js
  metrics,             // GET /api/metrics JSON (or null)
}
```

Rules for every section:
- Render markup inside `root` only. All CSS selectors start with `.s-<id>`; use shared `.c-*`, `.l-*`, `.u-*` classes freely.
- Wrap all GSAP/ScrollTrigger work in `ctx.gsap.context(() => { ... }, root)`.
- Every async call shows a loading state in the design language and an error state with a retry button.
  Requests triggered by input are debounced (300–400 ms) and the previous one aborted (`AbortController`).
- When `ctx.reduced` is true: no scroll-scrubbed movement, no parallax, no autoplay loops; content appears in its final state
  (opacity fades of ≤ 200 ms are fine).
- Keyboard: every control reachable and operable; visible focus (`--focus`); results announced with `aria-live="polite"`.
- Text never below 14px on screen; data labels use `--font-mono`.
- Plain English first, method name second: e.g. "Where the AI looked · Grad-CAM".

## 5. API (owner: backend)

All JSON. Numbers rounded to 4 significant digits where floats; 64×64 maps rounded to 3 decimals.
Errors: HTTP 4xx/5xx with `{"error": "<message>"}`.

### GET /api/health → `{"ok": true}`

### GET /api/metrics
`results/metrics.json` verbatim, plus `"axioms"` from `results/report_extras.json`. Shape (abridged):
```json
{
  "tabular": {"metrics": {"xgboost": {"accuracy","f1","roc_auc"}, "logistic_regression": {...}}, "n_train", "n_test",
              "counterfactual": {"flip_rate","mean_features_changed","mean_l1_std_distance","example": {...}},
              "agreement": {"SHAP vs LIME": {"top5_overlap","spearman"}, "SHAP vs glass-box": {...}, "LIME vs glass-box": {...}},
              "stability": {"LIME seed-to-seed spearman","SHAP seed-to-seed spearman","SHAP under 1% input noise","LIME under 1% input noise"},
              "faithfulness_top5_drop": {"SHAP","LIME","Glass-box coef","Random"}},
  "image":   {"metrics": {"accuracy","f1","roc_auc","val_auc"},
              "deletion_auc": {"Grad-CAM","Integrated Gradients","SmoothGrad","Vanilla gradient","Occlusion","Random"},
              "sanity_spearman": {"Grad-CAM","Integrated Gradients","SmoothGrad","Vanilla gradient","Occlusion"},
              "method_agreement_example": {"methods": [...6], "matrix": [[...6]x6]}},
  "text":    {"metrics": {"accuracy","macro_f1"}, "n_classes", "n_train", "n_test",
              "agreement_top3": {"SHAP vs LIME", ...}, "comprehensiveness_top3": {"SHAP","LIME","Occlusion","Linear contribution","Random"},
              "lime_seed_stability"},
  "axioms":  {"shap_additivity_max_abs_error","shap_n","ig_completeness_mean_rel_error","ig_n"}
}
```

### GET /api/tabular/meta
```json
{
  "class_names": ["benign", "malignant"],
  "base_prob": 0.3956,
  "features": [{"name": "mean_radius", "label": "mean radius", "min": 6.98, "max": 28.11, "median": 13.3, "std": 3.5, "step": 0.1057}],
  "global_order": ["worst_concave_points", "..."],
  "patients": [{"index": 0, "label": "malignant", "prob": 0.9979}]
}
```
`features` = all 30 in dataset order (`label` = name with underscores as spaces; `step` = (max−min)/200);
`global_order` = feature names by mean |SHAP| on the test set, descending; `patients` = all 143 test patients.

### POST /api/tabular/explain
Body: `{"index": 0}` or `{"index": 0, "values": {"worst_area": 900.0, ...}}` (values override that patient's
features), optional `"lime_seed": 42`.
```json
{
  "index": 0, "label": "malignant", "prob": 0.9979, "glass_prob": 0.999,
  "base_value": -0.424, "base_prob": 0.3956, "output_value": 6.155,
  "values": {"mean_radius": 20.57, "...": 0},
  "shap": [{"feature": "mean_concave_points", "label": "mean concave points", "value": 0.112, "shap": 1.2}],
  "lime": {"seed": 42, "score": 0.39, "weights": [{"rule": "area_error > 45.39", "feature": "area_error", "weight": 0.121}]},
  "counterfactual": {"flipped": true, "target": "benign", "start_prob": 0.9979, "end_prob": 0.4092, "l1_std_distance": 6.24,
                     "changes": [{"feature": "worst_perimeter", "label": "worst perimeter", "from": 151.7, "to": 100.23, "delta_sd": -1.5}]},
  "card": "markdown"
}
```
`shap` = all 30 features sorted by |shap| descending (log-odds of malignant; `sum(shap) + base_value = output_value`).
`lime.weights` = top 10 by |weight| toward malignant. `card` = `xai.narrative.tabular_card` output.

### GET /api/tabular/global
```json
{
  "importance": [{"feature": "worst_concave_points", "label": "...", "mean_abs_shap": 1.1, "permutation": 0.01, "permutation_std": 0.003, "glass_coef": 0.9}],
  "beeswarm": {"features": ["...top 12 names"], "rows": [{"feature": "...", "shap": [143 values], "value_norm": [143 values 0..1]}]}
}
```
`importance` = all 30 sorted by mean |SHAP| descending; `glass_coef` = |logistic-regression coefficient| on standardised features.

### GET /api/tabular/pdp?feature=worst_concave_points
```json
{"feature": "worst_concave_points", "label": "...", "grid": [30 values], "pdp": [30 probs], "ice": [[30 probs] x 40], "rug": [143 test values]}
```
Grid = 30 evenly spaced points between the training min and max; ICE = the first 40 test patients; PDP = mean over all test patients.

### POST /api/tabular/lime-stability
Body `{"index": 0, "runs": 5}` → `{"runs": [[{"feature","label","weight"} x 8] x runs], "spearman": 0.90, "shap_spearman": 1.0}`
(each run = LIME with seed 42+k, top 8 features by |weight|; `spearman` = mean pairwise rank correlation over all 30 features).

### GET /api/image/samples
`{"samples": [{"index": 12, "label": "pneumonia", "prob": 0.9991, "png": "data:image/png;base64,..."}]}` —
24 test images, 12 per class (first of each class in test order), each a 64×64 grayscale PNG; `prob` = P(pneumonia).

### POST /api/image/explain
Body: `{"index": 12}` or `{"image": "data:image/png;base64,..."}` (an upload: grayscale, centre-cropped, resized to 64×64),
optional `"target": 0|1` (class to explain; default = predicted).
```json
{
  "index": 12, "image": [[64 floats 0..1] x 64],
  "prob": {"normal": 0.0001, "pneumonia": 0.9999}, "predicted": "pneumonia", "target": "pneumonia",
  "methods": [{"key": "gradcam", "name": "Grad-CAM", "plain": "Where the AI's last layer focused", "signed": false}],
  "maps": {"gradcam": [[64]x64], "ig": [[...]], "smoothgrad": [[...]], "saliency": [[...]], "occlusion": [[...]], "lime": [[...]]},
  "deletion": {"x": [0, 5, ..., 100], "curves": {"gradcam": [21 probs], "...": [], "random": [21 probs]}, "auc": {"gradcam": 0.71, "random": 0.83}},
  "card": "markdown"
}
```
Method keys in this order: `gradcam` (Grad-CAM, unsigned), `ig` (Integrated Gradients, signed), `smoothgrad` (SmoothGrad, unsigned),
`saliency` (Vanilla gradient, unsigned), `occlusion` (Occlusion, signed), `lime` (LIME superpixels, signed).
Unsigned maps are scaled to 0..1 (divide by max); signed maps to −1..1 (divide by max |v|). Results are cached per (index, target).

### POST /api/image/sanity
Body `{"index": 12, "method": "smoothgrad", "target"?}` →
`{"method": "smoothgrad", "trained": [[64]x64], "random": [[64]x64], "spearman": 0.45}` (same scaling as above;
`random` comes from the same CNN with re-initialised weights, `xai.image.randomized_copy`).

### GET /api/text/meta
`{"classes": [22 names], "examples": [{"id": 3, "label": "pneumonia", "text": "..."}]}` — test examples 3, 10, 25, 60, 100, 150.

### POST /api/text/explain
Body `{"text": "...", "cls"?: "pneumonia", "contrast"?: "gastroesophageal reflux disease"}` (defaults: top class; runner-up).
```json
{
  "top": [{"cls": "pneumonia", "prob": 0.956}],
  "cls": "pneumonia", "contrast": "gastroesophageal reflux disease",
  "tokens": [{"t": "I'm", "w": "i"}, {"t": " ", "w": null}, {"t": "sweating", "w": "sweating"}],
  "scores": {"shap": {"sweating": 0.52}, "lime": {}, "occlusion": {}, "linear": {}},
  "contrastive": {"sweating": 0.4},
  "global": [{"word": "heart", "coef": 3.79}],
  "card": "markdown"
}
```
`tokens` splits the text so that joining every `t` gives the text back; `w` = the lower-case word key used in
`scores` (null for spaces/punctuation). Word scores follow `xai.text` (bigram scores already split onto words).
`global` = top 12 positive coefficients of `cls`. Empty text → 400.

## 6. Shared libraries (foundation)

`js/lib/api.js` — `export const api = { get(path, opts), post(path, body, opts) }`; `ApiError` with `.status`.

`js/lib/dom.js` — `h(tag, attrs, ...children)` (attrs: class, style object, on<Event> handlers, data-*, aria-*),
`qs(sel, root)`, `qsa(sel, root)`, `esc(str)`, `md(str)` (tiny markdown: **bold**, *italic*, paragraphs) →
safe HTML string, `fmt.pct(v, digits=1)`, `fmt.num(v, digits=2)`, `fmt.signed(v, digits=2)`, `fmt.feature(name)`,
`debounce(fn, ms)`, `clamp(v, a, b)`.

`js/lib/color.js` — reads tokens at runtime; `attr(v)` for v in −1..1 → `rgb(...)` across the five `--attr-*` stops
(−1 = strongest AGAINST, +1 = strongest FOR); `heat(v)` for v in 0..1 → from `--attr-0` to `--attr-pos-2`
(unsigned maps are evidence FOR the explained class, so they share the FOR side); `attrRGB(v)` / `heatRGB(v)` →
`[r, g, b]` for canvas; `token(name)` → the computed value.

`js/lib/motion.js` (all honour `reduced`: final state, no movement):
- `reveal(targets, {trigger, start='top 80%', stagger=0.06, delay=0})` — the standard entrance from DESIGN.md.
- `lines(el, {trigger})` — SplitText line-by-line headline entrance; reverts SplitText on resize.
- `count(el, to, {decimals=0, prefix='', suffix='', duration, trigger})` — number ticker in `--font-mono`.
- `scan(el, {trigger, direction='down'})` — the scan/mask reveal signature from DESIGN.md.
- `pin(el, {end='+=100%', scrub=true, timeline})` — pinned, scroll-scrubbed timeline helper.
- `onEnter(el, fn, {start='top 75%', once=true})` — run `fn` when the element enters.

## 7. Charts (owner: charts) — `js/charts/<name>.js`, re-exported by `js/charts/index.js`

Each is `create<Name>(container, options) → { update(options), play(), destroy() }`. The chart renders SVG (or canvas
where stated) sized to the container width (ResizeObserver; height from options or the chart's natural height),
draws in its final layout but hidden until `play()` animates it in (sections call `play()` from a ScrollTrigger);
`update()` animates from the current state to new data (bars grow/shrink, labels re-sort with Flip-like motion);
colours only from `ctx.color`/tokens; labels in `--font-mono`; numbers formatted by the caller-supplied `format`.

| chart | options | used by |
|---|---|---|
| `createWaterfall` | `{base, output, items:[{label, value, contribution}], max: 10, format, valueFormat}` — horizontal SHAP waterfall from `base` to `output`, rest grouped as "N other features", FOR = `attr(+1)`, AGAINST = `attr(−1)` | tumour |
| `createDiverging` | `{items:[{label, value}], domain, format, labelWidth}` — bars left/right of a zero line | tumour (LIME), symptoms (words, contrast) |
| `createRank` | `{items:[{label, value, tone?:'accent'|'muted'|'baseline', note?}], better:'high'|'low', domain, format}` — one-sided bars sorted, best highlighted | trust, tumour (global importance), symptoms (top-5) |
| `createBeeswarm` | `{features, rows:[{feature, shap[], value_norm[]}]}` — dots per patient, x = SHAP, colour = feature value low→high | tumour |
| `createLines` | `{x[], series:[{key, label, values[], tone:'accent'|'pos'|'neg'|'muted'|'faint', dashed?}], xLabel, yLabel, yDomain, area?}` — line chart; `faint` series are many thin lines (ICE) | xray (deletion), tumour (PDP/ICE) |
| `createHeatmap` | canvas: `{image:[[64]x64], map:[[64]x64]|null, signed, alpha=0.65, mode:'overlay'|'map'}` + `reveal()` (develops the map with the scan motion) and `setMap(map, signed)` (crossfade) | xray, hero (optional) |
| `createGauge` | `{value 0..1, label, sublabel}` — probability arc with ticker | tumour, xray |
| `createMatrix` | `{labels, values:[[n]x n] 0..1}` — agreement matrix, cells coloured by `heat` | trust |

`dev/charts.html` shows every chart with mock data for testing.

## 8. Testing

- Backend: start on your own port, hit every endpoint, check shapes and timings.
- Front end: serve with `python v2/server.py --port <yours>` (it serves `web/` even if you only test the page) and use
  `python v2/tools/shot.py --cdp-port <yours+800>`; read the screenshots; fix every console error.
- Do not start anything on port 8503 (reserved for the integrated build) or 8501/8502 (other apps).
