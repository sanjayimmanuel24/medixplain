# MediXplain v2

A ground-up redesign of the dashboard. v1 (`../app.py`, Streamlit, http://localhost:8502) is unchanged.

```bash
python v2/server.py      # run from the medixplain folder, then open http://127.0.0.1:8503
```

- `server.py` + `services.py`: a Starlette JSON API over the same models in `../xai/` (no new installs).
- `web/`: hand-built page (`index.html`, `css/app.css`, `js/app.js`) with GSAP + Lenis and self-hosted fonts in `web/vendor` and `web/fonts`, so it works offline.
- Look: "reading room". Images glow out of the dark, warm amber means evidence **for** the prediction and cool blue means evidence **against**, in all three labs.
- Hero: 4,096 particles, one per pixel of a real test X-ray, in three frames: noise → the X-ray → the CNN's real Grad-CAM map swept in by a scan line.
- `tools/shot.py` takes headless screenshots and prints console errors; `CONTRACT.md` documents the API.
