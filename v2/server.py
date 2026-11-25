"""MediXplain v2 server: the page at / and the JSON API at /api (CONTRACT.md §5).

    python v2/server.py              # http://127.0.0.1:8503
    python v2/server.py --port 8511  # any other port
"""
from __future__ import annotations

import argparse
import contextlib
import logging
import sys
import time
from pathlib import Path

import uvicorn
from starlette.applications import Starlette
from starlette.concurrency import run_in_threadpool
from starlette.middleware.gzip import GZipMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Mount, Route
from starlette.staticfiles import StaticFiles

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import services  # noqa: E402

log = logging.getLogger("medixplain.v2")
S: services.Services | None = None


def ok(data, status=200):
    return JSONResponse(services.clean(data), status_code=status)


def fail(message, status=400):
    return JSONResponse({"error": str(message)}, status_code=status)


async def call(fn, *args, **kwargs):
    """Run a model call in the thread pool, one at a time, with clean errors."""
    def locked():
        with services.LOCK:
            return fn(*args, **kwargs)
    t = time.perf_counter()
    try:
        data = await run_in_threadpool(locked)
    except (ValueError, KeyError, TypeError) as e:
        return fail(e, 400)
    except Exception as e:  # noqa: BLE001
        log.exception("model call failed")
        return fail(f"server error: {e}", 500)
    log.info("%s %.0f ms", getattr(fn, "__name__", "call"), (time.perf_counter() - t) * 1000)
    return ok(data)


async def body(request: Request) -> dict:
    try:
        data = await request.json()
    except Exception:  # noqa: BLE001
        return {}
    return data if isinstance(data, dict) else {}


# ---------------------------------------------------------------- routes
async def health(_):
    return ok({"ok": True})


async def metrics(_):
    return await call(services.metrics)


async def tab_meta(_):
    return await call(S.tabular.meta)


async def tab_explain(request):
    b = await body(request)
    return await call(S.tabular.explain, int(b.get("index", 0)), b.get("values"), int(b.get("lime_seed", 42)))


async def tab_global(_):
    return await call(S.tab_global)


async def tab_pdp(request):
    return await call(S.tabular.pdp, request.query_params.get("feature", "worst_concave_points"))


async def tab_lime_stability(request):
    b = await body(request)
    return await call(S.tabular.lime_stability, int(b.get("index", 0)), int(b.get("runs", 5)))


async def img_samples(_):
    return await call(S.image_samples)


async def img_explain(request):
    b = await body(request)
    return await call(S.images.explain, b.get("index"), b.get("image"), b.get("target"))


async def img_sanity(request):
    b = await body(request)
    return await call(S.images.sanity, b.get("index", 0), b.get("method", "gradcam"), b.get("target"))


async def txt_meta(_):
    return await call(S.texts.meta)


async def txt_explain(request):
    b = await body(request)
    return await call(S.texts.explain, b.get("text", ""), b.get("cls"), b.get("contrast"))


async def not_found(_):
    return fail("unknown API route", 404)


routes = [
    Route("/api/health", health),
    Route("/api/metrics", metrics),
    Route("/api/tabular/meta", tab_meta),
    Route("/api/tabular/explain", tab_explain, methods=["POST"]),
    Route("/api/tabular/global", tab_global),
    Route("/api/tabular/pdp", tab_pdp),
    Route("/api/tabular/lime-stability", tab_lime_stability, methods=["POST"]),
    Route("/api/image/samples", img_samples),
    Route("/api/image/explain", img_explain, methods=["POST"]),
    Route("/api/image/sanity", img_sanity, methods=["POST"]),
    Route("/api/text/meta", txt_meta),
    Route("/api/text/explain", txt_explain, methods=["POST"]),
    Route("/api/{rest:path}", not_found, methods=["GET", "POST"]),
    Mount("/", StaticFiles(directory=HERE / "web", html=True), name="web"),
]


@contextlib.asynccontextmanager
async def lifespan(_app):
    """Load every model once before the first request (Starlette 1.x has no on_startup)."""
    global S
    t = time.perf_counter()
    S = await run_in_threadpool(services.Services)
    log.info("models loaded in %.1f s", time.perf_counter() - t)
    yield


app = Starlette(routes=routes, lifespan=lifespan)
app.add_middleware(GZipMiddleware, minimum_size=1024)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8503)
    ap.add_argument("--host", default="127.0.0.1")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    print(f"MediXplain v2 -> http://{a.host}:{a.port}")
    uvicorn.run(app, host=a.host, port=a.port, log_level="warning")
