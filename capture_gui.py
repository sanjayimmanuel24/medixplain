"""Capture GUI screenshots for the report through headless Edge/Chrome (DevTools protocol).

    streamlit run app.py          # in another terminal
    python capture_gui.py
"""
import base64
import json
import subprocess
import tempfile
import time
import urllib.request
from pathlib import Path

import websocket

from xai.common import FIGURES

BROWSERS = [r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
            r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
            r"C:\Program Files\Google\Chrome\Application\chrome.exe"]
APP = "http://localhost:8502"
PORT = 9333

# (file name, page index, tab to click or None, seconds to wait after load)
SHOTS = [
    ("gui_overview.png", 0, None, 12),
    ("gui_tumour_shap.png", 1, None, 14),
    ("gui_tumour_counterfactual.png", 1, 2, 6),
    ("gui_xray_maps.png", 2, None, 22),
    ("gui_xray_sanity.png", 2, 2, 14),
    ("gui_text.png", 3, None, 14),
    ("gui_trust_lab.png", 4, None, 10),
]


class CDP:
    def __init__(self, ws_url):
        self.ws = websocket.create_connection(ws_url, timeout=60)
        self.n = 0

    def call(self, method, **params):
        self.n += 1
        self.ws.send(json.dumps({"id": self.n, "method": method, "params": params}))
        while True:
            msg = json.loads(self.ws.recv())
            if msg.get("id") == self.n:
                return msg.get("result", {})


def main():
    exe = next(b for b in BROWSERS if Path(b).exists())
    profile = tempfile.mkdtemp(prefix="mx_cdp_")
    proc = subprocess.Popen([exe, "--headless=new", "--disable-gpu", "--hide-scrollbars",
                             f"--remote-debugging-port={PORT}", f"--user-data-dir={profile}",
                             "--window-size=1500,1000", "about:blank"])
    try:
        for _ in range(50):
            try:
                tabs = json.load(urllib.request.urlopen(f"http://127.0.0.1:{PORT}/json"))
                break
            except OSError:
                time.sleep(0.3)
        page = next(t for t in tabs if t["type"] == "page")
        cdp = CDP(page["webSocketDebuggerUrl"])
        cdp.call("Emulation.setDeviceMetricsOverride", width=1500, height=1000, deviceScaleFactor=1.5, mobile=False)
        cdp.call("Emulation.setEmulatedMedia", features=[{"name": "prefers-color-scheme", "value": "light"}])
        current = None
        for name, idx, tab, wait in SHOTS:
            if idx != current:
                cdp.call("Page.navigate", url=f"{APP}/?page={idx}")
                current = idx
            if tab is not None:
                r = cdp.call("Runtime.evaluate", returnByValue=True, expression=(
                    f"(() => {{ const t = document.querySelectorAll('[role=\"tab\"]'); "
                    f"if (t.length > {tab}) {{ t[{tab}].click(); return t.length; }} return -t.length; }})()"))
                print("  tabs found:", r.get("result", {}).get("value"))
            time.sleep(wait)
            # dismiss Streamlit's own promotional popup so it does not cover the page
            cdp.call("Runtime.evaluate", expression=(
                "[...document.querySelectorAll('button')].filter(b => b.textContent.trim() === \"Don't show again\")"
                ".forEach(b => b.click())"))
            time.sleep(1)
            data = cdp.call("Page.captureScreenshot", format="png")["data"]
            (FIGURES / name).write_bytes(base64.b64decode(data))
            print("saved", name)
    finally:
        proc.terminate()


if __name__ == "__main__":
    main()
