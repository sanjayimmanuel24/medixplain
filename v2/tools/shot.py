"""Screenshot a page in headless Edge/Chrome and print its console errors.

    python v2/tools/shot.py --url http://127.0.0.1:8503/ --out shots/home.png
    python v2/tools/shot.py --url http://127.0.0.1:8503/ --out shots/s.png --scroll 0 --scroll 1800 --scroll 3600
    python v2/tools/shot.py --url ... --out ... --eval "document.querySelector('#tumour').scrollIntoView()"

Every capture is a viewport screenshot (default 1440x900). With several --scroll values the
files are numbered (s-1.png, s-2.png, ...). Console errors and uncaught exceptions are printed
at the end, so a blank section can be traced to its error. Each caller should use its own
--cdp-port so parallel runs do not collide.
"""
import argparse
import base64
import json
import subprocess
import tempfile
import time
import urllib.request
from pathlib import Path

import websocket

BROWSERS = [r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
            r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
            r"C:\Program Files\Google\Chrome\Application\chrome.exe"]


class CDP:
    def __init__(self, ws_url):
        self.ws = websocket.create_connection(ws_url, timeout=120)
        self.n = 0
        self.events = []

    def call(self, method, **params):
        self.n += 1
        self.ws.send(json.dumps({"id": self.n, "method": method, "params": params}))
        while True:
            msg = json.loads(self.ws.recv())
            if msg.get("id") == self.n:
                return msg.get("result", {})
            if "method" in msg:
                self.events.append(msg)

    def pump(self, seconds):
        """Collect events (console messages, exceptions) while time passes."""
        end = time.time() + seconds
        self.ws.settimeout(0.2)
        while time.time() < end:
            try:
                msg = json.loads(self.ws.recv())
                if "method" in msg:
                    self.events.append(msg)
            except websocket.WebSocketTimeoutException:
                pass
        self.ws.settimeout(120)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--width", type=int, default=1440)
    ap.add_argument("--height", type=int, default=900)
    ap.add_argument("--scale", type=float, default=1.0, help="device scale factor")
    ap.add_argument("--wait", type=float, default=4.0, help="seconds after load before the first capture")
    ap.add_argument("--scroll", type=int, action="append", help="window.scrollTo y before a capture (repeatable)")
    ap.add_argument("--settle", type=float, default=1.5, help="seconds after each scroll")
    ap.add_argument("--eval", action="append", default=[], help="JavaScript to run before capturing (repeatable)")
    ap.add_argument("--reduced-motion", action="store_true")
    ap.add_argument("--cdp-port", type=int, default=9340)
    args = ap.parse_args()

    exe = next(b for b in BROWSERS if Path(b).exists())
    profile = tempfile.mkdtemp(prefix="mx_shot_")
    proc = subprocess.Popen([exe, "--headless=new", "--disable-gpu", "--hide-scrollbars", "--mute-audio",
                             f"--remote-debugging-port={args.cdp_port}", f"--user-data-dir={profile}",
                             f"--window-size={args.width},{args.height}", "about:blank"],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        tabs = None
        for _ in range(60):
            try:
                tabs = json.load(urllib.request.urlopen(f"http://127.0.0.1:{args.cdp_port}/json"))
                break
            except OSError:
                time.sleep(0.25)
        page = next(t for t in tabs if t["type"] == "page")
        cdp = CDP(page["webSocketDebuggerUrl"])
        cdp.call("Runtime.enable")
        cdp.call("Log.enable")
        cdp.call("Emulation.setDeviceMetricsOverride", width=args.width, height=args.height,
                 deviceScaleFactor=args.scale, mobile=False)
        feats = [{"name": "prefers-color-scheme", "value": "dark"}]
        if args.reduced_motion:
            feats.append({"name": "prefers-reduced-motion", "value": "reduce"})
        cdp.call("Emulation.setEmulatedMedia", features=feats)
        cdp.call("Page.enable")
        cdp.call("Page.navigate", url=args.url)
        cdp.pump(args.wait)
        for js in args.eval:
            r = cdp.call("Runtime.evaluate", expression=js, awaitPromise=True, returnByValue=True)
            if "exceptionDetails" in r:
                print("eval error:", r["exceptionDetails"].get("text"), r["exceptionDetails"].get("exception", {}).get("description", ""))
            cdp.pump(args.settle)
        positions = args.scroll or [None]
        for k, y in enumerate(positions, 1):
            if y is not None:
                cdp.call("Runtime.evaluate", expression=f"window.scrollTo(0, {y}); window.__lenis && window.__lenis.scrollTo({y}, {{immediate: true}});")
                cdp.pump(args.settle)
            data = cdp.call("Page.captureScreenshot", format="png")["data"]
            path = out if len(positions) == 1 else out.with_name(f"{out.stem}-{k}{out.suffix}")
            path.write_bytes(base64.b64decode(data))
            print("saved", path)
        problems = []
        for ev in cdp.events:
            m = ev.get("method")
            p = ev.get("params", {})
            if m == "Runtime.exceptionThrown":
                d = p.get("exceptionDetails", {})
                problems.append("EXCEPTION " + (d.get("exception", {}).get("description") or d.get("text", "")))
            elif m == "Runtime.consoleAPICalled" and p.get("type") in ("error", "warning", "assert"):
                problems.append(p["type"].upper() + " " + " ".join(str(a.get("value", a.get("description", ""))) for a in p.get("args", [])))
            elif m == "Log.entryAdded" and p.get("entry", {}).get("level") in ("error", "warning"):
                e = p["entry"]
                problems.append(f"LOG {e.get('level')} {e.get('text')} {e.get('url', '')}")
        print("console problems:", len(problems))
        for line in problems[:40]:
            print("  " + line[:400])
    finally:
        proc.terminate()


if __name__ == "__main__":
    main()
