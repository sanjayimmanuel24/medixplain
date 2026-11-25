// MediXplain v2 — one module: helpers, charts, hero, three labs, Trust Lab, motion.
const { gsap, ScrollTrigger, SplitText, ScrambleTextPlugin, Lenis } = window;
gsap.registerPlugin(ScrollTrigger, SplitText, ScrambleTextPlugin);
const reduced = matchMedia("(prefers-reduced-motion: reduce)").matches;
const $ = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => [...r.querySelectorAll(s)];
const NS = "http://www.w3.org/2000/svg";
const css = (n) => getComputedStyle(document.documentElement).getPropertyValue(n).trim();

// ---------------------------------------------------------------- api + text helpers
async function api(path, body, signal) {
  const r = await fetch("/api" + path, body ? { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body), signal } : { signal });
  const j = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(j.error || `HTTP ${r.status}`);
  return j;
}
const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const md = (s) => s.split(/\n\n+/).map((p) => "<p>" + esc(p).replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>").replace(/(^|[\s(])_(.+?)_(?=[\s.,)]|$)/g, "$1<em>$2</em>").replace(/\*(.+?)\*/g, "<em>$1</em>") + "</p>").join("");
const pct = (v, d = 1) => (v * 100).toFixed(d) + "%";
const num = (v, d = 2) => (Math.abs(v) >= 1000 ? Math.round(v).toLocaleString() : (+v).toFixed(d));
const debounce = (fn, ms) => { let t; return (...a) => { clearTimeout(t); t = setTimeout(() => fn(...a), ms); }; };
const errorBox = (el, msg, retry) => { el.innerHTML = `<div class="error">${esc(msg)} <button class="btn btn--ghost">retry</button></div>`; $("button", el).onclick = retry; };

// ---------------------------------------------------------------- colour: warm = FOR, cool = AGAINST
const hex = (h) => [1, 3, 5].map((i) => parseInt(h.slice(i, i + 2), 16));
const STOPS = ["--attr-neg-2", "--attr-neg-1", "--attr-0", "--attr-pos-1", "--attr-pos-2"].map((n) => hex(css(n)));
const mix = (a, b, t) => a.map((v, i) => Math.round(v + (b[i] - v) * t));
function attrRGB(v) { // v in -1..1
  const x = (Math.max(-1, Math.min(1, v)) + 1) * 2; const i = Math.min(3, Math.floor(x));
  return mix(STOPS[i], STOPS[i + 1], x - i);
}
const HEAT = [[27, 34, 43], [168, 97, 42], [255, 181, 71], [255, 238, 200]];
function heatRGB(v) { // v in 0..1
  const x = Math.max(0, Math.min(1, v)) * 3; const i = Math.min(2, Math.floor(x));
  return mix(HEAT[i], HEAT[i + 1], x - i);
}
const rgb = (c, a = 1) => `rgba(${c[0]},${c[1]},${c[2]},${a})`;

// ---------------------------------------------------------------- motion helpers
function whenVisible(el, fn, start = "top 82%") {
  if (reduced) return fn();
  ScrollTrigger.create({ trigger: el, start, once: true, onEnter: fn });
}
function countUp(el) {
  const to = parseFloat(el.dataset.count), dec = +(el.dataset.dec || 0), suf = el.dataset.suf || "";
  const o = { v: 0 };
  const set = () => (el.textContent = o.v.toFixed(dec) + suf);
  if (reduced) { o.v = to; return set(); }
  set();
  whenVisible(el, () => gsap.to(o, { v: to, duration: 1.6, ease: "expo.out", onUpdate: set }));
}
function headline(el) {
  if (reduced) return;
  const split = SplitText.create(el, { type: "lines", mask: "lines", linesClass: "line" });
  gsap.from(split.lines, { yPercent: 105, duration: 1.1, ease: "expo.out", stagger: 0.09, scrollTrigger: { trigger: el, start: "top 85%", once: true } });
}
function rise(targets, trigger) {
  if (reduced) return;
  gsap.from(targets, { y: 40, autoAlpha: 0, duration: 1, ease: "expo.out", stagger: 0.08, scrollTrigger: { trigger: trigger || targets, start: "top 85%", once: true } });
}

// ---------------------------------------------------------------- charts (hand-made SVG)
function svg(w, h) { const s = document.createElementNS(NS, "svg"); s.setAttribute("viewBox", `0 0 ${w} ${h}`); s.setAttribute("role", "img"); return s; }
function el(tag, attrs, parent) { const e = document.createElementNS(NS, tag); for (const k in attrs) e.setAttribute(k, attrs[k]); parent && parent.appendChild(e); return e; }

// horizontal bars; signed → zero line in the middle, colours by sign; else one-sided
function hbars(host, items, { signed = false, format = (v) => num(v, 3), labelW = 190, max, W = 640 } = {}) {
  const rowH = 30, H = items.length * rowH + 8;
  const s = svg(W, H); host.replaceChildren(s);
  const valW = 70, x0 = labelW, x1 = W - valW;
  const m = max || Math.max(1e-9, ...items.map((d) => Math.abs(d.value)));
  const zero = signed ? (x0 + x1) / 2 : x0;
  const span = signed ? (x1 - x0) / 2 : x1 - x0;
  if (signed) el("line", { x1: zero, x2: zero, y1: 0, y2: H, stroke: css("--line-strong") }, s);
  const bars = [];
  items.forEach((d, i) => {
    const y = i * rowH + 6, w = (Math.abs(d.value) / m) * span;
    const t = el("text", { x: labelW - 12, y: y + 15, "text-anchor": "end" }, s); t.textContent = d.label.length > 26 ? d.label.slice(0, 25) + "…" : d.label;
    if (d.tone === "baseline") t.setAttribute("fill", css("--text-faint"));
    let fill;
    if (signed) fill = rgb(attrRGB(d.value >= 0 ? 0.55 + 0.45 * (Math.abs(d.value) / m) : -0.55 - 0.45 * (Math.abs(d.value) / m)));
    else fill = d.tone === "baseline" ? "transparent" : d.tone === "accent" ? css("--accent") : css("--line-strong");
    const x = signed ? (d.value >= 0 ? zero : zero - w) : x0;
    const r = el("rect", { x, y, width: Math.max(1, w), height: 18, rx: 3, fill }, s);
    if (d.tone === "baseline") { r.setAttribute("stroke", css("--text-faint")); r.setAttribute("stroke-dasharray", "4 3"); }
    const v = el("text", { x: W - 4, y: y + 15, "text-anchor": "end", fill: d.tone === "accent" ? css("--accent") : css("--text") }, s); v.textContent = format(d.value);
    bars.push({ r, x, w, signed, pos: d.value >= 0, zero });
  });
  const play = () => {
    if (reduced) return;
    bars.forEach((b, i) => {
      const from = b.signed ? { attr: { width: 0, x: b.pos ? b.zero : b.zero } } : { attr: { width: 0 } };
      gsap.fromTo(b.r, from, { attr: { width: Math.max(1, b.w), x: b.x }, duration: 0.9, ease: "expo.out", delay: i * 0.035 });
    });
  };
  return { play };
}

// SHAP waterfall: base → output, top 9 features + the rest grouped
function waterfall(host, d) {
  const top = d.shap.slice(0, 9), rest = d.shap.slice(9);
  const items = top.map((s) => ({ label: `${s.label} = ${num(s.value, s.value > 100 ? 0 : 3)}`, v: s.shap }));
  if (rest.length) items.push({ label: `${rest.length} other measurements`, v: rest.reduce((a, s) => a + s.shap, 0) });
  let cum = d.base_value; const segs = items.map((it) => { const a = cum; cum += it.v; return { ...it, a, b: cum }; });
  const lo = Math.min(d.base_value, ...segs.map((s) => Math.min(s.a, s.b))), hi = Math.max(d.base_value, ...segs.map((s) => Math.max(s.a, s.b)));
  const W = 760, labelW = 250, rowH = 32, H = segs.length * rowH + 56, x0 = labelW, x1 = W - 60;
  const X = (v) => x0 + ((v - lo) / (hi - lo || 1)) * (x1 - x0);
  const s = svg(W, H); host.replaceChildren(s);
  const g = el("g", {}, s);
  el("line", { x1: X(d.base_value), x2: X(d.base_value), y1: 0, y2: H - 40, stroke: css("--line-strong"), "stroke-dasharray": "3 4" }, g);
  el("line", { x1: X(d.output_value), x2: X(d.output_value), y1: 0, y2: H - 40, stroke: css("--accent"), "stroke-dasharray": "3 4" }, g);
  const t1 = el("text", { x: X(d.base_value), y: H - 22, "text-anchor": "middle" }, g); t1.textContent = `average ${num(d.base_value)}`;
  const t2 = el("text", { x: X(d.output_value), y: H - 6, "text-anchor": "middle", fill: css("--accent") }, g); t2.textContent = `this patient ${num(d.output_value)}`;
  const rects = [];
  segs.forEach((sg, i) => {
    const y = i * rowH + 4, pos = sg.v >= 0;
    const lab = el("text", { x: labelW - 14, y: y + 17, "text-anchor": "end" }, g); lab.textContent = sg.label;
    const xa = X(Math.min(sg.a, sg.b)), w = Math.max(2, Math.abs(X(sg.b) - X(sg.a)));
    const r = el("rect", { x: xa, y, width: w, height: 22, rx: 3, fill: rgb(attrRGB(pos ? 1 : -1)) }, g);
    const v = el("text", { x: pos ? xa + w + 8 : xa - 8, y: y + 17, "text-anchor": pos ? "start" : "end", fill: css("--text") }, g);
    v.textContent = (pos ? "+" : "−") + num(Math.abs(sg.v));
    rects.push({ r, xa, w, start: X(sg.a), pos, v });
  });
  if (!reduced) rects.forEach((o, i) => {
    gsap.fromTo(o.r, { attr: { x: o.start, width: 0 } }, { attr: { x: o.xa, width: o.w }, duration: 0.8, ease: "expo.out", delay: 0.05 * i });
    gsap.from(o.v, { autoAlpha: 0, duration: 0.4, delay: 0.05 * i + 0.4 });
  });
}

function gauge(host, value, label) {
  const R = 50, C = 2 * Math.PI * R;
  if (!host.firstChild) {
    host.innerHTML = `<svg viewBox="0 0 120 120"><circle cx="60" cy="60" r="${R}" fill="none" stroke="${css("--line")}" stroke-width="8"/><circle class="g-arc" cx="60" cy="60" r="${R}" fill="none" stroke-width="8" stroke-linecap="round" transform="rotate(-90 60 60)" stroke-dasharray="${C}" stroke-dashoffset="${C}"/></svg><div><div class="gauge__val">0%</div><div class="gauge__lab"></div></div>`;
    host._v = { v: 0 };
  }
  const arc = $(".g-arc", host), val = $(".gauge__val", host);
  $(".gauge__lab", host).textContent = label;
  arc.setAttribute("stroke", rgb(attrRGB(value * 2 - 1)));
  gsap.to(arc, { attr: { "stroke-dashoffset": C * (1 - value) }, duration: reduced ? 0 : 1, ease: "expo.out" });
  gsap.to(host._v, { v: value, duration: reduced ? 0 : 1, ease: "expo.out", onUpdate: () => (val.textContent = pct(host._v.v)) });
}

function lines(host, x, series, { yLabel = "", active } = {}) {
  const W = 420, H = 240, L = 36, B = 28, T = 8, Rr = 8;
  const X = (i) => L + (x[i] / x[x.length - 1]) * (W - L - Rr), Y = (v) => T + (1 - v) * (H - T - B);
  const s = svg(W, H); host.replaceChildren(s);
  [0, 0.5, 1].forEach((v) => { el("line", { x1: L, x2: W - Rr, y1: Y(v), y2: Y(v), stroke: css("--line") }, s); const t = el("text", { x: L - 6, y: Y(v) + 4, "text-anchor": "end" }, s); t.textContent = v; });
  [0, 50, 100].forEach((v) => { const t = el("text", { x: L + (v / 100) * (W - L - Rr), y: H - 8, "text-anchor": "middle" }, s); t.textContent = v + "%"; });
  const paths = [];
  series.forEach((sr) => {
    const dpath = sr.values.map((v, i) => `${i ? "L" : "M"}${X(i).toFixed(1)},${Y(v).toFixed(1)}`).join("");
    const on = sr.key === active;
    const p = el("path", { d: dpath, fill: "none", stroke: on ? css("--accent") : sr.key === "random" ? css("--text-faint") : css("--line-strong"), "stroke-width": on ? 2.5 : 1.25, "stroke-dasharray": sr.key === "random" ? "5 4" : "none" }, s);
    if (on) paths.push(p); else paths.unshift(p);
  });
  if (!reduced) paths.forEach((p, i) => { if (p.getAttribute("stroke-dasharray") !== "none") return; const L2 = p.getTotalLength(); gsap.fromTo(p, { attr: { "stroke-dasharray": L2, "stroke-dashoffset": L2 } }, { attr: { "stroke-dashoffset": 0 }, duration: 1.2, ease: "power2.out", delay: i * 0.05, onComplete: () => p.setAttribute("stroke-dasharray", "none") }); });
  const lab = el("text", { x: L, y: T + 2, "dominant-baseline": "hanging" }, s); lab.textContent = yLabel;
}

// ---------------------------------------------------------------- canvas: X-ray + map
function drawXray(canvas, image, map, signed, { alpha = 0.78, reveal = 1 } = {}) {
  const ctx = canvas.getContext("2d"), N = image.length, W = canvas.width, H = canvas.height;
  const base = new OffscreenCanvas(N, N), bc = base.getContext("2d"), bi = bc.createImageData(N, N);
  const over = new OffscreenCanvas(N, N), oc = over.getContext("2d"), oi = oc.createImageData(N, N);
  for (let y = 0; y < N; y++) for (let x = 0; x < N; x++) {
    const k = (y * N + x) * 4, g = image[y][x] * 255;
    bi.data[k] = g * 0.93; bi.data[k + 1] = g * 0.95; bi.data[k + 2] = g; bi.data[k + 3] = 255;
    if (map) {
      const v = map[y][x], c = signed ? attrRGB(v) : heatRGB(v), a = signed ? Math.min(1, Math.abs(v) * 1.4) : Math.min(1, v * 1.15);
      oi.data[k] = c[0]; oi.data[k + 1] = c[1]; oi.data[k + 2] = c[2]; oi.data[k + 3] = a * 255;
    }
  }
  bc.putImageData(bi, 0, 0); oc.putImageData(oi, 0, 0);
  ctx.imageSmoothingEnabled = true; ctx.imageSmoothingQuality = "high";
  ctx.fillStyle = "#000"; ctx.fillRect(0, 0, W, H);
  ctx.globalAlpha = 1; ctx.drawImage(base, 0, 0, W, H);
  if (map) {
    ctx.save(); ctx.beginPath(); ctx.rect(0, 0, W, H * reveal); ctx.clip();
    ctx.globalAlpha = alpha; ctx.drawImage(over, 0, 0, W, H); ctx.restore();
    if (reveal < 1) { ctx.globalAlpha = 1; ctx.fillStyle = css("--accent-2"); ctx.fillRect(0, H * reveal - 1, W, 2); }
  }
  ctx.globalAlpha = 1;
}

// ---------------------------------------------------------------- shell: Lenis, nav, progress
let lenis = null;
function shell() {
  if (!reduced) {
    lenis = new Lenis({ autoRaf: false, lerp: 0.1 });
    window.__lenis = lenis;
    lenis.on("scroll", ScrollTrigger.update);
    gsap.ticker.add((t) => lenis.raf(t * 1000));
    gsap.ticker.lagSmoothing(0);
  }
  $$('a[href^="#"]').forEach((a) => a.addEventListener("click", (e) => {
    const t = $(a.getAttribute("href")); if (!t) return;
    e.preventDefault(); lenis ? lenis.scrollTo(t, { offset: -40, duration: 1.4 }) : t.scrollIntoView();
  }));
  gsap.to("#progress", { scaleX: 1, ease: "none", scrollTrigger: { trigger: document.body, start: "top top", end: "bottom bottom", scrub: 0.3 } });
  $$(".nav__links a").forEach((a) => ScrollTrigger.create({ trigger: a.getAttribute("href"), start: "top 50%", end: "bottom 50%", onToggle: (st) => a.classList.toggle("is-on", st.isActive) }));
  $$("[data-count]").forEach(countUp);
  $$(".h2").forEach(headline);
  $$(".models__grid .mcard").forEach((c) => c.addEventListener("pointermove", (e) => { const r = c.getBoundingClientRect(); c.style.setProperty("--mx", e.clientX - r.left + "px"); c.style.setProperty("--my", e.clientY - r.top + "px"); }));
  rise($$(".mcard"), ".models__grid");
  $$(".lab, .trust").forEach((sec) => rise($$(".panel, .thumbs", sec), sec.querySelector(".lab__grid, .trust__grid, .thumbs") || sec));
  rise($$(".finding"), "#findings");
}

// ---------------------------------------------------------------- hero: 4,096 particles, three frames
async function hero() {
  const canvas = $("#hero-canvas"), ctx = canvas.getContext("2d"), stage = $("#hero-stage");
  const DPR = Math.min(2, devicePixelRatio || 1);
  let W, H;
  const N = 64, P = N * N;
  const px = new Float32Array(P), py = new Float32Array(P), nx = new Float32Array(P), ny = new Float32Array(P);
  const inten = new Float32Array(P).fill(0.3), heat = new Float32Array(P);
  const state = { form: 0, heat: 0, scan: -1, drift: 0 };
  const mouse = { x: -1e4, y: -1e4 };
  let cell = 8, ox = 0, oy = 0;
  function layout() {
    W = canvas.clientWidth; H = canvas.clientHeight; canvas.width = W * DPR; canvas.height = H * DPR;
    const size = Math.min(H * 0.84, W * 0.44); cell = size / N;
    ox = W > 900 ? W - size - W * 0.04 : (W - size) / 2; oy = (H - size) / 2 + 20;
  }
  layout(); addEventListener("resize", layout);
  let seed = 7; const rnd = () => ((seed = (seed * 16807) % 2147483647) / 2147483647);
  for (let i = 0; i < P; i++) { nx[i] = rnd(); ny[i] = rnd(); }
  canvas.addEventListener("pointermove", (e) => { const r = canvas.getBoundingClientRect(); mouse.x = e.clientX - r.left; mouse.y = e.clientY - r.top; });
  canvas.addEventListener("pointerleave", () => { mouse.x = mouse.y = -1e4; });
  const bone = hex(css("--text")), warm = attrRGB(1), cool = attrRGB(-1);
  function frame(t) {
    ctx.setTransform(DPR, 0, 0, DPR, 0, 0);
    ctx.fillStyle = "#07090c"; ctx.fillRect(0, 0, W, H);
    ctx.globalCompositeOperation = "lighter";
    const f = state.form, hz = state.heat, sc = state.scan;
    for (let i = 0; i < P; i++) {
      const gx = i % N, gy = (i / N) | 0;
      const tx = ox + gx * cell + cell / 2, ty = oy + gy * cell + cell / 2;
      const wob = Math.sin(t * 0.0006 + i * 0.37) * (1 - f) * 18;
      let x = nx[i] * W + wob + (tx - nx[i] * W) * f, y = ny[i] * H + Math.cos(t * 0.0005 + i) * (1 - f) * 18 + (ty - ny[i] * H) * f;
      y += state.drift * (inten[i] - 0.5) * 120;
      const dx = x - mouse.x, dy = y - mouse.y, d2 = dx * dx + dy * dy;
      if (d2 < 14400) { const k = (1 - d2 / 14400) * 26; x += (dx / Math.sqrt(d2 + 1)) * k; y += (dy / Math.sqrt(d2 + 1)) * k; }
      px[i] = x; py[i] = y;
      const v = inten[i];
      const scanned = sc < 0 ? 0 : Math.max(0, Math.min(1, (sc * (N + 8) - gy) / 6));
      const h = heat[i] * hz * scanned;
      const c = h > 0.02 ? mix(mix(bone, cool, 0.15), warm, Math.min(1, h * 1.3)) : bone;
      const a = (0.08 + (0.85 * v * v) * f + 0.06 * (1 - f)) * (1 - 0.35 * h) + h * 0.9;
      const s = Math.max(1, cell * (0.28 + 0.62 * v * f + 0.25 * h));
      ctx.fillStyle = rgb(c, Math.min(1, a));
      ctx.fillRect(x - s / 2, y - s / 2, s, s);
    }
    if (sc >= 0 && sc < 1) {
      const yy = oy + sc * (N + 8) * cell;
      ctx.globalCompositeOperation = "source-over";
      const gr = ctx.createLinearGradient(0, yy - 40, 0, yy); gr.addColorStop(0, "rgba(111,211,224,0)"); gr.addColorStop(1, "rgba(111,211,224,.22)");
      ctx.fillStyle = gr; ctx.fillRect(ox - 20, yy - 40, N * cell + 40, 40);
      ctx.fillStyle = css("--accent-2"); ctx.fillRect(ox - 20, yy, N * cell + 40, 1.5);
    }
    ctx.globalCompositeOperation = "source-over";
  }
  let running = true;
  ScrollTrigger.create({ trigger: "#hero", start: "top top", end: "bottom top", onToggle: (s) => (running = s.isActive) });
  gsap.ticker.add(() => running && frame(performance.now()));

  const copy = $$(".hero__copy > *");
  if (!reduced) {
    const split = SplitText.create(".hero__title", { type: "lines", mask: "lines" });
    gsap.from(split.lines, { yPercent: 110, duration: 1.4, ease: "expo.out", stagger: 0.12, delay: 0.2 });
    gsap.from([copy[0], copy[2], copy[3], ".hero__legend", ".hero__cue"], { autoAlpha: 0, y: 24, duration: 1.2, ease: "expo.out", stagger: 0.1, delay: 0.5 });
  }
  let data;
  try { data = await api("/image/explain", { index: 0 }); } catch { stage.textContent = "offline · noise only"; return; }
  for (let y = 0; y < N; y++) for (let x = 0; x < N; x++) { inten[y * N + x] = data.image[y][x]; heat[y * N + x] = data.maps.gradcam[y][x]; }
  const setStage = (s) => gsap.to(stage, { duration: 0.6, scrambleText: { text: s, chars: "01·", speed: 0.6 } });
  if (reduced) { Object.assign(state, { form: 1, heat: 1, scan: 1 }); stage.textContent = "03 · explanation"; return; }
  gsap.timeline({ delay: 0.3 })
    .call(() => setStage("01 · noise"))
    .to(state, { form: 1, duration: 2.6, ease: "expo.inOut" }, 0.2)
    .call(() => setStage("02 · signal"), null, 1.6)
    .to(state, { scan: 1, duration: 2.0, ease: "power1.inOut", onStart: () => (state.heat = 1) }, 3.0)
    .call(() => setStage("03 · explanation"), null, 3.6)
    .set(state, { scan: 1.2 });
  gsap.to(state, { drift: 1, ease: "none", scrollTrigger: { trigger: "#hero", start: "top top", end: "bottom top", scrub: true } });
  gsap.to(".hero__copy", { yPercent: -30, autoAlpha: 0, ease: "none", scrollTrigger: { trigger: "#hero", start: "top top", end: "bottom 30%", scrub: true } });
}

// ---------------------------------------------------------------- problem: words light up on scroll
function problem() {
  const split = SplitText.create("#problem-text", { type: "words", wordsClass: "word" });
  if (reduced) return;
  gsap.to(split.words, { opacity: 1, stagger: 0.1, ease: "none", scrollTrigger: { trigger: "#problem", start: "top 70%", end: "bottom 75%", scrub: true } });
}

// ---------------------------------------------------------------- tumour lab
async function tumour() {
  const sel = $("#tum-patient"), wf = $("#tum-waterfall");
  let meta, index = 0, values = {}, ctrl, feats = {};
  try { meta = await api("/tabular/meta"); } catch (e) { return errorBox(wf, e.message, tumour); }
  meta.features.forEach((f) => (feats[f.name] = f));
  sel.innerHTML = meta.patients.map((p) => `<option value="${p.index}">#${String(p.index).padStart(3, "0")} · ${p.label} · P(malignant) ${pct(p.prob)}</option>`).join("");
  const run = async (rebuildSliders) => {
    ctrl?.abort(); ctrl = new AbortController();
    wf.style.opacity = 0.5;
    let d;
    try { d = await api("/tabular/explain", { index, values }, ctrl.signal); } catch (e) { if (e.name !== "AbortError") errorBox(wf, e.message, () => run(true)); return; }
    wf.style.opacity = 1;
    gauge($("#tum-gauge"), d.prob, `P(malignant) · ${d.label}`);
    $("#tum-glass").textContent = `glass-box model says ${pct(d.glass_prob)}`;
    waterfall(wf, d);
    const cf = d.counterfactual;
    $("#tum-cf").innerHTML = cf.flipped
      ? `<div class="cf__head">Becomes <em>${esc(cf.target)}</em> (${pct(cf.start_prob)} → ${pct(cf.end_prob)}) if:</div>` + cf.changes.slice(0, 6).map((c) => `<div class="cf__row"><b>${esc(c.label)}</b><span>${num(c.from)} <i>→ ${num(c.to)}</i></span></div>`).join("")
      : `<div class="cf__head">No small change flips this prediction.</div>`;
    if (!reduced) gsap.from($$(".cf__row", $("#tum-cf")), { x: -12, autoAlpha: 0, stagger: 0.05, duration: 0.5, ease: "expo.out" });
    $("#tum-card").innerHTML = md(d.card);
    if (rebuildSliders) sliders(d);
  };
  function sliders(d) {
    const box = $("#tum-sliders"); box.innerHTML = "";
    d.shap.slice(0, 5).forEach((s) => {
      const f = feats[s.feature];
      const row = document.createElement("label"); row.className = "slider";
      row.innerHTML = `<span>${esc(f.label)}</span><output class="mono">${num(s.value)}</output><input type="range" min="${f.min}" max="${f.max}" step="${f.step}" value="${s.value}" aria-label="${esc(f.label)}">`;
      const inp = $("input", row), out = $("output", row);
      inp.addEventListener("input", () => { out.textContent = num(+inp.value); values[s.feature] = +inp.value; go(); });
      box.appendChild(row);
    });
  }
  const go = debounce(() => run(false), 220);
  const pick = (i) => { index = (i + meta.patients.length) % meta.patients.length; sel.value = index; values = {}; run(true); };
  sel.onchange = () => pick(+sel.value);
  $("#tum-prev").onclick = () => pick(index - 1);
  $("#tum-next").onclick = () => pick(index + 1);
  $("#tum-reset").onclick = () => pick(index);
  whenVisible($("#tumour"), () => run(true), "top 90%");
}

// ---------------------------------------------------------------- x-ray lab
async function xray() {
  const canvas = $("#xr-canvas"), loading = $("#xr-loading"), seg = $("#xr-methods");
  let samples, cur = { index: 0 }, data = null, method = "gradcam", compare = false;
  try { samples = (await api("/image/samples")).samples; } catch (e) { return errorBox($("#xr-thumbs"), e.message, xray); }
  $("#xr-thumbs").innerHTML = samples.map((s) => `<button class="thumb" role="option" data-i="${s.index}" aria-selected="false" title="${s.label} · P(pneumonia) ${pct(s.prob)}"><img src="${s.png}" alt="${s.label} X-ray"><i style="background:${rgb(attrRGB(s.label === "pneumonia" ? 1 : -1))}"></i></button>`).join("");
  $$(".thumb").forEach((b) => (b.onclick = () => load({ index: +b.dataset.i })));
  const pill = document.createElement("span"); pill.className = "seg__pill";
  const movePill = () => { const on = $('[aria-selected="true"]', seg); if (on) { pill.style.left = on.offsetLeft + "px"; pill.style.width = on.offsetWidth + "px"; } };
  function paint(revealIt) {
    if (!data) return;
    const m = data.methods.find((x) => x.key === method);
    $("#xr-plain").textContent = `${m.plain} · ${m.name}`;
    if (revealIt && !reduced) {
      const o = { r: 0 };
      gsap.to(o, { r: 1, duration: 0.9, ease: "power2.inOut", onUpdate: () => drawXray(canvas, data.image, data.maps[method], m.signed, { reveal: o.r }) });
    } else drawXray(canvas, data.image, data.maps[method], m.signed);
    lines($("#xr-deletion"), data.deletion.x, Object.entries(data.deletion.curves).map(([key, values]) => ({ key, values })), { yLabel: `P(${data.target})`, active: method });
    if (compare) sanity();
  }
  async function load(src) {
    cur = src; compare = false; setCompare(false);
    $$(".thumb").forEach((b) => b.setAttribute("aria-selected", String(+b.dataset.i === src.index)));
    loading.classList.add("is-on");
    try { data = await api("/image/explain", src); } catch (e) { loading.classList.remove("is-on"); return errorBox($("#xr-card"), e.message, () => load(src)); }
    loading.classList.remove("is-on");
    if (!seg.children.length) {
      seg.innerHTML = data.methods.map((m) => `<button role="tab" data-k="${m.key}" aria-selected="${m.key === method}">${m.name}</button>`).join("");
      seg.appendChild(pill);
      $$("button", seg).forEach((b) => (b.onclick = () => { method = b.dataset.k; $$("button", seg).forEach((x) => x.setAttribute("aria-selected", String(x === b))); movePill(); paint(true); }));
      requestAnimationFrame(movePill);
    }
    gauge($("#xr-gauge"), data.prob.pneumonia, `P(pneumonia) · ${data.predicted}`);
    $("#xr-card").innerHTML = md(data.card);
    paint(true);
  }
  function setCompare(on) {
    compare = on; $("#xr-compare").hidden = !on; $("#xr-sanity").setAttribute("aria-pressed", String(on));
    if (!on) $("#xr-sanity-out").textContent = "";
  }
  async function sanity() {
    if (cur.index == null) { $("#xr-sanity-out").textContent = "pick a test X-ray for the honesty test"; return setCompare(false); }
    $("#xr-sanity-out").textContent = "scrambling the AI…";
    try {
      const r = await api("/image/sanity", { index: cur.index, method });
      const m = data.methods.find((x) => x.key === method);
      drawXray($("#xr-random"), data.image, r.random, m.signed);
      const verdict = r.spearman < 0.15 ? "passes: the map depends on what the AI learned" : r.spearman < 0.3 ? "mostly passes" : "partly fails: the map survives a scrambled AI";
      $("#xr-sanity-out").textContent = `similarity ${num(r.spearman)} · ${verdict}`;
    } catch (e) { $("#xr-sanity-out").textContent = e.message; }
  }
  $("#xr-sanity").onclick = () => { setCompare(!compare); if (compare) sanity(); };
  $("#xr-split").oninput = (e) => $("#xr-compare").style.setProperty("--split", e.target.value + "%");
  $("#xr-upload").onchange = (e) => {
    const f = e.target.files[0]; if (!f) return;
    const rd = new FileReader(); rd.onload = () => load({ image: rd.result }); rd.readAsDataURL(f);
  };
  addEventListener("resize", movePill);
  whenVisible($("#xray"), () => load({ index: samples[0].index }), "top 90%");
}

// ---------------------------------------------------------------- symptoms lab
async function symptoms() {
  const ta = $("#sym-text"), seg = $("#sym-methods"), hl = $("#sym-highlight");
  const METHODS = [["shap", "SHAP"], ["lime", "LIME"], ["occlusion", "Leave-one-out"], ["linear", "Exact contribution"]];
  let meta, data, method = "shap", ctrl;
  try { meta = await api("/text/meta"); } catch (e) { return errorBox(hl, e.message, symptoms); }
  $("#sym-examples").innerHTML = meta.examples.map((x, i) => `<button class="chip" data-i="${i}">${esc(x.label)}</button>`).join("");
  $$(".chip").forEach((c) => (c.onclick = () => { $$(".chip").forEach((x) => x.classList.toggle("is-on", x === c)); ta.value = meta.examples[+c.dataset.i].text; run(); }));
  seg.innerHTML = METHODS.map(([k, n]) => `<button role="tab" data-k="${k}" aria-selected="${k === method}">${n}</button>`).join("");
  const pill = document.createElement("span"); pill.className = "seg__pill"; seg.appendChild(pill);
  const movePill = () => { const on = $('[aria-selected="true"]', seg); pill.style.left = on.offsetLeft + "px"; pill.style.width = on.offsetWidth + "px"; };
  $$("button", seg).forEach((b) => (b.onclick = () => { method = b.dataset.k; $$("button", seg).forEach((x) => x.setAttribute("aria-selected", String(x === b))); movePill(); paint(); }));
  function paint() {
    if (!data) return;
    const sc = data.scores[method], m = Math.max(1e-9, ...Object.values(sc).map(Math.abs));
    hl.innerHTML = data.tokens.map((t) => {
      if (!t.w || !(t.w in sc) || Math.abs(sc[t.w]) < 1e-6) return esc(t.t);
      const v = sc[t.w] / m, c = attrRGB(v);
      return `<span class="w" title="${num(sc[t.w], 3)}" style="background:${rgb(c, Math.min(0.9, 0.25 + Math.abs(v) * 0.7))};color:${Math.abs(v) > 0.5 ? "#07090c" : "inherit"}">${esc(t.t)}</span>`;
    }).join("");
    if (!reduced) gsap.from($$(".w", hl), { backgroundColor: "rgba(0,0,0,0)", duration: 0.8, stagger: 0.02, ease: "power2.out" });
    $("#sym-method-name").textContent = `· ${METHODS.find((x) => x[0] === method)[1]}`;
    const words = Object.entries(sc).sort((a, b) => Math.abs(b[1]) - Math.abs(a[1])).slice(0, 8).map(([label, value]) => ({ label, value }));
    hbars($("#sym-words"), words, { signed: true, labelW: 130, format: (v) => num(v, 3) }).play();
    const con = Object.entries(data.contrastive).sort((a, b) => Math.abs(b[1]) - Math.abs(a[1])).slice(0, 8).map(([label, value]) => ({ label, value }));
    hbars($("#sym-contrast"), con, { signed: true, labelW: 130, format: (v) => num(v, 2) }).play();
  }
  async function run() {
    const text = ta.value.trim(); if (!text) return;
    ctrl?.abort(); ctrl = new AbortController();
    hl.style.opacity = 0.5;
    try { data = await api("/text/explain", { text }, ctrl.signal); } catch (e) { if (e.name !== "AbortError") errorBox(hl, e.message, run); return; }
    hl.style.opacity = 1;
    $("#sym-a").innerHTML = `<em>${esc(data.cls)}</em>`; $("#sym-b").innerHTML = `<em>${esc(data.contrast)}</em>`;
    hbars($("#sym-top"), data.top.map((t, i) => ({ label: t.cls, value: t.prob, tone: i === 0 ? "accent" : "muted" })), { labelW: 150, W: 360, format: (v) => pct(v, 0), max: 1 }).play();
    $("#sym-card").innerHTML = md(data.card);
    paint();
  }
  ta.addEventListener("input", debounce(run, 380));
  addEventListener("resize", movePill);
  whenVisible($("#symptoms"), () => { $$(".chip")[0].click(); requestAnimationFrame(movePill); }, "top 90%");
}

// ---------------------------------------------------------------- trust lab
async function trust() {
  let m;
  try { m = await api("/metrics"); } catch (e) { return errorBox($("#tr-tab"), e.message, trust); }
  const mk = (obj, better, best) => Object.entries(obj).map(([label, value]) => ({ label, value, tone: /random/i.test(label) ? "baseline" : label === best ? "accent" : "muted" })).sort((a, b) => (better === "low" ? a.value - b.value : b.value - a.value));
  const bestOf = (obj, better) => Object.entries(obj).filter(([k]) => !/random/i.test(k)).sort((a, b) => (better === "low" ? a[1] - b[1] : b[1] - a[1]))[0][0];
  const charts = [
    ["#tr-tab", m.tabular.faithfulness_top5_drop, "high"],
    ["#tr-img", m.image.deletion_auc, "low"],
    ["#tr-txt", m.text.comprehensiveness_top3, "high"],
    ["#tr-sanity", m.image.sanity_spearman, "low"],
  ];
  charts.forEach(([id, obj, better]) => {
    const c = hbars($(id), mk(obj, better, bestOf(obj, better)), { labelW: 190, format: (v) => num(v, 3) });
    whenVisible($(id), c.play);
  });
}

// ---------------------------------------------------------------- outro
function outro() {
  if (reduced) return;
  const split = SplitText.create("#outro-line", { type: "chars,words" });
  gsap.from(split.chars, { yPercent: 60, autoAlpha: 0, rotate: 6, duration: 1, ease: "expo.out", stagger: 0.018, scrollTrigger: { trigger: "#outro", start: "top 70%", once: true } });
}

// ---------------------------------------------------------------- boot
document.fonts.ready.then(async () => {
  shell();
  hero();
  problem();
  outro();
  for (const fn of [tumour, xray, symptoms, trust]) {
    try { await fn(); } catch (e) { console.error(fn.name, e); }
  }
  ScrollTrigger.refresh();
});
