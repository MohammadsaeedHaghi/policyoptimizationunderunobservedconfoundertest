"""Shared plumbing for the per-experiment pilot reports (IST, SUPPORT2).

House style matches the main report: pure-ASCII output, math-definition captions, and the
family color code (estimator = color, uncertainty set = line style + marker).
"""
import numpy as np

MC = {"Oracle": "#111111",
      "IPW-O-W": "#d62728", "IPW-O-X": "#d62728", "IPW-X-X": "#d62728",
      "DoublyRobust-O-W": "#1f77b4", "DoublyRobust-O-X": "#1f77b4", "DoublyRobust-X-X": "#1f77b4",
      "Hajek-O-X": "#ff7f0e", "Direct-X-X": "#17becf", "Kallus": "#7f7f7f"}
def mdash(m): return {"O-W": "", "O-X": "7 3", "X-X": "2 3"}.get(m[-3:], "")
def mmark(m): return {"O-W": "c", "O-X": "s", "X-X": "t"}.get(m[-3:], "c")
def ser(m, xs, ys, lab=None): return (lab or m, MC.get(m, "#7f7f7f"), xs, ys, mdash(m), mmark(m))

def esc(s): return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

CSS = """
:root{--bg:#f5f6fb;--surface:#ffffff;--fg:#0f172a;--muted:#667085;--border:#e9ebf3;
 --accent:#334155;--accent-soft:#f1f5f9;--good:#0a7d33;--warn:#b45309;--bad:#b91c1c;--sh:0 6px 18px rgba(16,24,40,.09);}
@media (prefers-color-scheme: dark){:root{--bg:#0e1420;--surface:#161d2b;--fg:#e6eaf2;--muted:#93a0b4;
 --border:#263043;--accent:#9fb2c9;--accent-soft:#1d2636;--good:#4ade80;--warn:#fbbf24;--bad:#f87171;--sh:0 6px 18px rgba(0,0,0,.35);}}
:root[data-theme="dark"]{--bg:#0e1420;--surface:#161d2b;--fg:#e6eaf2;--muted:#93a0b4;--border:#263043;
 --accent:#9fb2c9;--accent-soft:#1d2636;--good:#4ade80;--warn:#fbbf24;--bad:#f87171;--sh:0 6px 18px rgba(0,0,0,.35);}
:root[data-theme="light"]{--bg:#f5f6fb;--surface:#ffffff;--fg:#0f172a;--muted:#667085;--border:#e9ebf3;
 --accent:#334155;--accent-soft:#f1f5f9;--good:#0a7d33;--warn:#b45309;--bad:#b91c1c;--sh:0 6px 18px rgba(16,24,40,.09);}
*{box-sizing:border-box}
body{font-family:Inter,-apple-system,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;background:var(--bg);
 color:var(--fg);margin:0 auto;max-width:980px;padding:0 22px 80px;line-height:1.6;}
code,.mono{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;font-size:.85em;}
.hero{color:#fff;border-radius:20px;padding:26px 30px;margin:18px 0 20px;box-shadow:var(--sh);}
.hero h1{margin:0;font-weight:800;letter-spacing:-.03em;font-size:1.6rem;}
.hero p{margin:.5rem 0 0;color:#e3e8f0;max-width:840px;}
h2{font-size:1.28rem;font-weight:750;margin:34px 0 6px;} h3{font-size:1.0rem;font-weight:700;margin:18px 0 4px;}
p{margin:.5rem 0;} .muted{color:var(--muted);}
.card{background:var(--surface);border:1px solid var(--border);border-radius:16px;padding:14px 18px;margin:12px 0;box-shadow:var(--sh);}
.good{border-left:4px solid var(--good);} .warn{border-left:4px solid var(--warn);} .bad{border-left:4px solid var(--bad);}
table{border-collapse:collapse;width:100%;margin:10px 0;font-variant-numeric:tabular-nums;font-size:.88rem;}
.tw{overflow-x:auto;}
th{font-size:.74rem;letter-spacing:.04em;text-transform:uppercase;color:var(--muted);text-align:right;padding:6px 9px;border-bottom:2px solid var(--border);}
th:first-child,td:first-child{text-align:left;} td{padding:6px 9px;border-bottom:1px solid var(--border);text-align:right;}
td.g{color:var(--good);font-weight:700;} td.b{color:var(--bad);font-weight:700;}
.fig{margin:14px 0;background:var(--surface);border:1px solid var(--border);border-radius:14px;padding:10px 10px 4px;}
.figrow{display:grid;grid-template-columns:repeat(auto-fit,minmax(330px,1fr));gap:12px;}
svg.chart{width:100%;height:auto;display:block;}
.ct{font-size:12.5px;font-weight:700;fill:var(--fg);} .tk{font-size:10px;fill:var(--muted);}
.al{font-size:11px;fill:var(--muted);font-weight:600;} .hm{font-size:8.6px;font-weight:600;}
.grid{stroke:var(--border);stroke-width:1;} .ax{stroke:var(--muted);stroke-width:1.2;}
.figcap{font-size:.8rem;line-height:1.5;margin:4px 4px 14px;}
.leg{display:flex;flex-wrap:wrap;gap:4px 14px;padding:6px 8px 8px;}
.li{font-size:.78rem;color:var(--muted);display:inline-flex;align-items:center;gap:6px;font-weight:600;}
.sw{width:14px;height:4px;border-radius:2px;display:inline-block;}
"""

def _ticks(lo, hi, n=5):
    if hi <= lo: hi = lo + 1
    raw = (hi - lo) / n
    mag = 10 ** np.floor(np.log10(raw)); r = raw / mag
    step = (1 if r <= 1.5 else 2 if r <= 3 else 5 if r <= 7 else 10) * mag
    t0 = np.ceil(lo / step) * step
    return [round(v, 10) for v in np.arange(t0, hi + step / 2, step)]

def linechart(series, W=560, H=330, xlab="", ylab="", title="", hlines=None, legend=True, xticks=None):
    padL, padR, padT, padB = 52, 14, 30, 42
    xs_all = [x for it in series for x in it[2]]
    ys_all = [y for it in series for y in it[3] if y == y] + [h[2] for h in (hlines or [])]
    x0, x1 = min(xs_all), max(xs_all); ylo, yhi = min(ys_all), max(ys_all)
    ypad = 0.08 * (yhi - ylo + 1e-9); ylo -= ypad; yhi += ypad
    X = lambda v: padL + (v - x0) / (x1 - x0 + 1e-12) * (W - padL - padR)
    Y = lambda v: H - padB - (v - ylo) / (yhi - ylo + 1e-12) * (H - padT - padB)
    p = [f'<svg viewBox="0 0 {W} {H}" class="chart">', f'<text x="{padL}" y="16" class="ct">{esc(title)}</text>']
    for t in _ticks(ylo + ypad, yhi - ypad):
        if ylo <= t <= yhi:
            p.append(f'<line x1="{padL}" y1="{Y(t):.1f}" x2="{W-padR}" y2="{Y(t):.1f}" class="grid"/>')
            p.append(f'<text x="{padL-6}" y="{Y(t)+3.5:.1f}" class="tk" text-anchor="end">{t:g}</text>')
    for t in (xticks if xticks is not None else _ticks(x0, x1, 6)):
        if x0 - 1e-9 <= t <= x1 + 1e-9:
            p.append(f'<text x="{X(t):.1f}" y="{H-padB+16}" class="tk" text-anchor="middle">{t:g}</text>')
    for lab, col, yv, dash in (hlines or []):
        p.append(f'<line x1="{padL}" y1="{Y(yv):.1f}" x2="{W-padR}" y2="{Y(yv):.1f}" stroke="{col}" stroke-width="1.4" stroke-dasharray="{dash}"/>')
        p.append(f'<text x="{W-padR-2}" y="{Y(yv)-4:.1f}" class="tk" text-anchor="end" fill="{col}">{esc(lab)}</text>')
    for it in series:
        lab, col, xs, ys, dash = it[:5]; mk = it[5] if len(it) > 5 else "c"
        pts = " ".join(f"{X(a):.1f},{Y(b):.1f}" for a, b in zip(xs, ys) if b == b)
        d = f' stroke-dasharray="{dash}"' if dash else ""
        p.append(f'<polyline points="{pts}" fill="none" stroke="{col}" stroke-width="2"{d}/>')
        for a, b in zip(xs, ys):
            if b != b: continue
            cx, cy = X(a), Y(b)
            if mk == "s": p.append(f'<rect x="{cx-2.3:.1f}" y="{cy-2.3:.1f}" width="4.6" height="4.6" fill="{col}"/>')
            elif mk == "t": p.append(f'<polygon points="{cx:.1f},{cy-2.9:.1f} {cx-2.7:.1f},{cy+2.3:.1f} {cx+2.7:.1f},{cy+2.3:.1f}" fill="{col}"/>')
            else: p.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="2.3" fill="{col}"/>')
    p.append(f'<line x1="{padL}" y1="{H-padB}" x2="{W-padR}" y2="{H-padB}" class="ax"/>')
    p.append(f'<line x1="{padL}" y1="{padT}" x2="{padL}" y2="{H-padB}" class="ax"/>')
    p.append(f'<text x="{(padL+W-padR)/2:.0f}" y="{H-8}" class="al" text-anchor="middle">{esc(xlab)}</text>')
    p.append(f'<text x="14" y="{(padT+H-padB)/2:.0f}" class="al" text-anchor="middle" transform="rotate(-90 14 {(padT+H-padB)/2:.0f})">{esc(ylab)}</text>')
    p.append('</svg>')
    leg = ""
    if legend:
        items = "".join(f'<span class="li"><span class="sw" style="background:{c}"></span>{esc(l)}</span>' for l, c, *_ in series)
        leg = f'<div class="leg">{items}</div>'
    return f'<figure class="fig">{"".join(p)}{leg}</figure>'

def heatmap(rows, cols, Mv, title, rlab, clab, vmin, vmax, W=620, H=300):
    padL, padR, padT, padB = 64, 86, 30, 40
    cw = (W - padL - padR) / len(cols); ch = (H - padT - padB) / len(rows)
    def color(v):
        if v != v: return "#bbb"
        t = max(0.0, min(1.0, (v - vmin) / (vmax - vmin + 1e-12)))
        c0, c1, c2 = (68, 1, 84), (33, 145, 140), (253, 231, 37)
        a, b = (c0, c1) if t < 0.5 else (c1, c2); u = t * 2 if t < 0.5 else t * 2 - 1
        return "#%02x%02x%02x" % tuple(int(a[i] + (b[i] - a[i]) * u) for i in range(3))
    p = [f'<svg viewBox="0 0 {W} {H}" class="chart">', f'<text x="{padL}" y="16" class="ct">{esc(title)}</text>']
    for i, r in enumerate(rows):
        p.append(f'<text x="{padL-6}" y="{padT+ch*(i+0.5)+3.5:.1f}" class="tk" text-anchor="end">{esc(str(r))}</text>')
        for j, c in enumerate(cols):
            v = Mv[i][j]
            tcol = "#fff" if (v == v and (v - vmin) / (vmax - vmin + 1e-12) < 0.55) else "#111"
            p.append(f'<rect x="{padL+cw*j:.1f}" y="{padT+ch*i:.1f}" width="{cw:.1f}" height="{ch:.1f}" fill="{color(v)}" stroke="rgba(0,0,0,.12)" stroke-width="0.5"/>')
            p.append(f'<text x="{padL+cw*(j+0.5):.1f}" y="{padT+ch*(i+0.5)+3.2:.1f}" class="hm" text-anchor="middle" fill="{tcol}">{"--" if v != v else f"{v:.2f}"}</text>')
    for j, c in enumerate(cols):
        p.append(f'<text x="{padL+cw*(j+0.5):.1f}" y="{H-padB+14}" class="tk" text-anchor="middle">{esc(str(c))}</text>')
    p.append(f'<text x="{(padL+W-padR)/2:.0f}" y="{H-8}" class="al" text-anchor="middle">{esc(clab)}</text>')
    p.append(f'<text x="16" y="{(padT+H-padB)/2:.0f}" class="al" text-anchor="middle" transform="rotate(-90 16 {(padT+H-padB)/2:.0f})">{esc(rlab)}</text>')
    cbx, cbw = W - padR + 18, 14
    for k in range(60):
        t = 1 - k / 59; yy = padT + (H - padT - padB) * k / 60
        p.append(f'<rect x="{cbx}" y="{yy:.1f}" width="{cbw}" height="{(H-padT-padB)/60+0.6:.2f}" fill="{color(vmin+t*(vmax-vmin))}"/>')
    p.append(f'<text x="{cbx+cbw+4}" y="{padT+8}" class="tk">{vmax:g}</text>')
    p.append(f'<text x="{cbx+cbw+4}" y="{H-padB}" class="tk">{vmin:g}</text>')
    p.append('</svg>')
    return f'<figure class="fig">{"".join(p)}</figure>'

def surface_block(J, title_prefix, vmin=None):
    """Standard results block for a lip_gamma_2d JSON: two heatmaps + best table + L=3 slice."""
    Gk, Lk = J["gammas"], J["Lgrid"]; s = J["surface"]
    vmin = vmin if vmin is not None else min(J["never_treat"], min(min(r.values()) for r in s["IPW-O-W"].values()))
    out = ['<div class="figrow">']
    for m in ("IPW-O-W", "DoublyRobust-O-W"):
        Mv = [[s[m][g][l] for l in Lk] for g in Gk]
        out.append(heatmap(["G=" + g for g in Gk], Lk, Mv,
                           f"{title_prefix}: {m} test E[Y] (oracle {J['oracle']:.2f})",
                           "Gamma", "Lipschitz L", vmin, J["oracle"], W=560, H=280))
    out.append("</div>")
    sl = [ser(m, list(range(len(Lk))), [s[m][g]["3"] if "3" in s[m][g] else float("nan") for g in Gk])
          for m in J["methods"]]
    gl = [float(g) for g in Gk]
    sl = [ser(m, gl, [s[m][g]["3"] for g in Gk]) for m in J["methods"]]
    out.append(linechart(sl, title=f"{title_prefix}: E[Y] vs Gamma at L=3", xlab="Gamma", ylab="test E[Y]",
                         hlines=[("oracle", "#111", J["oracle"], "5 4"),
                                 ("naive DR", MC["DoublyRobust-X-X"], J["naive_dr"], "2 3"),
                                 ("never-treat", "#888", J["never_treat"], "2 3")], xticks=gl))
    bt = "".join(f"<tr><td>{m}</td><td>{J['best'][m]['value']:.3f}</td><td>{J['best'][m]['gamma']}</td>"
                 f"<td>{J['best'][m]['L']}</td></tr>" for m in J["methods"])
    out.append(f'<div class="tw"><table><tr><th>method</th><th>best E[Y]</th><th>&Gamma;</th><th>L</th></tr>{bt}</table></div>'
               f'<p class="muted">oracle {J["oracle"]:.3f} | naive DR {J["naive_dr"]:.3f} | '
               f'never-treat {J["never_treat"]:.3f} | all-treat {J["all_treat"]:.3f} | '
               f'{len(J["seeds"])} seeds, N={J["N_train"]}/{J["N_test"]}, deploy={J.get("deploy", "knn")}</p>')
    return "".join(out)

def page(title, hero_grad, hero_html, body_html):
    h = (f'<!doctype html><html lang="en"><head><meta charset="utf-8">'
         f'<meta name="viewport" content="width=device-width, initial-scale=1">'
         f'<title>{esc(title)}</title><style>{CSS}</style></head><body>'
         f'<div class="hero" style="background:{hero_grad}">{hero_html}</div>{body_html}</body></html>')
    return h.encode("ascii", "xmlcharrefreplace").decode("ascii")
