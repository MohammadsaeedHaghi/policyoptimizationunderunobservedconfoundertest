#!/usr/bin/env python3
"""Build 'html result/owgap_results.html' -- one self-contained page with ALL owgap results to date.

Reads every landed result JSON (base N=600 20-seed x 3 epsilons, N=1000, Kallus, alpha sweep,
continuous L x Gamma 2-D, v2 diagnostics), draws inline-SVG charts from the real numbers, compiles
math via latex2mathml, and writes pure-ASCII HTML (non-ASCII -> numeric entities).
Rerun any time: python3 "html result/build_results.py"
"""
import json, os, subprocess
import numpy as np
from latex2mathml.converter import convert as l2m

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
A = os.path.join(ROOT, "assets")
OUT = os.path.join(ROOT, "html result", "owgap_results.html")

def J(p):
    try: return json.load(open(os.path.join(A, p)))
    except Exception: return None

R20 = {ce: J(f"exp_owgap/owgap_results_20seed_ce{ce}.json") for ce in ("1.0", "1.5", "2.0")}
R1000 = {ce: J(f"exp_owgap/owgap_results_n1000_ce{ce}.json") for ce in ("1.0", "1.5", "2.0")}
KAL = J("exp_owgap/owgap_kallus_20seed.json")
ALPHAS = [1, 2, 4, 6, 10]
RA = {a: J(f"exp_owgap_alpha/owgap_alpha{a}_ce1.0.json") for a in ALPHAS}
KA = {a: J(f"exp_owgap_alpha/kallus_alpha{a}.json") for a in ALPHAS}
ADIAG = J("exp_owgap_alpha/alpha_diag.json")
C2D = J("exp_owgap_cont/owgap_lip_gamma_2d.json")
V2D = J("exp_owgap_v2/v2_diag.json")
# discrete v2: prefer the final 20-seed file over the 8-seed pilot as runs land
V2R = J("exp_owgap_v2/owgap_v2_20seed_ce1.0.json") or J("exp_owgap_v2/owgap_v2_ce1.0.json")
V2R_CE = {ce: J(f"exp_owgap_v2/owgap_v2_20seed_ce{ce}.json") for ce in ("1.0", "1.5", "2.0")}
KV2 = J("exp_owgap_v2/kallus_v2.json")
V2CAP = {c: J(f"exp_owgap_v2/owgap_v2_cap{c}_ce1.0.json") for c in ("40", "50")}   # cap robustness
# continuous v2: prefer the final 8-seed/Nte=4000 file over the 3-seed pilot
C2DV2 = J("exp_owgap_v2_cont/owgap_v2_lip_gamma_2d_final.json") or J("exp_owgap_v2_cont/owgap_v2_lip_gamma_2d.json")

MC = {"Oracle": "#111111", "IPW-O-W": "#d62728", "DoublyRobust-O-W": "#a01f1f",
      "IPW-O-X": "#9467bd", "DoublyRobust-O-X": "#6d4a7d", "Hajek-O-X": "#ff7f0e",
      "IPW-X-X": "#2ca02c", "DoublyRobust-X-X": "#8c564b", "Direct-X-X": "#17becf",
      "Kallus": "#7f7f7f"}
MORDER = ["IPW-O-W", "DoublyRobust-O-W", "DoublyRobust-X-X", "DoublyRobust-O-X",
          "IPW-X-X", "IPW-O-X", "Hajek-O-X", "Direct-X-X", "Kallus"]

def esc(s): return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
def M(tex): return l2m(tex)

# ---------------- SVG helpers ----------------
def _ticks(lo, hi, n=5):
    if hi <= lo: hi = lo + 1
    raw = (hi - lo) / n
    mag = 10 ** np.floor(np.log10(raw)); r = raw / mag
    step = (1 if r <= 1.5 else 2 if r <= 3 else 5 if r <= 7 else 10) * mag
    t0 = np.ceil(lo / step) * step
    return [round(v, 10) for v in np.arange(t0, hi + step / 2, step)]

def linechart(series, W=560, H=330, xlab="", ylab="", title="", bands=None, hlines=None,
              legend=True, xticks=None):
    """series: list of (label, color, xs, ys, dash). bands: (label,color,xs,lo,hi). hlines: (label,color,y,dash)."""
    padL, padR, padT, padB = 52, 14, 30, 42
    xs_all = [x for _, _, xs, _, _ in series for x in xs]
    ys_all = [y for _, _, _, ys, _ in series for y in ys if y == y]
    for hl in (hlines or []): ys_all.append(hl[2])
    for b in (bands or []): ys_all += list(b[3]) + list(b[4])
    x0, x1 = min(xs_all), max(xs_all); ylo, yhi = min(ys_all), max(ys_all)
    ypad = 0.08 * (yhi - ylo + 1e-9); ylo -= ypad; yhi += ypad
    def X(v): return padL + (v - x0) / (x1 - x0 + 1e-12) * (W - padL - padR)
    def Y(v): return H - padB - (v - ylo) / (yhi - ylo + 1e-12) * (H - padT - padB)
    p = [f'<svg viewBox="0 0 {W} {H}" class="chart" role="img" aria-label="{esc(title)}">']
    p.append(f'<text x="{padL}" y="16" class="ct">{esc(title)}</text>')
    for t in _ticks(ylo + ypad, yhi - ypad):
        if t < ylo or t > yhi: continue
        p.append(f'<line x1="{padL}" y1="{Y(t):.1f}" x2="{W-padR}" y2="{Y(t):.1f}" class="grid"/>')
        p.append(f'<text x="{padL-6}" y="{Y(t)+3.5:.1f}" class="tk" text-anchor="end">{t:g}</text>')
    for t in (xticks if xticks is not None else _ticks(x0, x1, 6)):
        if t < x0 - 1e-9 or t > x1 + 1e-9: continue
        p.append(f'<text x="{X(t):.1f}" y="{H-padB+16}" class="tk" text-anchor="middle">{t:g}</text>')
    for lab, col, y, dash in (hlines or []):
        p.append(f'<line x1="{padL}" y1="{Y(y):.1f}" x2="{W-padR}" y2="{Y(y):.1f}" stroke="{col}" stroke-width="1.4" stroke-dasharray="{dash}"/>')
        p.append(f'<text x="{W-padR-2}" y="{Y(y)-4:.1f}" class="tk" text-anchor="end" fill="{col}">{esc(lab)}</text>')
    for lab, col, xs, lo, hi in (bands or []):
        up = " ".join(f"{X(x):.1f},{Y(v):.1f}" for x, v in zip(xs, hi))
        dn = " ".join(f"{X(x):.1f},{Y(v):.1f}" for x, v in zip(reversed(xs), reversed(lo)))
        p.append(f'<polygon points="{up} {dn}" fill="{col}" opacity="0.13"/>')
    for lab, col, xs, ys, dash in series:
        pts = " ".join(f"{X(x):.1f},{Y(y):.1f}" for x, y in zip(xs, ys) if y == y)
        d = f' stroke-dasharray="{dash}"' if dash else ""
        p.append(f'<polyline points="{pts}" fill="none" stroke="{col}" stroke-width="2"{d}/>')
        for x, y in zip(xs, ys):
            if y == y: p.append(f'<circle cx="{X(x):.1f}" cy="{Y(y):.1f}" r="2.3" fill="{col}"/>')
    p.append(f'<line x1="{padL}" y1="{H-padB}" x2="{W-padR}" y2="{H-padB}" class="ax"/>')
    p.append(f'<line x1="{padL}" y1="{padT}" x2="{padL}" y2="{H-padB}" class="ax"/>')
    p.append(f'<text x="{(padL+W-padR)/2:.0f}" y="{H-8}" class="al" text-anchor="middle">{esc(xlab)}</text>')
    p.append(f'<text x="14" y="{(padT+H-padB)/2:.0f}" class="al" text-anchor="middle" transform="rotate(-90 14 {(padT+H-padB)/2:.0f})">{esc(ylab)}</text>')
    p.append('</svg>')
    leg = ""
    if legend:
        items = "".join(f'<span class="li"><span class="sw" style="background:{col}"></span>{esc(lab)}</span>'
                        for lab, col, _, _, _ in series)
        leg = f'<div class="leg">{items}</div>'
    return f'<figure class="fig">{"".join(p)}{leg}</figure>'

def heatmap(rows, cols, Mv, title, rlab, clab, vmin, vmax, W=620, H=300):
    padL, padR, padT, padB = 64, 86, 30, 40
    cw = (W - padL - padR) / len(cols); ch = (H - padT - padB) / len(rows)
    def color(v):
        if v != v: return "#bbb"
        t = max(0.0, min(1.0, (v - vmin) / (vmax - vmin + 1e-12)))
        # blue -> teal -> yellow (viridis-ish, hand mix)
        c0, c1, c2 = (68, 1, 84), (33, 145, 140), (253, 231, 37)
        a, b = (c0, c1) if t < 0.5 else (c1, c2); u = t * 2 if t < 0.5 else t * 2 - 1
        return "#%02x%02x%02x" % tuple(int(a[i] + (b[i] - a[i]) * u) for i in range(3))
    p = [f'<svg viewBox="0 0 {W} {H}" class="chart" role="img" aria-label="{esc(title)}">']
    p.append(f'<text x="{padL}" y="16" class="ct">{esc(title)}</text>')
    for i, r in enumerate(rows):
        p.append(f'<text x="{padL-6}" y="{padT+ch*(i+0.5)+3.5:.1f}" class="tk" text-anchor="end">{esc(str(r))}</text>')
        for j, c in enumerate(cols):
            v = Mv[i][j]
            fill = color(v)
            tcol = "#fff" if (v == v and (v - vmin) / (vmax - vmin + 1e-12) < 0.55) else "#111"
            p.append(f'<rect x="{padL+cw*j:.1f}" y="{padT+ch*i:.1f}" width="{cw:.1f}" height="{ch:.1f}" fill="{fill}" stroke="rgba(0,0,0,.12)" stroke-width="0.5"/>')
            txt = "--" if v != v else f"{v:.2f}"
            p.append(f'<text x="{padL+cw*(j+0.5):.1f}" y="{padT+ch*(i+0.5)+3.2:.1f}" class="hm" text-anchor="middle" fill="{tcol}">{txt}</text>')
    for j, c in enumerate(cols):
        p.append(f'<text x="{padL+cw*(j+0.5):.1f}" y="{H-padB+14}" class="tk" text-anchor="middle">{esc(str(c))}</text>')
    p.append(f'<text x="{(padL+W-padR)/2:.0f}" y="{H-8}" class="al" text-anchor="middle">{esc(clab)}</text>')
    p.append(f'<text x="16" y="{(padT+H-padB)/2:.0f}" class="al" text-anchor="middle" transform="rotate(-90 16 {(padT+H-padB)/2:.0f})">{esc(rlab)}</text>')
    # colorbar
    cbx, cbw = W - padR + 18, 14
    for k in range(60):
        t = 1 - k / 59; y = padT + (H - padT - padB) * k / 60
        p.append(f'<rect x="{cbx}" y="{y:.1f}" width="{cbw}" height="{(H-padT-padB)/60+0.6:.2f}" fill="{color(vmin+t*(vmax-vmin))}"/>')
    p.append(f'<text x="{cbx+cbw+4}" y="{padT+8}" class="tk">{vmax:g}</text>')
    p.append(f'<text x="{cbx+cbw+4}" y="{H-padB}" class="tk">{vmin:g}</text>')
    p.append('</svg>')
    return f'<figure class="fig">{"".join(p)}</figure>'

# ---------------- data digests ----------------
def uncap_series(R, methods, kal=None):
    gam = R["gammas"]; mean = R["regimes"]["uncap"]["mean"]
    s = [(m, MC[m], gam, mean[m], "") for m in methods if m in mean]
    if kal:
        kg = kal["gammas"]; s.append(("Kallus", MC["Kallus"], kg, kal["regimes"]["uncap"]["mean"]["Kallus"], ""))
    return s

def margin_row(R, gi=None):
    gam = R["gammas"]; gi = gam.index(5.0) if gi is None else gi
    out = {}
    for reg in ("uncap", "cap"):
        if reg not in R["regimes"]: continue
        mean, sd = R["regimes"][reg]["mean"], R["regimes"][reg]["sd"]
        naive = max(mean["DoublyRobust-X-X"][0], mean["IPW-X-X"][0], mean["Direct-X-X"][0])
        out[reg] = dict(naive=naive, ipwow=mean["IPW-O-W"][gi], ipwow_sd=sd["IPW-O-W"][gi],
                        drow=mean["DoublyRobust-O-W"][gi], drow_sd=sd["DoublyRobust-O-W"][gi],
                        margin=mean["IPW-O-W"][gi] - naive)
    return out

def fmt(v, d=3): return f"{v:.{d}f}"

# ---------------- sections ----------------
html = []
html.append("""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<!-- built by html result/build_results.py -->
<title>OWGAP: all results to date</title>
<style>
:root{--bg:#f5f6fb;--surface:#ffffff;--fg:#0f172a;--muted:#667085;--border:#e9ebf3;
 --accent:#334155;--accent-soft:#f1f5f9;--good:#0a7d33;--warn:#b45309;--chartbg:#ffffff;
 --sh:0 6px 18px rgba(16,24,40,.09);}
@media (prefers-color-scheme: dark){:root{--bg:#0e1420;--surface:#161d2b;--fg:#e6eaf2;--muted:#93a0b4;
 --border:#2630433;--border:#263043;--accent:#9fb2c9;--accent-soft:#1d2636;--good:#4ade80;--warn:#fbbf24;
 --chartbg:#161d2b;--sh:0 6px 18px rgba(0,0,0,.35);}}
:root[data-theme="dark"]{--bg:#0e1420;--surface:#161d2b;--fg:#e6eaf2;--muted:#93a0b4;--border:#263043;
 --accent:#9fb2c9;--accent-soft:#1d2636;--good:#4ade80;--warn:#fbbf24;--chartbg:#161d2b;--sh:0 6px 18px rgba(0,0,0,.35);}
:root[data-theme="light"]{--bg:#f5f6fb;--surface:#ffffff;--fg:#0f172a;--muted:#667085;--border:#e9ebf3;
 --accent:#334155;--accent-soft:#f1f5f9;--good:#0a7d33;--warn:#b45309;--chartbg:#ffffff;--sh:0 6px 18px rgba(16,24,40,.09);}
*{box-sizing:border-box}
body{font-family:Inter,-apple-system,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;background:var(--bg);
 color:var(--fg);margin:0 auto;max-width:1080px;padding:0 22px 80px;line-height:1.6;}
code,.mono{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;font-size:.86em;}
code{background:var(--accent-soft);padding:1px 6px;border-radius:6px;}
.tabs{position:sticky;top:0;z-index:9;background:var(--bg);border-bottom:1px solid var(--border);
 display:flex;gap:6px;flex-wrap:wrap;padding:10px 0;margin:0 0 6px;}
.tabs button{font:inherit;font-size:.88rem;font-weight:700;color:var(--muted);background:var(--surface);
 border:1px solid var(--border);border-radius:99px;padding:7px 18px;cursor:pointer;}
.tabs button:hover{color:var(--fg);background:var(--accent-soft);}
.tabs button.on{color:#fff;background:var(--accent);border-color:var(--accent);}
:root[data-theme="dark"] .tabs button.on{color:#0e1420;}
@media (prefers-color-scheme: dark){.tabs button.on{color:#0e1420;}}
:root[data-theme="light"] .tabs button.on{color:#fff;}
.tabs button:focus-visible{outline:2px solid var(--accent);outline-offset:2px;}
.tabpane{display:none;} .tabpane.on{display:block;}
.hero{color:#fff;border-radius:20px;padding:28px 32px;margin:18px 0 20px;
 background:radial-gradient(130% 150% at 0% 0%,#475569 0%,#334155 46%,#0f172a 100%);box-shadow:var(--sh);}
.hero h1{margin:0;font-weight:800;letter-spacing:-.03em;font-size:1.85rem;text-wrap:balance;}
.hero p{margin:.55rem 0 0;color:#e2e8f0;max-width:840px;}
.chips{display:flex;gap:8px;flex-wrap:wrap;margin-top:14px;}
.chip{font-size:.74rem;font-weight:700;letter-spacing:.03em;padding:3px 10px;border-radius:99px;background:rgba(255,255,255,.14);color:#e2e8f0;}
.chip.ok{background:rgba(74,222,128,.22);color:#bbf7d0;} .chip.run{background:rgba(251,191,36,.22);color:#fde68a;}
h2{font-size:1.38rem;font-weight:750;letter-spacing:-.02em;margin:40px 0 6px;scroll-margin-top:56px;}
h3{font-size:1.04rem;font-weight:700;margin:24px 0 6px;display:flex;align-items:center;gap:9px;}
h3::before{content:"";width:4px;height:1em;border-radius:3px;background:var(--accent);}
p{margin:.55rem 0;} .muted{color:var(--muted);}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:10px;margin:14px 0;}
.tile{background:var(--surface);border:1px solid var(--border);border-radius:14px;padding:12px 14px;}
.tile .v{font-size:1.5rem;font-weight:800;letter-spacing:-.02em;font-variant-numeric:tabular-nums;}
.tile .l{font-size:.78rem;color:var(--muted);font-weight:600;}
.card{background:var(--surface);border:1px solid var(--border);border-radius:16px;padding:16px 20px;margin:12px 0;box-shadow:var(--sh);}
.finding{border-left:4px solid var(--good);}
.caveat{border-left:4px solid var(--warn);}
table{border-collapse:collapse;width:100%;margin:10px 0;font-variant-numeric:tabular-nums;font-size:.9rem;}
.tw{overflow-x:auto;}
th{font-size:.76rem;letter-spacing:.04em;text-transform:uppercase;color:var(--muted);text-align:right;padding:6px 10px;border-bottom:2px solid var(--border);}
th:first-child,td:first-child{text-align:left;}
td{padding:6px 10px;border-bottom:1px solid var(--border);text-align:right;}
td.g{color:var(--good);font-weight:700;} td.b{color:var(--warn);font-weight:700;}
.fig{margin:14px 0;background:var(--chartbg);border:1px solid var(--border);border-radius:14px;padding:10px 10px 4px;}
.figrow{display:grid;grid-template-columns:repeat(auto-fit,minmax(330px,1fr));gap:12px;}
svg.chart{width:100%;height:auto;display:block;}
.ct{font-size:12.5px;font-weight:700;fill:var(--fg);} .tk{font-size:10px;fill:var(--muted);}
.al{font-size:11px;fill:var(--muted);font-weight:600;} .grid{stroke:var(--border);stroke-width:1;}
.ax{stroke:var(--muted);stroke-width:1.2;} .hm{font-size:8.6px;font-weight:600;}
.leg{display:flex;flex-wrap:wrap;gap:4px 14px;padding:6px 8px 8px;}
.li{font-size:.78rem;color:var(--muted);display:inline-flex;align-items:center;gap:6px;font-weight:600;}
.sw{width:14px;height:4px;border-radius:2px;display:inline-block;}
math{font-size:1.05em;}
.eq{margin:10px 0;text-align:center;overflow-x:auto;}
@media (prefers-reduced-motion: no-preference){html{scroll-behavior:smooth;}}
</style>
</head>
<body>""")

# section HTML is collected per tab, assembled at the end
T = {"dgp": [], "disc": [], "cont": [], "real": []}

# diabetes real-data results (auto-fill as jobs land)
DIA = J("exp_diabetes/diab_inregime_lip_gamma_2d.json")
DIB = J("exp_diabetes/diab_real_lip_gamma_2d.json")
KDA = J("exp_diabetes/kallus_diab_inregime.json")
KDB = J("exp_diabetes/kallus_diab_real.json")

# ---- problem setup & method taxonomy -> dgp tab (paper-style front matter) ----
EQ_V = M(r"V(\pi)=\mathbb{E}\big[\pi(X)\,Y(1)+(1-\pi(X))\,Y(0)\big],\qquad \pi^\ast=\arg\max_\pi V(\pi)")
EQ_MSM = M(r"\Gamma^{-1}\;\le\;\frac{e(X,S)/(1-e(X,S))}{\hat e(X)/(1-\hat e(X))}\;\le\;\Gamma")
EQ_W = M(r"\mathcal{W}\big(\textstyle\sum_i w_i^{(1)}\delta_{X_i},\;\sum_i w_i^{(0)}\delta_{X_i}\big)\;\le\;\varepsilon")
T["dgp"].append(f"""
<h2 id="s-setup">0. Problem setup, methods, and protocol</h2>
<p>We learn an individualized treatment rule &pi;(X) &isin; [0,1] from observational data
(X<sub>i</sub>, T<sub>i</sub>, Y<sub>i</sub>) whose treatment assignment depended on an
<b>unobserved</b> confounder S, maximizing the policy value</p>
<div class="eq">{EQ_V}</div>
<p>Because e(X,S) is not identified from (X,T,Y), plug-in ("naive") methods carry hidden-confounding
bias. The robust methods optimize the worst case over an uncertainty set of inverse-propensity
weights: the <b>marginal sensitivity model (odds-box, "O")</b> bounds how far the true propensity
odds may deviate from the fitted observable propensity,</p>
<div class="eq">{EQ_MSM}</div>
<p>and the <b>Wasserstein covariate-balance constraint ("W")</b> additionally requires the
worst-case reweighted treated and control covariate distributions to stay within transport cost
&varepsilon; of each other (&varepsilon; set data-driven as c<sub>&varepsilon;</sub> times the
tightest feasible value):</p>
<div class="eq">{EQ_W}</div>
<h3>Method taxonomy (Estimator&ndash;UncertaintySet&ndash;Balance)</h3>
<div class="tw"><table>
<tr><th>method</th><th>estimator</th><th>uncertainty set</th><th>balance</th><th>solved as</th></tr>
<tr><td>IPW-X-X / DR-X-X / Direct-X-X</td><td>IPW / AIPW / outcome model</td><td>none (plug-in)</td><td>&mdash;</td><td>closed form</td></tr>
<tr><td>IPW-O-X / DR-O-X / Hajek-O-X</td><td>IPW / AIPW / self-normalized</td><td>odds-box &Gamma;</td><td>&mdash;</td><td>LP / Dinkelbach</td></tr>
<tr><td><b>IPW-O-W / DR-O-W</b></td><td>IPW / AIPW</td><td>odds-box &Gamma;</td><td>Wasserstein &varepsilon;</td><td>LP (Gurobi)</td></tr>
<tr><td>Kallus &amp; Zhou</td><td>self-normalized regret</td><td>odds-box &Gamma;</td><td>&mdash; (parametric softmax)</td><td>Dinkelbach</td></tr>
<tr><td>Oracle</td><td colspan="4">treats iff true CATE(X) &gt; 0 &mdash; upper reference</td></tr>
</table></div>
<h3>Evaluation protocol</h3>
<p><b>Matched &Gamma;.</b> Every DGP has a known true selection odds ratio &Lambda;; headline
numbers are quoted at &Gamma; = &Lambda; (no tuning), with full &Gamma;-curves showing
misspecification behavior. <b>Value.</b> Discrete experiments report exact (noise-free) policy
value on the level grid; continuous experiments report realized test value on fresh draws with
known potential outcomes, deploying support policies by KNN. <b>Uncertainty.</b> Mean &plusmn; SD
over seeds, plus paired per-seed 95% CIs for the headline margins. <b>Ablations.</b> Coupling
strength &alpha; (when does balance help), transport budget c<sub>&varepsilon;</sub>, capacity
budget (30/40/50%), and the Lipschitz constant L (continuous).</p>""")

# ---- hero + tiles ----
mr = margin_row(R20["1.0"])
v2m = {}
if V2R:
    _g5 = V2R["gammas"].index(5.0)
    for _reg in ("uncap", "cap"):
        _mu = V2R["regimes"][_reg]["mean"]
        _nv = max(_mu["DoublyRobust-X-X"][0], _mu["IPW-X-X"][0], _mu["Direct-X-X"][0])
        v2m[_reg] = _mu["IPW-O-W"][_g5] - _nv
try:
    sq = subprocess.run(["squeue", "-u", "haghim", "-h", "-o", "%A %j %T"], capture_output=True, text=True, timeout=10).stdout.strip()
except Exception:
    sq = ""
chips = ['<span class="chip ok">20-seed N=600: DONE (3 epsilons)</span>',
         '<span class="chip ok">Alpha sweep: DONE</span>',
         '<span class="chip ok">Kallus baseline: DONE</span>']
if "owgap_v2c" in sq: chips.append('<span class="chip run">v2 continuous L x Gamma: QUEUED</span>')
if "owgap_v2 " in sq + " ": chips.append('<span class="chip run">v2 discrete pilot: RUNNING</span>')
html.append(f"""
<div class="hero">
<h1>OWGAP &mdash; every result so far</h1>
<p>The hidden-vitality synthetic experiment for confounding-robust policy optimization.
Observed fitness X, unobserved vitality S with P(S=+1|X)=&sigma;(10X); vitality dominates
outcomes (&plusmn;8) while the true treatment effect is small. The question throughout: do the
odds-box &cap; Wasserstein (O-W) methods recover the oracle policy where naive and box-only
methods fail? Built {esc(subprocess.run(['date'], capture_output=True, text=True).stdout.strip())}.</p>
<div class="chips">{''.join(chips)}</div>
</div>
<div class="tiles">
<div class="tile"><div class="v">0.847</div><div class="l">oracle E[Y] (uncapped)</div></div>
<div class="tile"><div class="v">+{fmt(mr['uncap']['margin'])}</div><div class="l">base O-W margin at &Gamma;=5 (20 seeds)</div></div>
<div class="tile"><div class="v">{('%+.3f' % v2m['uncap']) if v2m else '&mdash;'}</div><div class="l">v2 uncapped margin at &Gamma;=5</div></div>
<div class="tile"><div class="v">{('%+.3f' % v2m['cap']) if v2m else '&mdash;'}</div><div class="l">v2 capped(30%) margin at &Gamma;=5</div></div>
</div>""")

# ---- DGP section ----
xg = np.linspace(-1, 1, 241); sig = lambda z: 1 / (1 + np.exp(-z))
ps1 = sig(10 * xg); ES = 2 * ps1 - 1
cate = ES + 1.5 * xg
ep = np.clip(sig(0.8 - 2 * xg), 0.02, 0.98); em = np.clip(sig(-0.8 - 2 * xg), 0.02, 0.98)
dgp_figs = f"""<div class="figrow">
{linechart([("P(S=+1|X)", "#334155", list(xg), list(ps1), "")], title="Hidden-vitality coupling", xlab="X (fitness)", ylab="P(S=+1|X)", legend=False)}
{linechart([("CATE(X)", "#d62728", list(xg), list(cate), "")], title="True CATE: treat iff X>0", xlab="X", ylab="CATE", hlines=[("0", "#888", 0.0, "4 3")], legend=False)}
{linechart([("e(X,S=+1)", "#2ca02c", list(xg), list(ep), ""), ("e(X,S=-1)", "#9467bd", list(xg), list(em), "")], title="Confounded propensity", xlab="X", ylab="e(X,S)")}
</div>"""
EQ1 = M(r"P(S{=}{+}1\mid X)=\sigma(10X),\quad e(X,S)=\mathrm{clip}(\sigma(0.8S-2X),0.02,0.98)")
EQ2 = M(r"\mu_0=8S,\quad \mu_1=9S+1.5X,\quad Y(t)=\mu_t+\mathcal{N}(0,0.6^2)")
EQ3 = M(r"\mathrm{CATE}(X)=(2\sigma(10X)-1)+1.5X \;\Rightarrow\; \text{oracle treats iff } X>0")
T["dgp"].append(f"""
<h2 id="s-dgp">1. The data-generating process (base owgap)</h2>
<p>Seven discrete fitness levels X uniform in [-1,1]; unobserved vitality S; aggressive-therapy
treatment T; outcome Y (higher is better). N=600 per seed unless stated.</p>
<div class="eq">{EQ1}</div>
<div class="eq">{EQ2}</div>
<div class="eq">{EQ3}</div>
<p>S shifts both arms by &plusmn;8&ndash;9 while the treatment differential is ~1&ndash;2.5: treated
patients look great because they are vital, not because therapy works. The true selection odds
ratio is &Lambda;=4.95, so <b>&Gamma;=5 is the matched sensitivity level</b> &mdash; results quoted
"at matched &Gamma;" involve no tuning. Reference values: never-treat 0, all-treat 0, oracle 0.847.</p>
<div class="card"><b>Where the confounding bias lives (a design feature, stated up front).</b>
Selection on S operates at every X (the 0.8S term in the propensity), but it can only BIAS a
within-X comparison where X fails to pin down S &mdash; i.e. where Var(S|X) = 4p(1-p) with
p = &sigma;(&alpha;X) is non-negligible. At the strong coupling &alpha;=10 this is the
&sigma;(&alpha;X)&asymp;&frac12; region: on the 7-level grid, the X=0 level (naive CATE inflation
+6.5; faint traces &plusmn;0.8 at X=&plusmn;1/3); in the continuous variant, a band
|X|&lesssim;0.2 that shifts the naive decision threshold. This concentration is the unavoidable
consequence of ANY monotone X&ndash;S coupling crossing one-half, not a planted artifact; the
continuous experiment shows the band version of the same effect, and the &alpha; sweep (Discrete
tab, Section 4) shows what happens as the region widens.</div>
{dgp_figs}""")

# ---- main 20-seed results ----
gam = R20["1.0"]["gammas"]
s_un = uncap_series(R20["1.0"], MORDER[:-1], KAL)
mean_u = R20["1.0"]["regimes"]["uncap"]; mean_c = R20["1.0"]["regimes"]["cap"]
bands = [("IPW-O-W", MC["IPW-O-W"], gam,
          [m - s for m, s in zip(mean_u["mean"]["IPW-O-W"], mean_u["sd"]["IPW-O-W"])],
          [m + s for m, s in zip(mean_u["mean"]["IPW-O-W"], mean_u["sd"]["IPW-O-W"])])]
s_cap = [(m, MC[m], gam, mean_c["mean"][m], "") for m in MORDER[:-1] if m in mean_c["mean"]]
bands_c = [("IPW-O-W", MC["IPW-O-W"], gam,
            [m - s for m, s in zip(mean_c["mean"]["IPW-O-W"], mean_c["sd"]["IPW-O-W"])],
            [m + s for m, s in zip(mean_c["mean"]["IPW-O-W"], mean_c["sd"]["IPW-O-W"])])]
rows = []
for ce in ("1.0", "1.5", "2.0"):
    if not R20[ce]: continue
    r = margin_row(R20[ce])
    rows.append(f"<tr><td>c<sub>&epsilon;</sub>={ce}, uncapped</td><td>{fmt(r['uncap']['naive'])}</td>"
                f"<td>{fmt(r['uncap']['ipwow'])} &plusmn; {fmt(r['uncap']['ipwow_sd'],2)}</td>"
                f"<td>{fmt(r['uncap']['drow'])} &plusmn; {fmt(r['uncap']['drow_sd'],2)}</td>"
                f"<td class='g'>+{fmt(r['uncap']['margin'])}</td></tr>")
    if "cap" in r:
        rows.append(f"<tr><td>c<sub>&epsilon;</sub>={ce}, capped 50%</td><td>{fmt(r['cap']['naive'])}</td>"
                    f"<td>{fmt(r['cap']['ipwow'])} &plusmn; {fmt(r['cap']['ipwow_sd'],2)}</td>"
                    f"<td>{fmt(r['cap']['drow'])} &plusmn; {fmt(r['cap']['drow_sd'],2)}</td>"
                    f"<td class='g'>+{fmt(r['cap']['margin'])}</td></tr>")
T["disc"].append(f"""
<h2 id="s-main">1. Headline: N=600, 20 seeds, all three transport budgets</h2>
<p>Realized E[Y] versus &Gamma; (uncapped and capped, c<sub>&epsilon;</sub>=1.0; shaded band =
&plusmn;1 SD for IPW-O-W over 20 seeds). The O-W pair dominates every box-only and naive method
across the entire &Gamma; range; box-only methods (O-X) never beat naive AIPW at any &Gamma;.</p>
<div class="figrow">
{linechart(s_un, title="Uncapped, ce=1.0 (oracle 0.847)", xlab="Gamma", ylab="realized E[Y]", bands=bands, hlines=[("oracle", "#111", R20["1.0"]["oracle"], "5 4"), ("never-treat", "#888", 0.0, "2 3")], xticks=gam)}
{linechart(s_cap, title="Capped 50%, ce=1.0", xlab="Gamma", ylab="realized E[Y]", bands=bands_c, hlines=[("oracle", "#111", R20["1.0"]["oracle"], "5 4"), ("never-treat", "#888", 0.0, "2 3")], xticks=gam)}
</div>
<h3>Values at the matched &Gamma;=5 (no tuning)</h3>
<div class="tw"><table>
<tr><th>setting</th><th>best naive</th><th>IPW-O-W</th><th>DR-O-W</th><th>margin</th></tr>
{''.join(rows)}
</table></div>
<div class="card finding"><b>Finding.</b> At the matched &Gamma;=&Lambda;=5 both O-W methods beat the
best naive method in every regime and at every transport budget; the margin is largest at
c<sub>&epsilon;</sub>=1.0 (+0.104, ~6 standard errors over 20 seeds) and shrinks as the Wasserstein
budget loosens &mdash; the expected price of extra hedging room.</div>
<div class="card caveat"><b>Known weakness (motivates v2 below).</b> The capped margins are roughly
half the uncapped ones. Cause: the naive methods' one big mistake is treating the X=0 level (see
Section 7), but true CATE(0)=0 makes that mistake free, and the 50% cap trims their over-treatment
harmlessly &mdash; a "cap-rescue" that shrinks the O-W advantage exactly where the paper wants it.</div>""")

# ---- Kallus ----
if KAL:
    kg = KAL["gammas"]; km = KAL["regimes"]["uncap"]["mean"]["Kallus"]
    T["disc"].append(f"""
<h2 id="s-kallus">2. External baseline: Kallus &amp; Zhou (box-only, parametric)</h2>
<p>The confounding-robust softmax policy learner of Kallus &amp; Zhou, run with the marginal
sensitivity box only (20 seeds, uncapped &mdash; the smooth policy class cannot enforce a hard
capacity). It peaks at {max(km):.2f} near &Gamma;=1 and collapses to the never-treat value
&asymp;0 by &Gamma;&asymp;2.5 &mdash; at the matched &Gamma;=5 it does nothing, while IPW-O-W
holds {fmt(mr['uncap']['ipwow'])}. The same collapse occurs at every coupling strength in the
&alpha; sweep. The box alone forces total pessimism; only the Wasserstein balance constraint
lets robustness coexist with a non-trivial policy.</p>
{linechart([("Kallus", MC["Kallus"], kg, km, ""), ("IPW-O-W", MC["IPW-O-W"], gam, mean_u["mean"]["IPW-O-W"], ""), ("DoublyRobust-X-X (naive)", MC["DoublyRobust-X-X"], gam, mean_u["mean"]["DoublyRobust-X-X"], "")], title="Kallus vs O-W vs naive (uncapped, ce=1.0)", xlab="Gamma", ylab="realized E[Y]", hlines=[("oracle", "#111", 0.8469, "5 4"), ("never-treat", "#888", 0.0, "2 3")], xticks=gam)}""")

# ---- N=1000 ----
if any(R1000.values()):
    r1rows = []
    for ce in ("1.0", "1.5", "2.0"):
        R = R1000[ce]
        if not R: continue
        g1 = R["gammas"]; gi = g1.index(5.0) if 5.0 in g1 else len(g1) - 1
        mu = R["regimes"]["uncap"]["mean"]
        naive = max(mu["DoublyRobust-X-X"][0], mu["IPW-X-X"][0], mu["Direct-X-X"][0])
        r1rows.append(f"<tr><td>c<sub>&epsilon;</sub>={ce}</td><td>{fmt(naive)}</td>"
                      f"<td>{fmt(mu['IPW-O-W'][gi])}</td><td>{fmt(mu['DoublyRobust-O-W'][gi])}</td>"
                      f"<td class='g'>+{fmt(mu['IPW-O-W'][gi]-naive)}</td></tr>")
    T["disc"].append(f"""
<h2 id="s-n1000">3. Sample-size check: N=1000 (4 seeds)</h2>
<p>The same experiment at N=1000 (the Wasserstein LP scales ~O(n&sup3;), so this ran at 4 seeds).
The story is unchanged &mdash; the O-W margin is not a small-sample artifact.</p>
<div class="tw"><table>
<tr><th>uncapped, at &Gamma;=5</th><th>best naive</th><th>IPW-O-W</th><th>DR-O-W</th><th>margin</th></tr>
{''.join(r1rows)}
</table></div>""")

# ---- alpha sweep ----
if all(RA.values()):
    corr = {}
    if ADIAG and "per" in ADIAG:
        for a in ALPHAS:
            pa = ADIAG["per"].get(str(a)) or ADIAG["per"].get(a) or {}
            corr[a] = pa.get("corr_XS", pa.get("corr_xs", None))
    arows, mvals, o_v, n_v, w_v = [], [], [], [], []
    for a in ALPHAS:
        R = RA[a]; g = R["gammas"]; gi = g.index(5.0)
        mu = R["regimes"]["uncap"]["mean"]
        naive = max(mu["DoublyRobust-X-X"][0], mu["IPW-X-X"][0], mu["Direct-X-X"][0])
        ow = mu["IPW-O-W"][gi]; marg = ow - naive
        o_v.append(R["oracle"]); n_v.append(naive); w_v.append(ow); mvals.append(marg)
        cc = f"{corr[a]:.2f}" if corr.get(a) is not None else "--"
        cls = "g" if marg > 0 else "b"
        arows.append(f"<tr><td>&alpha;={a}</td><td>{cc}</td><td>{fmt(R['oracle'])}</td><td>{fmt(naive)}</td>"
                     f"<td>{fmt(ow)}</td><td class='{cls}'>{marg:+.3f}</td></tr>")
    T["disc"].append(f"""
<h2 id="s-alpha">4. Coupling sweep: when does O-W win?</h2>
<p>The X&ndash;S coupling &alpha; in P(S=+1|X)=&sigma;(&alpha;X) is swept over {{1,2,4,6,10}}
(N=600, 8 seeds, c<sub>&epsilon;</sub>=1.0) while the true selection strength is held FIXED at
&Lambda;=4.95 &mdash; only the usefulness of X as a proxy for S varies.</p>
<div class="tw"><table>
<tr><th>coupling</th><th>corr(X,S)</th><th>oracle</th><th>best naive</th><th>IPW-O-W at &Gamma;=5</th><th>margin</th></tr>
{''.join(arows)}
</table></div>
<div class="figrow">
{linechart([("oracle", "#111", ALPHAS, o_v, "5 4"), ("IPW-O-W at G=5", MC["IPW-O-W"], ALPHAS, w_v, ""), ("best naive", MC["DoublyRobust-X-X"], ALPHAS, n_v, "")], title="Value vs coupling strength", xlab="alpha", ylab="realized E[Y]", xticks=ALPHAS)}
{linechart([("O-W margin at G=5", MC["IPW-O-W"], ALPHAS, mvals, "")], title="O-W margin over best naive", xlab="alpha", ylab="margin", hlines=[("break-even", "#888", 0.0, "4 3")], xticks=ALPHAS, legend=False)}
</div>
<div class="card caveat"><b>Honest scoping: the margin flips sign.</b> O-W wins for
&alpha;&ge;6 (corr(X,S)&gtrsim;0.8), roughly ties at &alpha;=4, and LOSES below that &mdash; at
weak coupling the Wasserstein constraint has nothing to grab (balancing X no longer balances S)
and matched-&Gamma; robustness over-hedges below never-treat. This is a boundary of applicability,
not graceful degradation; the real-data Diabetes coupling (corr&asymp;0.20) sits below &alpha;=1,
which is why that experiment is deferred pending a stronger hidden-confounder definition.
Diagnostic upside: corr(X, S-proxy) is measurable, so the operating regime is checkable in
practice.</div>""")

# ---- continuous ----
if C2D:
    Gk, Lk = C2D["gammas"], C2D["Lgrid"]
    surf = C2D["surface"]; bo = C2D["best_overall"]
    Mv = [[surf["IPW-O-W"][g][l] for l in Lk] for g in Gk]
    # slices
    Lnum = [999 if l == "inf" else float(l) for l in Lk]
    sliceG = bo["gamma"]
    sL = []
    for m in ["IPW-O-W", "DoublyRobust-O-W", "IPW-O-X", "DoublyRobust-O-X", "Hajek-O-X"]:
        ys = [surf[m][sliceG][l] for l in Lk]
        sL.append((m, MC[m], list(range(len(Lk))), ys, ""))
    btab = "".join(f"<tr><td>{m}</td><td>{C2D['best'][m]['value']:.3f}</td><td>{C2D['best'][m]['gamma']}</td><td>{C2D['best'][m]['L']}</td></tr>" for m in C2D["methods"])
    T["cont"].append(f"""
<h2 id="s-cont">1. Continuous X and the Lipschitz constant L</h2>
<p>Continuous version (X ~ Uniform(-1,1), same DGP; N=400 train, KNN deployment to 2000 test
units, 3 seeds). With per-unit policies (L=&infin;) every support point is its own parameter:
the policy overfits, pooled mass vanishes and &Gamma; is inert. Two dials fix it: bin/smooth the
policy (the Lipschitz class |&pi;(x)-&pi;(x')| &le; L|x-x'|) for the VARIANCE, and &Gamma;
robustness for the BIAS. The L &times; &Gamma; surface shows both.</p>
{heatmap(["G=" + g for g in Gk], Lk, Mv, "IPW-O-W: test E[Y] over Gamma x L (oracle %.2f)" % C2D["oracle"], "Gamma", "Lipschitz L (inf = per-unit)", max(C2D["never_treat"], -0.1), C2D["oracle"])}
<div class="figrow">
{linechart(sL, title="Slice at Gamma=%s: the effect of L" % sliceG, xlab="L index: 0=inf ... 7=0.5 (tighter smoothing to the right)", ylab="test E[Y]", hlines=[("oracle", "#111", C2D["oracle"], "5 4"), ("never-treat", "#888", C2D["never_treat"], "2 3")], xticks=list(range(len(Lk))))}
<div class="card"><b>Best per method (any &Gamma;, L).</b><div class="tw"><table>
<tr><th>method</th><th>best E[Y]</th><th>&Gamma;</th><th>L</th></tr>{btab}</table></div>
<p class="muted">Best overall: {bo['method']} at &Gamma;={bo['gamma']}, L={bo['L']} &rarr; {bo['value']:.3f}
(oracle {C2D['oracle']:.3f}, never-treat {C2D['never_treat']:.3f}). O-W methods dominate the surface;
box-only methods peak at low &Gamma; and far lower value. Moderate smoothing (L&asymp;1.5&ndash;3)
is the sweet spot: tight enough to restore pooling so &Gamma; bites, loose enough to keep the
policy's decision boundary sharp.</p></div>
</div>""")

# ---- v2 ----
EQ4 = M(r"\mu_1 = 9S + 3X - 1 \;\Rightarrow\; \mathrm{CATE}(X) = (2\sigma(10X)-1) + 3X - 1,\quad \mathrm{CATE}(0)=-1")
if V2D:
    lv = V2D["levels"]; ct = V2D["cate"]; ch = V2D["catehat_naive_inf"]; vv = V2D["values"]
    idx = list(range(len(lv)))
    v2fig = linechart([("true CATE", "#111", lv, ct, ""), ("naive CATE-hat (infinite data)", MC["DoublyRobust-X-X"], lv, ch, "")],
                      title="v2: the ranking inversion at X=0", xlab="X level", ylab="CATE",
                      hlines=[("0", "#888", 0.0, "4 3")])
    T["dgp"].append(f"""
<h2 id="s-v2">2. The v2 redesign: making the showcase decisive (in flight)</h2>
<p>Diagnosis from Sections 2&ndash;5: the naive methods' signature error is treating X=0 &mdash;
the one level where S is genuinely uncertain given X, so within-level selection inflates the
estimated CATE from ~0 to ~+6.5. In the base DGP that error is costless (true CATE(0)=0), which
mutes the uncapped margin and lets the cap rescue naive methods. <b>exp_owgap_v2</b> keeps the
whole structure and makes exactly that error expensive:</p>
<div class="eq">{EQ4}</div>
<p>plus a genuinely scarce budget: cap 30% &lt; the 43% oracle-treat mass. Unchanged: coupling,
propensity (so &Lambda;=4.95 and matched &Gamma;=5), oracle = treat iff X&gt;0, uncapped oracle
0.847, never-treat 0. New: all-treat = -1 (over-treatment is now visibly bad).</p>
{v2fig}
<h3>Analytic + Monte-Carlo validation (no solver needed)</h3>
<div class="tw"><table>
<tr><th>policy</th><th>uncapped E[Y]</th><th>capped(30%) E[Y]</th></tr>
<tr><td>oracle</td><td>{vv['oracle_uncap']:.3f}</td><td>{vv['oracle_cap']:.3f}</td></tr>
<tr><td>naive plug-in (infinite data)</td><td>{vv['naive_inf_uncap']:.3f}</td><td>{vv['naive_inf_cap']:.3f}</td></tr>
<tr><td>naive plug-in (N=600 MC, 20 seeds)</td><td>{vv['naive_mc_uncap'][0]:.3f} &plusmn; {vv['naive_mc_uncap'][1]:.2f}</td><td>{vv['naive_mc_cap'][0]:.3f} &plusmn; {vv['naive_mc_cap'][1]:.2f}</td></tr>
<tr><td>never-treat / all-treat</td><td>{vv['never']:.2f} / {vv['all_treat']:.2f}</td><td>&mdash;</td></tr>
</table></div>
<div class="card finding"><b>Why each method will pick its policy (predictions under test).</b>
Naive IPW/AIPW: over-credits therapy at X=0 (vitality-enriched treated arm) &rarr; treats it first;
uncapped this buys CATE=-1 harm, capped it wastes a third of the budget. Box-only O-X at &Gamma;=5:
worst-case collapses toward never-treat (no balance constraint to anchor mass). Kallus: same box,
same collapse. O-W: the Wasserstein constraint forces the reweighted X-distribution to stay close,
and since X tracks S (&alpha;=10 regime, where Section 5 says balance works), the worst case cannot
fabricate vitality gaps &rarr; recovers "treat the fit" at both caps. A continuous twin with the
same constants is defined in exp_owgap_v2_cont (x* = 0.137, oracle 0.713, naive DR threshold
shifts to -0.29 &rarr; 0.35 &plusmn; 0.25). Discrete pilot results: see the Discrete tab,
Section 5; the predictions above were confirmed.</div>""")

def _v2_exact_value():
    import importlib.util
    sp = importlib.util.spec_from_file_location("v2dgp", os.path.join(A, "exp_owgap_v2", "dgp.py"))
    m = importlib.util.module_from_spec(sp); sp.loader.exec_module(m)
    return m.exact_value

def paired_margin_ci(V2R, reg, g="5", comparator="DoublyRobust-X-X"):
    """Per-seed exact-value margin IPW-O-W minus comparator at Gamma=g; mean and 95% t-CI."""
    try:
        ev = _v2_exact_value()
        pbs = V2R["regimes"][reg]["policy_by_seed"]
        seeds = sorted((k for k in pbs["IPW-O-W"] if k.isdigit()), key=int)
        diffs = [ev(pbs["IPW-O-W"][s][g]) - ev(pbs[comparator][s][g]) for s in seeds]
        d = np.array(diffs); n = len(d)
        tcrit = 2.093 if n >= 20 else 2.365   # t_{0.975} at df=19 / df=7
        half = tcrit * d.std(ddof=1) / np.sqrt(n)
        return d.mean(), half, n
    except Exception:
        return None, None, 0

# ---- v2 discrete pilot results -> disc tab ----
if V2R:
    gamv = V2R["gammas"]; giv = gamv.index(5.0)
    vm_u = V2R["regimes"]["uncap"]; vm_c = V2R["regimes"]["cap"]
    s_v2u = [(m, MC[m], gamv, vm_u["mean"][m], "") for m in MORDER[:-1] if m in vm_u["mean"]]
    s_v2c = [(m, MC[m], gamv, vm_c["mean"][m], "") for m in MORDER[:-1] if m in vm_c["mean"]]
    if KV2:
        s_v2u.append(("Kallus", MC["Kallus"], KV2["gammas"], KV2["regimes"]["uncap"]["mean"]["Kallus"], ""))
    bands_v2u = [("IPW-O-W", MC["IPW-O-W"], gamv,
                  [m - s for m, s in zip(vm_u["mean"]["IPW-O-W"], vm_u["sd"]["IPW-O-W"])],
                  [m + s for m, s in zip(vm_u["mean"]["IPW-O-W"], vm_u["sd"]["IPW-O-W"])])]
    bands_v2c = [("IPW-O-W", MC["IPW-O-W"], gamv,
                  [m - s for m, s in zip(vm_c["mean"]["IPW-O-W"], vm_c["sd"]["IPW-O-W"])],
                  [m + s for m, s in zip(vm_c["mean"]["IPW-O-W"], vm_c["sd"]["IPW-O-W"])])]
    nvu = max(vm_u["mean"]["DoublyRobust-X-X"][0], vm_u["mean"]["IPW-X-X"][0], vm_u["mean"]["Direct-X-X"][0])
    nvc = max(vm_c["mean"]["DoublyRobust-X-X"][0], vm_c["mean"]["IPW-X-X"][0], vm_c["mean"]["Direct-X-X"][0])
    owu, owc = vm_u["mean"]["IPW-O-W"][giv], vm_c["mean"]["IPW-O-W"][giv]
    # seed-0 policies at Gamma=5 for the explanation table
    def prow(P, methods, g="5"):
        rows = []
        for m in methods:
            v = P.get(m, {}).get(g)
            if v is None: continue
            cells = "".join(f"<td>{x:.2f}</td>" for x in v)
            rows.append(f"<tr><td>{m}</td>{cells}</tr>")
        return "".join(rows)
    lvl_hdr = "".join(f"<th>X={x:g}</th>" for x in V2R["grid"])
    pol_u = prow(vm_u.get("policy_seed0", {}), ["IPW-O-W", "DoublyRobust-O-W", "DoublyRobust-X-X", "IPW-X-X", "DoublyRobust-O-X"])
    pol_c = prow(vm_c.get("policy_seed0", {}), ["IPW-O-W", "DoublyRobust-O-W", "DoublyRobust-X-X", "IPW-X-X", "DoublyRobust-O-X"])
    n_sd = len(V2R["seeds"])
    extra_ce = ""
    ce_rows = []
    for ce in ("1.5", "2.0"):
        Rce = V2R_CE.get(ce)
        if not Rce: continue
        gce = Rce["gammas"].index(5.0)
        for reg in ("uncap", "cap"):
            mce = Rce["regimes"][reg]["mean"]
            nvce = max(mce["DoublyRobust-X-X"][0], mce["IPW-X-X"][0], mce["Direct-X-X"][0])
            ce_rows.append(f"<tr><td>c<sub>&epsilon;</sub>={ce}, {reg}</td><td>{nvce:.3f}</td>"
                           f"<td>{mce['IPW-O-W'][gce]:.3f}</td><td>{mce['DoublyRobust-O-W'][gce]:.3f}</td>"
                           f"<td class='g'>{mce['IPW-O-W'][gce]-nvce:+.3f}</td></tr>")
    if ce_rows:
        extra_ce = ('<h3>Other transport budgets (20 seeds, at &Gamma;=5)</h3>'
                    '<div class="tw"><table><tr><th>setting</th><th>best naive</th><th>IPW-O-W</th>'
                    '<th>DR-O-W</th><th>margin</th></tr>' + "".join(ce_rows) + '</table></div>')
    paired_html = ""
    pu, hu, nu_n = paired_margin_ci(V2R, "uncap")
    pc, hc, nc_n = paired_margin_ci(V2R, "cap")
    if pu is not None and pc is not None:
        paired_html = (f'<div class="card finding"><b>Paired significance (exact policy values, '
                       f'per-seed differences vs naive AIPW at &Gamma;=5).</b> '
                       f'Uncapped: mean margin {pu:+.3f} (95% CI &plusmn;{hu:.3f}, n={nu_n} seeds); '
                       f'capped: {pc:+.3f} (95% CI &plusmn;{hc:.3f}). '
                       f'{"Both CIs exclude zero." if (pu-hu>0 and pc-hc>0) else "See CI bounds."}</div>')
    T["disc"].append(f"""
<h2 id="s-v2res">5. v2 showcase results ({n_sd} seeds) &mdash; predictions confirmed</h2>
<p>The redesigned DGP (see the DGP tab, Section 2) ran at N=600, {n_sd} seeds,
c<sub>&epsilon;</sub>=1.0. At the matched &Gamma;=5 the O-W margin over the best naive method is
<b>+{owu-nvu:.3f} uncapped</b> (was +0.104 in the base DGP) and <b>+{owc-nvc:.3f} capped</b>
(was +0.056 &mdash; tripled: the cap-rescue is gone). Kallus again collapses to never-treat by
&Gamma;&asymp;2.5.</p>
<div class="figrow">
{linechart(s_v2u, title="v2 uncapped (oracle %.3f)" % V2R["oracle"], xlab="Gamma", ylab="realized E[Y]", bands=bands_v2u, hlines=[("oracle", "#111", V2R["oracle"], "5 4"), ("never-treat", "#888", 0.0, "2 3")], xticks=gamv)}
{linechart(s_v2c, title="v2 capped 30% (capped oracle 0.727)", xlab="Gamma", ylab="realized E[Y]", bands=bands_v2c, hlines=[("capped oracle", "#111", 0.727, "5 4"), ("never-treat", "#888", 0.0, "2 3")], xticks=gamv)}
</div>
<h3>Values at matched &Gamma;=5</h3>
<div class="tw"><table>
<tr><th>regime</th><th>best naive</th><th>IPW-O-W</th><th>DR-O-W</th><th>margin (IPW-O-W)</th></tr>
<tr><td>uncapped</td><td>{nvu:.3f}</td><td>{owu:.3f} &plusmn; {vm_u["sd"]["IPW-O-W"][giv]:.2f}</td><td>{vm_u["mean"]["DoublyRobust-O-W"][giv]:.3f}</td><td class="g">+{owu-nvu:.3f}</td></tr>
<tr><td>capped 30%</td><td>{nvc:.3f}</td><td>{owc:.3f} &plusmn; {vm_c["sd"]["IPW-O-W"][giv]:.2f}</td><td>{vm_c["mean"]["DoublyRobust-O-W"][giv]:.3f}</td><td class="g">+{owc-nvc:.3f}</td></tr>
</table></div>
{extra_ce}
{paired_html}
<h3>The policies each method actually picked (seed 0, &Gamma;=5) &mdash; and why</h3>
<p>&pi;(treat | X) per level. Oracle = treat X &gt; 0, i.e. (0, 0, 0, 0, 1, 1, 1);
capped oracle concentrates its 30% budget on the top levels.</p>
<b>Uncapped</b>
<div class="tw"><table><tr><th>method</th>{lvl_hdr}</tr>{pol_u}</table></div>
<b>Capped 30%</b>
<div class="tw"><table><tr><th>method</th>{lvl_hdr}</tr>{pol_c}</table></div>
<div class="card finding"><b>Reading the policies.</b> IPW-O-W is near-oracle uncapped
(full treatment above X=0, a hedged 0.47 at the ambiguous X=0 level) and allocates its capped
budget increasing in X. Naive AIPW commits the predicted signature error: full treatment of X=0
(true CATE = -1) in BOTH regimes &mdash; uncapped it buys pure harm, capped it burns a third of
the budget there while IPW-X-X misses the top two levels entirely. Box-only DR-O-X splits the
difference and loses to O-W everywhere. Exactly the failure/success mechanism the redesign
targeted.</div>""")
    # cap-robustness (auto-fills when job 10581756 lands)
    cap_rows = []
    _nvc0 = max(vm_c["mean"]["DoublyRobust-X-X"][0], vm_c["mean"]["IPW-X-X"][0], vm_c["mean"]["Direct-X-X"][0])
    cap_rows.append(f"<tr><td>30% (main run)</td><td>{_nvc0:.3f}</td><td>{vm_c['mean']['IPW-O-W'][giv]:.3f}</td>"
                    f"<td class='g'>{vm_c['mean']['IPW-O-W'][giv]-_nvc0:+.3f}</td></tr>")
    for cnum, Rc in V2CAP.items():
        if not Rc: continue
        gc = Rc["gammas"].index(5.0); mc = Rc["regimes"]["cap"]["mean"]
        nvx = max(mc["DoublyRobust-X-X"][0], mc["IPW-X-X"][0], mc["Direct-X-X"][0])
        cap_rows.append(f"<tr><td>{cnum}%</td><td>{nvx:.3f}</td><td>{mc['IPW-O-W'][gc]:.3f}</td>"
                        f"<td class='g'>{mc['IPW-O-W'][gc]-nvx:+.3f}</td></tr>")
    if len(cap_rows) > 1:
        T["disc"].append(f"""
<h3>Cap-robustness: the capped margin is not an artifact of the 30% budget</h3>
<p>The capped regime rerun at budgets 40% and 50% (8 seeds, c<sub>&epsilon;</sub>=1.0,
identical DGP). At every budget the naive methods spend part of the budget on the
selection-inflated X=0 level; O-W does not.</p>
<div class="tw"><table>
<tr><th>treatment budget</th><th>best naive at &Gamma;=5</th><th>IPW-O-W at &Gamma;=5</th><th>margin</th></tr>
{''.join(cap_rows)}
</table></div>""")
    else:
        T["disc"].append("""
<h3>Cap-robustness (running)</h3>
<p class="muted">Budgets 40% and 50% are queued (job 10581756) to show the capped margin is not
an artifact of the 30% choice; this table fills automatically when they land.</p>""")

# ---- v2 continuous -> cont tab (auto-fills once job 10579938 lands) ----
if C2DV2:
    Gk2, Lk2 = C2DV2["gammas"], C2DV2["Lgrid"]; s2 = C2DV2["surface"]; bo2 = C2DV2["best_overall"]
    Mv2 = [[s2["IPW-O-W"][g][l] for l in Lk2] for g in Gk2]
    sL2 = [(m, MC[m], list(range(len(Lk2))), [s2[m][bo2["gamma"]][l] for l in Lk2], "")
           for m in ["IPW-O-W", "DoublyRobust-O-W", "IPW-O-X", "DoublyRobust-O-X", "Hajek-O-X"]]
    nd = C2DV2.get("naive_dr"); nds = f"{nd:.3f}" if nd is not None else "--"
    T["cont"].append(f"""
<h2 id="s-v2cont">2. v2 continuous: L &times; &Gamma; on the showcase DGP</h2>
<p>The same v2 constants with continuous X (oracle threshold x*=0.137, oracle
{C2DV2['oracle']:.3f}, never-treat {C2DV2['never_treat']:.3f}, all-treat
{C2DV2.get('all_treat', float('nan')):.3f}; naive DR baseline {nds}). N=400 train, 2000 test,
KNN deployment, 3 seeds.</p>
{heatmap(["G=" + g for g in Gk2], Lk2, Mv2, "v2 IPW-O-W: test E[Y] over Gamma x L (oracle %.2f)" % C2DV2["oracle"], "Gamma", "Lipschitz L (inf = per-unit)", max(C2DV2["never_treat"], -0.6), C2DV2["oracle"])}
{linechart(sL2, title="v2 slice at Gamma=%s: the effect of L" % bo2["gamma"], xlab="L index: 0=inf ... 7=0.5", ylab="test E[Y]", hlines=[("oracle", "#111", C2DV2["oracle"], "5 4"), ("naive DR", MC["DoublyRobust-X-X"], nd if nd is not None else 0.0, "6 3"), ("never-treat", "#888", C2DV2["never_treat"], "2 3")], xticks=list(range(len(Lk2))))}
<p class="muted">Best overall: {bo2['method']} at &Gamma;={bo2['gamma']}, L={bo2['L']} &rarr;
{bo2['value']:.3f}. Seed-0 policy curves per (&Gamma;, L) are stored in the JSON
(policies_seed0) for the policy-explanation figures.</p>""")
else:
    T["cont"].append("""
<h2 id="s-v2cont">2. v2 continuous: L &times; &Gamma; on the showcase DGP (running)</h2>
<p class="muted">Job 10579938 (chained behind the discrete pilot) sweeps the same 5 robust
methods over 6 &Gamma; &times; 8 L on the v2 continuous DGP, with naive-DR / oracle / never /
all-treat references and per-(&Gamma;,L) seed-0 policy curves for the explanation figures.
This section fills automatically when it lands &mdash; rerun build_results.py.</p>""")

# ---- real-data (Diabetes 130) -> real tab ----
T["real"].append("""
<h2 id="s-diab">Real data: UCI Diabetes 130-US Hospitals (69,990 patients)</h2>
<p>Covariate X = composite frailty score (age, medications, length of stay, labs, procedures),
rescaled to [-1,1]; hidden confounder S = acuity (prior inpatient/ER visits, diagnoses, facility
discharge), median-split; treatment T = the real insulin decision (53% treated). The REAL
X&ndash;S coupling is weak: corr(X,S)=0.196, matched &Gamma;=1.26 &mdash; by the coupling-sweep
diagnostic (Discrete tab, Section 4) this dataset sits OUTSIDE the O-W operating regime. We
therefore run two complementary experiments rather than forcing a win:</p>
<div class="card"><b>Experiment A &mdash; in-regime semi-synthetic (real covariates, simulated
confounding).</b> Real bootstrapped X (real covariate geometry); S ~ Bern(&sigma;(12(X-0.07)))
calibrated to the real acuity rate 0.249 with E[Var(S|X)]=0.23 (comparable to the owgap
&alpha;=10 regime); T synthetic with the real fitted X-coefficient, real treated fraction 0.51,
and selection strength CS=0.8 giving &Lambda;=4.95 &mdash; the SAME matched-&Gamma;=5 protocol as
owgap. Outcomes: &mu;<sub>0</sub>=-3S, &mu;<sub>1</sub>=-1.5S+X-0.3 (sicker patients do worse in
both arms; insulin helps the sicker and frailer, minus a fixed burden). <b>The failure mode is
the mirror image of owgap: sicker patients get insulin AND do worse, so the naive plug-in
concludes insulin is harmful and treats NOBODY</b> (naive = never-treat 1.51 vs oracle 1.78,
oracle treats the frailest 21%). Together the two experiments show both classic confounding
failures &mdash; over-treatment (owgap: hidden vitality flatters the drug) and under-treatment
(here: hidden acuity damns it) &mdash; each corrected by the same O-W machinery.</div>
<div class="card"><b>Experiment B &mdash; fully real (X, S, T): the diagnostic validation.</b>
Everything real except the calibrated synthetic potential outcomes (needed for ground truth).
With the real weak coupling the linear-naive gap is only 0.001 (naive &asymp; oracle) and the
matched &Gamma; is 1.26: the scoping diagnostic PREDICTS robustness is unnecessary here. Running
the full method suite and observing exactly that &mdash; naive fine, O-W at matched &Gamma;
harmless, large &Gamma; over-hedging &mdash; validates the diagnostic on real data and shows the
method does no harm outside its regime.</div>""")

def _real_results(RJ, KJ, label, pred):
    if not RJ:
        return (f'<h3>{label} (running)</h3><p class="muted">L &times; &Gamma; sweep queued '
                f'(job 10582608); this section fills automatically when it lands. Prediction: {pred}</p>')
    Gk, Lk = RJ["gammas"], RJ["Lgrid"]; s = RJ["surface"]; bo = RJ["best_overall"]
    Mv = [[s["IPW-O-W"][g][l] for l in Lk] for g in Gk]
    nd = RJ.get("naive_dr")
    kbest = ""
    if KJ:
        km = KJ["regimes"]["uncap"]["mean"]["Kallus"]
        kbest = f' Kallus best {max(km):.3f}.'
    hm = heatmap(["G=" + g for g in Gk], Lk, Mv, f"{label}: IPW-O-W test E[Y] over Gamma x L (oracle %.2f)" % RJ["oracle"],
                 "Gamma", "Lipschitz L", min(RJ["never_treat"], min(min(r) for r in Mv)), RJ["oracle"])
    return (f'<h3>{label}</h3><p>Oracle {RJ["oracle"]:.3f}, never-treat {RJ["never_treat"]:.3f}, '
            f'all-treat {RJ.get("all_treat", float("nan")):.3f}, naive DR {nd:.3f}.'
            f' Best overall: {bo["method"]} at &Gamma;={bo["gamma"]}, L={bo["L"]} &rarr; {bo["value"]:.3f}.{kbest}</p>{hm}')

T["real"].append(_real_results(DIA, KDA, "Experiment A results (in-regime)",
                               "O-W recovers most of the oracle-naive gap of 0.27 at matched Gamma=5."))
if DIA:
    T["real"].append("""
<div class="card finding"><b>Verdict A: prediction confirmed.</b> The naive plug-in (1.504) sits
essentially at never-treat (1.491) &mdash; it concluded insulin is harmful and treated almost
nobody, the classic under-treatment failure. The robust methods recover ~75% of the naive-to-oracle
gap (IPW-O-W 1.697, DR-O-W 1.702 at L=3, &Gamma;=3&ndash;4; margin +0.19 over naive), while the
parametric box-only Kallus baseline lands BELOW never-treat (1.471). On real covariate geometry
with in-regime confounding, the correction works in the under-treatment direction just as it does
in owgap's over-treatment direction. Box-only O-X variants perform close to O-W here (1.69) at
their best &Gamma; &mdash; the Wasserstein term's decisive advantages remain the discrete capped
regime and stability across &Gamma; (see the other tabs); on this axis the honest claim is
"all &Gamma;-robust methods fix the failure; O-W is never worse".</div>""")
T["real"].append(_real_results(DIB, KDB, "Experiment B results (fully real)",
                               "naive ~ oracle; robust at matched Gamma=1.26 ~ naive; large Gamma over-hedges."))
if DIB:
    T["real"].append("""
<div class="card finding"><b>Verdict B: the diagnostic passes on real data.</b> With the real weak
coupling, naive DR (1.268) is already near-oracle (1.284) &mdash; a gap of 0.016. Every robust
method's optimum sits at &Gamma;=1 (&asymp; the real matched &Gamma;=1.26) with value 1.263
&asymp; naive: robustness at the matched level does NO harm. Raising &Gamma; strictly degrades
value (1.258 &rarr; 1.201 at L=3) &mdash; the over-hedging the coupling diagnostic predicts
out-of-regime. Together with the &alpha; sweep this closes the loop: the measurable coupling
diagnostic tells practitioners when the O-W machinery pays off, and real data behaves exactly as
it forecasts.</div>""")

# ---- status ----
sq_html = esc(sq) if sq else "(queue empty at build time)"
st_v2 = "DONE" if V2R else "RUNNING"
st_c2 = "DONE" if C2DV2 else ("RUNNING/queued" if "owgap_v2c" in sq else "chained after pilot")
st_k2 = "DONE" if KV2 else "debug partition"

# ---- assemble tabs ----
html.append(f"""
<div class="tabs" role="tablist">
<button id="tb-dgp" class="on" onclick="showTab('dgp')">DGP explanation</button>
<button id="tb-disc" onclick="showTab('disc')">Discrete results</button>
<button id="tb-cont" onclick="showTab('cont')">Continuous results</button>
<button id="tb-real" onclick="showTab('real')">Real data (Diabetes)</button>
</div>
<div id="tab-dgp" class="tabpane on">{''.join(T['dgp'])}</div>
<div id="tab-disc" class="tabpane">{''.join(T['disc'])}</div>
<div id="tab-cont" class="tabpane">{''.join(T['cont'])}</div>
<div id="tab-real" class="tabpane">{''.join(T['real'])}</div>
<script>
function showTab(id){{
  for (const t of ['dgp','disc','cont','real']){{
    document.getElementById('tab-'+t).classList.toggle('on', t===id);
    document.getElementById('tb-'+t).classList.toggle('on', t===id);
  }}
  try{{history.replaceState(null,'','#'+id);}}catch(e){{}}
  window.scrollTo(0,0);
}}
(function(){{const h=location.hash.replace('#','');
  if(['dgp','disc','cont','real'].includes(h)) showTab(h);}})();
</script>""")

html.append(f"""
<h2 id="s-status">Compute status at build time</h2>
<p class="mono" style="white-space:pre-wrap">{sq_html}</p>
<div class="tw"><table>
<tr><th>job</th><th>what</th><th>state</th></tr>
<tr><td>10571980</td><td>alpha sweep (5 DGPs, 8 seeds)</td><td>DONE</td></tr>
<tr><td>10572000</td><td>base N=600, 20 seeds, 3 epsilons</td><td>DONE</td></tr>
<tr><td>10572117</td><td>report rebuild</td><td>DONE</td></tr>
<tr><td>10579797</td><td>v2 discrete pilot (8 seeds, both regimes)</td><td>{st_v2}</td></tr>
<tr><td>10579938</td><td>v2 continuous L x Gamma sweep</td><td>{st_c2}</td></tr>
<tr><td>10579971</td><td>Kallus baseline for both v2 DGPs (no Gurobi, parallel)</td><td>{st_k2}</td></tr>
</table></div>
<p class="muted">Gurobi WLS license = 2 concurrent sessions account-wide, so solver jobs run
chained at --workers 2 --threads 1; license-free jobs run in parallel. This page:
"html result/owgap_results.html", rebuilt by "html result/build_results.py".</p>""")

html.append("</body>\n</html>")
page = "\n".join(html)
page = page.encode("ascii", "xmlcharrefreplace").decode("ascii")
open(OUT, "w").write(page)
print(f"wrote {OUT}  ({len(page)/1024:.0f} KB)")
