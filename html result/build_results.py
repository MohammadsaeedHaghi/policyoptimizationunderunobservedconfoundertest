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
# continuous v2: prefer the Shapley-deployment rerun, then the final KNN 8-seed file, then the pilot
C2DV2 = (J("exp_owgap_v2_cont/owgap_v2_lip_gamma_2d_shapley.json")
         or J("exp_owgap_v2_cont/owgap_v2_lip_gamma_2d_final.json")
         or J("exp_owgap_v2_cont/owgap_v2_lip_gamma_2d.json"))

def dep_lab(RJ, short=False):
    """Human name of the off-support deployment used by a 2-D result file."""
    if RJ and RJ.get("deploy") == "shapley":
        return "Shapley extension" if short else \
            "the Shapley extension (closed-form Lipschitz min-max interpolant, exact at the support points)"
    return "KNN" if short else "KNN averaging over the k=50 nearest support points"

# COLOR CODE (user rule): estimator family = COLOR; uncertainty set = LINE STYLE + MARKER.
#   IPW = red, DoublyRobust = blue, Hajek = orange, Direct = teal, Kallus = gray.
#   O-W = solid + circle, O-X = dashed + square, X-X = dotted + triangle, Kallus = dash-dot + diamond.
MC = {"Oracle": "#111111",
      "IPW-O-W": "#d62728", "IPW-O-X": "#d62728", "IPW-X-X": "#d62728",
      "DoublyRobust-O-W": "#1f77b4", "DoublyRobust-O-X": "#1f77b4", "DoublyRobust-X-X": "#1f77b4",
      "Hajek-O-X": "#ff7f0e", "Direct-X-X": "#17becf", "Kallus": "#7f7f7f"}
MDASH = {"O-W": "", "O-X": "7 3", "X-X": "2 3"}
MMARK = {"O-W": "c", "O-X": "s", "X-X": "t"}
def mdash(m):
    if m == "Kallus": return "10 3 2 3"
    return MDASH.get(m[-3:], "")
def mmark(m):
    if m == "Kallus": return "d"
    return MMARK.get(m[-3:], "c")
def ser(m, xs, ys, lab=None):
    """Series tuple with the family color code applied."""
    return (lab or m, MC.get(m, "#7f7f7f"), xs, ys, mdash(m), mmark(m))
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
    xs_all = [x for item in series for x in item[2]]
    ys_all = [y for item in series for y in item[3] if y == y]
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
    for item in series:
        lab, col, xs, ys, dash = item[:5]
        mk = item[5] if len(item) > 5 else "c"
        pts = " ".join(f"{X(x):.1f},{Y(y):.1f}" for x, y in zip(xs, ys) if y == y)
        d = f' stroke-dasharray="{dash}"' if dash else ""
        p.append(f'<polyline points="{pts}" fill="none" stroke="{col}" stroke-width="2"{d}/>')
        for x, y in zip(xs, ys):
            if y != y: continue
            cx, cy = X(x), Y(y)
            if mk == "s":
                p.append(f'<rect x="{cx-2.3:.1f}" y="{cy-2.3:.1f}" width="4.6" height="4.6" fill="{col}"/>')
            elif mk == "t":
                p.append(f'<polygon points="{cx:.1f},{cy-2.9:.1f} {cx-2.7:.1f},{cy+2.3:.1f} {cx+2.7:.1f},{cy+2.3:.1f}" fill="{col}"/>')
            elif mk == "d":
                p.append(f'<polygon points="{cx:.1f},{cy-3.1:.1f} {cx-3.1:.1f},{cy:.1f} {cx:.1f},{cy+3.1:.1f} {cx+3.1:.1f},{cy:.1f}" fill="{col}"/>')
            else:
                p.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="2.3" fill="{col}"/>')
    p.append(f'<line x1="{padL}" y1="{H-padB}" x2="{W-padR}" y2="{H-padB}" class="ax"/>')
    p.append(f'<line x1="{padL}" y1="{padT}" x2="{padL}" y2="{H-padB}" class="ax"/>')
    p.append(f'<text x="{(padL+W-padR)/2:.0f}" y="{H-8}" class="al" text-anchor="middle">{esc(xlab)}</text>')
    p.append(f'<text x="14" y="{(padT+H-padB)/2:.0f}" class="al" text-anchor="middle" transform="rotate(-90 14 {(padT+H-padB)/2:.0f})">{esc(ylab)}</text>')
    p.append('</svg>')
    leg = ""
    if legend:
        items = "".join(f'<span class="li"><span class="sw" style="background:{col}"></span>{esc(lab)}</span>'
                        for lab, col, *_ in series)
        leg = f'<div class="leg">{items}</div>'
    return f'<figure class="fig">{"".join(p)}{leg}</figure>'

def scatterchart(pts, lines, W=560, H=330, xlab="", ylab="", title="", legend=True):
    """pts: list of (label, color, xs, ys) scatter series; lines: list of (label, color, xs, ys, dash)."""
    padL, padR, padT, padB = 52, 14, 30, 42
    xs_all = [x for _, _, xs, _ in pts for x in xs] + [x for _, _, xs, _, _ in lines for x in xs]
    ys_all = [y for _, _, _, ys in pts for y in ys] + [y for _, _, _, ys, _ in lines for y in ys]
    x0, x1 = min(xs_all), max(xs_all); ylo, yhi = min(ys_all), max(ys_all)
    ypad = 0.06 * (yhi - ylo + 1e-9); ylo -= ypad; yhi += ypad
    def X(v): return padL + (v - x0) / (x1 - x0 + 1e-12) * (W - padL - padR)
    def Y(v): return H - padB - (v - ylo) / (yhi - ylo + 1e-12) * (H - padT - padB)
    p = [f'<svg viewBox="0 0 {W} {H}" class="chart" role="img" aria-label="{esc(title)}">']
    p.append(f'<text x="{padL}" y="16" class="ct">{esc(title)}</text>')
    for t in _ticks(ylo + ypad, yhi - ypad):
        if t < ylo or t > yhi: continue
        p.append(f'<line x1="{padL}" y1="{Y(t):.1f}" x2="{W-padR}" y2="{Y(t):.1f}" class="grid"/>')
        p.append(f'<text x="{padL-6}" y="{Y(t)+3.5:.1f}" class="tk" text-anchor="end">{t:g}</text>')
    for t in _ticks(x0, x1, 6):
        if t < x0 - 1e-9 or t > x1 + 1e-9: continue
        p.append(f'<text x="{X(t):.1f}" y="{H-padB+16}" class="tk" text-anchor="middle">{t:g}</text>')
    for lab, col, xs, ys in pts:
        for x, y in zip(xs, ys):
            p.append(f'<circle cx="{X(x):.1f}" cy="{Y(y):.1f}" r="1.7" fill="{col}" opacity="0.45"/>')
    for lab, col, xs, ys, dash in lines:
        ptsl = " ".join(f"{X(x):.1f},{Y(y):.1f}" for x, y in zip(xs, ys))
        d = f' stroke-dasharray="{dash}"' if dash else ""
        p.append(f'<polyline points="{ptsl}" fill="none" stroke="{col}" stroke-width="2"{d}/>')
    p.append(f'<line x1="{padL}" y1="{H-padB}" x2="{W-padR}" y2="{H-padB}" class="ax"/>')
    p.append(f'<line x1="{padL}" y1="{padT}" x2="{padL}" y2="{H-padB}" class="ax"/>')
    p.append(f'<text x="{(padL+W-padR)/2:.0f}" y="{H-8}" class="al" text-anchor="middle">{esc(xlab)}</text>')
    p.append(f'<text x="14" y="{(padT+H-padB)/2:.0f}" class="al" text-anchor="middle" transform="rotate(-90 14 {(padT+H-padB)/2:.0f})">{esc(ylab)}</text>')
    p.append('</svg>')
    leg = ""
    if legend:
        items = "".join(f'<span class="li"><span class="sw" style="background:{col}"></span>{esc(lab)}</span>'
                        for lab, col, *_ in list(pts) + list(lines))
        leg = f'<div class="leg">{items}</div>'
    return f'<figure class="fig">{"".join(p)}{leg}</figure>'

def barchart(cats, series, title, ylab, W=560, H=330, catlab="X level"):
    """Grouped bars over categorical x. series: list of (label, color, values)."""
    padL, padR, padT, padB = 52, 14, 30, 42
    ys_all = [v for _, _, vs in series for v in vs] + [0.0]
    ylo, yhi = min(ys_all), max(ys_all)
    ypad = 0.08 * (yhi - ylo + 1e-9); ylo -= ypad; yhi += ypad
    def Y(v): return H - padB - (v - ylo) / (yhi - ylo + 1e-12) * (H - padT - padB)
    ncat, nser = len(cats), len(series)
    slot = (W - padL - padR) / ncat; bw = slot * 0.8 / max(nser, 1)
    p = [f'<svg viewBox="0 0 {W} {H}" class="chart" role="img" aria-label="{esc(title)}">']
    p.append(f'<text x="{padL}" y="16" class="ct">{esc(title)}</text>')
    for t in _ticks(ylo + ypad, yhi - ypad):
        if t < ylo or t > yhi: continue
        p.append(f'<line x1="{padL}" y1="{Y(t):.1f}" x2="{W-padR}" y2="{Y(t):.1f}" class="grid"/>')
        p.append(f'<text x="{padL-6}" y="{Y(t)+3.5:.1f}" class="tk" text-anchor="end">{t:g}</text>')
    y0 = Y(0.0)
    for si, (lab, col, vs) in enumerate(series):
        for ci, v in enumerate(vs):
            x = padL + slot * ci + slot * 0.1 + bw * si
            yt = Y(max(v, 0.0)); hgt = abs(y0 - Y(v))
            p.append(f'<rect x="{x:.1f}" y="{yt:.1f}" width="{bw:.1f}" height="{max(hgt,0.5):.1f}" fill="{col}" opacity="0.9"/>')
    p.append(f'<line x1="{padL}" y1="{y0:.1f}" x2="{W-padR}" y2="{y0:.1f}" class="ax"/>')
    for ci, c in enumerate(cats):
        p.append(f'<text x="{padL+slot*(ci+0.5):.1f}" y="{H-padB+16}" class="tk" text-anchor="middle">{esc(str(c))}</text>')
    p.append(f'<line x1="{padL}" y1="{padT}" x2="{padL}" y2="{H-padB}" class="ax"/>')
    p.append(f'<text x="{(padL+W-padR)/2:.0f}" y="{H-8}" class="al" text-anchor="middle">{esc(catlab)}</text>')
    p.append(f'<text x="14" y="{(padT+H-padB)/2:.0f}" class="al" text-anchor="middle" transform="rotate(-90 14 {(padT+H-padB)/2:.0f})">{esc(ylab)}</text>')
    p.append('</svg>')
    items = "".join(f'<span class="li"><span class="sw" style="background:{col}"></span>{esc(lab)}</span>' for lab, col, _ in series)
    return f'<figure class="fig">{"".join(p)}<div class="leg">{items}</div></figure>'

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

def _vir(t):
    """viridis-ish 0..1 -> hex (same mix as heatmap's colormap)."""
    if t != t: return "#bbbbbb"
    t = max(0.0, min(1.0, t))
    c0, c1, c2 = (68, 1, 84), (33, 145, 140), (253, 231, 37)
    a, b = (c0, c1) if t < 0.5 else (c1, c2); u = t * 2 if t < 0.5 else t * 2 - 1
    return "#%02x%02x%02x" % tuple(int(a[i] + (b[i] - a[i]) * u) for i in range(3))

def stripemap(rowlabs, Mv, xvals, title, xlab="x", W=800, rowh=12, seps=(), xticks=(-1, -0.5, 0, 0.5, 1)):
    """Policy-stripe figure: one row per (label, policy curve), cell color = pi in [0,1].
    Adjacent equal-color cells are run-length merged (policies are mostly plateaus)."""
    n, m = len(rowlabs), len(xvals)
    padL, padR, padT, padB = 118, 56, 30, 40
    H = int(padT + padB + rowh * n)
    cw = (W - padL - padR) / m
    p = [f'<svg viewBox="0 0 {W} {H}" class="chart" role="img" aria-label="{esc(title)}">']
    p.append(f'<text x="{padL}" y="16" class="ct">{esc(title)}</text>')
    for i, (rl, row) in enumerate(zip(rowlabs, Mv)):
        y = padT + rowh * i
        p.append(f'<text x="{padL-5}" y="{y+rowh*0.5+3:.1f}" class="sk" text-anchor="end">{esc(str(rl))}</text>')
        j = 0
        while j < m:
            vj = round(float(row[j]), 1); k = j
            while k + 1 < m and round(float(row[k+1]), 1) == vj: k += 1
            p.append(f'<rect x="{padL+cw*j:.1f}" y="{y:.1f}" width="{cw*(k-j+1)+0.3:.1f}" height="{rowh+0.3:.1f}" fill="{_vir(vj)}"/>')
            j = k + 1
    for si in seps:
        y = padT + rowh * si
        p.append(f'<line x1="{padL}" y1="{y:.1f}" x2="{W-padR}" y2="{y:.1f}" stroke="#ffffff" stroke-width="2.4"/>')
    yb = padT + rowh * n
    x0, x1 = xvals[0], xvals[-1]
    for t in xticks:
        if t < x0 - 1e-9 or t > x1 + 1e-9: continue
        xx = padL + ((t - x0) / (x1 - x0 + 1e-12) * (m - 1) + 0.5) * cw
        p.append(f'<line x1="{xx:.1f}" y1="{yb}" x2="{xx:.1f}" y2="{yb+4}" class="ax"/>')
        p.append(f'<text x="{xx:.1f}" y="{yb+16}" class="tk" text-anchor="middle">{t:g}</text>')
    p.append(f'<text x="{(padL+W-padR)/2:.0f}" y="{H-8}" class="al" text-anchor="middle">{esc(xlab)}</text>')
    cbx, cbw, ch = W - padR + 16, 12, rowh * n
    for k in range(40):
        tt = 1 - k / 39; y = padT + ch * k / 40
        p.append(f'<rect x="{cbx}" y="{y:.1f}" width="{cbw}" height="{ch/40+0.6:.2f}" fill="{_vir(tt)}"/>')
    p.append(f'<text x="{cbx+cbw+3}" y="{padT+9}" class="tk">1</text>')
    p.append(f'<text x="{cbx+cbw+3}" y="{padT+ch}" class="tk">0</text>')
    p.append('</svg>')
    return f'<figure class="fig">{"".join(p)}</figure>'

def mean_policy_matrix(R, reg, m):
    """Mean pi(level) across seeds, one row per Gamma. Returns (gamma_keys, matrix)."""
    pbs = R["regimes"][reg]["policy_by_seed"][m]
    seeds = [k for k in pbs if k.isdigit()]
    gks = list(pbs[seeds[0]].keys())
    nl = len(pbs[seeds[0]][gks[0]])
    Mv = [[float(np.mean([pbs[s][gk][j] for s in seeds])) for j in range(nl)] for gk in gks]
    return gks, Mv

# ---------------- interactive policy viewer ----------------
PD = {}   # widget-id -> dataset, embedded as JSON for the in-page pi(X) plotter

def pol_widget_html(wid, d, defaults=None):
    """Controls + plot container for one policy dataset; drawing happens in pwDraw() (JS).
    Methods are checkbox chips: any subset can be overlaid on the same axes."""
    df = defaults or {}
    dm = df.get("m", [d["methods"][0]])
    if isinstance(dm, str): dm = [dm]
    def opts(vals, dv):
        o = []
        for v in vals:
            sel = ' selected' if str(v) == str(dv) else ''
            o.append(f'<option value="{v}"{sel}>{v}</option>')
        return "".join(o)
    chips = []
    for m in d["methods"]:
        chk = " checked" if m in dm else ""
        chips.append(f'<label class="mchip"><input type="checkbox" data-m="{m}"{chk}>'
                     f'<span class="sw" style="background:{MC.get(m, "#7f7f7f")}"></span>{m}</label>')
    c = [f'<div class="polw" id="pw-{wid}">']
    c.append(f'<div class="ctl"><span class="ctt">overlay methods:</span>'
             f'<span class="mck" id="pw-{wid}-m">{"".join(chips)}</span>'
             f'<button type="button" class="mbtn" data-sel="all">all</button>'
             f'<button type="button" class="mbtn" data-sel="none">none</button></div>')
    c.append('<div class="ctl">')
    c.append(f'<label>&Gamma; <select id="pw-{wid}-g">{opts(d["gammas"], df.get("g", d["gammas"][0]))}</select></label>')
    if d["kind"] == "2d":
        c.append(f'<label>L <select id="pw-{wid}-l">{opts(d["Ls"], df.get("l", d["Ls"][0]))}</select></label>')
    if d["kind"] == "disc":
        c.append(f'<label>regime <select id="pw-{wid}-r">{opts(d["regimes"], df.get("r", "uncap"))}</select></label>')
    c.append(f'</div><div id="pw-{wid}-plot" class="fig"></div></div>')
    return "".join(c)


SV_JS = """
function svSvg(xs, series, hls){
  const W=760,H=340,pL=56,pR=14,pT=24,pB=42;
  let ys=[]; for(const s of series) ys=ys.concat(s.ys.filter(v=>v===v));
  for(const h of hls) ys.push(h[1]);
  let ylo=Math.min(...ys), yhi=Math.max(...ys);
  const pad=0.08*(yhi-ylo+1e-9); ylo-=pad; yhi+=pad;
  const x0=xs[0], x1=xs[xs.length-1];
  const X=v=>pL+(v-x0)/(x1-x0+1e-12)*(W-pL-pR);
  const Y=v=>H-pB-(v-ylo)/(yhi-ylo+1e-12)*(H-pT-pB);
  let s='<svg viewBox="0 0 '+W+' '+H+'" class="chart">';
  for(let i=0;i<=5;i++){const t=ylo+pad+i*(yhi-ylo-2*pad)/5;
    s+='<line x1="'+pL+'" y1="'+Y(t).toFixed(1)+'" x2="'+(W-pR)+'" y2="'+Y(t).toFixed(1)+'" class="grid"/>'
      +'<text x="'+(pL-6)+'" y="'+(Y(t)+3.5).toFixed(1)+'" class="tk" text-anchor="end">'+t.toFixed(2)+'</text>';}
  for(const g of xs) s+='<text x="'+X(g).toFixed(1)+'" y="'+(H-pB+16)+'" class="tk" text-anchor="middle">'+g+'</text>';
  const hcol={'oracle':'var(--fg)','naive DR':MCJS['DoublyRobust-X-X'],'never-treat':'#888888'};
  const hdash={'oracle':'5 4','naive DR':'2 3','never-treat':'2 3'};
  for(const [lab,v] of hls){
    s+='<line x1="'+pL+'" y1="'+Y(v).toFixed(1)+'" x2="'+(W-pR)+'" y2="'+Y(v).toFixed(1)+'" style="stroke:'+hcol[lab]+'" stroke-width="1.4" stroke-dasharray="'+hdash[lab]+'"/>'
      +'<text x="'+(W-pR-2)+'" y="'+(Y(v)-4).toFixed(1)+'" class="tk" text-anchor="end" style="fill:'+hcol[lab]+'">'+lab+'</text>';}
  for(const se of series){
    const pts=xs.map((x,i)=>X(x).toFixed(1)+','+Y(se.ys[i]).toFixed(1)).join(' ');
    s+='<polyline points="'+pts+'" fill="none" style="stroke:'+se.col+'" stroke-width="2.2"'+(se.dash?' stroke-dasharray="'+se.dash+'"':'')+'/>';
    for(let i=0;i<xs.length;i++) s+=pwMark(X(xs[i]),Y(se.ys[i]),se.col,se.mk||'c',3.0);
  }
  s+='<line x1="'+pL+'" y1="'+(H-pB)+'" x2="'+(W-pR)+'" y2="'+(H-pB)+'" class="ax"/><line x1="'+pL+'" y1="'+pT+'" x2="'+pL+'" y2="'+(H-pB)+'" class="ax"/>';
  s+='<text x="'+((pL+W-pR)/2)+'" y="'+(H-8)+'" class="al" text-anchor="middle">Gamma</text>';
  s+='<text x="14" y="'+((pT+H-pB)/2)+'" class="al" text-anchor="middle" transform="rotate(-90 14 '+((pT+H-pB)/2)+')">test E[Y]</text>';
  return s+'</svg>';
}
function svDraw(){
  const d=SURFD; const e=document.getElementById('sv-cont-l'); if(!e||!d) return;
  const l=e.value; const xs=d.gammas.map(Number);
  const series=d.methods.map(m=>({lab:m+'  (L='+l+')', col:MCJS[m]||'#7f7f7f',
    ys:d.gammas.map(g=>d.surface[m][g][l]), dash:DASHJS[m]||'', mk:MARKJS[m]||'c'}));
  const leg='<div class="leg">'+series.map(se=>'<span class="li"><span class="sw" style="background:'+se.col+'"></span>'+se.lab+'</span>').join('')+'</div>';
  document.getElementById('sv-cont-plot').innerHTML=svSvg(xs,series,[['oracle',d.oracle],['naive DR',d.naive],['never-treat',d.never]])+leg;
}
document.addEventListener('change',e=>{ if(e.target && e.target.id==='sv-cont-l') svDraw(); });
svDraw();
"""

PW_JS = """
const MCJS = %(MC)s;
const DASHJS = %(DASH)s;
const MARKJS = %(MARK)s;
function pwMark(cx, cy, col, mk, r){
  if (mk==='s') return '<rect x="'+(cx-r).toFixed(1)+'" y="'+(cy-r).toFixed(1)+'" width="'+(2*r)+'" height="'+(2*r)+'" style="fill:'+col+'"/>';
  if (mk==='t') return '<polygon points="'+cx.toFixed(1)+','+(cy-1.3*r).toFixed(1)+' '+(cx-1.2*r).toFixed(1)+','+(cy+r).toFixed(1)+' '+(cx+1.2*r).toFixed(1)+','+(cy+r).toFixed(1)+'" style="fill:'+col+'"/>';
  if (mk==='d') return '<polygon points="'+cx.toFixed(1)+','+(cy-1.4*r).toFixed(1)+' '+(cx-1.4*r).toFixed(1)+','+cy.toFixed(1)+' '+cx.toFixed(1)+','+(cy+1.4*r).toFixed(1)+' '+(cx+1.4*r).toFixed(1)+','+cy.toFixed(1)+'" style="fill:'+col+'"/>';
  return '<circle cx="'+cx.toFixed(1)+'" cy="'+cy.toFixed(1)+'" r="'+r+'" style="fill:'+col+'"/>';
}
function pwSvg(xs, series){
  const W=760,H=340,pL=52,pR=14,pT=24,pB=42;
  const x0=xs[0],x1=xs[xs.length-1],ylo=-0.06,yhi=1.06;
  const X=v=>pL+(v-x0)/(x1-x0)*(W-pL-pR);
  const Y=v=>H-pB-(v-ylo)/(yhi-ylo)*(H-pT-pB);
  let s='<svg viewBox="0 0 '+W+' '+H+'" class="chart">';
  for (const t of [0,0.25,0.5,0.75,1]){
    s+='<line x1="'+pL+'" y1="'+Y(t).toFixed(1)+'" x2="'+(W-pR)+'" y2="'+Y(t).toFixed(1)+'" class="grid"/>';
    s+='<text x="'+(pL-6)+'" y="'+(Y(t)+3.5).toFixed(1)+'" class="tk" text-anchor="end">'+t+'</text>';
  }
  for (const t of [-1,-0.5,0,0.5,1]){
    if (t<x0-1e-9||t>x1+1e-9) continue;
    s+='<text x="'+X(t).toFixed(1)+'" y="'+(H-pB+16)+'" class="tk" text-anchor="middle">'+t+'</text>';
  }
  for (const se of series){
    const pts=xs.map((x,i)=>X(x).toFixed(1)+','+Y(se.ys[i]).toFixed(1)).join(' ');
    s+='<polyline points="'+pts+'" fill="none" style="stroke:'+se.col+'" stroke-width="2.2"'+
       (se.dash?' stroke-dasharray="'+se.dash+'"':'')+'/>';
    if (xs.length<=9){
      for (let i=0;i<xs.length;i++)
        s+=pwMark(X(xs[i]), Y(se.ys[i]), se.col, se.mk||'c', 3.2);
    }
  }
  s+='<line x1="'+pL+'" y1="'+(H-pB)+'" x2="'+(W-pR)+'" y2="'+(H-pB)+'" class="ax"/>';
  s+='<line x1="'+pL+'" y1="'+pT+'" x2="'+pL+'" y2="'+(H-pB)+'" class="ax"/>';
  s+='<text x="'+((pL+W-pR)/2)+'" y="'+(H-8)+'" class="al" text-anchor="middle">X</text>';
  s+='<text x="14" y="'+((pT+H-pB)/2)+'" class="al" text-anchor="middle" transform="rotate(-90 14 '+
     ((pT+H-pB)/2)+')">pi(X) = treatment probability</text>';
  return s+'</svg>';
}
function pwScat(xs, series, naive){
  const W=760,H=300,pL=52,pR=14,pT=24,pB=42;
  const x0=Math.min(...xs),x1=Math.max(...xs),ylo=-0.06,yhi=1.06;
  const X=v=>pL+(v-x0)/(x1-x0)*(W-pL-pR);
  const Y=v=>H-pB-(v-ylo)/(yhi-ylo)*(H-pT-pB);
  let s='<svg viewBox="0 0 '+W+' '+H+'" class="chart">';
  s+='<text x="'+pL+'" y="14" class="ct">Raw pointwise policy pi(X_i) at the '+xs.length+
     ' support points (seed 0) - solver output BEFORE extension</text>';
  for (const t of [0,0.5,1]){
    s+='<line x1="'+pL+'" y1="'+Y(t).toFixed(1)+'" x2="'+(W-pR)+'" y2="'+Y(t).toFixed(1)+'" class="grid"/>';
    s+='<text x="'+(pL-6)+'" y="'+(Y(t)+3.5).toFixed(1)+'" class="tk" text-anchor="end">'+t+'</text>';
  }
  for (const t of [-1,-0.5,0,0.5,1]){
    if (t<x0-1e-9||t>x1+1e-9) continue;
    s+='<text x="'+X(t).toFixed(1)+'" y="'+(H-pB+16)+'" class="tk" text-anchor="middle">'+t+'</text>';
  }
  if (naive)
    for (let i=0;i<xs.length;i++)
      s+='<g opacity="0.25">'+pwMark(X(xs[i]), Y(naive[i]/100), MCJS['DoublyRobust-X-X']||'#1f77b4', 't', 1.7)+'</g>';
  for (const se of series)
    for (let i=0;i<xs.length;i++)
      s+='<g opacity="0.8">'+pwMark(X(xs[i]), Y(se.ys[i]/100), se.col, se.mk||'c', 1.9)+'</g>';
  s+='<line x1="'+pL+'" y1="'+(H-pB)+'" x2="'+(W-pR)+'" y2="'+(H-pB)+'" class="ax"/>';
  s+='<line x1="'+pL+'" y1="'+pT+'" x2="'+pL+'" y2="'+(H-pB)+'" class="ax"/>';
  s+='<text x="'+((pL+W-pR)/2)+'" y="'+(H-8)+'" class="al" text-anchor="middle">X_i (training support)</text>';
  s+='<text x="14" y="'+((pT+H-pB)/2)+'" class="al" text-anchor="middle" transform="rotate(-90 14 '+
     ((pT+H-pB)/2)+')">pi(X_i)</text>';
  return s+'</svg>';
}
function pwDraw(wid){
  const d=PD[wid]; if(!d) return;
  const gv=(s)=>{const e=document.getElementById('pw-'+wid+'-'+s); return (e&&e.tagName==='SELECT')?e.value:null;};
  const g=gv('g'), l=gv('l'), reg=gv('r')||'uncap';
  const ms=Array.from(document.querySelectorAll('#pw-'+wid+'-m input:checked')).map(e=>e.dataset.m);
  let series=[], note='';
  if (d.kind==='2d'){
    series.push({lab:'oracle', col:'var(--fg)', ys:d.refs.oracle, dash:'6 4'});
    series.push({lab:'naive DR plug-in', col:MCJS['DoublyRobust-X-X']||'#1f77b4', ys:d.refs.naive, dash:'2 3'});
    for (const mm of ms)
      series.push({lab:mm+'  (G='+g+', L='+l+')', col:MCJS[mm]||'#7f7f7f', ys:d.pol[mm][g][l], dash:DASHJS[mm]||'', mk:MARKJS[mm]||'c'});
  } else {
    series.push({lab:reg==='uncap'?'oracle':'capped oracle', col:'var(--fg)',
                 ys:reg==='uncap'?d.refs.oracle_uncap:d.refs.oracle_cap, dash:'6 4'});
    for (const mm of ms){
      const c=d.pol[reg][mm]&&d.pol[reg][mm][g];
      if (c) series.push({lab:mm+'  (G='+g+', '+reg+')', col:MCJS[mm]||'#7f7f7f', ys:c, dash:DASHJS[mm]||'', mk:MARKJS[mm]||'c'});
      else note='<p class="muted" style="padding:0 8px 8px">'+mm+' has no '+reg+' variant (uncapped-only method).</p>';
    }
  }
  if (!ms.length) note='<p class="muted" style="padding:0 8px 8px">Select at least one method above to overlay it.</p>';
  let leg='<div class="leg">'+series.map(se=>'<span class="li"><span class="sw" style="background:'+
          se.col+'"></span>'+se.lab+'</span>').join('')+'</div>';
  let vals='';
  if (d.kind==='disc' && ms.length===1 && series.length>1){
    const ys=series[series.length-1].ys;
    vals='<div class="muted mono" style="padding:0 8px 8px">pi = ['+ys.map(v=>Math.round(v*100)/100).join(', ')+']</div>';
  }
  let head='';
  if (d.kind==='2d' && d.sup){
    const ss=[];
    for (const mm of ms)
      if (d.sup[mm] && d.sup[mm][g] && d.sup[mm][g][l]) ss.push({col:MCJS[mm]||'#7f7f7f', ys:d.sup[mm][g][l], mk:MARKJS[mm]||'c'});
    head=pwScat(d.supX, ss, d.supNaive||null);
  }
  document.getElementById('pw-'+wid+'-plot').innerHTML=head+pwSvg(d.grid,series)+leg+vals+note;
}
document.addEventListener('change',e=>{const w=e.target.closest('.polw'); if(w) pwDraw(w.id.slice(3));});
document.addEventListener('click',e=>{
  const b=e.target.closest('.mbtn'); if(!b) return;
  const w=b.closest('.polw');
  w.querySelectorAll('.mck input').forEach(i=>{i.checked=(b.dataset.sel==='all');});
  pwDraw(w.id.slice(3));
});
for (const k of Object.keys(PD)) pwDraw(k);
"""

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
.sk{font-size:8.4px;fill:var(--muted);}
.ctl{display:flex;gap:10px 18px;flex-wrap:wrap;align-items:center;margin:12px 0 4px;}
.ctl label{font-size:.84rem;font-weight:600;color:var(--muted);display:inline-flex;align-items:center;gap:7px;}
.ctl select{font:inherit;font-size:.86rem;padding:4px 10px;border-radius:9px;border:1px solid var(--border);
 background:var(--surface);color:var(--fg);}
.ctt{font-size:.84rem;font-weight:600;color:var(--muted);}
.mck{display:inline-flex;flex-wrap:wrap;gap:4px 8px;}
.mchip{display:inline-flex;align-items:center;gap:5px;font-size:.78rem;font-weight:600;color:var(--muted);
 border:1px solid var(--border);border-radius:99px;padding:2px 10px;cursor:pointer;background:var(--surface);}
.mchip input{accent-color:var(--accent);width:13px;height:13px;margin:0;cursor:pointer;}
.mchip:has(input:checked){color:var(--fg);border-color:var(--accent);background:var(--accent-soft);}
.mbtn{font:inherit;font-size:.75rem;font-weight:700;color:var(--muted);background:var(--surface);
 border:1px solid var(--border);border-radius:8px;padding:2px 10px;cursor:pointer;}
.mbtn:hover{color:var(--fg);background:var(--accent-soft);}
.figcap{font-size:.8rem;line-height:1.5;margin:4px 4px 14px;}
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
T = {"dgp": [], "disc": [], "cont": [], "real": [], "coup": []}

# diabetes real-data results (auto-fill as jobs land; prefer the Shapley-deployment reruns)
DIA = J("exp_diabetes/diab_inregime_lip_gamma_2d_shapley.json") or J("exp_diabetes/diab_inregime_lip_gamma_2d.json")
DIB = J("exp_diabetes/diab_real_lip_gamma_2d_shapley.json") or J("exp_diabetes/diab_real_lip_gamma_2d.json")
KDA = J("exp_diabetes/kallus_diab_inregime.json")
KDB = J("exp_diabetes/kallus_diab_real.json")

# ---- problem setup & method taxonomy -> dgp tab (paper-style front matter) ----
EQ_V = M(r"V(\pi)=\mathbb{E}\big[\pi(X)\,Y(1)+(1-\pi(X))\,Y(0)\big],\qquad \pi^\ast=\arg\max_\pi V(\pi)")
EQ_MSM = M(r"\Gamma^{-1}\;\le\;\frac{e(X,S)/(1-e(X,S))}{\hat e(X)/(1-\hat e(X))}\;\le\;\Gamma")
EQ_W = M(r"\mathcal{W}\big(\textstyle\sum_i w_i^{(1)}\delta_{X_i},\;\sum_i w_i^{(0)}\delta_{X_i}\big)\;\le\;\varepsilon")
T["disc"].append(f"""
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
known potential outcomes, deploying support policies off-support via {dep_lab(C2DV2)}.
<b>Uncertainty.</b> Mean &plusmn; SD
over seeds, plus paired per-seed 95% CIs for the headline margins. <b>Ablations.</b> Coupling
strength &alpha; (when does balance help), transport budget c<sub>&varepsilon;</sub>, capacity
budget (30/40/50%), and the Lipschitz constant L (continuous).</p>
<p><b>Propensity clipping (disclosure).</b> The DGP clips P(T=1|X,S) to [0.02, 0.98]; the clip
binds at extreme |X|, where the realized selection odds ratio falls BELOW
&Lambda; = e<sup>2&middot;0.8</sup> = 4.95. The matched &Gamma;=5 therefore upper-bounds the
realized confounding at every X &mdash; the protocol errs conservative, never favorable.</p>
<h3>Reproducibility settings (complete)</h3>
<div class="tw"><table>
<tr><th>item</th><th>discrete experiment</th><th>continuous experiment</th></tr>
<tr><td>sample</td><td>N=600 per seed; seeds 0&ndash;19</td><td>N=400 train, 4000 test (fresh draws, seed+1000); seeds 0&ndash;7</td></tr>
<tr><td>&Gamma; grid</td><td>{{1, 1.5, 2, 2.5, 3, 4, 5, 6, 8}}</td><td>{{1, 2, 3, 4, 6, 8}} (matched 4.95 bracketed by 4, 6)</td></tr>
<tr><td>policy class</td><td>per-unit &pi; &isin; [0,1], tied within level</td><td>per-unit &pi;; Lipschitz L &isin; {{&infin;, 10, 5, 3, 2, 1.5, 1, 0.5}} via consecutive-sorted-pair constraints</td></tr>
<tr><td>capacity</td><td>uncapped + capped 30% (ablation: 40%, 50%)</td><td>uncapped</td></tr>
<tr><td>Wasserstein budget</td><td colspan="2">per-arm &epsilon; = tightest feasible &times; c<sub>&epsilon;</sub>, c<sub>&epsilon;</sub> &isin; {{1.0, 1.5, 2.0}} (all reported)</td></tr>
<tr><td>outcome model</td><td colspan="2">&mu;&#770;<sub>t</sub> cross-fitted (2 folds) for DR / Direct methods</td></tr>
<tr><td>evaluation</td><td>exact (noise-free) policy value on the level grid</td><td>realized mean of &pi;Y(1)+(1-&pi;)Y(0) on test draws; off-support deployment via the closed-form Shapley operator (exact at support)</td></tr>
<tr><td>solver</td><td colspan="2">Gurobi (WLS) dual LPs; 2 workers &times; 1 thread; ~2.1 h per continuous 6&times;8&times;5-method sweep (8 seeds), discrete 20-seed &times; 3-budget suite overnight on the same license</td></tr>
<tr><td>baseline (Kallus &amp; Zhou)</td><td colspan="2">parametric softmax, odds-box only, Dinkelbach inner solve (no LP); 20 seeds</td></tr>
</table></div>""")

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
chips = ['<span class="chip ok">Discrete: 20 seeds x 3 budgets DONE</span>',
         '<span class="chip ok">Continuous L x Gamma (Shapley): DONE</span>',
         '<span class="chip ok">Coupling sweep: DONE</span>',
         '<span class="chip ok">Diabetes A/B: DONE</span>']
if "ceps" in sq: chips.append('<span class="chip run">continuous c-eps 1.5/2.0: RUNNING</span>')
cbest = C2DV2["best_overall"]["value"] if C2DV2 else float("nan")
html.append(f"""
<div class="hero">
<h1>OWGAP &mdash; the showcase experiments</h1>
<p>Hidden-vitality synthetic experiments for confounding-robust policy optimization. Observed
fitness X, unobserved vitality S with P(S=+1|X)=&sigma;(10X); vitality dominates outcomes
(&plusmn;8&ndash;9) while the true treatment effect is a few units and therapy carries a fixed
burden. One discrete and one continuous experiment, plus real-data validation. The question
throughout: do the odds-box &cap; Wasserstein (O-W) methods recover the oracle policy where
naive and box-only methods fail? Built {esc(subprocess.run(['date'], capture_output=True, text=True).stdout.strip())}.</p>
<div class="chips">{''.join(chips)}</div>
</div>
<div class="tiles">
<div class="tile"><div class="v">0.847</div><div class="l">discrete oracle E[Y] (uncapped; capped 0.727)</div></div>
<div class="tile"><div class="v">{('%+.3f' % v2m['uncap']) if v2m else '&mdash;'}</div><div class="l">discrete uncapped O-W margin at matched &Gamma;=5</div></div>
<div class="tile"><div class="v">{('%+.3f' % v2m['cap']) if v2m else '&mdash;'}</div><div class="l">discrete capped(30%) O-W margin at &Gamma;=5</div></div>
<div class="tile"><div class="v">{cbest:.3f}</div><div class="l">continuous best O-W (oracle 0.737, naive 0.390)</div></div>
</div>""")

# ---- DGP section (v2 = THE DGP; the burden-free base variant appears only in Section 2's rationale) ----
xg = np.linspace(-1, 1, 241); sig = lambda z: 1 / (1 + np.exp(-z))
ps1 = sig(10 * xg); ES = 2 * ps1 - 1
cate = ES + 3.0 * xg - 1.0
ep = np.clip(sig(0.8 - 2 * xg), 0.02, 0.98); em = np.clip(sig(-0.8 - 2 * xg), 0.02, 0.98)
def figcap(fig, cap):
    return f'<div>{fig}<p class="muted figcap">{cap}</p></div>'

dgp_figs = '<div class="figrow">' + figcap(
    linechart([("P(S=+1|X)", "#334155", list(xg), list(ps1), "")], title="Hidden-vitality coupling", xlab="X (fitness)", ylab="P(S=+1|X)", legend=False),
    "P(S=+1 | X) = &sigma;(10X). Var(S | X) = 4&sigma;(10X)(1-&sigma;(10X)), maximal at X=0.") + figcap(
    linechart([("CATE(X)", "#d62728", list(xg), list(cate), "")], title="True CATE: treat iff X>0 (CATE(0)=-1)", xlab="X", ylab="CATE", hlines=[("0", "#888", 0.0, "4 3")], legend=False),
    "CATE(X) = E[Y(1) - Y(0) | X] = (2&sigma;(10X)-1) + 3X - 1; "
    "oracle &pi;*(X) = 1{CATE(X) &gt; 0}.") + figcap(
    linechart([("P(T=1 | X, S=+1)", "#2ca02c", list(xg), list(ep), ""), ("P(T=1 | X, S=-1)", "#9467bd", list(xg), list(em), "")], title="Confounded propensity", xlab="X", ylab="P(T=1 | X,S)"),
    "P(T=1 | X,S) = clip(&sigma;(0.8S - 2X), 0.02, 0.98); &Lambda; = exp(2&middot;0.8) = 4.95.") + '</div>'

# mu_t(X,S) lines and one realized draw of the potential outcomes Y(t)
mu1p, mu1m = 3 * xg + 8, 3 * xg - 10          # mu1 = 9S + 3X - 1 at S = +1 / -1
mu0p, mu0m = np.full_like(xg, 8.0), np.full_like(xg, -8.0)
eY0 = 8 * ES; eY1 = 9 * ES + 3 * xg - 1       # E[Y(t)|X]: mu_t averaged over S|X
import importlib.util as _ilu
_sp = _ilu.spec_from_file_location("v2dgp_plot", os.path.join(A, "exp_owgap_v2", "dgp.py"))
_dm = _ilu.module_from_spec(_sp); _sp.loader.exec_module(_dm)
_obs, _full = _dm.generate(600, 0)
_rj = np.random.default_rng(1).uniform(-0.06, 0.06, 600)
_Xj = _obs["X"].ravel() + _rj
mu_y_figs = '<div class="figrow">' + figcap(
    linechart([("mu1(X, S=+1)", "#d62728", list(xg), list(mu1p), ""),
               ("mu0(X, S=+1)", "#1f77b4", list(xg), list(mu0p), ""),
               ("mu1(X, S=-1)", "#d62728", list(xg), list(mu1m), "5 4"),
               ("mu0(X, S=-1)", "#1f77b4", list(xg), list(mu0m), "5 4")],
              title="Mean potential outcomes mu_t(X,S)", xlab="X", ylab="mu_t(X,S)"),
    "&mu;<sub>0</sub>(X,S) = 8S, &nbsp;&mu;<sub>1</sub>(X,S) = 9S + 3X - 1; "
    "&mu;<sub>1</sub> - &mu;<sub>0</sub> = S + 3X - 1.") + figcap(
    scatterchart([("Y(1) draws", "#d62728", list(_Xj), list(_full["Y1"])),
                  ("Y(0) draws", "#1f77b4", list(_Xj), list(_full["Y0"]))],
                 [("E[Y(1)|X]", "#a01f1f", list(xg), list(eY1), "6 4"),
                  ("E[Y(0)|X]", "#144d73", list(xg), list(eY0), "6 4")],
                 title="Potential outcomes Y(t), one draw (N=600, seed 0)", xlab="X (jittered)", ylab="Y(t)"),
    "Y(t) = &mu;<sub>t</sub>(X,S) + N(0, 0.6&sup2;); dashed: "
    "E[Y(t)|X] = E<sub>S|X</sub>[&mu;<sub>t</sub>(X,S)], with "
    "E[Y(1)|X] - E[Y(0)|X] = CATE(X).") + '</div>'

# what the analyst sees (no S) + the selection composition behind it
_Yo, _To = _obs["Y"], _obs["T"]
pT1 = ps1 * ep / (ps1 * ep + (1 - ps1) * em)                      # P(S=+1 | T=1, X)
pT0 = ps1 * (1 - ep) / (ps1 * (1 - ep) + (1 - ps1) * (1 - em))    # P(S=+1 | T=0, X)
obs_figs = '<div class="figrow">' + figcap(
    scatterchart([("treated (T=1)", "#d62728", list(_Xj[_To == 1]), list(_Yo[_To == 1])),
                  ("untreated (T=0)", "#1f77b4", list(_Xj[_To == 0]), list(_Yo[_To == 0]))],
                 [],
                 title="What the analyst sees: (X, T, Y), S hidden (N=600, seed 0)", xlab="X (jittered)", ylab="Y"),
    "Observed data: Y = Y(T). E[Y | T=1, X=0] - E[Y | T=0, X=0] &asymp; +5.5, while "
    "CATE(0) = -1.") + figcap(
    linechart([("P(S=+1 | T=1, X)", "#d62728", list(xg), list(pT1), ""),
               ("P(S=+1 | T=0, X)", "#1f77b4", list(xg), list(pT0), ""),
               ("P(S=+1 | X)", "#334155", list(xg), list(ps1), "4 3")],
              title="Vitality composition of the two arms", xlab="X", ylab="P(S=+1 | T, X)"),
    "P(S{=}{+}1 | T{=}t, X) &propto; P(S{=}{+}1|X) &middot; P(T{=}t|X,S{=}{+}1); the "
    "T=1/T=0 gap is the within-level selection, maximal at X=0.") + '</div>'
EQ0 = M(r"X \sim \mathrm{Unif}\{-1,\,-\tfrac{2}{3},\,\ldots,\,1\}\ \text{(7 levels; continuous variant: } X\sim\mathrm{Unif}[-1,1]\text{)},\qquad S\mid X \in \{\pm 1\},\ \ P(S{=}{+}1\mid X)=\sigma(10X)")
EQ1 = M(r"T\mid X,S \sim \mathrm{Bernoulli}(e(X,S)),\qquad e(X,S)=\mathrm{clip}(\sigma(0.8S-2X),\,0.02,\,0.98)")
EQ2 = M(r"\mu_0=8S,\quad \mu_1=9S+3X-1,\quad Y(t)=\mu_t+\mathcal{N}(0,0.6^2)")
EQ3 = M(r"\mathrm{CATE}(X)=(2\sigma(10X)-1)+3X-1 \;\Rightarrow\; \text{oracle treats iff } X>0,\;\; \mathrm{CATE}(0)=-1")
T["dgp"].append(f"""
<h2 id="s-dgp">1. The data-generating process</h2>
<p>Aggressive therapy under hidden vitality, with a fixed therapy burden: fitness X is observed,
vitality S is unobserved, Y is the outcome (higher is better). N=600 per seed unless stated.</p>
<div class="eq">{EQ0}</div>
<div class="eq">{EQ1}</div>
<div class="eq">{EQ2}</div>
<div class="eq">{EQ3}</div>
<p>S shifts both arms by &plusmn;8&ndash;9 while the treatment differential is a few units:
treated patients look great because they are vital, not because therapy works. The -1 is a fixed
treatment burden (toxicity/cost), so over-treatment is visibly bad (all-treat = -1). The true
selection odds ratio is &Lambda;=e<sup>2&middot;0.8</sup>=4.95, so <b>&Gamma;=5 is the matched
sensitivity level</b> &mdash; results quoted "at matched &Gamma;" involve no tuning. Reference
values: never-treat 0, all-treat -1, oracle 0.847 uncapped / 0.727 under the 30% capacity cap
(the cap is genuinely scarce: 30% &lt; the 43% oracle-treat mass).</p>
<div class="card"><b>Where the confounding bias lives (a design feature, stated up front).</b>
Selection on S operates at every X (the 0.8S term in the propensity), but it can only BIAS a
within-X comparison where X fails to pin down S &mdash; i.e. where Var(S|X) = 4p(1-p) with
p = &sigma;(&alpha;X) is non-negligible. At the strong coupling &alpha;=10 this is the
&sigma;(&alpha;X)&asymp;&frac12; region: on the 7-level grid, the X=0 level (naive CATE inflation
+6.5; faint traces &plusmn;0.8 at X=&plusmn;1/3); in the continuous variant, a band
|X|&lesssim;0.2 that shifts the naive decision threshold. This concentration is the unavoidable
consequence of ANY monotone X&ndash;S coupling crossing one-half, not a planted artifact; the
continuous experiment shows the band version of the same effect, and the coupling sweep (Discrete
tab, Section 4) shows what happens as the region widens.</div>
{dgp_figs}
{mu_y_figs}
{obs_figs}""")

# (base-DGP headline, base Kallus, and N=1000 sections removed 2026-07-26: the report is
#  v2-only; the burden-free variant survives as the design rationale in the DGP tab, and the
#  base result JSONs remain on disk under assets/exp_owgap/.)

# ---- alpha sweep (appended to the disc tab AFTER the v2 results + policies; see below) ----
def _alpha_section():
    if not all(RA.values()): return None
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
    return f"""
<h2 id="s-alpha">4. Coupling sweep: when does O-W win?</h2>
<p>The X&ndash;S coupling &alpha; in P(S=+1|X)=&sigma;(&alpha;X) is swept over {{1,2,4,6,10}}
(N=600, 8 seeds, c<sub>&epsilon;</sub>=1.0, on the burden-free outcome variant &mdash; the
coupling and propensity are identical to the showcase DGP, so the boundary transfers) while the
true selection strength is held FIXED at &Lambda;=4.95: only the usefulness of X as a proxy for
S varies.</p>
<div class="tw"><table>
<tr><th>coupling</th><th>corr(X,S)</th><th>oracle</th><th>best naive</th><th>IPW-O-W at &Gamma;=5</th><th>margin</th></tr>
{''.join(arows)}
</table></div>
<div class="figrow">
{linechart([("oracle", "#111", ALPHAS, o_v, "5 4"), ser("IPW-O-W", ALPHAS, w_v, lab="IPW-O-W at G=5"), ser("DoublyRobust-X-X", ALPHAS, n_v, lab="best naive")], title="Value vs coupling strength", xlab="alpha", ylab="realized E[Y]", xticks=ALPHAS)}
{linechart([("O-W margin at G=5", MC["IPW-O-W"], ALPHAS, mvals, "")], title="O-W margin over best naive", xlab="alpha", ylab="margin", hlines=[("break-even", "#888", 0.0, "4 3")], xticks=ALPHAS, legend=False)}
</div>
<div class="card caveat"><b>Honest scoping: the margin flips sign.</b> O-W wins for
&alpha;&ge;6 (corr(X,S)&gtrsim;0.8), roughly ties at &alpha;=4, and LOSES below that &mdash; at
weak coupling the Wasserstein constraint has nothing to grab (balancing X no longer balances S)
and matched-&Gamma; robustness over-hedges below never-treat. This is a boundary of
applicability, not graceful degradation. Diagnostic upside: corr(X, S-proxy) is measurable, so
the operating regime is checkable in practice &mdash; and the Real-data tab's Experiment B shows
the diagnostic passing on the actual Diabetes coupling (corr&asymp;0.20, correctly predicted
out-of-regime).</div>"""

# (base continuous section removed 2026-07-26: the report is v2-only; the base L x Gamma
#  JSON remains under assets/exp_owgap_cont/.)

# ---- v2 ----
EQ4 = M(r"\mu_1 = 9S + 3X - 1 \;\Rightarrow\; \mathrm{CATE}(X) = (2\sigma(10X)-1) + 3X - 1,\quad \mathrm{CATE}(0)=-1")
if V2D:
    lv = V2D["levels"]; ct = V2D["cate"]; ch = V2D["catehat_naive_inf"]; vv = V2D["values"]
    idx = list(range(len(lv)))
    # finite-sample companion: naive CATE-hat per level, mean +- SD over 20 seeds of N=600
    _LVv = np.asarray(_dm.LEVELS, float)
    _ch20 = []
    for _sd in range(20):
        _o, _ = _dm.generate(600, _sd)
        _Xs, _Ts, _Ys = _o["X"].ravel(), _o["T"], _o["Y"]
        _lv = np.searchsorted(_LVv, _Xs - 1e-9)
        _ch20.append([float(_Ys[(_lv == j) & (_Ts == 1)].mean() - _Ys[(_lv == j) & (_Ts == 0)].mean())
                      for j in range(len(_LVv))])
    _ch20 = np.asarray(_ch20)
    _chm, _chs = _ch20.mean(0), _ch20.std(0)
    v2fig = '<div class="figrow">' + figcap(
        linechart([("true CATE", "#111", lv, ct, ""), ("naive CATE-hat (infinite data)", MC["DoublyRobust-X-X"], lv, ch, "2 3", "t")],
                  title="The ranking inversion at X=0", xlab="X level", ylab="CATE",
                  hlines=[("0", "#888", 0.0, "4 3")]),
        "Black: CATE(X). Blue: the infinite-data naive limit "
        "E[Y | T=1, X] - E[Y | T=0, X]; the difference is confounding bias, "
        "non-negligible only at X=0 (-1 &rarr; +5.5).") + figcap(
        linechart([("true CATE", "#111", lv, ct, ""),
                   ("naive CATE-hat, mean of 20 seeds", MC["DoublyRobust-X-X"], list(_LVv), [float(v) for v in _chm], "2 3", "t")],
                  title="Finite sample: bias dwarfs noise (20 seeds, N=600)", xlab="X level", ylab="CATE",
                  hlines=[("0", "#888", 0.0, "4 3")],
                  bands=[("+-1 SD", MC["DoublyRobust-X-X"], list(_LVv),
                          [float(m - s) for m, s in zip(_chm, _chs)],
                          [float(m + s) for m, s in zip(_chm, _chs)])]),
        "Same comparison at N=600: mean &plusmn; 1 SD of the within-level difference over 20 "
        "seeds. At six levels the band straddles the truth (sampling noise, shrinks with N); "
        "at X=0 the entire band sits near +5.5 while CATE(0) = -1 &mdash; bias &Gt; noise, so "
        "more data cannot fix it.") + '</div>'
    T["dgp"].append(f"""
<h2 id="s-v2">2. Design rationale: why the burden and the scarce cap exist</h2>
<p>The DGP above was reached by an explicit design iteration, worth reporting because it explains
what the experiment isolates. A first, burden-free variant (&mu;<sub>1</sub> = 9S + 1.5X, i.e.
THETA=0, so CATE(0)=0, with a loose 50% cap) produced the same estimation failure but a muted
experiment: the naive methods' signature error is treating X=0 &mdash; the one level where S is
genuinely uncertain given X, so within-level selection inflates the estimated CATE from -1 (there:
~0) to ~+5.5 &mdash; but with CATE(0)=0 that error was COSTLESS uncapped, and under the loose cap
the trimming even rescued the naive methods ("cap-rescue": margins +0.104 uncapped, +0.056
capped). The final DGP keeps the entire structure and makes exactly that error expensive:</p>
<div class="eq">{EQ4}</div>
<p>plus a genuinely scarce budget: cap 30% &lt; the 43% oracle-treat mass. Unchanged by the
redesign: coupling, propensity (so &Lambda;=4.95 and matched &Gamma;=5), oracle = treat iff
X&gt;0, uncapped oracle 0.847, never-treat 0. New: all-treat = -1. (A burden-free variant rerun
at N=1000 also confirmed the O-W margins are not a small-sample artifact.)</p>
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
same constants is defined in exp_owgap_v2_cont (x* = 0.137, oracle 0.737, naive DR threshold
shifts to -0.29). Results: see the Discrete tab, Section 1; every prediction above was
confirmed.</div>""")

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
    s_v2u = [ser(m, gamv, vm_u["mean"][m]) for m in MORDER[:-1] if m in vm_u["mean"]]
    s_v2c = [ser(m, gamv, vm_c["mean"][m]) for m in MORDER[:-1] if m in vm_c["mean"]]
    if KV2:
        s_v2u.append(ser("Kallus", KV2["gammas"], KV2["regimes"]["uncap"]["mean"]["Kallus"]))
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
<h2 id="s-v2res">1. Headline results: N=600, {n_sd} seeds, both regimes</h2>
<p>The showcase DGP (DGP tab, Sections 1&ndash;2) at N=600, {n_sd} seeds,
c<sub>&epsilon;</sub>=1.0. At the matched &Gamma;=5 the O-W margin over the best naive method is
<b>+{owu-nvu:.3f} uncapped</b> and <b>+{owc-nvc:.3f} capped</b> (the burden-free design variant
gave +0.104 / +0.056 &mdash; the capped margin tripled once the X=0 mistake carried a real
cost). The external Kallus &amp; Zhou baseline (box-only, parametric softmax; uncapped only,
since a smooth policy class cannot enforce a hard capacity) peaks at
{(max(KV2["regimes"]["uncap"]["mean"]["Kallus"]) if KV2 else 0):.2f} near &Gamma;=1 and
collapses to never-treat by &Gamma;&asymp;2.5 &mdash; the box alone forces total pessimism; only
the Wasserstein balance constraint lets robustness coexist with a non-trivial policy.</p>
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

# ---- discrete selected policies: every method x Gamma ----
if V2R and V2D:
    lvl_labs = [f"{x:.2g}" for x in V2R["grid"]]
    # capped(30%) oracle: greedy fill of the budget by true CATE (levels have mass 1/7 each)
    cap_orc = [0.0] * len(V2D["cate"]); rem = 0.3
    for j in sorted(range(len(V2D["cate"])), key=lambda i: -V2D["cate"][i]):
        if V2D["cate"][j] <= 0 or rem <= 1e-9: break
        take = min(1.0, rem * len(V2D["cate"])); cap_orc[j] = take; rem -= take / len(V2D["cate"])
    orc_u_s = "(" + ", ".join("0" if x <= 0 else "1" for x in V2D["cate"]) + ")"
    orc_c_s = "(" + ", ".join(f"{v:g}" for v in cap_orc) + ")"
    POL_METHODS = ["IPW-O-W", "DoublyRobust-O-W", "IPW-O-X", "DoublyRobust-O-X", "Hajek-O-X",
                   "IPW-X-X", "DoublyRobust-X-X", "Direct-X-X"]
    disc_pol = {}
    for reg in ("uncap", "cap"):
        disc_pol[reg] = {}
        for m in POL_METHODS:
            if m not in V2R["regimes"][reg].get("policy_by_seed", {}): continue
            gks, Mv = mean_policy_matrix(V2R, reg, m)
            disc_pol[reg][m] = {gk: row for gk, row in zip(gks, Mv)}
    if KV2:
        gksK, MvK = mean_policy_matrix(KV2, "uncap", "Kallus")
        disc_pol["uncap"]["Kallus"] = {gk: row for gk, row in zip(gksK, MvK)}
    PD["disc"] = {"kind": "disc", "grid": [float(x) for x in V2R["grid"]],
                  "gammas": gks, "regimes": ["uncap", "cap"],
                  "methods": POL_METHODS + (["Kallus"] if KV2 else []),
                  "pol": disc_pol,
                  "refs": {"oracle_uncap": [0.0 if c <= 0 else 1.0 for c in V2D["cate"]],
                           "oracle_cap": cap_orc}}
    T["disc"].append(f"""
<h2 id="s-pols">2. Selected policies: &pi;(X) vs X for every method and &Gamma; (mean over {len(V2R['seeds'])} seeds)</h2>
<p>Tick any set of methods to overlay them, then pick a &Gamma; and the regime; the plot shows
each selected method's policy as treatment probability &pi;(X) over the 7 levels, <b>averaged
over all {len(V2R['seeds'])} seeds</b>, with the oracle for that regime as the dashed reference
(uncapped oracle {orc_u_s}, capped(30%) oracle {orc_c_s}). With a single method ticked, the
exact &pi; vector is printed under the plot.</p>
{pol_widget_html("disc", PD["disc"], defaults={"m": ["IPW-O-W", "DoublyRobust-X-X"], "g": "5", "r": "uncap"})}
<p>What to look for: the naive X-X curves do not move with &Gamma; (plug-ins ignore it) and
always treat the selection-inflated X=0 level (true CATE = -1); the box-only O-X curves drain
toward never-treat as &Gamma; grows &mdash; worst-case pessimism with nothing to anchor it &mdash;
and Kallus drains the same way; the O-W curves stay anchored near the oracle across the whole
&Gamma; range, hedging only the ambiguous X=0 level. Under the cap the question becomes WHERE
each method spends its 30% budget: O-W concentrates it on the top levels, the naive methods burn
roughly a third of it on X=0. This &Gamma;-stability of the selected policy &mdash; not just of
the value &mdash; is the Wasserstein constraint's visible fingerprint.</p>""")

# ---- mechanism section: why each method picks its policy (from make_mechanism.py) ----
MECH = J("exp_owgap_v2/mechanism.json")
if MECH:
    ml = [f"{x:.2g}" for x in MECH["levels"]]
    sc, dis, cur, GGm = MECH["scores"], MECH["dists"], MECH["curves"], MECH["gamma_grid"]
    tc = MECH["true_cate"]
    T["disc"].append(f"""
<h2 id="s-mech">3. Why each method picks its policy: the worst-case mechanics</h2>
<p>Everything below is computed on the seed-0 draw (N=600) with the empirically fitted weights
w&#770;<sub>i</sub> = 1/e&#770;(T<sub>i</sub>|X<sub>i</sub>), the MSM box
[1+(w&#770;-1)/&Gamma;, 1+&Gamma;(w&#770;-1)], and &mdash; for the box&cap;W rows &mdash; the
per-arm Hajek-normalized reweighted level distribution constrained to W<sub>1</sub>-distance
&le; &epsilon; from the pooled empirical distribution (&epsilon; = the fitted weights' own
distance, i.e. the tightest budget). This is the mechanism of the production LPs in a form
solvable in closed form / by a small LP; exact numbers differ slightly from the full solver.</p>
<h3>The ranking each criterion induces</h3>
<p>Per-level score at the matched &Gamma;=5: the worst-case change in the objective from
treating level j versus not treating it (holding the rest of the policy at the oracle);
true CATE shown for reference.</p>
<div class="figrow">
{barchart(ml, [("naive plug-in score", MC["DoublyRobust-X-X"], sc["naive"]), ("true CATE", "#94a3b8", tc)], "Naive: E[Y|T=1,X] - E[Y|T=0,X]", "score")}
{barchart(ml, [("box-only worst-case score", MC["DoublyRobust-O-X"], sc["box"]), ("true CATE", "#94a3b8", tc)], "Odds-box only (G=5)", "score")}
{barchart(ml, [("box + W worst-case score", MC["IPW-O-W"], sc["boxw"]), ("true CATE", "#94a3b8", tc)], "Odds-box + Wasserstein (G=5)", "score")}
</div>
<div class="card finding"><b>Reading the rankings.</b> The naive score reproduces true CATE at
every level except X=0, where selection inflates it to {sc["naive"][3]:+.1f} (truth -1)
&rarr; naive methods treat X=0 first. The box-only score is PATHOLOGICAL: with &Gamma;=5 of
per-unit freedom and no balance constraint, the worst case is dominated by weight inflation on
whichever arm holds the mass &mdash; it awards {sc["box"][0]:+.1f} to treating the frailest
level (true CATE -5) simply because the control arm's worst case there is even worse. Its
ranking is unrelated to CATE, which is why pure box methods are erratic at moderate &Gamma; and
the self-normalized ones (Hajek, Kallus) flee to never-treat. The box&cap;W score recovers the
true SIGN at all seven levels; at X=0 it is mildly positive ({sc["boxw"][3]:+.2f}), which is
exactly why IPW-O-W hedges there (&pi;(0) &asymp; 0.47) instead of committing.</div>
<h3>What the adversary is allowed to do to the treated arm</h3>
{barchart(ml, [("pooled empirical", "#94a3b8", dis["empirical"]), ("fitted weights", "#334155", dis["fitted"]),
               ("box adversary", MC["DoublyRobust-O-X"], dis["box_adversary"]),
               ("box + W adversary", MC["IPW-O-W"], dis["boxw_adversary"])],
          "Treated-arm reweighted level distribution (oracle policy, G=5)", "P(level)", W=760, catlab="X level")}
<p class="muted figcap">Box alone: the adversary piles {100*sum(dis["box_adversary"][:4]):.0f}%
of treated-arm mass onto the four lowest levels (fabricating a frail treated arm). Adding the
W constraint pins the reweighted distribution to the empirical one &mdash; the box&cap;W
adversary's column is visually identical to the fitted weights'.</p>
<h3>Worst-case value of a FIXED policy vs &Gamma;</h3>
<div class="figrow">
{linechart([("oracle", "#111111", GGm, cur["oracle"]["box"], ""), ("naive policy", MC["DoublyRobust-X-X"], GGm, cur["naive"]["box"], ""), ("never-treat", "#888888", GGm, cur["never"]["box"], "4 3")], title="Odds-box only: worst case dives for EVERY policy", xlab="Gamma", ylab="worst-case value", xticks=GGm)}
{linechart([("oracle", "#111111", GGm, cur["oracle"]["boxw"], ""), ("naive policy", MC["DoublyRobust-X-X"], GGm, cur["naive"]["boxw"], ""), ("never-treat", "#888888", GGm, cur["never"]["boxw"], "4 3")], title="Box + Wasserstein: an informative band", xlab="Gamma", ylab="worst-case value", xticks=GGm)}
</div>
<div class="card finding"><b>Reading the curves.</b> Box only: by &Gamma;=5 every policy's worst
case sits near {cur["oracle"]["box"][6]:.0f} &mdash; differences between good and bad policies
are swamped by arm-level weight inflation, so the criterion carries almost no signal (and
regret-normalized box methods collapse onto the one certain policy, never-treat). Box&cap;W:
worst cases stay within ~1 unit of the true values at every &Gamma;
(oracle {cur["oracle"]["boxw"][6]:+.2f} vs never-treat {cur["never"]["boxw"][6]:+.2f} at
&Gamma;=5), so maximizing it still distinguishes policies &mdash; robustness without
paralysis. This pair of panels is the entire O-W argument in two pictures.</div>""")

_a = _alpha_section()
if _a: T["disc"].append(_a)

# ---- v2 continuous -> cont tab ----
if C2DV2:
    Gk2, Lk2 = C2DV2["gammas"], C2DV2["Lgrid"]; s2 = C2DV2["surface"]; bo2 = C2DV2["best_overall"]
    Mv2 = [[s2["IPW-O-W"][g][l] for l in Lk2] for g in Gk2]
    sL2 = [ser(m, list(range(len(Lk2))), [s2[m][bo2["gamma"]][l] for l in Lk2])
           for m in ["IPW-O-W", "DoublyRobust-O-W", "IPW-O-X", "DoublyRobust-O-X", "Hajek-O-X"]]
    nd = C2DV2.get("naive_dr"); nds = f"{nd:.3f}" if nd is not None else "--"
    T["cont"].append(f"""
<h2 id="s-v2cont">1. The L &times; &Gamma; surface: continuous X and the Lipschitz constant</h2>
<p>The showcase DGP with continuous X ~ Uniform(-1,1) (identical constants to the discrete
experiment; oracle threshold x*=0.137, oracle {C2DV2['oracle']:.3f}, never-treat
{C2DV2['never_treat']:.3f}, all-treat {C2DV2.get('all_treat', float('nan')):.3f}; naive DR
baseline {nds}). N={C2DV2['N_train']} train, {C2DV2['N_test']} test, {len(C2DV2['seeds'])}
seeds; off-support deployment: {dep_lab(C2DV2)}. With per-unit policies (L=&infin;) every
support point is its own parameter: the policy overfits and &Gamma; is inert. Two dials fix it:
the Lipschitz class |&pi;(x)-&pi;(x')| &le; L|x-x'| for the VARIANCE, and &Gamma; robustness
for the BIAS. The L &times; &Gamma; surface shows both.</p>
{heatmap(["G=" + g for g in Gk2], Lk2, Mv2, "v2 IPW-O-W: test E[Y] over Gamma x L (oracle %.2f)" % C2DV2["oracle"], "Gamma", "Lipschitz L (inf = per-unit)", max(C2DV2["never_treat"], -0.6), C2DV2["oracle"])}
<h3>E[Y] vs &Gamma;, with L selectable</h3>
<div class="svw"><div class="ctl"><label>Lipschitz L
<select id="sv-cont-l">{''.join('<option value="%s"%s>%s</option>' % (l, ' selected' if l == '3' else '', l) for l in Lk2)}</select></label>
<span class="ctt">all five robust methods at the chosen L; naive / oracle / never-treat as dashed references</span></div>
<div id="sv-cont-plot" class="fig"></div></div>
{linechart(sL2, title="v2 slice at Gamma=%s: the effect of L" % bo2["gamma"], xlab="L index: 0=inf ... 7=0.5", ylab="test E[Y]", hlines=[("oracle", "#111", C2DV2["oracle"], "5 4"), ("naive DR", MC["DoublyRobust-X-X"], nd if nd is not None else 0.0, "6 3"), ("never-treat", "#888", C2DV2["never_treat"], "2 3")], xticks=list(range(len(Lk2))))}
<p class="muted">Best overall: {bo2['method']} at &Gamma;={bo2['gamma']}, L={bo2['L']} &rarr;
{bo2['value']:.3f}. The full seed-0 policy curve for EVERY (&Gamma;, L) cell of this surface is
shown in Section 2 below.</p>
<div class="card"><b>Why DR is less damaged than IPW at L=&infin; (and why neither survives).</b>
With per-unit &pi; and no Lipschitz coupling, the IPW score for unit i is that unit's own
inverse-weighted outcome &mdash; one draw of Y whose scale is dominated by the hidden S
(&plusmn;8&ndash;9), so the per-unit policy tracks the unit's vitality draw, not CATE(x). Two
symptoms confirm it: the L=&infin; IPW value is
{C2DV2['surface']['IPW-O-W']['4']['inf']:+.2f} IDENTICALLY across &Gamma;=1&ndash;8 and across
IPW-O-X / Hajek-O-X / IPW-O-W &mdash; box, self-normalization, and even the W constraint change
nothing, because with one observation per variable there is no pooled mass for the uncertainty
machinery to act on &mdash; and it sits below never-treat
({C2DV2['never_treat']:+.2f}): deployment turns the 0/1 speckle into a mid-band mixture that
partially treats the frail region. The AIPW score instead anchors each unit on
&mu;&#770;<sub>1</sub>(X<sub>i</sub>) - &mu;&#770;<sub>0</sub>(X<sub>i</sub>), a cross-fit
regression pooled over all N points, and the adversarial weight multiplies only the residual
Y<sub>i</sub> - &mu;&#770;(X<sub>i</sub>) rather than the raw outcome &mdash; less noise and
less adversarial leverage per decision &mdash; hence
{C2DV2['surface']['DoublyRobust-O-W']['8']['inf']:+.2f} instead of
{C2DV2['surface']['IPW-O-W']['4']['inf']:+.2f}. But pooling the SCORE is not pooling the
POLICY: per-unit residual wiggles still flip decisions, deployment mixes them, and
&mu;&#770;'s pooled signal carries the naive bias exactly in the confounded band &mdash; so
DR at L=&infin; is still below never-treat. Division of labor:
&mu;&#770;/DR pools information, L pools the policy, &Gamma;&cap;W fixes the bias; only cells
with all three reach &asymp;0.55.</div>""")
else:
    T["cont"].append("""
<h2 id="s-v2cont">2. v2 continuous: L &times; &Gamma; on the showcase DGP (running)</h2>
<p class="muted">Job 10579938 (chained behind the discrete pilot) sweeps the same 5 robust
methods over 6 &Gamma; &times; 8 L on the v2 continuous DGP, with naive-DR / oracle / never /
all-treat references and per-(&Gamma;,L) seed-0 policy curves for the explanation figures.
This section fills automatically when it lands &mdash; rerun build_results.py.</p>""")

# ---- continuous transport-budget ablation: c_eps in {1.0, 1.5, 2.0} (mirrors the discrete suite) ----
C2DV2_CE = {"1.0": C2DV2,
            "1.5": J("exp_owgap_v2_cont/owgap_v2_lip_gamma_2d_shapley_ce1.5.json"),
            "2.0": J("exp_owgap_v2_cont/owgap_v2_lip_gamma_2d_shapley_ce2.0.json")}
if C2DV2:
    ce_rows2 = []
    for ce, R in C2DV2_CE.items():
        if not R: continue
        s = R["surface"]; bo = R["best_overall"]
        ce_rows2.append(f"<tr><td>c<sub>&epsilon;</sub>={ce}</td><td>{R['naive_dr']:.3f}</td>"
                        f"<td>{s['IPW-O-W']['4']['3']:.3f}</td><td>{s['DoublyRobust-O-W']['4']['3']:.3f}</td>"
                        f"<td>{s['IPW-O-W']['6']['3']:.3f}</td>"
                        f"<td>{bo['method']} at &Gamma;={bo['gamma']}, L={bo['L']}</td>"
                        f"<td class='g'>{bo['value']:.3f}</td></tr>")
    CU = J("exp_owgap_v2_cont/cont_uncertainty.json")
    if CU:
        cu_rows = []
        for ce in ("1.0", "1.5", "2.0"):
            r = CU.get(ce)
            if not r: continue
            c = r["cells"]
            k = "IPW-O-W@G4,L3"
            cu_rows.append(f"<tr><td>c<sub>&epsilon;</sub>={ce}</td>"
                           f"<td>{c['naive']['mean']:.3f} &plusmn; {c['naive']['sd']:.2f}</td>"
                           f"<td>{c[k]['mean']:.3f} &plusmn; {c[k]['sd']:.2f}</td>"
                           f"<td class='g'>{c[k]['margin_mean']:+.3f} &plusmn; {c[k]['margin_ci95']:.3f}</td></tr>")
        n8 = CU.get("1.0", {}).get("n_seeds", 8)
        T["cont"].append(f"""
<h3>Uncertainty: per-seed values and paired margins (continuous)</h3>
<p>Per-seed test values recomputed from the persisted raw support policies (same Shapley
deployment, deterministic test draws; recomputed means reproduce the stored surface cells).
Paired margin = per-seed difference IPW-O-W@&Gamma;4,L3 minus naive DR; 95% t-interval,
n={n8} seeds.</p>
<div class="tw"><table>
<tr><th>budget</th><th>naive DR (mean &plusmn; SD)</th><th>IPW-O-W @&Gamma;4,L3</th><th>paired margin &plusmn; 95% CI</th></tr>
{''.join(cu_rows)}
</table></div>""")
    missing_ce = [ce for ce, R in C2DV2_CE.items() if not R]
    miss_html = ("" if not missing_ce else
                 f'<p class="muted">c<sub>&epsilon;</sub> = {", ".join(missing_ce)} queued (job 10608156); '
                 f'rows fill automatically when the runs land.</p>')
    T["cont"].append(f"""
<h3>Transport-budget ablation: c<sub>&epsilon;</sub> &isin; {{{{1.0, 1.5, 2.0}}}}</h3>
<p>The same 8-seed L &times; &Gamma; sweep at all three Wasserstein budgets, mirroring the
discrete suite's ablation. &epsilon; enters only the O-W methods (the box-only O-X columns are
&epsilon;-free), and the matched &Gamma;=&Lambda;=4.95 is bracketed by the &Gamma;=4 and
&Gamma;=6 grid points; cells quoted at the L=3 sweet spot. Oracle {C2DV2['oracle']:.3f},
never-treat {C2DV2['never_treat']:.3f} throughout.</p>
<div class="tw"><table>
<tr><th>budget</th><th>naive DR</th><th>IPW-O-W @&Gamma;4,L3</th><th>DR-O-W @&Gamma;4,L3</th>
<th>IPW-O-W @&Gamma;6,L3</th><th>best overall cell</th><th>value</th></tr>
{''.join(ce_rows2)}
</table></div>
{miss_html}""")

# ---- interactive policy viewer datasets for any L x Gamma 2-D result (continuous + diabetes) ----
def policy_2d_dataset(RJ):
    """PD entry for the pi(X) widget: seed-0 policy curve per method x Gamma x L + refs,
    plus (when the run saved it) the RAW per-unit support policy for the verification scatter."""
    ps = RJ["policies_seed0"]
    d = {"kind": "2d", "grid": [float(x) for x in RJ["policy_grid"]],
         "gammas": RJ["gammas"], "Ls": RJ["Lgrid"], "methods": RJ["methods"],
         "pol": {m: ps[m] for m in RJ["methods"]},
         "refs": {"oracle": ps["_refs"]["oracle"], "naive": ps["_refs"]["naive_dr"]}}
    sup = RJ.get("policies_support_seed0") or (RJ.get("policies_support_by_seed") or {}).get("0")
    if sup and "_X" in sup:
        d["supX"] = [round(float(x), 3) for x in sup["_X"]]
        d["sup"] = {m: {g: {l: [int(round(float(v) * 100)) for v in sup[m][g][l]]
                            for l in RJ["Lgrid"] if l in sup[m][g]}
                        for g in RJ["gammas"]} for m in RJ["methods"]}
        if "_naive_dr" in sup:
            d["supNaive"] = [int(round(float(v) * 100)) for v in sup["_naive_dr"]]
    return d

def policy_2d_bestchart(RJ, title):
    ps = RJ["policies_seed0"]; pg, refs = RJ["policy_grid"], ps["_refs"]
    series = [("oracle policy", "#111111", pg, refs["oracle"], "5 4"),
              ("naive DR policy", MC["DoublyRobust-X-X"], pg, refs["naive_dr"], "2 3")]
    for m in ("IPW-O-W", "DoublyRobust-O-W"):
        b = RJ["best"].get(m) if "best" in RJ else None
        if b: series.append(ser(m, pg, ps[m][b["gamma"]][b["L"]], lab=f"{m} at G={b['gamma']}, L={b['L']}"))
    return linechart(series, title=title, xlab="x", ylab="pi(x)", W=760, H=320)

if C2DV2 and "policies_seed0" in C2DV2:
    bo2 = C2DV2["best_overall"]
    PD["cont"] = policy_2d_dataset(C2DV2)
    T["cont"].append(f"""
<h2 id="s-contpols">2. Policy curves: &pi;(x) vs x for every method, &Gamma;, and L</h2>
<p>The complete answer to "what policy did each method actually pick": choose a method, a
&Gamma;, and a Lipschitz constant L, and the plot shows the seed-0 learned policy
&pi;(x) over x &isin; [-1, 1] for that cell of the Section-1 surface, deployed off-support via
{dep_lab(C2DV2)}. Dashed references: the oracle (treat iff x &gt; 0.137) and the naive DR
plug-in (threshold shifted left to &asymp;-0.3 by hidden-vitality bias: it over-treats the
ambiguous band).{" Unlike KNN averaging, this deployment is exact at the support points, so the L=&infin; curves show the per-unit overfitting oscillation directly rather than a smoothed band." if C2DV2.get("deploy") == "shapley" else ""}</p>
<p><b>Verification plot.</b> The first panel shows the RAW pointwise solver output: the
{C2DV2['N_train']} per-unit values &pi;(X<sub>i</sub>) that the LP actually chose at the training
support (seed 0; faint brown dots = the naive plug-in's raw 0/1 decisions). The second panel is
the same policy after off-support extension. Because the Shapley operator is exact at support
points, the extended curve must pass through the dot cloud wherever the plot grid comes close to
a support point &mdash; at L=&infin; both panels oscillate together, and at L&le;3 the dots
already lie on a smooth ramp that the extension simply traces. Any systematic gap between the
two panels would indicate an extension bug.</p>
{pol_widget_html("cont", PD["cont"], defaults={"m": ["IPW-O-W", "DoublyRobust-O-W"], "g": bo2["gamma"], "l": bo2["L"]})}
<p>What to look for as you move the dropdowns: at <b>L = &infin;</b> the curve is jagged &mdash;
per-unit policies overfit each support point, and changing &Gamma; barely moves them (the
inert-&Gamma; column of the surface). At <b>moderate L (1.5&ndash;3)</b> the curve is a clean
step whose boundary sits near the oracle's. Raising &Gamma; drains the <b>box-only O-X</b>
curves toward &pi;&equiv;0 (worst-case collapse to never-treat), while the <b>O-W</b> curves keep
a stable treat-region &mdash; the Wasserstein anchor at work; their boundary drifts only slightly
right (more conservative) with &Gamma;. Best overall cell: {bo2['method']} at
&Gamma;={bo2['gamma']}, L={bo2['L']}.</p>
{policy_2d_bestchart(C2DV2, "Best-cell policies vs oracle and naive (seed 0)")}""")
    if J("exp_owgap_v2_cont/kallus_v2_cont.json"):
        KVC = J("exp_owgap_v2_cont/kallus_v2_cont.json")
        gksK, MvK = mean_policy_matrix(KVC, "uncap", "Kallus")
        T["cont"].append(f"""
<h3>Kallus baseline policies (parametric; evaluated on the 7-point level grid)</h3>
<p class="muted">The softmax policy class is a function of x, shown here on the coarse level grid
(mean over {len(KVC['seeds'])} seeds). Same collapse as in the discrete case: by
&Gamma;&asymp;2.5 the policy is all-purple (never-treat).</p>
{heatmap(["G=" + g for g in gksK], [f"{x:.2g}" for x in KVC["grid"]], MvK,
         "Kallus: mean policy pi(x) per Gamma", "Gamma", "x (level grid)", 0.0, 1.0, W=470, H=300)}""")

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

# selected policies for both diabetes experiments: every method x Gamma x L
if DIA and "policies_seed0" in DIA:
    PD["diabA"] = policy_2d_dataset(DIA)
    T["real"].append(f"""
<h3>Experiment A: selected policies &mdash; &pi;(x) vs frailty x, by method, &Gamma;, and L</h3>
<p>Same viewer as the Continuous tab (seed 0; &pi;(x) = probability of giving insulin). The
oracle treats the frailest patients (x above &asymp; the 79th percentile of frailty); the naive
DR curve is nearly &pi;&equiv;0 &mdash; the under-treatment failure ("insulin looks harmful").
Move &Gamma; toward the matched value 5 to watch the O-W curve recover a stable frail-side
treat-region; at L=&infin; nothing works, and box-only methods overshoot toward &pi;&equiv;0 at
large &Gamma;.</p>
{pol_widget_html("diabA", PD["diabA"], defaults={"m": ["IPW-O-W", "DoublyRobust-O-W"], "g": "4", "l": "3"})}
{policy_2d_bestchart(DIA, "Experiment A: best-cell policies vs oracle and naive (seed 0)")}""")
if DIB and "policies_seed0" in DIB:
    PD["diabB"] = policy_2d_dataset(DIB)
    T["real"].append(f"""
<h3>Experiment B: selected policies &mdash; &pi;(x) vs x, by method, &Gamma;, and L</h3>
<p>The fully-real coupling (matched &Gamma; = 1.26). Here the correct behavior is to CHANGE
LITTLE: naive is already near-oracle, so the best robust curves are the low-&Gamma; ones that
track the naive/oracle boundary; raising &Gamma; visibly erodes the treat-region &mdash; the
over-hedging the diagnostic predicts out-of-regime.</p>
{pol_widget_html("diabB", PD["diabB"], defaults={"m": "IPW-O-W", "g": "1", "l": "3"})}
{policy_2d_bestchart(DIB, "Experiment B: best-cell policies vs oracle and naive (seed 0)")}""")

# ---- coupled synthetic -> coup tab ----
def _cj(p):
    try: return json.load(open(os.path.join(A, p)))
    except Exception: return None
CB = {0.0: _cj("exp_msmbench/msmbench_lip_gamma_2d_pilot.json"),
      2.5: _cj("exp_coupled_synth/coupled_beta2.5.json"),
      5.0: _cj("exp_coupled_synth/coupled_beta5.json"),
      10.0: _cj("exp_coupled_synth/coupled_beta10.json")}
CK = {("2.5", "10"): _cj("exp_coupled_synth/coupled_k2.5b10.json"),
      ("4", "10"): _cj("exp_coupled_synth/coupled_k4b10.json"),
      ("4", "5"): _cj("exp_coupled_synth/coupled_k4b5.json")}
_CCORR = {0.0: -0.007, 2.5: 0.578, 5.0: 0.756, 10.0: 0.840}
if any(CB.values()) or any(v for v in CK.values()):
    EQC = M(r"Y(a) = (2a{-}1)X + (2a{-}1) - 2\sin(2(2a{-}1)X) - 2\kappa(2U{-}1)(1+0.5X) + \mathcal{N}(0,1),\;\; X \sim \mathrm{Unif}[-2,2],\; U \mid x \sim \mathrm{Bern}(\sigma(\beta x))")
    EQP = M(r"P(T{=}1 \mid x, U) = \sigma(0.5 + 1.5x + 0.8(2U{-}1)) \Rightarrow \Gamma^{*} = e^{1.6} = 4.95 \text{ exactly, } \forall x, \beta, \kappa")
    cbx, cbI, cbD = [], [], []
    brows = []
    for b in sorted(k for k, v in CB.items() if v):
        r = CB[b]; bx_, bw_ = r["best"]["IPW-O-X"], r["best"]["IPW-O-W"]
        cbx.append(_CCORR[b]); cbI.append(bw_["value"] - bx_["value"])
        cbD.append(r["best"]["DoublyRobust-O-W"]["value"] - r["best"]["DoublyRobust-O-X"]["value"])
        brows.append(f"<tr><td>&beta;={b:g}, &kappa;=1</td><td>{_CCORR[b]:.2f}</td>"
                     f"<td>{r['oracle']:+.2f}</td><td>{r['naive_dr']:+.2f}</td>"
                     f"<td>{bx_['value']:.3f}</td><td>{bw_['value']:.3f}</td>"
                     f"<td>{bw_['value']-bx_['value']:+.3f}</td></tr>")
    krows = []
    for (kk, bb), r in CK.items():
        if not r: continue
        bx_, bw_ = r["best"]["IPW-O-X"], r["best"]["IPW-O-W"]
        hl = " style='font-weight:700'" if (kk, bb) == ("2.5", "10") else ""
        krows.append(f"<tr{hl}><td>&kappa;={kk}, &beta;={bb}</td><td>0.84</td>"
                     f"<td>{r['oracle']:+.2f}</td><td>{r['naive_dr']:+.2f}</td>"
                     f"<td>{bx_['value']:.3f}</td><td>{bw_['value']:.3f}</td>"
                     f"<td class='g'>{bw_['value']-bx_['value']:+.3f}</td></tr>")
    kI = {1.0: CB[10.0]["best"]["IPW-O-W"]["value"] - CB[10.0]["best"]["IPW-O-X"]["value"]}
    kD = {1.0: CB[10.0]["best"]["DoublyRobust-O-W"]["value"] - CB[10.0]["best"]["DoublyRobust-O-X"]["value"]}
    for (kk, bb), r in CK.items():
        if r and bb == "10":
            kI[float(kk)] = r["best"]["IPW-O-W"]["value"] - r["best"]["IPW-O-X"]["value"]
            kD[float(kk)] = r["best"]["DoublyRobust-O-W"]["value"] - r["best"]["DoublyRobust-O-X"]["value"]
    kgap_fig = linechart([ser("IPW-O-W", sorted(kI), [kI[k] for k in sorted(kI)], lab="IPW: O-W minus O-X"),
                          ser("DoublyRobust-O-W", sorted(kD), [kD[k] for k in sorted(kD)], lab="DR: O-W minus O-X")],
                         title="The gap opens with the leverage dial (beta = 10)",
                         xlab="kappa (confounder outcome leverage)", ylab="O-W minus O-X",
                         hlines=[("0", "#888", 0.0, "4 3")], W=620, xticks=[1, 2.5, 4])
    _rh = CK[("2.5", "10")]
    khead_fig = barchart([""],
                         [(m, MC[m], [v]) for m, v in
                          [("DoublyRobust-X-X", _rh["naive_dr"]),
                           ("Hajek-O-X", _rh["best"]["Hajek-O-X"]["value"]),
                           ("IPW-O-X", _rh["best"]["IPW-O-X"]["value"]),
                           ("DoublyRobust-O-X", _rh["best"]["DoublyRobust-O-X"]["value"]),
                           ("DoublyRobust-O-W", _rh["best"]["DoublyRobust-O-W"]["value"]),
                           ("IPW-O-W", _rh["best"]["IPW-O-W"]["value"])]],
                         "Every method at the headline setting (kappa=2.5, beta=10); oracle %.2f" % _rh["oracle"],
                         "test E[Y]", W=620, catlab="") if CK.get(("2.5", "10")) else ""
    gap_fig = linechart([ser("IPW-O-W", cbx, cbI, lab="IPW: O-W minus O-X (kappa=1)"),
                         ser("DoublyRobust-O-W", cbx, cbD, lab="DR: O-W minus O-X (kappa=1)")],
                        title="W-term contribution vs measured coupling (kappa = 1)",
                        xlab="corr(x, U)", ylab="O-W minus O-X",
                        hlines=[("0", "#888", 0.0, "4 3")], W=680)
    T["coup"].append(f"""
<h2 id="s-coup">The coupled-confounder synthetic (second synthetic experiment)</h2>
<p>Complements the showcase on every axis: UNDER-treatment failure direction, an oscillating
heterogeneous CATE, and two designed properties. <b>(1) &Gamma;* is known by construction</b>
&mdash; the propensity pins the hidden-confounder odds ratio algebraically, so every method
runs at the single matched &Gamma; = &Gamma;* = 4.95 with nothing swept or tuned; only the
Lipschitz dial L varies. <b>(2) Two disclosed dials</b>: coupling &beta; (can X track the
confounder?) and leverage &kappa; (does the confounder dominate the outcome scale?). Outcome
functional forms follow Kallus-Mao-Zhou (2019).</p>
<div class="eq">{EQC}</div>
<div class="eq">{EQP}</div>
<h3>The coupling dial at &kappa; = 1: the W-term switches on at the predicted boundary</h3>
<div class="tw"><table>
<tr><th>setting</th><th>corr(x,U)</th><th>oracle</th><th>naive DR</th><th>best IPW-O-X</th>
<th>best IPW-O-W</th><th>O-W gap</th></tr>
{"".join(brows)}
</table></div>
{gap_fig}
<p>The flip from ~0 to positive lands between corr 0.58 and 0.76 &mdash; the same boundary the
showcase's coupling sweep identified, measured on an unrelated DGP family. At &beta; = 0
(U &perp; X) no X-balance device can constrain the confounder; O-W ties box-only within noise
&mdash; do no harm, as required.</p>
<h3>The leverage dial: the gap opens when the confounder dominates outcomes</h3>
<div class="tw"><table>
<tr><th>setting</th><th>corr(x,U)</th><th>oracle</th><th>naive DR</th><th>best IPW-O-X</th>
<th>best IPW-O-W</th><th>O-W gap</th></tr>
{"".join(krows)}
</table></div>
{kgap_fig}
{khead_fig}
<div class="card finding"><b>Headline (quick pilots, n=200, 3 seeds, &Gamma; = &Gamma;*
throughout).</b> At &kappa; = 2.5, &beta; = 10 IPW-O-W is the best method on the board
(-1.292; box-only IPW -1.704, naive -2.59, Hajek -2.62), with the like-for-like gap
<b>+0.411</b> (per-seed +0.305 / +0.073 / +0.856, all positive) and +0.727 at &kappa; = 4.
Mechanism in one sentence: the W-term pays exactly when the confounder is both
outcome-dominant (&kappa;) and X-trackable (&beta;) &mdash; and the low-dial ties are shown on
the same page. Honest note: at high &kappa; the DR box-only variant also recovers (the outcome
model is the other route to exploiting coupling); O-W is best overall and model-free.</div>
<p class="muted">Full experiment report with all construction figures, the known-&Gamma;*
algebra, and per-seed analyses: assets/exp_coupled_synth/coupled_report.html
(<a href="https://claude.ai/code/artifact/c61ab76f-d4cc-4624-bc21-3b96f25a2da8">published
copy</a>). Paper-grade next steps: n=400 at 8-20 seeds, capped variant, Kallus/sharp
baselines.</p>""")

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
<button id="tb-coup" onclick="showTab('coup')">Coupled synthetic</button>
</div>
<div id="tab-dgp" class="tabpane on">{''.join(T['dgp'])}</div>
<div id="tab-disc" class="tabpane">{''.join(T['disc'])}</div>
<div id="tab-cont" class="tabpane">{''.join(T['cont'])}</div>
<div id="tab-real" class="tabpane">{''.join(T['real'])}</div>
<div id="tab-coup" class="tabpane">{''.join(T['coup'])}</div>
<script>
function showTab(id){{
  for (const t of ['dgp','disc','cont','real','coup']){{
    document.getElementById('tab-'+t).classList.toggle('on', t===id);
    document.getElementById('tb-'+t).classList.toggle('on', t===id);
  }}
  try{{history.replaceState(null,'','#'+id);}}catch(e){{}}
  window.scrollTo(0,0);
}}
(function(){{const h=location.hash.replace('#','');
  if(['dgp','disc','cont','real','coup'].includes(h)) showTab(h);}})();
</script>""")

# ---- policy-viewer data + plotter (must come after the widget divs) ----
def _r3(o):
    if isinstance(o, list): return [_r3(x) for x in o]
    if isinstance(o, dict): return {k: _r3(v) for k, v in o.items()}
    if isinstance(o, float): return round(o, 3)
    return o
SURF = None
if C2DV2:
    SURF = {"gammas": C2DV2["gammas"], "Ls": C2DV2["Lgrid"], "methods": C2DV2["methods"],
            "surface": C2DV2["surface"], "oracle": C2DV2["oracle"],
            "naive": C2DV2["naive_dr"], "never": C2DV2["never_treat"]}
html.append("<script>\nconst PD = " + json.dumps(_r3(PD), separators=(",", ":")) + ";\n"
            + "const SURFD = " + json.dumps(_r3(SURF), separators=(",", ":")) + ";\n"
            + (PW_JS % {"MC": json.dumps(MC),
                        "DASH": json.dumps({m: mdash(m) for m in MORDER}),
                        "MARK": json.dumps({m: mmark(m) for m in MORDER})})
            + SV_JS + "\n</script>")

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
