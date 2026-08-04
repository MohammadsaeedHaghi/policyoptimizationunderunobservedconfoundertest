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


def legend_swatch(col, dash="", mk="c", w=30, h=12):
    """Legend key that shows the LINE STYLE and MARKER, not just the colour.

    The palette encodes the estimator family as colour, so DoublyRobust-O-W and
    DoublyRobust-O-X are both blue and a plain colour block made them indistinguishable in the
    legend even though their plotted lines differ. This draws the actual dash pattern with the
    method's marker on top, so the legend key matches what is on the chart.
    """
    cy = h / 2.0
    da = f' stroke-dasharray="{dash}"' if dash else ""
    p = [f'<svg width="{w}" height="{h}" viewBox="0 0 {w} {h}" style="vertical-align:middle;flex:none">',
         f'<line x1="1" y1="{cy}" x2="{w-1}" y2="{cy}" style="stroke:{col}" stroke-width="2.2"{da}/>']
    cx, r = w / 2.0, 3.0
    if mk == "n":
        pass                                              # line only -- matches a marker-free series
    elif mk == "s":
        p.append(f'<rect x="{cx-r}" y="{cy-r}" width="{2*r}" height="{2*r}" style="fill:{col}"/>')
    elif mk == "t":
        p.append(f'<polygon points="{cx},{cy-1.3*r} {cx-1.2*r},{cy+r} {cx+1.2*r},{cy+r}" style="fill:{col}"/>')
    elif mk == "d":
        p.append(f'<polygon points="{cx},{cy-r} {cx+r},{cy} {cx},{cy+r} {cx-r},{cy}" style="fill:{col}"/>')
    else:
        p.append(f'<circle cx="{cx}" cy="{cy}" r="{r}" style="fill:{col}"/>')
    p.append('</svg>')
    return "".join(p)


def esc(s): return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

CSS = """
:root{--bg:#f5f6fb;--surface:#ffffff;--fg:#0f172a;--muted:#667085;--border:#e9ebf3;
 --accent:#334155;--accent-soft:#f1f5f9;--good:#0a7d33;--warn:#b45309;--bad:#b91c1c;--sh:0 6px 18px rgba(16,24,40,.09);}
/* LIGHT ONLY, deliberately. These reports were rendering dark for anyone whose OS was in dark
   mode (the old @media prefers-color-scheme block flipped every variable). The figures, chart
   gridlines and chip swatches are tuned for a light background, so the theme is pinned: the
   dark-mode media query and the [data-theme="dark"] override are both removed, and a viewer
   theme toggle stamping data-theme="dark" now leaves the palette unchanged. */
:root[data-theme="dark"], :root[data-theme="light"]{--bg:#f5f6fb;--surface:#ffffff;--fg:#0f172a;
 --muted:#667085;--border:#e9ebf3;--accent:#334155;--accent-soft:#f1f5f9;--good:#0a7d33;
 --warn:#b45309;--bad:#b91c1c;--sh:0 6px 18px rgba(16,24,40,.09);}
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
            if b != b or mk == "n": continue
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
        items = "".join(
            '<span class="li">%s%s</span>'
            % (legend_swatch(it[1], it[4] if len(it) > 4 else "", it[5] if len(it) > 5 else "c"), esc(it[0]))
            for it in series)
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
    hl = [("oracle", "#111", J["oracle"], "5 4"),
          ("naive DR", MC["DoublyRobust-X-X"], J["naive_dr"], "2 3"),
          ("never-treat", "#888", J["never_treat"], "2 3")]
    if len(Gk) > 1:
        gl = [float(g) for g in Gk]
        sl = [ser(m, gl, [s[m][g]["3"] for g in Gk]) for m in J["methods"]]
        out.append(linechart(sl, title=f"{title_prefix}: E[Y] vs Gamma at L=3", xlab="Gamma",
                             ylab="test E[Y]", hlines=hl, xticks=gl))
    else:
        g0 = Gk[0]; li = list(range(len(Lk)))
        sl = [ser(m, li, [s[m][g0][l] for l in Lk]) for m in J["methods"]]
        out.append(linechart(sl, title=f"{title_prefix}: E[Y] vs L at Gamma={g0}",
                             xlab="L index: " + " ".join(f"{i}={l}" for i, l in enumerate(Lk)),
                             ylab="test E[Y]", hlines=hl, xticks=li))
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


def bars(cats, vals, cols, title, ylab, W=680, H=340, hlines=None):
    """Single-series vertical bars with per-bar colors, value labels, optional hlines."""
    padL, padR, padT, padB = 56, 14, 30, 56
    ys_all = list(vals) + [0.0] + [h[2] for h in (hlines or [])]
    ylo, yhi = min(ys_all), max(ys_all)
    ypad = 0.10 * (yhi - ylo + 1e-9); ylo -= ypad; yhi += ypad
    Y = lambda v: H - padB - (v - ylo) / (yhi - ylo + 1e-12) * (H - padT - padB)
    n = len(cats); slot = (W - padL - padR) / n; bw = slot * 0.62
    p = [f'<svg viewBox="0 0 {W} {H}" class="chart">', f'<text x="{padL}" y="16" class="ct">{esc(title)}</text>']
    for t in _ticks(ylo + ypad, yhi - ypad):
        if ylo <= t <= yhi:
            p.append(f'<line x1="{padL}" y1="{Y(t):.1f}" x2="{W-padR}" y2="{Y(t):.1f}" class="grid"/>')
            p.append(f'<text x="{padL-6}" y="{Y(t)+3.5:.1f}" class="tk" text-anchor="end">{t:g}</text>')
    for lab, col, yv, dash in (hlines or []):
        p.append(f'<line x1="{padL}" y1="{Y(yv):.1f}" x2="{W-padR}" y2="{Y(yv):.1f}" stroke="{col}" stroke-width="1.4" stroke-dasharray="{dash}"/>')
        p.append(f'<text x="{W-padR-2}" y="{Y(yv)-4:.1f}" class="tk" text-anchor="end" fill="{col}">{esc(lab)}</text>')
    y0 = Y(0.0)
    for i, (c, v, col) in enumerate(zip(cats, vals, cols)):
        x = padL + slot * i + (slot - bw) / 2
        yt = Y(max(v, 0.0)); hgt = abs(y0 - Y(v))
        p.append(f'<rect x="{x:.1f}" y="{yt:.1f}" width="{bw:.1f}" height="{max(hgt,0.6):.1f}" fill="{col}" opacity="0.92"/>')
        p.append(f'<text x="{x+bw/2:.1f}" y="{yt-5:.1f}" class="tk" text-anchor="middle">{v:+.2f}</text>')
        p.append(f'<text x="{padL+slot*(i+0.5):.1f}" y="{H-padB+14}" class="tk" text-anchor="middle">{esc(str(c))}</text>')
    p.append(f'<line x1="{padL}" y1="{y0:.1f}" x2="{W-padR}" y2="{y0:.1f}" class="ax"/>')
    p.append(f'<line x1="{padL}" y1="{padT}" x2="{padL}" y2="{H-padB}" class="ax"/>')
    p.append(f'<text x="16" y="{(padT+H-padB)/2:.0f}" class="al" text-anchor="middle" transform="rotate(-90 16 {(padT+H-padB)/2:.0f})">{esc(ylab)}</text>')
    p.append('</svg>')
    return f'<figure class="fig">{"".join(p)}</figure>'


# ============================ interactive E[Y]-vs-Gamma widget ============================
# Static linechart() is fine for DGP figures, but the results charts carry 5-8 method series and
# become unreadable. This is the same chip pattern the policy viewers use: checkboxes per method
# (coloured by family), a Lipschitz-L selector, and optional flat baselines. Methods without an
# L axis (Sharp-O-X, Kallus) are passed via `extra` and are drawn whenever their chip is on.
def ev_widget_html(wid, data, defaults=None, title="", note=""):
    """HTML for one interactive E[Y]-vs-Gamma widget. Pair with EV_JS and an `EVD[wid]` payload."""
    df = defaults or {}
    on = df.get("m") or list(data["methods"])[:2]
    if isinstance(on, str): on = [on]
    names = list(data["methods"]) + list((data.get("extra") or {}).keys())
    chips = "".join(
        '<label class="mchip"><input type="checkbox" data-m="%s"%s>%s%s</label>'
        % (m, " checked" if m in on else "",
           legend_swatch(MC.get(m, "#7f7f7f"), mdash(m), mmark(m), w=24, h=10), m) for m in names)
    lsel = ""
    if data.get("Ls"):
        opts = "".join('<option value="%s"%s>%s</option>'
                       % (l, " selected" if str(l) == str(df.get("l", "3")) else "", l)
                       for l in data["Ls"])
        lsel = f'<label>Lipschitz L <select id="ev-{wid}-l">{opts}</select></label>'
    h = [f'<div class="evw" id="ev-{wid}">']
    if title: h.append(f'<h3>{esc(title)}</h3>')
    h.append(f'<div class="ctl"><span class="ctt">methods:</span><span class="mck" id="ev-{wid}-m">{chips}</span>'
             f'<button type="button" class="mbtn" data-sel="all">all</button>'
             f'<button type="button" class="mbtn" data-sel="none">none</button></div>')
    h.append(f'<div class="ctl">{lsel}</div><div id="ev-{wid}-plot" class="fig"></div>')
    if note: h.append(f'<p class="muted figcap">{note}</p>')
    h.append('</div>')
    return "".join(h)


EV_JS = r"""
function evSwatch(col, dash, mk){
  const w=30,h=12,cy=h/2,cx=w/2,r=3;
  let s='<svg width="'+w+'" height="'+h+'" viewBox="0 0 '+w+' '+h+'" style="vertical-align:middle;flex:none">';
  s+='<line x1="1" y1="'+cy+'" x2="'+(w-1)+'" y2="'+cy+'" style="stroke:'+col+'" stroke-width="2.2"'+(dash?' stroke-dasharray="'+dash+'"':'')+'/>';
  if(mk==='s') s+='<rect x="'+(cx-r)+'" y="'+(cy-r)+'" width="'+(2*r)+'" height="'+(2*r)+'" style="fill:'+col+'"/>';
  else if(mk==='t') s+='<polygon points="'+cx+','+(cy-1.3*r)+' '+(cx-1.2*r)+','+(cy+r)+' '+(cx+1.2*r)+','+(cy+r)+'" style="fill:'+col+'"/>';
  else if(mk==='d') s+='<polygon points="'+cx+','+(cy-r)+' '+(cx+r)+','+cy+' '+cx+','+(cy+r)+' '+(cx-r)+','+cy+'" style="fill:'+col+'"/>';
  else s+='<circle cx="'+cx+'" cy="'+cy+'" r="'+r+'" style="fill:'+col+'"/>';
  return s+'</svg>';
}

function evDraw(wid){
  const d = EVD[wid]; if(!d) return;
  const lsel = document.getElementById('ev-'+wid+'-l');
  const L = lsel ? lsel.value : null;
  const on = Array.from(document.querySelectorAll('#ev-'+wid+'-m input:checked')).map(e=>e.dataset.m);
  const gs = d.gammas.map(Number);
  let series = [];
  for (const m of on){
    if (d.surface && d.surface[m]){
      const ys = d.gammas.map(g => { const c = d.surface[m][g]; return (c && L!==null && c[L]!==undefined) ? c[L] : NaN; });
      series.push({lab:m, col:(MCJS[m]||'#7f7f7f'), ys:ys, dash:(DASHJS[m]||''), mk:(MARKJS[m]||'c')});
    } else if (d.extra && d.extra[m]){
      series.push({lab:m, col:(MCJS[m]||'#7f7f7f'), ys:d.gammas.map(g=>{const v=d.extra[m][g]; return v===undefined?NaN:v;}),
                   dash:(DASHJS[m]||''), mk:(MARKJS[m]||'c')});
    }
  }
  const hl = d.hlines || {};
  let vals = [];
  series.forEach(s=>s.ys.forEach(v=>{ if(v===v) vals.push(v); }));
  Object.values(hl).forEach(v=>{ if(v===v) vals.push(v); });
  if(!vals.length){ document.getElementById('ev-'+wid+'-plot').innerHTML='<p class="muted">no series selected</p>'; return; }
  const W=720,H=360,pL=58,pR=16,pT=24,pB=44;
  let ylo=Math.min(...vals), yhi=Math.max(...vals); const pad=0.08*(yhi-ylo+1e-9); ylo-=pad; yhi+=pad;
  const x0=Math.min(...gs), x1=Math.max(...gs);
  const X=v=>pL+(v-x0)/(x1-x0+1e-12)*(W-pL-pR);
  const Y=v=>H-pB-(v-ylo)/(yhi-ylo+1e-12)*(H-pT-pB);
  let s='<svg viewBox="0 0 '+W+' '+H+'" class="chart">';
  for(let t=0;t<=4;t++){ const yv=ylo+(yhi-ylo)*t/4;
    s+='<line x1="'+pL+'" y1="'+Y(yv).toFixed(1)+'" x2="'+(W-pR)+'" y2="'+Y(yv).toFixed(1)+'" class="grid"/>';
    s+='<text x="'+(pL-6)+'" y="'+(Y(yv)+4).toFixed(1)+'" class="tk" text-anchor="end">'+yv.toFixed(2)+'</text>'; }
  for(const g of gs){ s+='<text x="'+X(g).toFixed(1)+'" y="'+(H-pB+16)+'" class="tk" text-anchor="middle">'+g+'</text>'; }
  for(const [lab,v] of Object.entries(hl)){
    if(v!==v) continue;
    s+='<line x1="'+pL+'" y1="'+Y(v).toFixed(1)+'" x2="'+(W-pR)+'" y2="'+Y(v).toFixed(1)+'" style="stroke:#888" stroke-dasharray="4 4"/>';
    s+='<text x="'+(W-pR-4)+'" y="'+(Y(v)-4).toFixed(1)+'" class="tk" text-anchor="end">'+lab+'</text>'; }
  if(d.gstar!==undefined && d.gstar!==null){
    s+='<line x1="'+X(d.gstar).toFixed(1)+'" y1="'+pT+'" x2="'+X(d.gstar).toFixed(1)+'" y2="'+(H-pB)+'" style="stroke:#0a7d33" stroke-dasharray="3 3"/>';
    s+='<text x="'+(X(d.gstar)+4).toFixed(1)+'" y="'+(pT+12)+'" class="tk" style="fill:#0a7d33">matched G*</text>'; }
  for(const se of series){
    let pts=[];
    for(let i=0;i<gs.length;i++) if(se.ys[i]===se.ys[i]) pts.push(X(gs[i]).toFixed(1)+','+Y(se.ys[i]).toFixed(1));
    if(pts.length) s+='<polyline points="'+pts.join(' ')+'" fill="none" style="stroke:'+se.col+'" stroke-width="2.2"'+(se.dash?' stroke-dasharray="'+se.dash+'"':'')+'/>';
    for(let i=0;i<gs.length;i++) if(se.ys[i]===se.ys[i]) s+=pwMark(X(gs[i]),Y(se.ys[i]),se.col,se.mk,2.6);
  }
  s+='<line x1="'+pL+'" y1="'+(H-pB)+'" x2="'+(W-pR)+'" y2="'+(H-pB)+'" class="ax"/>';
  s+='<line x1="'+pL+'" y1="'+pT+'" x2="'+pL+'" y2="'+(H-pB)+'" class="ax"/>';
  s+='<text x="'+((pL+W-pR)/2)+'" y="'+(H-8)+'" class="al" text-anchor="middle">Gamma</text>';
  s+='<text x="14" y="'+((pT+H-pB)/2)+'" class="al" text-anchor="middle" transform="rotate(-90 14 '+((pT+H-pB)/2)+')">test E[Y]</text>';
  s+='</svg>';
  const leg='<div class="leg">'+series.map(se=>'<span class="li">'+evSwatch(se.col,se.dash||'',se.mk||'c')+se.lab+'</span>').join('')+'</div>';
  document.getElementById('ev-'+wid+'-plot').innerHTML=s+leg;
}
document.addEventListener('change',e=>{const w=e.target.closest('.evw'); if(w) evDraw(w.id.slice(3));});
document.addEventListener('click',e=>{
  const b=e.target.closest('.mbtn'); if(!b) return;
  const w=b.closest('.evw'); if(!w) return;
  w.querySelectorAll('.mck input').forEach(i=>{i.checked=(b.dataset.sel==='all');});
  evDraw(w.id.slice(3));
});
for (const k of Object.keys(typeof EVD!=='undefined'?EVD:{})) evDraw(k);
"""
