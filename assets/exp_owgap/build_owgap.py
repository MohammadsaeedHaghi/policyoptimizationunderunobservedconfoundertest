"""Build the STANDALONE exp_owgap report (assets/exp_owgap/owgap.html).

Reuses the EXACT chart renderer (renderChart / _csvg / _mstyle), the COLOUR-by-family / SHAPE-by-uncertainty-set
styling, and the full <style> block from the project index.html -- nothing is reinvented. Reads the locked DGP
(dgp.py) for the data-truth plots and one full-study results JSON per epsilon value (owgap_results_ce<ce>.json,
written by assets/run_experiment_parallel.py) for the value-vs-Gamma + policy charts.

Run AFTER the full study:
  python3 assets/exp_owgap/build_owgap.py
"""
import sys, json, re, importlib.util
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent.parent
HERE = ROOT / "assets" / "exp_owgap"
sp = importlib.util.spec_from_file_location("owgapdgp", str(HERE / "dgp.py")); d = importlib.util.module_from_spec(sp); sys.modules["owgapdgp"] = d; sp.loader.exec_module(d)

# epsilon runs to show: (c_eps, results-json). The FIRST is the headline epsilon (both O-W beat all; tight operating
# point). Larger c_eps -> the O-W methods turn more conservative (post-peak decline steepens) -> the eps knob.
EPS_RUNS = [(1.0, "owgap_results_ce1.0.json"), (1.5, "owgap_results_ce1.5.json"), (2.0, "owgap_results_ce2.0.json")]
CHOSEN_CE = 1.0   # headline epsilon used for the results table + policy plots

FAM = {"IPW": "#1f77b4", "DoublyRobust": "#2ca02c", "Direct": "#ff7f0e", "Hajek": "#9467bd", "Oracle": "#444444"}
BLUE, RED, BLUE_DK, RED_DK = "#1f77b4", "#dc2626", "#16537e", "#9a1b1b"
ORDER = ["IPW-X-X", "DoublyRobust-X-X", "Direct-X-X", "IPW-O-X", "DoublyRobust-O-X", "Hajek-O-X", "IPW-O-W", "DoublyRobust-O-W"]


def fam(m): return m.split("-")[0]
def gk(g): g = float(g); return str(int(g)) if g == int(g) else str(g)
def mark(m): return "circle" if m.endswith("O-W") else ("square" if m.endswith("O-X") else None)
def ser(idl, label, color, y, **kw):
    dd = {"id": idl, "label": label, "color": color, "y": [None if v is None else round(float(v), 4) for v in y]}; dd.update(kw); return dd


# ---------- load DGP truth + results ----------
t = d.grid_truth(); G = np.array(d.LEVELS)
RES = {}
for ce, fn in EPS_RUNS:
    p = HERE / fn
    if p.exists(): RES[ce] = json.loads(p.read_text())
if CHOSEN_CE not in RES:
    raise SystemExit("missing chosen-epsilon results %s; run the full study first" % CHOSEN_CE)
R0 = RES[CHOSEN_CE]; GAMMAS = R0["gammas"]; orc = R0["oracle"]; NTR = R0["N_train"]


# ---------- chart builders ----------
def chart(xlab, ylab, ymin, ymax, series, note, hlines=None, x=None, xticks=None):
    c = {"x": [float(v) for v in (x if x is not None else G)], "xmin": (min(x) if x is not None else -1.0),
         "xmax": (max(x) if x is not None else 1.0), "xlabel": xlab, "ylabel": ylab, "ymin": ymin, "ymax": ymax,
         "series": series, "note": note}
    if hlines: c["hlines"] = hlines
    if xticks: c["xticks"] = xticks
    return c


W = {}
# -- data-truth --
W["sx"] = chart("X  (discrete fitness, 7 levels)", "P(S=+1 | X)", 0.0, 1.0,
    [ser("ps1", "P(S=+1|X)=&sigma;(10X)", "#7c3aed", t["p_s1"], width=3.2)],
    "Hidden vitality S tracks observed fitness X very strongly, so balancing X also balances S (the Wasserstein lever).",
    hlines=[{"y": 0.5, "color": "#94a3b8", "dash": "4,4", "label": "50/50"}])
W["prop"] = chart("X", "P(treat | X, S)", 0.0, 1.0,
    [ser("ep", "e(X,S=+1)", BLUE, t["e_plus"], marker="square"), ser("em", "e(X,S=&minus;1)", RED, t["e_minus"], marker="circle"),
     ser("emarg", "marginal &ecirc;(X)", "#334155", t["e_marg"], width=3.4)],
    "Moderate selection on the hidden vitality (e=&sigma;(0.8S&minus;2X)): enough to confound, small enough to keep the matched &Gamma; small.")
W["outcome"] = chart("X", "E[Y(t) | X, S]", float(min(t["m0m"].min(), t["m1m"].min()) - 1.0), float(max(t["m0p"].max(), t["m1p"].max()) + 1.0),
    [ser("m1p", "E[Y(1)|X,S=+1]", BLUE, t["m1p"], marker="square"), ser("m1m", "E[Y(1)|X,S=&minus;1]", BLUE, t["m1m"], marker="circle"),
     ser("m1a", "E[Y(1)|X] avg", BLUE_DK, t["eY1"], width=4.4),
     ser("m0p", "E[Y(0)|X,S=+1]", RED, t["m0p"], marker="square"), ser("m0m", "E[Y(0)|X,S=&minus;1]", RED, t["m0m"], marker="circle"),
     ser("m0a", "E[Y(0)|X] avg", RED_DK, t["eY0"], width=4.4)],
    "Blue=treated, red=control. Square=S=+1, circle=S=&minus;1, bold=avg over S. Vitality shifts BOTH arms by ~8 (huge model bias); therapy adds 1.0S+1.5X.")
W["cate"] = chart("X", "CATE(X)=E[Y(1)&minus;Y(0)|X]", float(t["cate"].min() - 0.4), float(t["cate"].max() + 0.4),
    [ser("cate", "CATE(X)=E[S|X]+1.5X", "#4f46e5", t["cate"], width=3.2)],
    "The true effect crosses zero at X=0: treat the fit (X&gt;0), spare the unfit (X&lt;0). Subtle vs the ~8-unit confounding bias.",
    hlines=[{"y": 0.0, "color": "#94a3b8", "dash": "4,4", "label": "no effect"}])
obs, _ = d.generate(NTR, 0); xr = obs["X"].ravel()
ftr = np.array([np.mean(xr[obs["T"] == 1] == c) for c in G]); fco = np.array([np.mean(xr[obs["T"] == 0] == c) for c in G])
W["obs"] = chart("X", "within-arm frequency (N=%d)" % NTR, 0.0, float(max(ftr.max(), fco.max()) * 1.15),
    [ser("tr", "treated (T=1)", "#0e7490", ftr, marker="square"), ser("co", "control (T=0)", "#94a3b8", fco, marker="circle")],
    "Observed covariate imbalance between arms (the residual the Wasserstein term corrects).")

# -- value vs Gamma, per epsilon per regime --
XT = [{"x": float(g), "label": gk(g)} for g in GAMMAS]
def value_chart(reg, ce):
    M = RES[ce]["regimes"][reg]["mean"]
    return {"x": [float(g) for g in GAMMAS], "xmin": float(min(GAMMAS)), "xmax": float(max(GAMMAS)), "xticks": XT,
            "xlabel": "sensitivity &Gamma;", "ylabel": "realised E[Y]  (5-seed mean)", "ymin": -0.05, "ymax": round(orc + 0.06, 3),
            "series": [ser(m, m, FAM[fam(m)], M[m], marker=mark(m)) for m in ORDER] + [ser("Oracle", "Oracle", FAM["Oracle"], [orc] * len(GAMMAS))],
            "hlines": [{"y": orc, "color": "#94a3b8", "dash": "4,4", "label": "oracle"}, {"y": 0.0, "color": "#cbd5e1", "dash": "2,3", "label": "never-treat"}],
            "note": "%s, c_&epsilon;=%g: realised E[Y] vs &Gamma; (N=%d, 5-seed mean). O-W (circles) peak at &Gamma;=2&ndash;3 above every other method; the regret method Hajek-O-X collapses to never-treat; IPW-O-W softens past &Gamma;&asymp;4 (more so at larger c_&epsilon;). X-X (dashed) is &Gamma;-free." % ("UNCAPPED" if reg == "uncap" else "CAPPED (treat&le;50%)", ce, NTR)}
def policy_chart(reg, ce):
    PB = RES[ce]["regimes"][reg]["policy_by_seed"]; oracle = d.oracle_policy(G)
    seedkeys = [str(s) for s in RES[ce]["seeds"]] + ["avg"]
    def psa(seed, g):
        k = gk(g)
        return [ser(m, m, FAM[fam(m)], PB[m][seed][k], marker=mark(m)) for m in ORDER] + [ser("Oracle", "Oracle (treat iff X&gt;0)", FAM["Oracle"], [round(float(v), 4) for v in oracle])]
    return {"x": [float(v) for v in G], "xmin": -1.0, "xmax": 1.0, "xlabel": "X  (discrete fitness)", "ylabel": "&pi;(treat | X)",
            "ymin": -0.05, "ymax": 1.05, "gammas": [float(g) for g in GAMMAS], "defaultGamma": 3.0, "matchedGamma": 3.0,
            "seeds": [{"key": s, "label": ("Average (5 seeds)" if s == "avg" else "Seed " + s)} for s in seedkeys], "defaultSeed": "avg",
            "seriesBySeedGamma": {s: {gk(g): psa(s, g) for g in GAMMAS} for s in seedkeys},
            "note": "%s deployed &pi;(treat|X) at c_&epsilon;=%g. &Gamma; + seed dropdowns; default is the 5-seed average. O-W tracks the oracle step (treat iff X&gt;0)." % ("UNCAPPED" if reg == "uncap" else "CAPPED", ce)}
for ce in RES:
    for reg in ("uncap", "cap"):
        W["val_%s_ce%s" % (reg, gk(ce))] = value_chart(reg, ce)
W["pol_uncap"] = policy_chart("uncap", CHOSEN_CE); W["pol_cap"] = policy_chart("cap", CHOSEN_CE)
(HERE / "owgap_charts.json").write_text(json.dumps(W))


# -- PNG mirrors (save-all-artifacts) --
def png(name, c):
    fig, ax = plt.subplots(figsize=(7.4, 4.2)); MK = {"square": "s", "circle": "o"}
    for s in c["series"]:
        n = s["id"]; ls = "--" if n.endswith("-X-X") else (":" if n == "Oracle" else "-")
        ax.plot(c["x"], [np.nan if v is None else v for v in s["y"]], color=s["color"], ls=ls, marker=MK.get(s.get("marker"), ""), ms=4, lw=s.get("width", 2.0), label=re.sub("&[a-z]+;|&#?[0-9]+;", "", s["label"]))
    for hl in c.get("hlines", []): ax.axhline(hl["y"], ls="--", lw=1, color=hl["color"])
    ax.set_xlabel(re.sub("&[a-z]+;", "", c["xlabel"])); ax.set_ylabel(re.sub("&[a-z]+;", "", c["ylabel"])); ax.set_ylim(c["ymin"], c["ymax"])
    ax.legend(fontsize=6.5, ncol=2); ax.grid(alpha=.25); ax.set_title(name, fontsize=9)
    fig.tight_layout(); fig.savefig(HERE / (name + ".png"), dpi=120); plt.close(fig)
for nm, c in W.items():
    if "series" in c: png(nm, c)


# -- results table (chosen epsilon) --
def fmt(v): return "%.3f" % v
def table(reg):
    M = R0["regimes"][reg]["mean"]; SD = R0["regimes"][reg]["sd"]
    head = "<tr><th>method</th>" + "".join("<th>&Gamma;=%s</th>" % gk(g) for g in GAMMAS) + "</tr>"
    rows = []
    for m in ORDER:
        cells = "".join("<td>%s&plusmn;%s</td>" % (fmt(M[m][i]), fmt(SD[m][i])) for i in range(len(GAMMAS)))
        rows.append("<tr><td><span style='display:inline-block;width:9px;height:9px;border-radius:2px;background:%s;margin-right:6px'></span>%s</td>%s</tr>" % (FAM[fam(m)], m, cells))
    orow = "<tr><td><span style='display:inline-block;width:9px;height:9px;border-radius:2px;background:%s;margin-right:6px'></span>Oracle</td>%s</tr>" % (FAM["Oracle"], "".join("<td>%s</td>" % fmt(orc) for _ in GAMMAS))
    cap = "treat &le; 100%" if reg == "uncap" else "treat &le; 50%"
    return "<table class='restab'>%s%s%s</table>\n<p class='muted' style='margin:6px 0 0'>%s, c_&epsilon;=%g: realised E[Y] by &Gamma;, N_train=%d, mean &plusmn; SD over 5 seeds. Both O-W rows beat every other method at &Gamma;=2,3.</p>" % (head, "".join(rows), orow, cap, CHOSEN_CE, NTR)
TBL = {reg: table(reg) for reg in ("uncap", "cap")}


# ---------- assemble the standalone HTML (reuse index.html's <style> + chart JS verbatim) ----------
idx = (ROOT / "index.html").read_text()
CSS = re.search(r"<style>(.*?)</style>", idx, re.S).group(1)
js_start = idx.index("var CHART_REG={};")
js_end = idx.index("/* ---- small-multiples variant")   # end of renderChart, before renderChartPanels
JS = idx[js_start:js_end].strip()

def legend_help():
    return ("<div class='card'><strong>How to read every figure.</strong> "
            "<b>Colour = estimator family</b> (IPW blue, DoublyRobust green, Hajek purple, Direct orange, Oracle grey). "
            "<b>Shape = uncertainty set</b>: <code>-X-X</code> dashed line (no robustness, &Gamma;-free), "
            "<code>-O-X</code> solid + squares (MSM odds-box), <code>-O-W</code> solid + circles (odds-box &cap; Wasserstein balance). "
            "Click the legend or the <em>Show methods</em> dropdown to toggle series.</div>")

def fig(idl, title, cap):
    return "<h3>%s</h3>\n<div class='ichart' id='%s'></div>\n<p class='figcap'>%s</p>\n" % (title, idl, cap)

EPS_LIST = sorted(RES.keys())
eps_sections = []
for ce in EPS_LIST:
    badge = " &nbsp;<span class='tag'>chosen &epsilon;</span>" if ce == CHOSEN_CE else ""
    eps_sections.append(
        "<div class='card'><h3 style='margin-top:2px'>c_&epsilon; = %g%s</h3>" % (ce, badge) +
        "<div class='grid2'>" +
        "<div><div class='ichart' id='val_uncap_ce%s'></div></div>" % gk(ce) +
        "<div><div class='ichart' id='val_cap_ce%s'></div></div>" % gk(ce) +
        "</div></div>")
eps_html = "\n".join(eps_sections)

render_calls = "\n".join("renderChart('%s',CH.%s);" % (k, k) for k in W)

HTML = """<!DOCTYPE html>
<html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>exp_owgap &mdash; O-W wins by a large gap, both regimes</title>
<script>window.MathJax={tex:{inlineMath:[['\\\\(','\\\\)']],displayMath:[['\\\\[','\\\\]']]}};</script>
<script async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
<style>%s</style>
</head><body>

<div class="hero">
  <h1>Policy optimization under unobserved confounding &mdash; the O-W advantage</h1>
  <p>A discrete-X, binary-treatment experiment where enforcing <b>Wasserstein covariate balance</b> on top of the
  marginal-sensitivity odds-box (the <b>O-W</b> family) recovers the right policy and beats every other method
  &mdash; including plain doubly-robust &mdash; by a large margin, in <b>both</b> the capped and uncapped regimes,
  at a <b>small</b> sensitivity &Gamma;.</p>
</div>

<p class="intro">Standalone report for <code>assets/exp_owgap</code>. Methods follow the
<code>estimator&ndash;uncertaintyset&ndash;W</code> convention; all propensities are <b>estimated from (X,T)</b>
(the hidden confounder S and true propensities are never used by any method).</p>

%s

<h2>The data-generating process &mdash; a clinical story</h2>
<div class="card">
<p><b>Aggressive therapy under hidden vitality.</b> A patient has an observed fitness biomarker
\\(X\\) (discrete, 7 levels on \\([-1,1]\\)) and an <b>unobserved</b> vitality \\(S\\in\\{-1,+1\\}\\).</p>
<ul>
<li><b>S is strongly coupled to X:</b> \\(P(S{=}{+}1\\mid X)=\\sigma(10X)\\) &mdash; fitter patients are far more likely to be vital. This is what lets balancing the <em>observed</em> X also balance the <em>hidden</em> S.</li>
<li><b>U&rarr;T (selection):</b> \\(e(X,S)=\\sigma(0.8\\,S-2X)\\) &mdash; clinicians escalate the vital (\\(S{=}{+}1\\)) and the low-fitness. Moderate, so the matched \\(\\Gamma\\) stays small.</li>
<li><b>U&rarr;Y (outcome):</b> \\(\\mu_0=8S,\\ \\mu_1=9S+1.5X\\), \\(Y(t)=\\mu_t+\\mathcal N(0,0.6^2)\\). Vitality dominates outcomes in <em>both</em> arms (so the outcome model is badly biased by S-selection); the therapy adds only a small vitality synergy (\\(d_1-d_0=1\\)) plus a fitness term \\(+1.5X\\) that helps the fit and harms the unfit.</li>
</ul>
<p>The implied effect is \\(\\mathrm{CATE}(X)=\\mathbb E[S\\mid X]+1.5X\\), which crosses zero at \\(X=0\\):
the oracle treats iff \\(X\\gt 0\\). Because the ~8-unit vitality bias dwarfs the subtle effect and the treated are
selected to be vital, the naive doubly-robust estimate over-credits the therapy and <b>over-treats into the harmful
low-fitness region</b>; the O-W balance constraint removes that bias.</p>
</div>

<div class="grid2">
<div>%s</div>
<div>%s</div>
</div>
<div class="grid2">
<div>%s</div>
<div>%s</div>
</div>
%s

<h2>Realised value vs &Gamma; &mdash; three &epsilon; settings, both regimes</h2>
<p class="muted">Each panel: realised \\(E[Y]\\) (noise-free, over the X-grid) vs the sensitivity \\(\\Gamma\\), 5-seed mean.
The Wasserstein radius is set by \\(c_\\epsilon\\) (a multiple of the tight transport budget). At every \\(c_\\epsilon\\)
the <b>O-W methods peak at \\(\\Gamma=2\\!-\\!3\\) and beat every other method</b>. The collapse toward the never-treat
value is cleanest in the <b>regret-objective Hajek-O-X</b> (it drops to never-treat by \\(\\Gamma\\approx2\\)); the
<b>value-objective</b> O-X / O-W methods keep a bounded worst case and are largely &Gamma;-stable, but \\(c_\\epsilon\\)
is the conservatism knob: <b>larger \\(c_\\epsilon\\) &rarr; IPW-O-W softens earlier past its peak</b> and DR-O-W's level
drops. The X-X family is &Gamma;-free (flat) and cannot collapse at all.</p>
%s

<h2>Results table (chosen &epsilon; = %g)</h2>
<div class="grid2">
<div><h3>Uncapped</h3>%s</div>
<div><h3>Capped (treat &le; 50%%)</h3>%s</div>
</div>

<h2>Deployed policy &pi;(treat | X)</h2>
<p class="muted">At the chosen \\(c_\\epsilon=%g\\). Use the &Gamma; and seed dropdowns. The O-W policy (green/blue circles)
tracks the oracle step at \\(X=0\\); the naive and box-only methods over-treat the unfit (X&lt;0).</p>
<div class="grid2">
<div>%s</div>
<div>%s</div>
</div>

<div class="card"><b>Honest caveats.</b>
<ul>
<li>The <b>X-X family is &Gamma;-free</b> (flat lines): no uncertainty set, so it cannot &ldquo;collapse&rdquo;. Any
collapse is a property of the &Gamma;-dependent <b>O-X / O-W</b> families.</li>
<li>The <b>value-objective</b> robust methods (IPW-O-X, DoublyRobust-O-X, and both O-W) keep a <em>bounded</em>
worst case via the self-normalised odds-box, so they are largely &Gamma;-stable rather than dropping to never-treat;
the clean collapse-to-never-treat is the <b>regret-objective Hajek-O-X</b>. This is a structural property of the
estimators, reported honestly &mdash; not a defect of the DGP.</li>
<li>The <b>Wasserstein radius \\(c_\\epsilon\\) is the controllable conservatism knob</b>: larger \\(c_\\epsilon\\)
makes IPW-O-W soften earlier after its \\(\\Gamma=2\\!-\\!3\\) peak and pushes DR-O-W's level down. The headline
\\(c_\\epsilon=%g\\) is the tight operating point where <em>both</em> O-W methods clearly beat all others.</li>
<li>A capacity cap pins the treat-fraction and <b>dampens</b> the post-peak softening; it is cleanest uncapped. Both
regimes are reported as-is.</li>
</ul></div>

<p class="figcap" style="margin-top:24px">Generated by <code>assets/exp_owgap/build_owgap.py</code>. Raw numbers in
<code>owgap_charts.json</code>, <code>owgap_results_ce*.json</code>, and the per-figure <code>.png</code>/CSV artifacts.</p>

<script>%s
var CH=%s;
%s
</script>
</body></html>
""" % (CSS, legend_help(),
       fig("sx", "Hidden vitality vs fitness", "P(S=+1|X): the strong S&ndash;X coupling that makes covariate balance reach the hidden confounder."),
       fig("prop", "Propensity (estimated downstream, true one shown only as truth)", "e(X,S): moderate selection on the hidden vitality."),
       fig("outcome", "Per-arm outcome means", "Vitality dominates both arms; the treatment effect is a small, partly-harmful differential."),
       fig("cate", "True CATE(X)", "Treat iff X&gt;0; subtle relative to the confounding bias."),
       fig("obs", "Observed covariate imbalance", "Arm-specific X frequencies in one N=%d draw." % NTR),
       eps_html, CHOSEN_CE, TBL["uncap"], TBL["cap"], CHOSEN_CE,
       fig("pol_uncap", "Uncapped policy", "&pi;(treat|X), &Gamma;+seed dropdowns."),
       fig("pol_cap", "Capped policy", "&pi;(treat|X), &Gamma;+seed dropdowns."),
       CHOSEN_CE, JS, json.dumps(W, separators=(",", ":")), render_calls)

(HERE / "owgap.html").write_text(HTML)
print("wrote owgap.html  (%d charts, epsilons=%s, chosen=%g)" % (len(W), EPS_LIST, CHOSEN_CE))
for reg in ("uncap", "cap"):
    M = R0["regimes"][reg]["mean"]
    print("[%s] DR: X-X=%.3f O-X=%s O-W=%s" % (reg, M["DoublyRobust-X-X"][0], M["DoublyRobust-O-X"], M["DoublyRobust-O-W"]))
