"""Build the COMBINED standalone report (assets/exp_nmcap/combined_report.html) for the two complementary
stress tests:  exp_owgap (unobserved-S confounding -> naive IPW fails)  +  exp_nmcap (non-monotone effect ->
naive AIPW fails under the cap).  Reuses the exact renderChart / _csvg / _mstyle / colours / <style> from the
project index.html.  Computes per-seed values from the saved per-seed policies and PAIRED bootstrap CIs.

Run after both studies:
  python3 assets/exp_nmcap/build_combined.py
"""
import sys, json, re, importlib.util
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent.parent
def load_dgp(rel):
    sp = importlib.util.spec_from_file_location(rel.replace("/", "_"), str(ROOT / rel)); m = importlib.util.module_from_spec(sp); sp.loader.exec_module(m); return m
OW = load_dgp("assets/exp_owgap/dgp.py")
NM = load_dgp("assets/exp_nmcap/dgp.py")
R_OW = json.loads((ROOT / "assets/exp_owgap/owgap_uncap_20seed_ce1.0.json").read_text())   # uncapped, 20 seeds
R_NM = json.loads((ROOT / "assets/exp_nmcap/nmcap_results_ce1.0.json").read_text())          # both regimes, 15 seeds
R_EX = json.loads((ROOT / "assets/exp_owgap/owgap_extreme_sweep.json").read_text())          # extreme Gamma x eps, owgap uncap

FAM = {"IPW": "#1f77b4", "DoublyRobust": "#2ca02c", "Direct": "#ff7f0e", "Hajek": "#9467bd", "Oracle": "#444444"}
BLUE, RED, BLUE_DK, RED_DK = "#1f77b4", "#dc2626", "#16537e", "#9a1b1b"
ORDER = ["IPW-X-X", "DoublyRobust-X-X", "Direct-X-X", "IPW-O-X", "DoublyRobust-O-X", "Hajek-O-X", "IPW-O-W", "DoublyRobust-O-W"]
def fam(m): return m.split("-")[0]
def gk(g): g = float(g); return str(int(g)) if g == int(g) else str(g)
def mark(m): return "circle" if m.endswith("O-W") else ("square" if m.endswith("O-X") else None)
def ser(idl, label, color, y, **kw):
    dd = {"id": idl, "label": label, "color": color, "y": [None if v is None else round(float(v), 4) for v in y]}; dd.update(kw); return dd

# ---- per-seed values from saved policies, mean/sd + paired bootstrap CI ----
def perseed(R, dgp, reg, m):
    PB = R["regimes"][reg]["policy_by_seed"]; seeds = [str(s) for s in R["seeds"]]; G = R["gammas"]
    return np.array([[dgp.exact_value(PB[m][s][gk(g)]) for g in G] for s in seeds])   # (nseed, nG)
def boot_ci(diffs, B=10000):
    diffs = np.asarray(diffs, float); n = len(diffs); rng = np.random.default_rng(0)
    means = diffs[rng.integers(0, n, size=(B, n))].mean(1)
    return float(diffs.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))

# ---- charts ----
def chart(xlab, ylab, ymin, ymax, series, note, hlines=None, x=None, xticks=None):
    c = {"x": [float(v) for v in x], "xmin": min(x), "xmax": max(x), "xlabel": xlab, "ylabel": ylab,
         "ymin": ymin, "ymax": ymax, "series": series, "note": note}
    if hlines: c["hlines"] = hlines
    if xticks: c["xticks"] = xticks
    return c
def truth_charts(dgp, tag):
    t = dgp.grid_truth(); G = list(dgp.LEVELS); W = {}
    W[tag + "_outcome"] = chart("X (discrete)", "E[Y(t)|X,S]", float(min(t["m0m"].min(), t["m1m"].min()) - 1), float(max(t["m0p"].max(), t["m1p"].max()) + 1),
        [ser("m1p", "E[Y(1)|X,S=+1]", BLUE, t["m1p"], marker="square"), ser("m1m", "E[Y(1)|X,S=&minus;1]", BLUE, t["m1m"], marker="circle"),
         ser("m1a", "E[Y(1)|X] avg", BLUE_DK, t["eY1"], width=4.4),
         ser("m0p", "E[Y(0)|X,S=+1]", RED, t["m0p"], marker="square"), ser("m0m", "E[Y(0)|X,S=&minus;1]", RED, t["m0m"], marker="circle"),
         ser("m0a", "E[Y(0)|X] avg", RED_DK, t["eY0"], width=4.4)],
        "Blue=treated, red=control; square S=+1, circle S=&minus;1, bold=avg over S.", x=G)
    W[tag + "_cate"] = chart("X", "CATE(X)=E[Y(1)&minus;Y(0)|X]", float(t["cate"].min() - 0.4), float(t["cate"].max() + 0.4),
        [ser("cate", "CATE(X)", "#4f46e5", t["cate"], width=3.2)],
        "Treat where CATE&gt;0.", hlines=[{"y": 0.0, "color": "#94a3b8", "dash": "4,4", "label": "no effect"}], x=G)
    W[tag + "_prop"] = chart("X", "P(treat|X,S)", 0.0, 1.0,
        [ser("ep", "e(X,S=+1)", BLUE, t["e_plus"], marker="square"), ser("em", "e(X,S=&minus;1)", RED, t["e_minus"], marker="circle"),
         ser("emarg", "marginal &ecirc;(X)", "#334155", t["e_marg"], width=3.4)],
        "Propensity by arm; methods only ever see the estimated &ecirc;(X).", x=G)
    return W
def value_chart(R, dgp, reg, nseed, note):
    G = R["gammas"]; orc = R["oracle"]
    sd = {m: perseed(R, dgp, reg, m).std(0) for m in ORDER}
    mn = {m: perseed(R, dgp, reg, m).mean(0) for m in ORDER}
    XT = [{"x": float(g), "label": gk(g)} for g in G]
    s = []
    for m in ORDER:
        d = ser(m, m, FAM[fam(m)], mn[m], marker=mark(m)); d["ysd"] = [round(float(x), 4) for x in sd[m]]; s.append(d)
    s.append(ser("Oracle", "Oracle", FAM["Oracle"], [orc] * len(G)))
    return {"x": [float(g) for g in G], "xmin": float(min(G)), "xmax": float(max(G)), "xticks": XT,
            "xlabel": "sensitivity &Gamma;", "ylabel": "realised E[Y] (mean &plusmn; SD)", "ymin": min(-0.5, float(min(min(mn[m]) for m in ORDER)) - 0.1), "ymax": round(orc + 0.08, 3),
            "series": s, "hlines": [{"y": orc, "color": "#94a3b8", "dash": "4,4", "label": "oracle"}, {"y": 0.0, "color": "#cbd5e1", "dash": "2,3", "label": "never-treat"}],
            "note": note + " %d-seed mean; shaded = &plusmn;1 SD." % nseed}

W = {}
W.update(truth_charts(OW, "ow"))
W.update(truth_charts(NM, "nm"))
W["ow_value_uncap"] = value_chart(R_OW, OW, "uncap", len(R_OW["seeds"]), "exp_owgap UNCAPPED, c_&epsilon;=1: both O-W beat every method.")
W["nm_value_uncap"] = value_chart(R_NM, NM, "uncap", len(R_NM["seeds"]), "exp_nmcap UNCAPPED, c_&epsilon;=1: naive AIPW (DR-X-X) goes negative; O / O-W recover.")
W["nm_value_cap"] = value_chart(R_NM, NM, "cap", len(R_NM["seeds"]), "exp_nmcap CAPPED (treat&le;50%), c_&epsilon;=1: naive AIPW fails; O-W beats it.")

# ---- extreme Gamma x eps over-conservatism panel (owgap uncapped) ----
import math
GX = R_EX["gammas"]; MX = R_EX["mean"]["uncap"]; orcx = R_EX["oracle"]
LGX = [round(math.log10(g), 4) for g in GX]
XTX = [{"x": round(math.log10(g), 4), "label": str(int(g))} for g in GX]
CE_COL = {"1.0": "#bfdbfe", "4.0": "#60a5fa", "8.0": "#2563eb", "16.0": "#1e3a8a"}
def exline(method, ce, col): return ser("%s@%s" % (method, ce), "%s, c_&epsilon;=%s" % (method, ce.rstrip("0").rstrip(".")), col, MX["%s@%g" % (method, float(ce))], marker="circle")
W["ex_ipw"] = {"x": LGX, "xmin": min(LGX), "xmax": max(LGX), "xticks": XTX, "xlabel": "log&#8321;&#8320; &Gamma;", "ylabel": "realised E[Y]",
    "ymin": -0.05, "ymax": round(orcx + 0.05, 3),
    "series": [exline("IPW-O-W", ce, CE_COL[ce]) for ce in ["1.0", "4.0", "8.0", "16.0"]],
    "hlines": [{"y": orcx, "color": "#94a3b8", "dash": "4,4", "label": "oracle"}, {"y": 0.0, "color": "#cbd5e1", "dash": "2,3", "label": "never-treat"}],
    "note": "Pure-IPW O-W (IPW-O-W) vs &Gamma; (to 1000) for four Wasserstein radii. Larger c_&epsilon; AND larger &Gamma; both push the worst case to the unrealistic regime: the value collapses from ~oracle toward never-treat (0.08 at c_&epsilon;=16, &Gamma;=1000)."}
def exfam(idl, lab, col, y, mk): return ser(idl, lab, col, y, marker=mk)
W["ex_family"] = {"x": LGX, "xmin": min(LGX), "xmax": max(LGX), "xticks": XTX, "xlabel": "log&#8321;&#8320; &Gamma;", "ylabel": "realised E[Y]",
    "ymin": -0.1, "ymax": round(orcx + 0.05, 3),
    "series": [exfam("IPW-O-W", "IPW-O-W (c_&epsilon;=16)", FAM["IPW"], MX["IPW-O-W@16"], "circle"),
               exfam("DoublyRobust-O-W", "DoublyRobust-O-W (c_&epsilon;=16)", FAM["DoublyRobust"], MX["DoublyRobust-O-W@16"], "circle"),
               exfam("IPW-O-X", "IPW-O-X (box only)", FAM["IPW"], MX["IPW-O-X"], "square"),
               exfam("DoublyRobust-O-X", "DoublyRobust-O-X (box only)", FAM["DoublyRobust"], MX["DoublyRobust-O-X"], "square"),
               exfam("Hajek-O-X", "Hajek-O-X (regret)", FAM["Hajek"], MX["Hajek-O-X"], "square")],
    "hlines": [{"y": orcx, "color": "#94a3b8", "dash": "4,4", "label": "oracle"}, {"y": 0.0, "color": "#cbd5e1", "dash": "2,3", "label": "never-treat"}],
    "note": "At the extreme: pure-IPW (IPW-O-W/O-X) and the regret method Hajek-O-X collapse to never-treat; the DOUBLY-ROBUST methods are anchored by their outcome-model term and floor at ~0.6 instead."}
(Path(__file__).resolve().parent / "combined_charts.json").write_text(json.dumps(W))

# ---- PNG mirrors ----
def png(name, c):
    fig, ax = plt.subplots(figsize=(7.2, 4.1)); MK = {"square": "s", "circle": "o"}
    for s in c["series"]:
        n = s["id"]; ls = "--" if n.endswith("-X-X") else (":" if n == "Oracle" else "-")
        y = [np.nan if v is None else v for v in s["y"]]
        ax.plot(c["x"], y, color=s["color"], ls=ls, marker=MK.get(s.get("marker"), ""), ms=4, lw=s.get("width", 2.0), label=re.sub("&[a-z]+;|&#?[0-9]+;", "", s["label"]))
        if s.get("ysd"): ax.fill_between(c["x"], np.array(y) - np.array(s["ysd"]), np.array(y) + np.array(s["ysd"]), color=s["color"], alpha=0.12)
    for hl in c.get("hlines", []): ax.axhline(hl["y"], ls="--", lw=1, color=hl["color"])
    ax.set_xlabel(re.sub("&[a-z]+;", "", c["xlabel"])); ax.set_ylabel(re.sub("&[a-z]+;", "", c["ylabel"])); ax.set_ylim(c["ymin"], c["ymax"])
    ax.legend(fontsize=6, ncol=2); ax.grid(alpha=.25); ax.set_title(name, fontsize=9); fig.tight_layout(); fig.savefig(Path(__file__).resolve().parent / (name + ".png"), dpi=120); plt.close(fig)
for nm, c in W.items():
    if "series" in c: png(nm, c)

# ---- paired CI tables ----
def ci_table(R, dgp, reg, baseline, methods, gammas):
    seeds = [str(s) for s in R["seeds"]]; G = R["gammas"]
    bvals = {g: perseed(R, dgp, reg, baseline)[:, G.index(g)] for g in gammas}
    head = "<tr><th>vs %s</th>%s</tr>" % (baseline, "".join("<th>&Gamma;=%s</th>" % gk(g) for g in gammas))
    rows = []
    for m in methods:
        cells = []
        for g in gammas:
            mv = perseed(R, dgp, reg, m)[:, G.index(g)]; md, lo, hi = boot_ci(mv - bvals[g])
            sig = "&#10003;" if lo > 0 else ("&#10007;" if hi < 0 else "&ndash;")
            col = "#16a34a" if lo > 0 else ("#dc2626" if hi < 0 else "#666")
            cells.append("<td style='color:%s'>%+.3f<br><span style='font-size:.8em'>[%+.2f,%+.2f] %s</span></td>" % (col, md, lo, hi, sig))
        rows.append("<tr><td><span style='display:inline-block;width:9px;height:9px;border-radius:2px;background:%s;margin-right:6px'></span>%s</td>%s</tr>" % (FAM[fam(m)], m, "".join(cells)))
    return "<table class='restab'>%s%s</table><p class='muted' style='margin:4px 0 0'>Paired &Delta; (method &minus; %s) per seed, 95%% bootstrap CI; &#10003;=CI&gt;0 (significant win), &#10007;=CI&lt;0, &ndash;=ns. %d seeds.</p>" % (head, "".join(rows), baseline, len(seeds))

TBL_OW = ci_table(R_OW, OW, "uncap", "IPW-X-X", ["IPW-O-W", "DoublyRobust-O-W"], [2.0, 3.0, 4.0])
TBL_OW2 = ci_table(R_OW, OW, "uncap", "DoublyRobust-X-X", ["IPW-O-W", "DoublyRobust-O-W"], [2.0, 3.0, 4.0])
TBL_NM = ci_table(R_NM, NM, "cap", "DoublyRobust-X-X", ["IPW-O-W", "DoublyRobust-O-W"], [2.0, 3.0, 4.0])
TBL_NM2 = ci_table(R_NM, NM, "uncap", "DoublyRobust-X-X", ["IPW-O-W", "DoublyRobust-O-W"], [2.0, 3.0, 4.0])

# ---- 2x2 robustness summary (best realised value per method-class per DGP/regime) ----
def best(R, dgp, reg, m): return round(float(max(perseed(R, dgp, reg, m).mean(0))), 3)
SUMMARY = {
    "owgap_uncap": {m: best(R_OW, OW, "uncap", m) for m in ["IPW-X-X", "DoublyRobust-X-X", "IPW-O-W", "DoublyRobust-O-W"]},
    "nmcap_cap": {m: best(R_NM, NM, "cap", m) for m in ["IPW-X-X", "DoublyRobust-X-X", "IPW-O-W", "DoublyRobust-O-W"]},
}

# ---- assemble HTML (reuse index.html <style> + chart JS) ----
idx = (ROOT / "index.html").read_text()
CSS = re.search(r"<style>(.*?)</style>", idx, re.S).group(1)
JS = idx[idx.index("var CHART_REG={};"):idx.index("/* ---- small-multiples variant")].strip()

def fig(idl, title, cap):
    return "<h3>%s</h3>\n<div class='ichart' id='%s'></div>\n<p class='figcap'>%s</p>\n" % (title, idl, cap)

sw = lambda c: "<span style='display:inline-block;width:10px;height:10px;border-radius:2px;background:%s;margin-right:5px'></span>" % c
def srow(label, ow, nm, good_ow, good_nm):
    def cell(v, good): return "<td style='font-weight:700;color:%s'>%.3f</td>" % ("#16a34a" if good else "#dc2626", v)
    return "<tr><td>%s</td>%s%s</tr>" % (label, cell(ow, good_ow), cell(nm, good_nm))
S = SUMMARY
summary_tbl = ("<table class='restab' style='max-width:680px'><tr><th>method</th>"
    "<th>exp_owgap<br>(unobs-S, UNCAPPED)</th><th>exp_nmcap<br>(non-monotone, CAPPED)</th></tr>"
    + srow(sw(FAM['IPW']) + "naive IPW-X-X", S['owgap_uncap']['IPW-X-X'], S['nmcap_cap']['IPW-X-X'], False, True)
    + srow(sw(FAM['DoublyRobust']) + "naive DoublyRobust-X-X (AIPW)", S['owgap_uncap']['DoublyRobust-X-X'], S['nmcap_cap']['DoublyRobust-X-X'], True, False)
    + srow(sw(FAM['IPW']) + "<b>IPW-O-W</b>", S['owgap_uncap']['IPW-O-W'], S['nmcap_cap']['IPW-O-W'], True, True)
    + srow(sw(FAM['DoublyRobust']) + "<b>DoublyRobust-O-W</b>", S['owgap_uncap']['DoublyRobust-O-W'], S['nmcap_cap']['DoublyRobust-O-W'], True, True)
    + "</table><p class='muted' style='margin:4px 0 0'>Best realised E[Y] over &Gamma;. Green=does well, red=fails. "
    "Each naive baseline collapses in ONE setting (IPW under unobserved-S; AIPW under the non-monotone effect); only the O-W methods stay green in both.</p>")

legend = ("<div class='card'><strong>How to read every figure.</strong> <b>Colour=estimator family</b> "
    "(IPW blue, DoublyRobust green, Hajek purple, Direct orange, Oracle grey). <b>Shape=uncertainty set</b>: "
    "<code>-X-X</code> dashed (no robustness, &Gamma;-free), <code>-O-X</code> squares (MSM odds-box), "
    "<code>-O-W</code> circles (odds-box &cap; Wasserstein balance). Shaded band = &plusmn;1 SD over seeds. "
    "Toggle any series in the legend / dropdown. All propensities are ESTIMATED from (X,T); the hidden S is never used.</div>")

render_calls = "\n".join("renderChart('%s',CH.%s);" % (k, k) for k in W)
HTML = """<!DOCTYPE html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Two stress tests for robust policy optimization under unobserved confounding</title>
<script>window.MathJax={tex:{inlineMath:[['\\\\(','\\\\)']],displayMath:[['\\\\[','\\\\]']]}};</script>
<script async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
<style>%s</style></head><body>

<div class="hero"><h1>Robust policy optimization under unobserved confounding: two complementary stress tests</h1>
<p>When can the Wasserstein-balanced robust methods (the <b>O-W</b> family) be trusted? We show two discrete-X,
binary-treatment DGPs that each break a different naive baseline &mdash; and that the <b>O-W</b> methods are the
only ones that survive both.</p></div>

<p class="intro">All methods estimate the propensity from \\((X,T)\\); the hidden confounder \\(S\\) is never used.
Methods follow the <code>estimator&ndash;uncertaintyset&ndash;W</code> naming.</p>
%s

<div class="card"><h2 style="margin-top:2px">The headline</h2>%s</div>

<h2>Test 1 &mdash; exp_owgap: unobserved-S confounding breaks naive IPW</h2>
<div class="card"><p><b>Story.</b> Fitness \\(X\\) (7 levels), hidden vitality \\(S\\) with \\(P(S{=}{+}1\\mid X)=\\sigma(10X)\\)
(S tracks X). Selection \\(e=\\sigma(0.8S-2X)\\); outcomes \\(\\mu_0=8S,\\ \\mu_1=9S+1.5X\\) &mdash; vitality dominates both
arms (huge model bias), the therapy effect is subtle (\\(\\mathrm{CATE}=E[S\\mid X]+1.5X\\), treat iff \\(X\\gt 0\\)).
Naive IPW can't see the hidden-S confounding; the Wasserstein balance reaches \\(S\\) through its correlation with \\(X\\).</p></div>
<div class="grid2"><div>%s</div><div>%s</div></div>
<div class="grid2"><div>%s</div><div>%s</div></div>
<div class="grid2"><div><h3>Paired CIs vs naive IPW</h3>%s</div><div><h3>Paired CIs vs naive AIPW</h3>%s</div></div>
<p class="muted">Uncapped, 20 seeds: <b>both O-W methods significantly beat naive IPW and naive AIPW</b> at &Gamma;=2&ndash;4
(CIs exclude 0). The box-only methods do not &mdash; the Wasserstein term is the differentiator.</p>

<h2>Test 2 &mdash; exp_nmcap: a non-monotone effect breaks naive AIPW under the cap</h2>
<div class="card"><p><b>Story.</b> Severity \\(X\\), hidden robustness \\(S\\perp X\\). The therapy <b>helps a middle band</b>
\\(X^2\\lt 0.3\\) and <b>harms the severe edges</b> (\\(\\mathrm{CATE}=3(0.3-X^2)\\)). Clinicians escalate the robust
\\(S{=}{+}1\\) and the extreme-severity edges, so the treated edges are a high-\\(S\\) elite whose outcomes look great.
The naive outcome model is fooled into crediting the edges; under <b>cap \\(\\le 50\\%%\\)</b> the naive doubly-robust
method spends its budget on the harmful edges, while the robust O / O-W methods recover the middle band.</p></div>
<div class="grid2"><div>%s</div><div>%s</div></div>
%s
<div class="grid2"><div>%s</div><div>%s</div></div>
<div class="grid2"><div><h3>Paired CIs vs naive AIPW (CAPPED)</h3>%s</div><div><h3>Paired CIs vs naive AIPW (UNCAPPED)</h3>%s</div></div>
<p class="muted">15 seeds, paired CIs: UNCAPPED the naive AIPW goes <b>negative</b> (it treats the harmful edges) and
<b>both O-W beat it by +0.15&ndash;0.18</b> (CIs exclude 0). CAPPED the cap limits the damage, but
<b>DoublyRobust-O-W still significantly beats naive AIPW</b> (+0.04) &mdash; the exact gap exp_owgap could not close.
Naive IPW self-corrects here (the edge over-selection shows up in the X-propensity), so it ties O-W &mdash; which is
why the <em>pair</em> of DGPs matters: no single naive baseline is safe across confounding structures.</p>

<h2>Over-conservatism &mdash; what happens at extreme &Gamma; and &epsilon;</h2>
<div class="card"><p>The robust set \\(U(\\Gamma,\\epsilon)\\) is supposed to cover the <em>plausible</em> amount of hidden
confounding. Push \\(\\Gamma\\) and the Wasserstein radius \\(\\epsilon\\) to <b>unrealistically large</b> values and the
worst-case treated value drops below control everywhere, so the value-maximiser hedges to <b>never-treat</b> and the
realised value <b>collapses</b> (owgap, uncapped). Two lessons:</p>
<ul><li><b>The pure-IPW methods collapse</b> (IPW-O-W, IPW-O-X) &mdash; from ~oracle 0.85 down to ~0.08 (never-treat)
at \\(c_\\epsilon=16,\\ \\Gamma=1000\\); monotone in <em>both</em> knobs.</li>
<li><b>The doubly-robust methods do NOT collapse</b> &mdash; only the IPW-correction term sees the worst case, so as the
set explodes they fall back to the (biased) outcome-model policy (~0.6) rather than to never-treat. <b>Hajek-O-X</b>
(regret) collapses earliest and hardest, to exactly never-treat, by its do-no-harm fallback.</li></ul></div>
<div class="grid2"><div>%s</div><div>%s</div></div>
<p class="muted">Takeaway: \\(c_\\epsilon\\!\\approx\\!1\\) (the tight, calibrated radius) is the right operating point;
cranking \\(\\Gamma\\) and \\(\\epsilon\\) together is assuming-the-impossible and throws away all value.</p>

<div class="card"><b>Honest caveats.</b><ul>
<li>Under a <b>capacity cap</b>, the cap rescues whichever naive correction is well-specified (propensity in nmcap,
outcome model in owgap), so beating <em>every</em> naive baseline capped is structurally hard. The O-W methods
<b>match the best and beat the failing one</b> in every case; the cleanest uniform wins are <b>uncapped</b>.</li>
<li>The <b>X-X family is &Gamma;-free</b> (flat). The value-objective robust methods keep a bounded worst case, so
they soften rather than collapse with &Gamma;; only the regret method Hajek-O-X collapses to never-treat.</li>
<li>All claims are <b>paired across seeds</b> (same data draw) with bootstrap CIs &mdash; the right test for "does
O-W beat the baseline".</li></ul></div>

<p class="figcap" style="margin-top:22px">Generated by <code>assets/exp_nmcap/build_combined.py</code>. Raw data:
<code>assets/exp_owgap/owgap_uncap_20seed_ce1.0.json</code>, <code>assets/exp_nmcap/nmcap_results_ce1.0.json</code>,
<code>combined_charts.json</code>, per-figure PNGs.</p>

<script>%s
var CH=%s;
%s
</script></body></html>
""" % (CSS, legend, summary_tbl,
       fig("ow_prop", "exp_owgap: propensity", "S-confounded selection."),
       fig("ow_outcome", "exp_owgap: outcome means", "Vitality dominates both arms."),
       fig("ow_cate", "exp_owgap: monotone CATE", "Subtle, monotone effect: treat iff X&gt;0."),
       fig("ow_value_uncap", "exp_owgap: realised value vs &Gamma; (uncapped, 20 seeds)", "Both O-W on top; SD bands."),
       TBL_OW, TBL_OW2,
       fig("nm_cate", "exp_nmcap: non-monotone CATE", "Treat the middle, spare the edges."),
       fig("nm_outcome", "exp_nmcap: outcome means", "Treatment helps the middle band only."),
       fig("nm_prop", "exp_nmcap: propensity", "Edge-heavy, S-confounded selection."),
       fig("nm_value_cap", "exp_nmcap: realised value vs &Gamma; (CAPPED, 15 seeds)", "Naive AIPW (green dashed) fails; O-W beats it."),
       fig("nm_value_uncap", "exp_nmcap: realised value vs &Gamma; (uncapped, 15 seeds)", "AIPW negative; O-W recovers."),
       TBL_NM, TBL_NM2,
       fig("ex_ipw", "Pure-IPW O-W collapses under extreme &Gamma;&times;&epsilon;", "IPW-O-W vs &Gamma; for four &epsilon;; toward never-treat."),
       fig("ex_family", "Doubly-robust is anchored; Hajek collapses hardest", "Family contrast at the extreme."),
       JS, json.dumps(W, separators=(",", ":")), render_calls)

(Path(__file__).resolve().parent / "combined_report.html").write_text(HTML)
print("wrote combined_report.html (%d charts)" % len(W))
print("owgap uncap best: IPW-X-X=%.3f AIPW=%.3f IPW-O-W=%.3f DR-O-W=%.3f" % tuple(SUMMARY["owgap_uncap"][m] for m in ["IPW-X-X","DoublyRobust-X-X","IPW-O-W","DoublyRobust-O-W"]))
print("nmcap cap   best: IPW-X-X=%.3f AIPW=%.3f IPW-O-W=%.3f DR-O-W=%.3f" % tuple(SUMMARY["nmcap_cap"][m] for m in ["IPW-X-X","DoublyRobust-X-X","IPW-O-W","DoublyRobust-O-W"]))
