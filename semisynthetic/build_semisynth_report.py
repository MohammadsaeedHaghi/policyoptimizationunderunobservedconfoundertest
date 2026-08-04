#!/usr/bin/env python3
"""HTML report for the risk-score-paper semi-synthetic replication, across datasets.

Reads summary_<dataset>.json for every dataset that has one.
"""
import json, sys, glob
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "semi experiments"))
from report_common import CSS, esc

DATASETS = ["bank_marketing", "wine_quality", "german_credit", "credit_default", "adult"]
SUM = {}
for d in DATASETS:
    f = HERE / ("summary_%s.json" % d)
    if f.exists():
        SUM[d] = json.loads(f.read_text())

OWF = ["IPW-O-W", "DoublyRobust-O-W", "Hajek-O-W"]
BASE = ["SharpHess", "Kallus"]
OTHER = ["IPW-O-X", "DoublyRobust-O-X", "Hajek-O-X", "IPW-X-X", "DoublyRobust-X-X", "Direct-X-X"]
SHORT = {"DoublyRobust-O-W": "DR-O-W", "DoublyRobust-O-X": "DR-O-X",
         "DoublyRobust-X-X": "DR-X-X", "SharpHess": "Hess"}

EXTRA = """
table.dt{border-collapse:collapse;font-size:12.5px;margin:10px 0;width:100%}
table.dt th,table.dt td{border:1px solid #e3e6ea;padding:5px 8px;text-align:right;
  white-space:nowrap;font-variant-numeric:tabular-nums}
table.dt th{background:#f4f6f8;font-weight:600}
table.dt td.l,table.dt th.l{text-align:left}
table.dt td.ref{color:#111}
table.dt .sd{font-size:10.5px;opacity:.65;font-weight:400}
table.dt tr.dsrow td{border-top:2px solid #b9c0c8;background:#f7f9fb;font-weight:700}
.scrollx{overflow-x:auto;max-width:100%}
.hsub{font-weight:400;font-size:.82rem;color:#667085}
"""
RAMP = [(0.0, (0xC0, 0x39, 0x2B)), (0.5, (0xB0, 0x71, 0x05)), (1.0, (0x0A, 0x7D, 0x33))]


def tint(t):
    t = 0.0 if t != t else min(max(t, 0.0), 1.0)
    for (a, ca), (b, cb) in zip(RAMP, RAMP[1:]):
        if t <= b:
            f = (t - a) / (b - a)
            return "#%02x%02x%02x" % tuple(int(round(ca[i] + f * (cb[i] - ca[i]))) for i in range(3))
    return "#0a7d33"


def cells(vals, order, sds=None):
    vs = [vals[m] for m in order if m in vals]
    lo, hi = (min(vs), max(vs)) if vs else (0.0, 1.0)
    out = []
    for m in order:
        if m not in vals: out.append("<td>&mdash;</td>"); continue
        v = vals[m]; t = 1.0 if hi <= lo else (v - lo) / (hi - lo)
        bold = ";font-weight:700" if abs(v - hi) < 1e-12 else ""
        txt = "%.4f" % v
        if sds and m in sds: txt += " <span class='sd'>&plusmn;%.3f</span>" % sds[m]
        out.append("<td style='color:%s%s'>%s</td>" % (tint(t), bold, txt))
    return "".join(out)


def norm(v, R):
    bc = max(R["never_treat"], R["all_treat"])
    return (v - bc) / (R["oracle"] - bc)


# ---------------------------------------------------- headline: O-W family vs Hess / Kallus
COLS = ["IPW-O-W", "IPW-O-W+C4", "DoublyRobust-O-W", "DoublyRobust-O-W+C4",
        "Hajek-O-W", "SharpHess", "Kallus"]
rows = []
for ds in DATASETS:
    if ds not in SUM: continue
    rows.append("<tr class='dsrow'><td class='l' colspan='%d'>%s</td></tr>"
                % (len(COLS) + 4, esc(ds)))
    for s in SUM[ds]:
        r, rr, R = s["rows"], s["rows_rat"], s["refs"]
        vals, sds = {}, {}
        for m in OWF + BASE:
            if m in r: vals[m] = r[m]["mean"]; sds[m] = r[m]["sd"]
        for m in ("IPW-O-W", "DoublyRobust-O-W"):
            if m in rr: vals[m + "+C4"] = rr[m]["mean"]; sds[m + "+C4"] = rr[m]["sd"]
        bc = max(R["never_treat"], R["all_treat"])
        rows.append("<tr><td class='l'>&gamma; = %.1f</td><td class='ref'>%.2f</td>%s"
                    "<td class='ref'>%.4f</td><td class='ref'>%.4f</td></tr>"
                    % (s["gamma"], s["Gamma"], cells(vals, COLS, sds), bc, R["oracle"]))
head = ("<tr><th class='l'>&nbsp;</th><th>&Gamma;</th>"
        + "".join("<th>%s</th>" % esc(SHORT.get(c.replace("+C4", ""), c.replace("+C4", ""))
                                      + ("+C4" if c.endswith("+C4") else "")) for c in COLS)
        + "<th>best const</th><th>oracle</th></tr>")
headline = "<div class='scrollx'><table class='dt'>%s%s</table></div>" % (head, "".join(rows))

# ---------------------------------------------------- normalised, pooled over datasets
pool = []
for g in (0.0, 1.0, 1.5, 2.0):
    agg = {}
    for c in COLS:
        vs = []
        for ds in SUM:
            for s in SUM[ds]:
                if abs(s["gamma"] - g) > 1e-12: continue
                base = c.replace("+C4", "")
                src = s["rows_rat"] if c.endswith("+C4") else s["rows"]
                if base in src: vs.append(norm(src[base]["mean"], s["refs"]))
        if vs: agg[c] = float(np.mean(vs))
    pool.append("<tr><td class='l'>&gamma; = %.1f</td><td class='ref'>%.2f</td>%s</tr>"
                % (g, float(np.exp(2 * g)), cells(agg, COLS)))
poolhead = ("<tr><th class='l'>&nbsp;</th><th>&Gamma;</th>"
            + "".join("<th>%s</th>" % esc(SHORT.get(c.replace("+C4", ""), c.replace("+C4", ""))
                                          + ("+C4" if c.endswith("+C4") else "")) for c in COLS)
            + "</tr>")
pooltbl = "<div class='scrollx'><table class='dt'>%s%s</table></div>" % (poolhead, "".join(pool))

# ---------------------------------------------------- transport margin
tm = []
for ds in DATASETS:
    if ds not in SUM: continue
    for s in SUM[ds]:
        m_ = s["transport_margin"]
        tm.append("<tr><td class='l'>%s</td><td>&gamma; = %.1f</td><td class='ref'>%.2f</td>"
                  "<td style='color:%s'>%s</td><td style='color:%s'>%s</td></tr>"
                  % (esc(ds), s["gamma"], s["Gamma"],
                     "#0a7d33" if m_.get("IPW-O-W|rat0", 0) > 0 else "#c0392b",
                     ("%+.4f" % m_["IPW-O-W|rat0"]) if "IPW-O-W|rat0" in m_ else "&mdash;",
                     "#0a7d33" if m_.get("DoublyRobust-O-W|rat0", 0) > 0 else "#c0392b",
                     ("%+.4f" % m_["DoublyRobust-O-W|rat0"]) if "DoublyRobust-O-W|rat0" in m_ else "&mdash;"))
tm_tbl = ("<div class='scrollx'><table class='dt'>"
          "<tr><th class='l'>dataset</th><th>&nbsp;</th><th>&Gamma;</th>"
          "<th>IPW-O-W &minus; IPW-O-X</th><th>DR-O-W &minus; DR-O-X</th></tr>"
          + "".join(tm) + "</table></div>")

# ---------------------------------------------------- C4 effect (O-W only)
c4 = []
for ds in DATASETS:
    if ds not in SUM: continue
    for s in SUM[ds]:
        e = s["c4_effect"]
        def g2(m, k):
            c = e.get(m)
            return ("%+.4f" % c[k]) if c and c.get(k) is not None else "&mdash;"
        c4.append("<tr><td class='l'>%s</td><td>&gamma; = %.1f</td>"
                  "<td>%s</td><td>%s</td><td class='ref'>%s</td><td class='ref'>%s</td></tr>"
                  % (esc(ds), s["gamma"], g2("IPW-O-W", "value_delta"),
                     g2("DoublyRobust-O-W", "value_delta"),
                     g2("IPW-O-W", "objective_delta"), g2("DoublyRobust-O-W", "objective_delta")))
c4_tbl = ("<div class='scrollx'><table class='dt'>"
          "<tr><th class='l'>dataset</th><th>&nbsp;</th><th>IPW-O-W value &Delta;</th>"
          "<th>DR-O-W value &Delta;</th><th>IPW-O-W obj &Delta;</th><th>DR-O-W obj &Delta;</th></tr>"
          + "".join(c4) + "</table></div>")

# ---------------------------------------------------- construction facts
cons = []
for ds in DATASETS:
    if ds not in SUM: continue
    m = SUM[ds][0]["meta"]; R = SUM[ds][0]["refs"]
    cons.append("<tr><td class='l'>%s</td><td>%d</td><td>%d</td><td>%.4f</td><td>%.4f</td>"
                "<td>%+.3f</td><td>%.3f</td></tr>"
                % (esc(ds), m["n_pool"], m["d"], m["cv_logloss"],
                   R["oracle"] - max(R["never_treat"], R["all_treat"]),
                   m["corr_x_u"], m["frac_tau_pos"]))
cons_tbl = ("<div class='scrollx'><table class='dt'>"
            "<tr><th class='l'>dataset</th><th>N pool</th><th>d</th><th>CV log-loss</th>"
            "<th>headroom</th><th>corr(x,u)</th><th>frac &tau;&gt;0</th></tr>"
            + "".join(cons) + "</table></div>")

nseeds = SUM[DATASETS[0]][0]["n_seeds"] if SUM else 0
html = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Risk-score-paper semi-synthetic construction, run for policy learning</title>
<style>{CSS}{EXTRA}</style></head><body>
<div class="hero" style="background:linear-gradient(135deg,#3b0764,#7e22ce)">
<h1>The risk-score paper's semi-synthetic construction &mdash; run for policy learning</h1>
<p>{len(SUM)} UCI datasets, their DGP, extended with a heterogeneous CATE so both potential
outcomes exist. Solved only at the <b>matched &Gamma; = e<sup>2&gamma;</sup></b>, with and without
their C4 "historical rationality" constraint. n<sub>train</sub> = 300, {nseeds} seeds per cell.
<b>Higher is better</b> &mdash; every value is negative only because
<i>Y</i><sup>0</sup> &isin; {{&minus;1,+1}} with &minus;1 the adverse outcome.</p></div>

<h2>How the benchmarks are built</h2>
<p><b>Theirs, exactly.</b> The UCI label is used twice &mdash; as the unobserved confounder
<i>U</i> and as the untreated potential outcome <i>Y</i><sup>0</sup> &isin; {{&minus;1, +1}}.
Treatment follows their Eq. (6), logit&thinsp;&pi;<sup>0</sup>(x,u) = &lambda;&#8242;x + &gamma;u
with &lambda;<sub>j</sub> ~ U(&minus;0.1, 0.1) and <b>no clipping</b>, &pi;<sup>0</sup> being the
probability of <i>no</i> treatment. Every dataset passes their admissibility screen: cross-validated
logistic log-loss inside [0.35, ln&thinsp;2 = 0.6931].</p>
<p><b>Ours.</b> Their construction only needs <i>Y</i><sup>0</sup>, since their target is a risk
score. Policy learning needs both arms, so <i>Y</i><sup>1</sup> = <i>Y</i><sup>0</sup> + &tau;(x)
+ &epsilon; with &tau;(x) = a&thinsp;(&sigma;(b&#8242;x &minus; c) &minus; &frac12;). The
&minus;&frac12; is load-bearing: without it &tau; is strictly positive, the CATE never changes
sign, and the oracle policy is "treat everyone" &mdash; measured headroom exactly
<b>+0.00000</b>.</p>
{cons_tbl}
<p class="muted">Because &pi;<sup>0</sup> is never clipped, the odds ratio between the two hidden
states equals e<sup>2&gamma;</sup> to ~1e&minus;13 at every x, so the matched &Gamma; is
<b>exact</b> here rather than the interval the paper settles for.
&Gamma; = e<sup>2&gamma;</sup> &isin; {{1, 7.39, 20.09, 54.60}}.</p>

<h2>O-W family against the published baselines
<span class="hsub">&mdash; best (c<sub>&epsilon;</sub>, L) cell, mean &plusmn; sd over {nseeds} seeds</span></h2>
{headline}

<h2>Same thing, normalised and pooled across datasets
<span class="hsub">&mdash; (V &minus; best constant) / (oracle &minus; best constant)</span></h2>
<p>0 = no better than the best constant policy, 1 = oracle. Negative means the method is worse
than not learning a policy at all.</p>
{pooltbl}

<h2>Transport margin
<span class="hsub">&mdash; O-W minus its own O-X twin at identical L, C4 off</span></h2>
<p>Same estimator, same box, same policy class; the only difference is the Wasserstein ball. This
is what the paper's Robust-vs-Robust-Baseline comparison measures.</p>
{tm_tbl}

<h2>The C4 "historical rationality" constraint
<span class="hsub">&mdash; on minus off, O-W family only</span></h2>
<p>Left: change in realised <b>test policy value</b>. Right: change in the solver's own
<b>worst-case objective</b>, which must be &ge; 0 &mdash; C4 restricts the inner minimisation, so
the certified worst case can only improve. A large positive objective change beside a negative
value change means the constraint bought certainty, not accuracy.</p>
{c4_tbl}

<h2>Protocol</h2>
<p>&gamma; &isin; {{0, 1, 1.5, 2}} (the paper's Table 1), each solved <b>only</b> at its matched
&Gamma;. c<sub>&epsilon;</sub> &isin; {{1, <b>1.025</b>, 2}} &mdash; 1.025 is the paper's own
&epsilon; multiplier. L &isin; {{&infin;, 3, 1}}. 300 training rows, up to 4,000 held-out rows with
both potential outcomes retained, {nseeds} seeds. Each (dataset, &gamma;, seed) cell ran as its own
SLURM job.</p>
<p class="muted">The solver's covariate is the scalar index b&#8242;x. Since &tau; depends on x only
through b&#8242;x, that index is a <b>sufficient statistic for the optimal policy</b>, so the
projection loses nothing about which decision is correct and the experiment isolates
confounding-robustness rather than dimension reduction.</p>
<p class="muted">Everything is persisted: per-cell result JSONs carrying every
&Gamma;&times;L&times;c<sub>&epsilon;</sub>&times;rationality cell with its solver objective,
the prepared population arrays, and the (a, b, c, &lambda;) parameters needed to reproduce any DGP
draw exactly. Nothing needs re-running. Regenerate the tables with
<code>aggregate_semisynth.py --dataset NAME</code> then <code>build_semisynth_report.py</code>.</p>
</body></html>"""

html = html.encode("ascii", "xmlcharrefreplace").decode("ascii")
(HERE / "semisynth_report.html").write_text(html)
print("wrote %s (%d KB, %d datasets)" % (HERE / "semisynth_report.html", len(html) // 1024, len(SUM)))
