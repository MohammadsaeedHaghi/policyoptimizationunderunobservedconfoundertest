#!/usr/bin/env python3
"""HTML report for the confounding-robust policy-learning experiments on IHDP, Twins and IST."""
import json, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "semi experiments"))
from report_common import CSS, esc, linechart

S = json.loads((HERE / "summary_rct.json").read_text())
OX = ["IPW-O-X", "DoublyRobust-O-X", "Hajek-O-X"]
OW = ["IPW-O-W", "DoublyRobust-O-W"]
XX = ["IPW-X-X", "DoublyRobust-X-X", "Direct-X-X"]
ORDER = OX + OW + XX + ["SharpHess", "Kallus"]
SHORT = {"DoublyRobust-O-X": "DR-O-X", "DoublyRobust-O-W": "DR-O-W",
         "DoublyRobust-X-X": "DR-X-X", "SharpHess": "Hess"}
TITLE = {"ihdp": "IHDP", "twins": "Twins", "ist": "IST"}
PROV = {"ihdp": ("simulated", "Y0/Y1 are the benchmark's own yf + ycf (NPCI surface A)."),
        "twins": ("observed", "Y0/Y1 are the genuinely observed 1-year survival of the lighter "
                              "and heavier twin -- no model anywhere."),
        "ist": ("imputed", "Y0/Y1 come from the T-learner imputation (AUC 0.79/0.80 on the "
                           "factual arm); the unobserved arm is a model output.")}

EXTRA = """
table.dt{border-collapse:collapse;font-size:12.5px;margin:10px 0;width:100%}
table.dt th,table.dt td{border:1px solid #e3e6ea;padding:5px 8px;text-align:right;
  white-space:nowrap;font-variant-numeric:tabular-nums}
table.dt th{background:#f4f6f8;font-weight:600}
table.dt td.l,table.dt th.l{text-align:left}
table.dt td.ref{color:#111}
table.dt .sd{font-size:10.5px;opacity:.65;font-weight:400}
table.dt tr.sumrow td{border-top:2px solid #b9c0c8;background:#f7f9fb}
.scrollx{overflow-x:auto;max-width:100%}
.hsub{font-weight:400;font-size:.82rem;color:#667085}
.prov{display:inline-block;font-size:11px;font-weight:700;padding:2px 8px;border-radius:10px;margin-left:8px}
.p-observed{background:#dcfce7;color:#166534}.p-simulated{background:#ede9fe;color:#5b21b6}
.p-imputed{background:#fef3c7;color:#92400e}
"""

RAMP = [(0.0, (0xC0, 0x39, 0x2B)), (0.5, (0xB0, 0x71, 0x05)), (1.0, (0x0A, 0x7D, 0x33))]


def tint(t):
    t = 0.0 if t != t else min(max(t, 0.0), 1.0)
    for (a, ca), (b, cb) in zip(RAMP, RAMP[1:]):
        if t <= b:
            f = (t - a) / (b - a)
            return "#%02x%02x%02x" % tuple(int(round(ca[i] + f * (cb[i] - ca[i]))) for i in range(3))
    return "#0a7d33"


def cells(vals, sds=None):
    vs = [vals[m] for m in ORDER if m in vals]
    lo, hi = (min(vs), max(vs)) if vs else (0.0, 1.0)
    out = []
    for m in ORDER:
        if m not in vals: out.append("<td>&mdash;</td>"); continue
        v = vals[m]
        t = 1.0 if hi <= lo else (v - lo) / (hi - lo)
        bold = ";font-weight:700" if abs(v - hi) < 1e-12 else ""
        txt = "%.3f" % v
        if sds and m in sds: txt += " <span class='sd'>&plusmn;%.3f</span>" % sds[m]
        out.append("<td style='color:%s%s'>%s</td>" % (tint(t), bold, txt))
    return "".join(out)


def table(key, caption):
    head = ("<tr><th class='l'>dataset</th>"
            + "".join("<th>%s</th>" % esc(SHORT.get(m, m)) for m in ORDER)
            + "<th>all-treat</th><th>oracle</th></tr>")
    body = []
    for s in S:
        vals = {m: s["rows"][m][key]["mean"] for m in ORDER
                if m in s["rows"] and s["rows"][m].get(key)}
        sds = {m: s["rows"][m][key]["sd"] for m in ORDER
               if m in s["rows"] and s["rows"][m].get(key) and "sd" in s["rows"][m][key]}
        body.append("<tr><td class='l'>%s</td>%s<td class='ref'>%.3f</td>"
                    "<td class='ref'>%.3f</td></tr>"
                    % (esc(TITLE[s["dataset"]]), cells(vals, sds),
                       s["refs"]["all_treat"], s["refs"]["oracle"]))
    # normalized row: fraction of the achievable gain recovered, averaged over the three
    agg = {}
    for m in ORDER:
        vs = [(s["rows"][m][key]["mean"] - s["refs"]["never_treat"])
              / (s["refs"]["oracle"] - s["refs"]["never_treat"])
              for s in S if m in s["rows"] and s["rows"][m].get(key)]
        if vs: agg[m] = float(np.mean(vs))
    allt = float(np.mean([(x["refs"]["all_treat"] - x["refs"]["never_treat"])
                          / (x["refs"]["oracle"] - x["refs"]["never_treat"]) for x in S]))
    body.append("<tr class='sumrow'><td class='l'><b>mean normalized</b></td>%s"
                "<td class='ref'><b>%.3f</b></td><td class='ref'>1.000</td></tr>"
                % (cells(agg), allt))
    return ("<div class='scrollx'><table class='dt'>%s%s</table></div>"
            "<p class='muted'>%s</p>" % (head, "".join(body), caption))


# ---------------------------------------------------------------- per-dataset Gamma curves
figs = []
for s in S:
    gs = [float(g) for g in s["gammas"]]
    ser = []
    pal = {"IPW-O-W": "#d62728", "DoublyRobust-O-W": "#1f77b4", "IPW-O-X": "#d62728",
           "DoublyRobust-O-X": "#1f77b4", "Hajek-O-X": "#ff7f0e"}
    for m in ["IPW-O-W", "IPW-O-X", "DoublyRobust-O-W", "DoublyRobust-O-X"]:
        row = s["surface"].get(m, {})
        ys = [row.get(g, {}).get("3") for g in s["gammas"]]
        if any(y is not None for y in ys):
            ser.append((m, pal[m], gs, [np.nan if y is None else y for y in ys],
                        "" if m.endswith("O-W") else "7 3", "n"))
    hb = s.get("hess_by_gamma", {})
    if hb:
        ser.append(("Hess et al.", "#9467bd", gs,
                    [hb.get(g, np.nan) for g in s["gammas"]], "", "n"))
    kb = s.get("kallus_by_gamma", {})
    if kb:
        ser.append(("Kallus", "#7f7f7f", gs,
                    [kb.get(g, np.nan) for g in s["gammas"]], "2 3", "n"))
    hl = [("oracle", "#111", s["refs"]["oracle"], "5 4"),
          ("never treat", "#888", s["refs"]["never_treat"], "2 3")]
    figs.append("<div>" + linechart(ser, title="%s: average test outcome vs Gamma (L = 3)"
                                    % TITLE[s["dataset"]], xlab="Gamma", ylab="test E[Y]",
                                    hlines=hl, xticks=gs)
                + "<p class='muted figcap'>Solid = O-W, dashed = its O-X twin. The DGP's exact "
                  "Gamma* = 4.</p></div>")

# ---------------------------------------------------------------- construction table
crows = []
for s in S:
    m = s["meta"]; R = s["refs"]; k, _note = PROV[s["dataset"]]
    crows.append("<tr><td class='l'>%s<span class='prov p-%s'>%s CF</span></td>"
                 "<td>%d</td><td>%s</td><td>%+.2f</td><td>%.3f</td>"
                 "<td>%.3f</td><td>%.3f</td><td>%.3f</td></tr>"
                 % (esc(TITLE[s["dataset"]]), k, k.upper(), m["n_total"], m["outcome_kind"][:4],
                    m["corr_x_S"], m["LAM_realised"], R["oracle"], R["never_treat"],
                    R["all_treat"]))

tm = []
for s in S:
    d = s["transport_margin"]
    tm.append("<tr><td class='l'>%s</td><td class='%s'>%+.4f</td><td class='%s'>%+.4f</td></tr>"
              % (esc(TITLE[s["dataset"]]),
                 "", d.get("IPW-O-W", {}).get("mean_over_grid", float("nan")),
                 "", d.get("DoublyRobust-O-W", {}).get("mean_over_grid", float("nan"))))

nseeds = S[0]["n_seeds"] if S else 0
html = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Confounding-robust policy learning on RCT-derived benchmarks</title>
<style>{CSS}{EXTRA}</style></head><body>
<div class="hero" style="background:linear-gradient(135deg,#134e4a,#0f766e)">
<h1>Confounding-robust policy learning on IHDP, Twins and IST</h1>
<p>Every method, three benchmarks built from real datasets with real potential outcomes and a
synthetic assignment of known strength. n<sub>train</sub> = 300, {nseeds} seeds,
&Gamma; &times; L &times; c<sub>&epsilon;</sub> swept, no capacity constraint.</p></div>

<h2>How the three benchmarks were built</h2>
<p>All three source datasets are randomised (or, for Twins, a matched pair), so there is <b>no
hidden confounding in them</b> &mdash; a sensitivity model would have nothing to be robust to and
every robust method would collapse onto its X-X twin. So only the <b>assignment</b> is replaced:
<i>e</i>(<i>x</i>,<i>S</i>) = &sigma;(1.2<i>x</i> + &frac12;ln&Lambda;&middot;<i>S</i>) with no
clipping, giving an odds ratio of exactly &Lambda; = 4 at every <i>x</i>. The covariates and
<b>both potential outcomes are the datasets' own</b>.</p>
<div class="scrollx"><table class="dt">
<tr><th class="l">benchmark</th><th>N</th><th>outcome</th><th>corr(x,S)</th>
<th>&Lambda; realised</th><th>oracle</th><th>never</th><th>all</th></tr>
{''.join(crows)}
</table></div>
<p class="muted">
<b>IHDP</b> &mdash; {esc(PROV['ihdp'][1])}
<b>Twins</b> &mdash; {esc(PROV['twins'][1])}
<b>IST</b> &mdash; {esc(PROV['ist'][1])}
&Lambda; realised is measured back out of the propensity actually used, confirming &Gamma;* = 4
is exact rather than approximate.</p>

<h2>Average test outcome at the matched &Gamma; = 4
<span class="hsub">&mdash; mean &plusmn; sd over {nseeds} seeds, best (L, c<sub>&epsilon;</sub>)</span></h2>
{table("matched_best_L",
       "The honest operating point: Gamma = 4 is the only value justified by the design, and it is "
       "the same for every method. The last row is the mean over the three benchmarks of "
       "(V - never) / (oracle - never), the fraction of the achievable gain recovered -- 0 = no "
       "better than never treating, 1 = oracle.")}

<div class="card bad"><b>Read this before the numbers: on all three benchmarks
&quot;treat everyone&quot; beats every method.</b> Normalised as
(V &minus; never) / (oracle &minus; never), the constant all-treat policy scores
<b>0.998 / 0.443 / 0.070</b> on IHDP / Twins / IST, while the best method scores
<b>0.997 / 0.294 / 0.046</b>. In raw units the oracle beats the better constant policy by only
<b>0.010</b> on IHDP, <b>0.010</b> on Twins and <b>0.127</b> on IST.
<br><br>That is a property of the benchmarks, not of the methods. When the optimal policy is
almost constant there is nothing for a policy learner to find, and every method &mdash; robust,
sharp, or unconfoundedness-assuming &mdash; lands in the same place, which is exactly the
flat ordering below. The same diagnosis applied to three UCI datasets earlier in this project and
they were rebuilt rather than reported.
<br><br><b>Why it happens here.</b> The treatment effect in these datasets is mostly a level
shift: IHDP's NPCI surface adds a near-constant benefit, Twins' heavier-twin survival advantage
is positive almost everywhere, and IST's aspirin effect is tiny (a 6-month mortality difference of
0.0099) relative to the outcome scale. None of them was designed to have a heterogeneous sign
flip, which is what a policy has to exploit. The IST row is the informative one: it has real
headroom (0.127) and every method recovers under 5% of it, i.e. all of them fail to find the
heterogeneity even though it exists.</div>

<h2>At each method's best cell
<span class="hsub">&mdash; oracle-tuned upper bound</span></h2>
{table("best",
       "Best cell anywhere on the whole Gamma x L x c_eps grid. Not an achievable protocol; "
       "reported so a loss at matched Gamma cannot be dismissed as bad tuning.")}

<h2>&Gamma; curves</h2>
<div class="figrow">{''.join(figs)}</div>

<h2>Transport margin</h2>
<p>O-W minus its own O-X twin at identical &Gamma; and <i>L</i>, averaged over the grid. Same
estimator, same box, same policy class &mdash; the only difference is the Wasserstein ball.</p>
<div class="scrollx"><table class="dt">
<tr><th class="l">benchmark</th><th>IPW</th><th>DoublyRobust</th></tr>
{''.join(tm)}
</table></div>

<h2>Protocol</h2>
<p>&Gamma; &isin; {{1, 2, 4, 8}} &times; <i>L</i> &isin; {{&infin;, 3, 1}} &times;
c<sub>&epsilon;</sub> &isin; {{1, 2}}, uncapped, {nseeds} seeds, 300 training rows and up to 4,000
held-out rows with both potential outcomes retained. Each (dataset, seed, c<sub>&epsilon;</sub>)
cell ran as its own SLURM job, 30 in parallel. The X-X baselines are swept over the same
<i>L</i> grid as everything else &mdash; earlier campaigns ran them only at
<i>L</i> = &infin;, which understated them.</p>
<p class="muted">Regenerate: <code>python3 prepare_rct.py</code>, then
<code>sbatch scripts/uci10/rct_array.sh</code>, then <code>python3 aggregate_rct.py</code> and
<code>python3 build_rct_report.py</code>.</p>
</body></html>"""

html = html.encode("ascii", "xmlcharrefreplace").decode("ascii")
(HERE / "rct_experiments.html").write_text(html)
print("wrote %s (%d KB)" % (HERE / "rct_experiments.html", len(html) // 1024))
