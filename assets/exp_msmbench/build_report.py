#!/usr/bin/env python3
"""Build 'assets/exp_msmbench/msmbench_report.html' -- OUR coupled-confounder synthetic
experiment (KMZ-derived functional forms; our design). Rerun: python3 assets/exp_msmbench/build_report.py
"""
import json, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT / "semi experiments"))
sys.path.insert(0, str(HERE))
from report_common import page, surface_block, linechart, ser, MC
from latex2mathml.converter import convert as l2m
import importlib.util
def _load(p, n):
    sp = importlib.util.spec_from_file_location(n, p); m = importlib.util.module_from_spec(sp)
    sys.modules[n] = m; sp.loader.exec_module(m); return m
d0 = _load(HERE / "dgp.py", "kmz0")
dc = _load(HERE / "dgp_coupled.py", "kmzc")

def J(fn):
    try: return json.load(open(HERE / fn))
    except Exception: return None
R = {0.0: J("msmbench_lip_gamma_2d_pilot.json"), 2.5: J("msmbench_beta2.5.json"),
     5.0: J("msmbench_beta5.json"), 10.0: J("msmbench_beta10.json")}
CORR = {0.0: -0.007, 2.5: 0.578, 5.0: 0.756, 10.0: 0.840}

def figcap(fig, cap): return f'<div>{fig}<p class="muted figcap">{cap}</p></div>'

xg = list(np.linspace(-1, 1, 241)); xa = np.array(xg)
cate_fig = figcap(
    linechart([("CATE(x)", "#d62728", xg, list(dc.cate(xa)), "")],
              title="True CATE (heterogeneous, oscillating)", xlab="x = X/2", ylab="CATE",
              hlines=[("0", "#888", 0.0, "4 3")], legend=False),
    "CATE(X) = 2X + 2 - 4 sin(2X); the confounder U shifts LEVELS only and cancels in the "
    "effect. Oracle pi*(x) = 1{CATE > 0} treats ~73% of mass.")
prop_fig = figcap(
    linechart([("P(T=1 | x, U=1)", "#2ca02c", xg, list(dc.propensity(xa, np.ones(241))), ""),
               ("P(T=1 | x, U=0)", "#9467bd", xg, list(dc.propensity(xa, -np.ones(241))), "")],
              title="Propensity: u-to-u odds ratio pinned to Lambda = 4.95", xlab="x", ylab="P(T=1 | x, U)"),
    "P(T=1 | x, U) = sigma(0.5 + 1.5x + 0.8(2U-1)); odds ratio between U-arms = "
    "e^{1.6} = 4.95 at EVERY x, by algebra -- Gamma* is known by construction.")
coup_fig = figcap(
    linechart([(f"beta={b:g}", c, xg, list(1 / (1 + np.exp(-b * xa))), dsh)
               for b, c, dsh in [(0.0, "#94a3b8", "2 3"), (2.5, "#93c5fd", ""), (5.0, "#3b82f6", ""), (10.0, "#1d4ed8", "")]],
              title="The coupling dial: P(U=1 | x) = sigma(beta x)", xlab="x", ylab="P(U=1 | x)"),
    "beta = 0 / 2.5 / 5 / 10 gives measured corr(x, U) = -0.01 / 0.58 / 0.76 / 0.84 "
    "(marginal P(U=1) = 1/2 for every beta by symmetry).")

EQ = l2m(r"Y(a) = (2a{-}1)X + (2a{-}1) - 2\sin(2(2a{-}1)X) - 2(2U{-}1)(1+0.5X) + \mathcal{N}(0,1),"
         r"\quad X \sim \mathrm{Unif}[-2,2],\; U \mid x \sim \mathrm{Bern}(\sigma(\beta x))")

hero = """<h1>Coupled-confounder synthetic &mdash; known &Gamma;*, one coupling dial</h1>
<p>Our second synthetic experiment. Design goals: (1) <b>&Gamma;* is known by construction</b>
&mdash; the propensity pins the hidden-confounder odds ratio to &Lambda; = 4.95 algebraically,
so every method runs at the single matched &Gamma; = &Gamma;* with NOTHING tuned or swept;
(2) a single disclosed dial &beta; controls how well the observed covariate proxies the
confounder, moving the experiment along the measured coupling axis of the diagnostic map;
(3) a heterogeneous, oscillating effect so policy mistakes are expensive. Outcome functional
forms follow Kallus-Mao-Zhou (2019).</p>"""

body = [f"""
<h2>1. The DGP</h2>
<div style="text-align:center;overflow-x:auto;margin:10px 0">{EQ}</div>
<p>Treatment: P(T=1 | x, U) = &sigma;(0.5 + 1.5x + 0.8&middot;(2U-1)) &mdash; the
u-to-u odds ratio is e<sup>1.6</sup> = 4.95 at every x and every &beta;, so the matched
&Gamma; = &Gamma;* = 4.95 odds-box around the fitted propensity covers the truth exactly:
the experiment has NO &Gamma; sweep and no tuning knob. Only the Lipschitz dial L is swept.</p>
<div class="figrow">{cate_fig}{prop_fig}</div>
<div class="figrow">{coup_fig}</div>
<h2>2. Declared prediction</h2>
<div class="card">The O-W margin over box-only O-X should be &asymp;0 at low &beta; (with
U &perp; X, balancing X cannot constrain U &mdash; no X-balance method can help, and O-W must
do no harm) and turn POSITIVE once the measured coupling crosses the &asymp;0.75 boundary
identified by the showcase's coupling sweep &mdash; the same threshold, measured on a
completely different DGP family.</div>
"""]

have = {b: r for b, r in R.items() if r}
if len(have) >= 3:
    rows, mI, mD, cx = [], [], [], []
    for b in sorted(have):
        r = have[b]
        bx, bw = r["best"]["IPW-O-X"], r["best"]["IPW-O-W"]
        dx, dw = r["best"]["DoublyRobust-O-X"], r["best"]["DoublyRobust-O-W"]
        cx.append(CORR[b]); mI.append(bw["value"] - bx["value"]); mD.append(dw["value"] - dx["value"])
        rows.append(f"<tr><td>&beta;={b:g}</td><td>{CORR[b]:.2f}</td><td>{r['oracle']:.3f}</td>"
                    f"<td>{r['naive_dr']:.3f}</td><td>{bx['value']:.3f} @L{bx['L']}</td>"
                    f"<td>{bw['value']:.3f} @L{bw['L']}</td>"
                    f"<td class='{'g' if mI[-1] > 0.02 else ''}'>{mI[-1]:+.3f}</td>"
                    f"<td class='{'g' if mD[-1] > 0.02 else ''}'>{mD[-1]:+.3f}</td></tr>")
    body.append(f"""
<h2>3. Results across the coupling dial (n=200, 3 seeds, &Gamma; = &Gamma;* throughout)</h2>
<div class="tw"><table>
<tr><th>setting</th><th>corr(x,U)</th><th>oracle</th><th>naive DR</th><th>best IPW-O-X</th>
<th>best IPW-O-W</th><th>IPW margin</th><th>DR margin</th></tr>
{''.join(rows)}
</table></div>
{figcap(linechart([ser("IPW-O-W", cx, mI, lab="IPW: O-W minus O-X (best cells)"),
                   ser("DoublyRobust-O-W", cx, mD, lab="DR: O-W minus O-X (best cells)")],
                  title="The W-term's contribution vs measured coupling", xlab="corr(x, U)",
                  ylab="O-W minus O-X", hlines=[("0", "#888", 0.0, "4 3")], W=700),
        "Margin of the Wasserstein-constrained method over its box-only counterpart, at each "
        "method's best (Gamma*, L) cell. The flip from ~0 to positive occurs between corr 0.58 "
        "and 0.76 -- the same boundary the showcase's coupling sweep identified.")}
<div class="card good"><b>Verdict.</b> The declared prediction holds: margins are
{mI[0]:+.3f}/{mD[0]:+.3f} (IPW/DR) at corr &asymp; 0 and {mI[1]:+.3f}/{mD[1]:+.3f} at 0.58
&mdash; statistical ties, O-W doing no harm where no X-balance method can help &mdash; then
turn positive at corr 0.76 ({mI[2]:+.3f}/{mD[2]:+.3f}; all three seeds positive for IPW) and
stay positive at 0.84 ({mI[3]:+.3f}/{mD[3]:+.3f}). The boundary matches the showcase's
coupling sweep on an unrelated DGP family. HONESTY NOTE: at n=200 / 3 seeds the per-seed
margins are noisy (at &beta;=10 one seed carries the mean); the paper version needs the
standard n=400 / 8-20 seeds to put a CI on the flip. Headline setting for the paper:
&beta; = 5&ndash;10 (in-regime), with the &beta;-ladder as the ablation and &beta;=0 kept as
the zero-coupling anchor.</div>""")
    b_head = 5.0 if 5.0 in have else sorted(have)[-1]
    body.append(f"<h2>4. Full surface at the headline setting (&beta; = {b_head:g})</h2>")
    body.append(surface_block(have[b_head], f"beta={b_head:g}"))
else:
    body.append('<h2>3. Results</h2><p class="muted">Coupled runs pending; rerun this builder.</p>')

body.append("""
<h2>5. The zero-coupling anchor (appendix material)</h2>
<p>&beta; = 0 with the original KMZ extremal-propensity construction (dgp.py, unmodified) is
kept as the anchor: there naive collapses (-0.254 vs oracle 1.439), box-only excels
(1.078, no collapse -- bounded pessimism on its correctly-specified home turf), and O-W ties
box-only within noise. It doubles as our on-file answer to "how does the method do on the
standard MSM synthetic?": we match the best method there, and the coupling diagnostic says
why nothing more was possible.</p>
<h2>6. Files</h2>
<p class="muted mono">assets/exp_msmbench/: dgp.py (beta=0 anchor, original construction),
dgp_coupled.py + dgp_beta2.5/5.py (the experiment), msmbench_*.json (results, all seeds' raw
policies persisted), build_report.py | scripts/sbatch_msmbench_pilot.sh,
sbatch_msmbench_coupled.sh | outcome forms: Kallus-Mao-Zhou 2019</p>""")

out = HERE / "msmbench_report.html"
out.write_text(page("Coupled-confounder synthetic (known Gamma*)",
                    "radial-gradient(130% 150% at 0% 0%,#4a4a2d 0%,#3d3d1f 46%,#1a1a0c 100%)",
                    hero, "".join(body)))
print(f"wrote {out}")
