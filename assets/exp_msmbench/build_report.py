#!/usr/bin/env python3
"""Build 'assets/exp_msmbench/msmbench_report.html' -- the KMZ'19 MSM-benchmark experiment
report. Auto-fills from msmbench_lip_gamma_2d_pilot.json when the run lands.
Rerun: python3 assets/exp_msmbench/build_report.py
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
_sp = importlib.util.spec_from_file_location("kmzdgp", HERE / "dgp.py")
d = importlib.util.module_from_spec(_sp); _sp.loader.exec_module(d)

try:
    J = json.load(open(HERE / "msmbench_lip_gamma_2d_pilot.json"))
except Exception:
    J = None

def figcap(fig, cap): return f'<div>{fig}<p class="muted figcap">{cap}</p></div>'

# ---------------- construction figures (analytic + one draw) ----------------
xg = list(np.linspace(-1, 1, 241))
cate_fig = figcap(
    linechart([("CATE(x)", "#d62728", xg, list(d.cate(np.array(xg))), "")],
              title="True CATE (heterogeneous, oscillating)", xlab="x = X/2", ylab="CATE",
              hlines=[("0", "#888", 0.0, "4 3")], legend=False),
    "CATE(X) = 2X + 2 - 4 sin(2X); U cancels (level-shift confounder). "
    "Oracle pi*(x) = 1{CATE > 0} treats ~73% of mass.")
prop_fig = figcap(
    linechart([("P(T=1 | x, U=1)", "#2ca02c", xg, list(d.propensity(np.array(xg), np.ones(241))), ""),
               ("P(T=1 | x, U=0)", "#9467bd", xg, list(d.propensity(np.array(xg), -np.ones(241))), ""),
               ("nominal e(1, x)", "#334155", xg, list(d.e_nom(np.array(xg))), "5 4")],
              title="MSM-extremal propensities around the nominal", xlab="x", ylab="P(T=1 | x, U)"),
    "e(1|x,u) = u/rho(x, 1/G*) + (1-u)/rho(x, G*), rho(x,g) = 1 + (1/e_nom - 1)g, "
    "G* = 4.95: the MSM holds EXACTLY with known Gamma*.")

# one confounded draw: naive binned CATE-hat vs truth
obs, full = d.generate(4000, 0)
x, T, Y = obs["X"].ravel(), obs["T"], obs["Y"]
bins = np.linspace(-1, 1, 13); mid = list(0.5 * (bins[1:] + bins[:-1]))
bi = np.clip(np.digitize(x, bins) - 1, 0, 11)
ch = [float(Y[(bi == b) & (T == 1)].mean() - Y[(bi == b) & (T == 0)].mean()) for b in range(12)]
inv_fig = figcap(
    linechart([("true CATE", "#111", mid, list(d.cate(np.array(mid))), ""),
               ("naive CATE-hat (one draw, n=4000)", MC["DoublyRobust-X-X"], mid, ch, "2 3", "t")],
              title="Naive bias: under-treatment, growing in x", xlab="x", ylab="CATE",
              hlines=[("0", "#888", 0.0, "4 3")]),
    "Treated units are enriched in U=1 (low-outcome) patients, so the naive contrast is biased "
    "DOWN by ~2 E[Delta(2U-1)](1+X) -- the mirror of owgap's over-treatment.")

EQ = l2m(r"Y(a) = (2a{-}1)X + (2a{-}1) - 2\sin(2(2a{-}1)X) - 2(2U{-}1)(1+0.5X) + \mathcal{N}(0,1),"
         r"\quad X \sim \mathrm{Unif}[-2,2],\; U \sim \mathrm{Bern}(\tfrac12) \perp X")

hero = """<h1>MSM benchmark &mdash; the literature's synthetic, on our pipeline</h1>
<p>The Kallus-Mao-Zhou (2019) synthetic DGP, as used by Dorn-Guo and by Hess/Frauen et al.
(ICLR 2026) -- adapted unchanged except x = X/2 and Gamma* = 4.95 so the matched-Gamma = 5
protocol is identical to every other experiment in the paper. Its structural signature:
<b>U is INDEPENDENT of X</b> -- measured corr(x, S) = -0.007 -- making this the ZERO-COUPLING
anchor of the diagnostic map, and the home turf of box-only methods (the MSM is exactly
correctly specified with known Gamma*).</p>"""

body = [f"""
<h2>1. The DGP</h2>
<div style="text-align:center;overflow-x:auto;margin:10px 0">{EQ}</div>
<p>True propensity = the exact MSM extremal around the nominal e(1,x) = &sigma;(0.75X + 0.5)
at &Gamma;* = 4.95 (their &Gamma;* convention IS our box parametrization; the implied
u-to-u odds ratio is &Gamma;*&sup2; &asymp; 24.5, fitted 26.7 on a draw). References
(4,000 test draws): oracle +1.41, all-treat +0.98, never-treat -0.97.</p>
<div class="figrow">{cate_fig}{prop_fig}</div>
<div class="figrow">{inv_fig}</div>
<h2>2. Protocol note: no Gamma sweep</h2>
<p>Because the DGP specifies &Gamma;* = 4.95 by construction, the methods are run at the single
matched value &Gamma; = &Gamma;* &mdash; there is nothing to tune and nothing to
misspecify; only the Lipschitz dial L is swept. (In the field experiments &Gamma;* is unknown
and the full &Gamma;-curves quantify misspecification; here that axis is moot.)</p>
<h2>3. Declared predictions (written before the run)</h2>
<div class="card"><ol>
<li><b>Naive under-treats</b>, increasingly with x, and loses substantial value.</li>
<li><b>Box-only O-X at matched &Gamma; = 5 performs WELL</b> &mdash; this DGP is its exactly
specified home turf; no collapse expected (moderate outcome scale, bias covered by the box).</li>
<li><b>O-W &asymp; O-X</b>: with U &perp; X the Wasserstein balance constraint has provably
nothing to grab &mdash; balancing X cannot constrain U. The W-term should be &asymp; free but
useless here. This is the coupling diagnostic's corr = 0 prediction; confirming it means one
measured quantity explains BOTH our showcase (box fails, W needed) and the literature's
benchmark (box suffices, W idle).</li>
</ol></div>
"""]

if J:
    body.append("<h2>4. Quick-pilot results (n=200 train, 3 seeds, Gamma = Gamma* = 4.95)</h2>")
    body.append(surface_block(J, "MSM-bench"))
    g0 = J["gammas"][0]
    difL = {l: (J["surface"]["IPW-O-W"][g0][l] - J["surface"]["IPW-O-X"][g0][l]) for l in J["Lgrid"]}
    body.append(f"""
<div class="card good"><b>Verdict vs predictions.</b> VERDICT_PLACEHOLDER
(IPW-O-W minus IPW-O-X at &Gamma;* by L: {", ".join(f"L={l}: {v:+.3f}" for l, v in difL.items())}.)</div>""")
else:
    body.append('<h2>4. Quick-pilot results</h2><p class="muted">Job 10621782 running (n=200 for '
                'fast turnaround); rerun this builder when it lands.</p>')

body.append("""
<h2>5. Files</h2>
<p class="muted mono">assets/exp_msmbench/: dgp.py, msmbench_lip_gamma_2d_pilot.json,
build_report.py (this page) | scripts/sbatch_msmbench_pilot.sh | source DGP: Kallus-Mao-Zhou
2019, arXiv 1810.02894; used by Hess/Frauen et al., ICLR 2026, arXiv 2502.13022</p>""")

out = HERE / "msmbench_report.html"
out.write_text(page("MSM benchmark (KMZ'19)", "radial-gradient(130% 150% at 0% 0%,#4a4a2d 0%,#3d3d1f 46%,#1a1a0c 100%)", hero, "".join(body)))
print(f"wrote {out}")
