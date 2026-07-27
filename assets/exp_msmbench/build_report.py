#!/usr/bin/env python3
"""Build 'assets/exp_msmbench/msmbench_report.html' -- the beta=0 ANCHOR page: the original
Kallus-Mao-Zhou MSM synthetic, unmodified, run once as the zero-coupling reference point.
The full coupled experiment lives in assets/exp_coupled_synth/ (coupled_report.html).
Rerun: python3 assets/exp_msmbench/build_report.py
"""
import json, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT / "semi experiments"))
from report_common import page, surface_block

J = json.load(open(HERE / "msmbench_lip_gamma_2d_pilot.json"))

hero = """<h1>Zero-coupling anchor: the standard MSM synthetic (KMZ'19), unmodified</h1>
<p>The original Kallus-Mao-Zhou synthetic (U ~ Bern(1/2) INDEPENDENT of X, exact MSM extremal
propensity, Gamma* = 4.95 known), run once at the single matched Gamma. Role: the corr(x,S) = 0,
kappa = 1 corner of our coupled-confounder experiment's two-dial design -- the point where the
Wasserstein term is PROVABLY useless (no X-balance device can constrain a confounder that is
independent of X) and the confounder's outcome leverage is mild, so the honest requirement on
O-W is only "do no harm". The full experiment -- coupling dial beta, leverage dial kappa, and
the opened O-W gap (+0.411 at the headline setting kappa=2.5, beta=10, every seed positive) --
is reported in assets/exp_coupled_synth/coupled_report.html
(<a href="https://claude.ai/code/artifact/c61ab76f-d4cc-4624-bc21-3b96f25a2da8"
style="color:#bbf7d0">published copy</a>).</p>"""

body = ["<h2>Results (n=200, 3 seeds, Gamma = Gamma* = 4.95)</h2>", surface_block(J, "beta=0 anchor"), """
<div class="card good"><b>Findings.</b> Naive collapses (-0.254 vs oracle 1.439); box-only
excels on its correctly-specified home turf (IPW-O-X 1.078 @ L=5, no collapse); O-W ties
box-only within noise (-0.012 / -0.002) -- with U independent of X, NO X-balance device can
constrain the confounder, so "do no harm" is the strongest achievable result, and O-W achieves
it.</div>
<div class="card"><b>Where this sits in the two-dial design.</b> This anchor is the
(beta = 0, kappa = 1) corner. Turning the COUPLING dial beta makes the confounder X-trackable
(the W-term switches on between corr 0.58 and 0.76 -- the boundary the showcase's coupling
sweep predicted); turning the LEVERAGE dial kappa makes the confounder dominate the outcome
scale (the O-W gap over box-only opens to +0.411 at kappa=2.5, beta=10 -- IPW-O-W best method
on the board -- and +0.727 at kappa=4). Two measured quantities explain the whole spectrum,
from this tie to the decisive win, with the low-dial corners honestly on the table.</div>
<p class="muted mono">Files: dgp.py (original construction, unmodified),
msmbench_lip_gamma_2d_pilot.json, build_report.py.</p>"""]

out = HERE / "msmbench_report.html"
out.write_text(page("Zero-coupling anchor (KMZ'19)",
                    "radial-gradient(130% 150% at 0% 0%,#4a4a2d 0%,#3d3d1f 46%,#1a1a0c 100%)",
                    hero, "".join(body)))
print(f"wrote {out}")
