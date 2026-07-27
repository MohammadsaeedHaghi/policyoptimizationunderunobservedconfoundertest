#!/usr/bin/env python3
"""Build 'semi experiments/support2_report.html' -- the SUPPORT2 pilot report (experiment R2).
Auto-fills from assets/exp_support2/s2{b,a}_lip_gamma_2d_pilot.json as the runs land.
Rerun: python3 "semi experiments/build_support2_report.py"
"""
import json, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from report_common import page, surface_block

ROOT = HERE.parent
def load(fn):
    try: return json.load(open(ROOT / "assets" / "exp_support2" / fn))
    except Exception: return None
JB = load("s2b_lip_gamma_2d_pilot.json")
JA = load("s2a_lip_gamma_2d_pilot.json")

hero = """<h1>SUPPORT2 pilot &mdash; hidden severity in a real ICU cohort</h1>
<p>Experiment R2. SUPPORT2 (UCI 880; the SUPPORT study cohort family, like the RHC dataset of
the MSM-&Gamma; sensitivity literature): 9,075 seriously ill hospitalized adults. Hidden
confounder S = APS physiology score &gt; median; T = the REAL DNR order (P = 0.352); x = the
observable "apparent severity" composite (CV S-proxy score, rank-uniform; corr(x, S) = 0.459).
Two sub-experiments: B keeps the real (x, S, T) at the measured matched &Lambda; = 2.20 &mdash;
the PREDICTED-BOUNDARY test of the coupling diagnostic; A amplifies to &Lambda; = 4.95 with
synthetic S/T on the same real covariates &mdash; the in-regime showcase. Outcomes synthetic in
both (severity-dominant, K/G/D/TH = 3/1.5/1/0.3), calibrated story: sicker patients receive DNR
orders and die more, so the naive analysis concludes the order itself is lethal.</p>"""

body = ["""
<h2>1. Construction facts (all measured)</h2>
<div class="tw"><table>
<tr><th>quantity</th><th>value</th><th>meaning</th></tr>
<tr><td>corr(x, S) on the 1-D composite</td><td>0.459</td><td>predicted-boundary coupling (alpha-sweep band 0.55-0.75 is above, 0.47 just below)</td></tr>
<tr><td>matched &Lambda; (B, real DNR)</td><td>2.20</td><td>the honest matched-&Gamma; for experiment B</td></tr>
<tr><td>real death rates by (S, T)</td><td>0.55/0.88 (S=0), 0.56/0.94 (S=1)</td><td>DNR patients die far more &mdash; the naive story</td></tr>
<tr><td>B reference values</td><td>oracle 0.407 | naive-binned 0.327 | never 0.008 | all -0.306</td><td>boundary-sized stakes, as the diagnostic predicts</td></tr>
<tr><td>A reference values</td><td>oracle 0.801 | never 0.053 | all -0.280</td><td>in-regime stakes at &Lambda;=4.95</td></tr>
</table></div>
<div class="card warn"><b>Disclosed design choices.</b> (1) The 1-D composite is the S-proxy
direction (a clinician's observable severity impression); a death-risk composite measured
corr(x, S) = -0.07 &mdash; sociodemographic mortality risk is orthogonal to APS &mdash; and was
rejected because it silently turns R2 into a deep out-of-regime experiment. (2) Real
within-treatment S &rarr; death is weak (+0.01/+0.07); the synthetic outcomes make severity
dominant BY DESIGN &mdash; the realism claims of B are the treatment decision, the coupling,
and the covariate geometry, not the outcome scale. (3) Expectation declared before the run:
B should show a SMALL, &Gamma;-stable O-W edge at matched &Gamma; &asymp; 2 (boundary regime);
a large win would actually be suspicious.</div>
"""]

body.append("<h2>2. Experiment B results: fully real (x, S, T), matched &Gamma; = 2.2</h2>")
if JB:
    body.append(surface_block(JB, "S2-B"))
    body.append('<div class="card good"><b>Verdict B.</b> VERDICT_B_PLACEHOLDER</div>')
else:
    body.append('<p class="muted">Job 10620864 (chained after the IST pilot); rerun this builder when it lands.</p>')

body.append("<h2>3. Experiment A results: amplified &Lambda; = 4.95 on real covariates</h2>")
if JA:
    body.append(surface_block(JA, "S2-A"))
    body.append('<div class="card good"><b>Verdict A.</b> VERDICT_A_PLACEHOLDER</div>')
else:
    body.append('<p class="muted">Second half of job 10620864; rerun this builder when it lands.</p>')

body.append("""
<h2>4. Files</h2>
<p class="muted mono">assets/exp_support2/: prepare_support2.py, support2_prepared.npz, dgp_b.py,
dgp_a.py, s2b/s2a_lip_gamma_2d_pilot.json | scripts/sbatch_support2_pilot.sh | raw data cached in
/scratch1/haghim/uci_cache/support2_*.pkl</p>""")

out = HERE / "support2_report.html"
out.write_text(page("SUPPORT2 pilot (recipes B + A)", "radial-gradient(130% 150% at 0% 0%,#2d4a5e 0%,#1f3a55 46%,#0c1a30 100%)", hero, "".join(body)))
print(f"wrote {out}")
