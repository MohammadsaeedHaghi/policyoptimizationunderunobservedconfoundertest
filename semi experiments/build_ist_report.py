#!/usr/bin/env python3
"""Build 'semi experiments/ist_report.html' -- the IST recipe-D pilot report (experiment R1).
Auto-fills from assets/exp_ist/ist_lip_gamma_2d_pilot.json when the run lands.
Rerun: python3 "semi experiments/build_ist_report.py"
"""
import json, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from report_common import page, surface_block, esc

ROOT = HERE.parent
try:
    J = json.load(open(ROOT / "assets" / "exp_ist" / "ist_lip_gamma_2d_pilot.json"))
except Exception:
    J = None

hero = """<h1>IST pilot &mdash; recipe D: real RCT outcomes, injected confounding</h1>
<p>Experiment R1. International Stroke Trial (Edinburgh DataShare, open licence): 19,285 usable
patients, T = randomized aspirin (P = 0.500 exactly), Y = REAL favorable 6-month outcome
(OCCODE &ge; 3) &mdash; no outcome synthesis anywhere. Hidden confounder S = fully alert at
baseline (RCONSC); confounding injected by biased subsampling of the RCT at
&Lambda; = 4.95 (fitted check: 5.30) &mdash; the same matched-&Gamma; = 5 protocol as the
synthetic showcase. Evaluation: Horvitz-Thompson pseudo-outcomes on an untouched 7,568-patient
RCT test split (unbiased since P(T) = 1/2).</p>"""

body = ["""
<h2>1. Construction facts (all measured)</h2>
<div class="tw"><table>
<tr><th>quantity</th><th>value</th><th>meaning</th></tr>
<tr><td>real aspirin effect (full RCT)</td><td>+0.0125</td><td>the true average effect is TINY</td></tr>
<tr><td>naive apparent effect (confounded pool)</td><td>+0.114</td><td>9x inflation from the injected selection</td></tr>
<tr><td>corr(X_obs, S) / AUC</td><td>0.567 / 0.861</td><td>upper-boundary coupling (between SUPPORT2 0.47 and Mushroom 0.71)</td></tr>
<tr><td>S &rarr; Y shift</td><td>+0.336</td><td>consciousness is strongly prognostic</td></tr>
<tr><td>fitted &Lambda; in kept pool</td><td>5.30 (target 4.95)</td><td>matched &Gamma; = 5 protocol verified</td></tr>
<tr><td>reference values (HT, test split)</td><td>never 0.354 | all 0.386 | oracle 0.387</td>
<td>aspirin mildly good for ~everyone &rarr; uncapped stakes = robustness WITHOUT paralysis</td></tr>
</table></div>
<div class="card"><b>What "promising" means here (declared before the run).</b> Because the real
effect is small and homogeneous, naive's inflated enthusiasm is DIRECTIONALLY right uncapped, so
no method can beat the oracle by much. The pilot's questions: (1) do the O-W methods HOLD near
the all-treat/oracle value (&asymp;0.386) at the matched &Gamma;=5 while box-only / Hajek
collapse toward never-treat (0.354)? &mdash; a ~+0.03 "robustness without paralysis" margin on
fully real outcomes; (2) is the L &times; &Gamma; surface sane on real data? The follow-up with
real value stakes is the CAPPED variant (allocating scarce aspirin by a distorted ranking),
not run in this pilot.</div>
"""]

if J:
    body.append("<h2>2. Pilot results (3 seeds)</h2>")
    body.append(surface_block(J, "IST-D"))
    body.append("""
<div class="card warn" id="verdict"><b>Verdict.</b> The confounding MECHANISM works exactly as designed (naive effect
estimate +0.114 vs RCT truth +0.0125), but the pilot returns an honest negative for the
uncapped VALUE story: total stakes (oracle - never-treat) are 0.016 on this evaluation, and
every method -- O-W, box-only, and naive alike -- lands within ~0.01 of the oracle
(DR-O-W 0.377, naive 0.369, never 0.367), differences at or below the Horvitz-Thompson noise
floor at 3 seeds. Aspirin's real effect is simply too small and too homogeneous for policy
VALUES to separate methods, no matter the construction. Two genuinely useful findings survive:
(1) with real bounded 0/1 outcomes the box-only methods do NOT collapse to never-treat --
worst-case pessimism is bounded when outcomes are non-negative, a methodological observation
worth one paper sentence; (2) the estimation-level story (robust bounds at matched Gamma cover
the truth ~0.01 while naive asserts +0.114) is intact and tellable. RECOMMENDATION: demote IST
from the headline portfolio to an optional comparability/diagnostic experiment; the capped
variant might create ranking stakes but the binned RCT CATE (max 0.059, noisy) caps the upside.
If a recipe-D slot is wanted, JTPA (real heterogeneous training effects + the budgeted-EWM
tradition) is the better host. SUPPORT2 and Mushroom carry the real-data section.</div>""")
else:
    body.append('<h2>2. Pilot results</h2><p class="muted">Job 10620820 running; rerun this '
                'builder when it lands.</p>')

body.append("""
<h2>3. Files</h2>
<p class="muted mono">assets/exp_ist/: prepare_ist.py, ist_prepared.npz, dgp.py,
ist_lip_gamma_2d_pilot.json | scripts/sbatch_ist_pilot.sh | raw CSV cached in
/scratch1/haghim/uci_cache/IST_corrected.csv</p>""")

out = HERE / "ist_report.html"
out.write_text(page("IST pilot (recipe D)", "radial-gradient(130% 150% at 0% 0%,#5b3a5e 0%,#462d55 46%,#1d1030 100%)", hero, "".join(body)))
print(f"wrote {out}")
