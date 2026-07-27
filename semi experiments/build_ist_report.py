#!/usr/bin/env python3
"""Build 'semi experiments/ist_report.html' -- the FULL IST recipe-D pilot report (experiment R1).

All construction figures are computed from assets/exp_ist/ist_prepared.npz (real data), results
from assets/exp_ist/ist_lip_gamma_2d_pilot.json. House style: math-definition captions, family
color code, MathML via latex2mathml, pure-ASCII output.
Rerun: python3 "semi experiments/build_ist_report.py"
"""
import json, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from report_common import page, surface_block, linechart, heatmap, ser, MC, esc
from latex2mathml.converter import convert as l2m

ROOT = HERE.parent
D = np.load(ROOT / "assets" / "exp_ist" / "ist_prepared.npz")
x, S, T, Y = D["x"], D["S"], D["T"], D["Y"].astype(float)
keep, test = D["keep"], D["test"]
BINS, CATE_SM = D["bins"], D["cate_sm"]
tr = ~test
try:
    J = json.load(open(ROOT / "assets" / "exp_ist" / "ist_lip_gamma_2d_pilot.json"))
except Exception:
    J = None

def figcap(fig, cap): return f'<div>{fig}<p class="muted figcap">{cap}</p></div>'

def binstat(mask, val, nb=15):
    edges = np.linspace(-1, 1, nb + 1); mid = 0.5 * (edges[1:] + edges[:-1])
    bi = np.clip(np.digitize(x, edges) - 1, 0, nb - 1)
    out = [val[mask & (bi == b)].mean() if (mask & (bi == b)).sum() > 15 else np.nan for b in range(nb)]
    return list(mid), out

# ---------------- construction figures (all from real data) ----------------
mid, pS = binstat(np.ones_like(T, bool), S.astype(float))
coup_fig = figcap(
    linechart([("P(S=1 | x)", "#334155", mid, pS, "")], title="Real coupling: hidden alertness vs the prognostic composite",
              xlab="x (prognostic composite, rank-uniform)", ylab="P(S=1 | x)", legend=False),
    "P(S=1 | x), binned. corr(x, S) = %.2f; with the full multivariate X_obs, corr = 0.57 (AUC 0.86)." % np.corrcoef(x, S)[0, 1])

_, y1r = binstat(tr & (T == 1), Y); _, y0r = binstat(tr & (T == 0), Y)
_, y1k = binstat(keep & (T == 1), Y); _, y0k = binstat(keep & (T == 0), Y)
arm_fig = figcap(
    linechart([("E[Y | T=1, x] RCT", MC["IPW-O-W"], mid, y1r, ""),
               ("E[Y | T=0, x] RCT", "#334155", mid, y0r, ""),
               ("E[Y | T=1, x] confounded", MC["IPW-O-W"], mid, y1k, "7 3"),
               ("E[Y | T=0, x] confounded", "#334155", mid, y0k, "7 3")],
              title="Arm means: RCT truth (solid) vs confounded pool (dashed)", xlab="x", ylab="P(favorable outcome)"),
    "Solid: unconfounded RCT arm means (their difference = true CATE(x), tiny). Dashed: the same "
    "quantities in the biased pool -- the treated curve is lifted and the control curve depressed "
    "by selection on hidden alertness.")

cate_true = [a - b if (a == a and b == b) else np.nan for a, b in zip(y1r, y0r)]
cate_conf = [a - b if (a == a and b == b) else np.nan for a, b in zip(y1k, y0k)]
inv_fig = figcap(
    linechart([("true CATE (RCT)", "#111", mid, cate_true, ""),
               ("naive CATE-hat (confounded pool)", MC["DoublyRobust-X-X"], mid, cate_conf, "2 3", "t")],
              title="The real-data ranking inversion", xlab="x", ylab="CATE",
              hlines=[("0", "#888", 0.0, "4 3")]),
    "Black: E[Y|T=1,x] - E[Y|T=0,x] on the RCT (unbiased). Blue: the same contrast in the "
    "confounded pool. The vertical gap (~+0.10 everywhere) is pure injected confounding bias; "
    "unlike the synthetic showcase it is FLAT in x, because P(S|x) never crosses 1/2 steeply.")

_, pS1 = binstat(keep & (T == 1), S.astype(float)); _, pS0 = binstat(keep & (T == 0), S.astype(float))
comp_fig = figcap(
    linechart([("P(S=1 | T=1, x) kept", MC["IPW-O-W"], mid, pS1, ""),
               ("P(S=1 | T=0, x) kept", "#334155", mid, pS0, "")],
              title="Selection composition in the confounded pool", xlab="x", ylab="P(S=1 | T, x)"),
    "The injected logging enriches the treated arm in alert patients at every x "
    "(target conditional odds ratio Lambda = 4.95; fitted 5.30).")

EQ_HT = l2m(r"\hat Y_i(1) = \frac{T_i Y_i}{1/2},\quad \hat Y_i(0) = \frac{(1-T_i) Y_i}{1/2}"
            r"\;\Rightarrow\; \mathbb{E}\big[\pi \hat Y(1) + (1-\pi)\hat Y(0)\big]"
            r" = \mathbb{E}\big[\pi Y(1) + (1-\pi) Y(0)\big]")

hero = """<h1>IST pilot &mdash; recipe D: real RCT outcomes, injected confounding</h1>
<p>Experiment R1. International Stroke Trial (Edinburgh DataShare, open licence): 19,285 usable
patients, T = randomized aspirin (P = 0.500 exactly), Y = REAL favorable 6-month outcome
(OCCODE &ge; 3) &mdash; no outcome synthesis anywhere. Hidden confounder S = fully alert at
baseline (RCONSC); confounding injected by biased subsampling of the RCT at &Lambda; = 4.95 --
the same matched-&Gamma; = 5 protocol as the synthetic showcase.</p>"""

body = [f"""
<h2>1. The data and the injected confounding</h2>
<p>x = prognostic composite (logistic P(favorable | X_obs) fitted on the TRAIN CONTROL arm only,
rank-uniform on [-1,1]); X_obs = age, sex, blood pressure, stroke subtype, 8 deficit indicators
&mdash; consciousness excluded. S = 1{{fully alert}} (P = 0.77, S&rarr;Y = +0.34). The
observational study is simulated by keeping patient i with probability q(T_i, S_i):
q(1,1) = q(0,0) = a, q(1,0) = q(0,1) = b, a/b = &radic;&Lambda;, which makes the
conditional-on-x selection odds ratio exactly &Lambda; and leaves 6,609 patients
(P(T | kept) = 0.605).</p>
<div class="figrow">{coup_fig}{comp_fig}</div>
<div class="figrow">{arm_fig}{inv_fig}</div>
<h2>2. Evaluation: fully real ground truth</h2>
<p>Policies are scored on the untouched 40% RCT test split via Horvitz-Thompson pseudo
potential outcomes (unbiased because T was randomized at exactly 1/2, independent of X):</p>
<div style="text-align:center;overflow-x:auto;margin:10px 0">{EQ_HT}</div>
<p>Learned policies deploy off-support via the Shapley extension, as in all continuous
experiments. Reference values on this evaluation: oracle 0.383, all-treat 0.377, naive DR
0.369, never-treat 0.367 &mdash; <b>the total spread is 0.016</b>, which is the pilot's central
fact (see the verdict).</p>
"""]

if J:
    body.append("<h2>3. Pilot results (3 seeds)</h2>")
    body.append(surface_block(J, "IST-D"))
    ps = J.get("policies_seed0")
    if ps:
        pg = J["policy_grid"]
        series = [("oracle policy", "#111111", pg, ps["_refs"]["oracle"], "5 4"),
                  ("naive DR policy", MC["DoublyRobust-X-X"], pg, ps["_refs"]["naive_dr"], "2 3")]
        for m in ("IPW-O-W", "DoublyRobust-O-W"):
            b = J["best"][m]
            series.append(ser(m, pg, ps[m][b["gamma"]][b["L"]], lab=f"{m} @ G={b['gamma']}, L={b['L']}"))
        body.append(figcap(
            linechart(series, title="Best-cell policies vs oracle and naive (seed 0)", xlab="x", ylab="pi(x)", W=760),
            "pi(x) = treatment probability. The oracle (binned RCT CATE > 0) treats most of the "
            "range; with near-zero heterogeneous effect, policy DIFFERENCES barely move the value."))
    body.append("""
<h2>4. Findings and verdict</h2>
<div class="card"><b>Finding 1 (methodological, positive): no box-only collapse on bounded
outcomes.</b> In the synthetic showcase the box-only methods dive to never-treat as &Gamma;
grows because the adversary can push weighted outcomes arbitrarily negative. Here Y &isin;
{0, 1}: worst-case pessimism is BOUNDED (a reweighted non-negative outcome stays non-negative),
so O-X values stay flat in &Gamma; instead of collapsing. One-sentence paper note: the
collapse phenomenon is an interaction of the odds-box with unbounded/signed outcomes, not a
universal property of box-only robustness.</div>
<div class="card"><b>Finding 2 (the estimation story survives).</b> Naive analysis of the
confounded pool asserts an aspirin effect of +0.114; the RCT truth is +0.0125. Robust
worst-case bounds at the matched &Gamma; cover the truth. As an ESTIMATION-level demonstration
on fully real outcomes this is intact -- it is the policy-VALUE experiment that is flat.</div>
<div class="card warn"><b>Verdict.</b> The confounding mechanism works exactly as designed, but
the uncapped value experiment returns an honest negative: total stakes (oracle - never-treat)
are 0.016 on this evaluation and every method -- O-W, box-only, naive -- lands within ~0.01 of
the oracle (DR-O-W 0.377, naive 0.369, never 0.367), at the Horvitz-Thompson noise floor for 3
seeds. Aspirin's real effect is too small and too homogeneous for policy values to separate
methods, regardless of construction. RECOMMENDATION: demote IST from the headline portfolio to
an optional comparability/diagnostic experiment; the capped variant might create ranking stakes
but the binned RCT CATE (max 0.059, noisy) caps the upside. If a recipe-D slot is wanted, JTPA
(real heterogeneous training effects + the budgeted-EWM tradition) is the better host. SUPPORT2
and Mushroom carry the real-data section.</div>""")
else:
    body.append('<h2>3. Pilot results</h2><p class="muted">Job 10620820 running; rerun this builder when it lands.</p>')

body.append("""
<h2>5. Files</h2>
<p class="muted mono">assets/exp_ist/: prepare_ist.py, ist_prepared.npz, dgp.py,
ist_lip_gamma_2d_pilot.json | scripts/sbatch_ist_pilot.sh | raw CSV cached in
/scratch1/haghim/uci_cache/IST_corrected.csv (source: datashare.ed.ac.uk/handle/10283/124)</p>""")

out = HERE / "ist_report.html"
out.write_text(page("IST pilot (recipe D)", "radial-gradient(130% 150% at 0% 0%,#5b3a5e 0%,#462d55 46%,#1d1030 100%)", hero, "".join(body)))
print(f"wrote {out}")
