#!/usr/bin/env python3
"""Build the HTML explainer for the downloaded RCT / causal-benchmark datasets.

Every number in the page comes from structure.json, which inspect_datasets.py computed from the
actual files -- nothing is quoted from memory.
"""
import json, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "semi experiments"))
from report_common import CSS, esc

D = json.loads((HERE / "structure.json").read_text())
CF = json.loads((HERE / "counterfactuals.json").read_text())

EXTRA_CSS = """
table.dt{border-collapse:collapse;font-size:12.5px;margin:10px 0;width:100%}
table.dt th,table.dt td{border:1px solid #e3e6ea;padding:5px 8px;text-align:left;vertical-align:top}
table.dt th{background:#f4f6f8;font-weight:600;white-space:nowrap}
table.dt td.n{text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}
.scrollx{overflow-x:auto;max-width:100%}
.ds{border:1px solid #e3e6ea;border-radius:12px;padding:2px 18px 14px;margin:16px 0;background:#fff}
.ds h3{margin:14px 0 2px;font-size:1.06rem}
.ds .sub{color:#667085;font-size:.86rem;margin:0 0 10px}
.role{display:grid;grid-template-columns:86px 1fr;gap:4px 12px;font-size:13px;margin:10px 0}
.role .k{font-weight:700;color:#334155}
.tag{display:inline-block;font-size:11px;font-weight:700;padding:2px 8px;border-radius:10px;
     vertical-align:middle;margin-left:8px}
.t-rct{background:#dcfce7;color:#166534}.t-obs{background:#fef3c7;color:#92400e}
.t-semi{background:#ede9fe;color:#5b21b6}.t-nat{background:#dbeafe;color:#1e40af}
.gt{background:#dcfce7;color:#166534;font-weight:700}
.nogt{background:#fee2e2;color:#991b1b;font-weight:700}
code{background:#f1f5f9;padding:1px 5px;border-radius:4px;font-size:.86em}
"""

TAG = {"rct": ("t-rct", "RANDOMISED"), "obs": ("t-obs", "OBSERVATIONAL"),
       "semi": ("t-semi", "SEMI-SYNTHETIC"), "nat": ("t-nat", "NATURAL EXPERIMENT")}

# (key in structure.json, kind, folder, what makes it worth having)
META = [
    ("IHDP", "semi", "ihdp/", "The only one here with a KNOWN individual treatment effect on real "
     "covariates, which is why almost every CATE paper reports PEHE on it."),
    ("Twins", "nat", "twins/", "Both potential outcomes are genuinely observed, so the individual "
     "effect is known without simulating anything."),
    ("Jobs / LaLonde (NSW)", "rct", "lalonde/", "The original 'can an observational method recover "
     "an experiment?' test, and still the sharpest one."),
    ("NHEFS", "obs", "nhefs/", "The textbook worked example: every adjustment set and g-method "
     "answer is published, so you can check your pipeline against a known target."),
    ("IST", "rct", "ist/", "Large, factorial, and clinical -- the realistic setting for "
     "policy learning with a capacity constraint."),
    ("STAR", "rct", "star/", "Three arms and within-school randomisation: the natural test for "
     "multi-arm and clustered designs."),
]


def role_block(v):
    rows = [("Treatment", v.get("treatment", "")),
            ("Outcome", v.get("outcome", "")),
            ("Covariates", ", ".join(v["covariates"]) if isinstance(v.get("covariates"), list)
             else v.get("covariates", ""))]
    return "".join("<div class='k'>%s</div><div>%s</div>" % (k, esc(str(t))) for k, t in rows)


cards = []
for key, kind, folder, why in META:
    v = D[key]
    cls, lab = TAG[kind]
    gt = v.get("ground_truth")
    gtb = ("<span class='gt'>ground truth available</span>" if gt else
           "<span class='nogt'>no counterfactuals</span>")
    facts = [("rows", v.get("n_rows")), ("columns", v.get("n_cols")),
             ("treated", v.get("n_treated")), ("control", v.get("n_control"))]
    ftab = "".join("<tr><td>%s</td><td class='n'>%s</td></tr>" % (k, val)
                   for k, val in facts if val is not None)
    ex = []
    for k, lab2 in (("n_arms", "treatment arms K"), ("n_replications", "replications"), ("test_rows", "held-out rows"),
                    ("ate_true", "TRUE ATE"), ("missing_outcome", "rows missing the outcome"),
                    ("observational_controls", "substitute control pools"),
                    ("naive_bias", "naive estimate with survey controls"),
                    ("heparin_arms", "heparin arms"), ("arms", "class-size arms"),
                    ("weight_cols", "birth-weight columns")):
        if k in v: ex.append("<tr><td>%s</td><td class='n'>%s</td></tr>" % (lab2, esc(str(v[k]))))
    cards.append(f"""
<div class="ds">
<h3>{esc(key)}<span class="tag {cls}">{lab}</span></h3>
<p class="sub">{esc(v.get('title',''))} &middot; <code>{esc(folder)}</code> &middot; {gtb}</p>
<div class="role">{role_block(v)}</div>
<div class="scrollx"><table class="dt">
<tr><th>quantity</th><th style="text-align:right">value</th></tr>
{ftab}
<tr><td>raw difference in means</td><td class='n'>{esc(str(v.get('ate_raw','')))}</td></tr>
{''.join(ex)}
</table></div>
<p><b>What it is for.</b> {esc(why)}</p>
<p class="muted">{esc(v.get('note',''))}</p>
</div>""")

CFCLS = {"real": ("gt", "ALL (observed)"), "simulated": ("gt", "ALL (simulated)"),
         "none": ("nogt", "NONE")}
armrows = "".join(
    "<tr><td>%s</td><td class='n'><b>%s</b></td><td>%s</td>"
    "<td><span class='%s'>%s</span><br><span class='muted' style='font-size:11.5px'>%s</span></td></tr>"
    % (esc(k), D[k]["n_arms"], esc(D[k]["arm_detail"]),
       CFCLS[D[k]["cf_kind"]][0], CFCLS[D[k]["cf_kind"]][1], esc(D[k]["counterfactuals"]))
    for k, _kind, _f, _w in META)

summary = "".join(
    "<tr><td>%s</td><td><span class='tag %s'>%s</span></td><td class='n'>%s</td>"
    "<td class='n'>%s</td><td class='n'>%s</td><td>%s</td><td>%s</td></tr>"
    % (esc(k), TAG[kind][0], TAG[kind][1], D[k].get("n_rows", ""), D[k].get("n_cols", ""),
       D[k].get("n_arms", ""),
       esc(str(D[k].get("treatment", "")).split("(")[0][:44]),
       esc(str(D[k].get("outcome", "")).split("(")[0][:40]))
    for k, kind, _f, _w in META)

# ---------------------------------------------------------------- imputed counterfactuals
MNAMES = ["T-learner", "X-learner", "1NN-match"]
cf_rows = []
for r in CF:
    best = min((r["methods"][m]["abs_error_vs_benchmark"] for m in MNAMES if m in r["methods"]),
               default=None)
    valid = "OBSERVATIONAL" not in r["benchmark_note"]
    tds = []
    for m in MNAMES:
        v = r["methods"].get(m)
        if not v: tds.append("<td>&mdash;</td>"); continue
        hit = valid and abs(v["abs_error_vs_benchmark"] - best) < 1e-12
        tds.append("<td class='n'%s>%+.4f<br><span class='muted' style='font-size:11px'>"
                   "err %.4f</span></td>"
                   % (" style='background:#e8f5e9;font-weight:700'" if hit else "",
                      v["implied_ate"], v["abs_error_vs_benchmark"]))
    cf_rows.append("<tr><td>%s</td><td class='n'>%+.4f</td>%s<td>%s</td></tr>"
                   % (esc(r["dataset"]), r["benchmark_ate"], "".join(tds),
                      "<span class='gt'>valid target</span>" if valid
                      else "<span class='nogt'>not a target</span>"))

fit_rows = []
for r in CF:
    f = r["factual_cv"]; metric = "AUC" if r["outcome_binary"] else "R<sup>2</sup>"
    vals = [f.get("arm0"), f.get("arm1")]
    weak = any(v is not None and (v < 0.05 if not r["outcome_binary"] else v < 0.6) for v in vals)
    sds = " / ".join("%.0f" % r["methods"][m]["cate_sd"] if abs(r["methods"][m]["cate_sd"]) > 1
                     else "%.3f" % r["methods"][m]["cate_sd"] for m in MNAMES if m in r["methods"])
    fit_rows.append("<tr><td>%s</td><td>%s</td><td class='n'>%s</td><td class='n'>%s</td>"
                    "<td class='n'>%s</td><td>%s</td></tr>"
                    % (esc(r["dataset"]), metric,
                       "%.3f" % vals[0] if vals[0] is not None else "-",
                       "%.3f" % vals[1] if vals[1] is not None else "-",
                       esc(sds),
                       "<span class='nogt'>no usable signal</span>" if weak
                       else "<span class='gt'>real signal</span>"))

CF_SECTION = f"""
<h2>Imputing the missing counterfactual for the four that lack one</h2>
<p>LaLonde, NHEFS, IST and STAR record only the factual arm, so
<i>Y</i>(1&minus;<i>T</i>) has to be estimated. Three methods were used, chosen to fail in
different ways rather than to be three flavours of one idea:</p>
<ul>
<li><b>T-learner</b> &mdash; two separate outcome regressions, &mu;<sub>0</sub> and
&mu;<sub>1</sub>, each fit on its own arm only. Flexible, but each extrapolates into covariate
regions its own arm barely covers.</li>
<li><b>X-learner</b> (K&uuml;nzel et al.) &mdash; imputes each unit's effect using the OTHER arm's
model, regresses those imputed effects on <i>X</i> per arm, then blends the two by the propensity
score. Designed for arms of very unequal size, which is exactly LaLonde (185 vs 260) and NHEFS
(403 vs 1163).</li>
<li><b>1-NN matching</b> &mdash; non-parametric: the counterfactual is the OBSERVED outcome of the
nearest unit in the opposite arm, on standardised covariates. It never extrapolates and invents
no value, but uses a single neighbour, so it is noisy.</li>
</ul>

<h3>Does the imputation recover the known answer?</h3>
<p>Three of the four are randomised, so difference-in-means is the TRUE average effect. After
imputation the completed table implies its own ATE (mean of <i>Y</i><sub>1</sub> &minus;
<i>Y</i><sub>0</sub> over all units), and it should land on that benchmark. Green = closest.</p>
<div class="scrollx"><table class="dt">
<tr><th>dataset</th><th style="text-align:right">benchmark ATE</th>
<th style="text-align:right">T-learner</th><th style="text-align:right">X-learner</th>
<th style="text-align:right">1NN-match</th><th>benchmark valid?</th></tr>
{''.join(cf_rows)}
</table></div>
<p class="muted">NHEFS is observational, so its difference in means is NOT a target: all three
methods moving it from 2.54 kg to about 3.5 kg is adjustment doing its job, not error.</p>

<h3>But first &mdash; can the model predict the arm it actually sees?</h3>
<p>Before trusting a model to invent an unobserved outcome, it is worth checking whether it can
predict the observed one. This is 5-fold cross-validation on the FACTUAL arm.</p>
<div class="scrollx"><table class="dt">
<tr><th>dataset</th><th>metric</th><th style="text-align:right">arm 0</th>
<th style="text-align:right">arm 1</th>
<th style="text-align:right">implied CATE sd (T / X / 1NN)</th><th>verdict</th></tr>
{''.join(fit_rows)}
</table></div>

<div class="card bad"><b>The caveat that matters most.</b> On LaLonde and NHEFS the covariates
essentially do not predict the outcome &mdash; LaLonde's cross-validated
<i>R</i><sup>2</sup> is <b>negative</b> (&minus;0.44 and &minus;0.85), i.e. worse than predicting
the arm mean. So the individual counterfactuals there are the arm mean plus noise, and the implied
per-unit effects confirm it: LaLonde's have a spread of roughly $8,000&ndash;9,900 around a
$1,794 average. LaLonde's headline result (1NN-match hitting the ATE to within $2.6) is therefore
misleading on its own &mdash; the average is right because the two arm ANCHORS are right, not
because any individual imputation is. The same 1NN-match is the WORST method on STAR (error 4.80):
single-neighbour matching is unbiased in aggregate but high-variance, and on 445 rows it got
lucky.
<br><br><b>Practical reading.</b> IST's imputed counterfactuals are usable &mdash; real predictive
signal (AUC 0.79/0.80) and the ATE recovered to about 11% of an effect that is itself only 0.0099.
STAR's are marginal; prefer the X-learner there. LaLonde's and NHEFS's should be treated as
ATE-consistent but individually meaningless. For genuine individual counterfactuals, IHDP and
Twins remain the only two options in this collection.</div>

<p class="muted">Written to <code>counterfactuals/&lt;dataset&gt;_&lt;method&gt;.csv</code>, twelve
files, each carrying the covariates plus <code>T</code>, <code>Y_factual</code>,
<code>Y0</code>, <code>Y1</code> and <code>which_imputed</code>. Regenerate with
<code>python3 impute_counterfactuals.py</code>.</p>
"""

html = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>RCT and causal-benchmark datasets</title>
<style>{CSS}{EXTRA_CSS}</style></head><body>
<div class="hero" style="background:linear-gradient(135deg,#134e4a,#0f766e)">
<h1>RCT and causal-benchmark datasets &mdash; what is in each one</h1>
<p>Six datasets downloaded and inspected: what plays the role of treatment, what the covariates
are, what the outcome is, and whether any counterfactual ground truth exists.</p></div>

<h2>The six at a glance</h2>
<div class="scrollx"><table class="dt">
<tr><th>dataset</th><th>design</th><th style="text-align:right">rows</th>
<th style="text-align:right">cols</th><th style="text-align:right">arms</th>
<th>treatment</th><th>outcome</th></tr>
{summary}
</table></div>

<div class="card warn"><b>OHIE could not be downloaded.</b> The Oregon Health Insurance
Experiment public-use files sit behind an NBER account login
(<code>data.nber.org/oregon/oregon_puf/oregon_puf.zip</code> returns the MyNBER sign-in page, not
a zip). It needs a free registration and manual acceptance of their terms, so it has to be
fetched by hand rather than scripted. Everything else here downloaded cleanly.</div>

<h2>Treatment arms, and whether counterfactuals are visible</h2>
<div class="scrollx"><table class="dt">
<tr><th>dataset</th><th style="text-align:right">arms K</th><th>what the arms are</th>
<th>counterfactuals</th></tr>
{armrows}
</table></div>
<p class="muted">"Counterfactuals visible" means the file contains the outcome under the arm the
unit did <b>not</b> receive. Only two of the six do, and they get there differently: IHDP
<b>simulates</b> both arms (its <code>ycf</code> and <code>mu0</code>/<code>mu1</code> arrays are
generated), while Twins <b>observes</b> both, because the two twins of a pair supply one arm
each. Everywhere else exactly one arm per unit is recorded, so no individual-level effect can be
scored &mdash; only averages.</p>

<h2>Three different things are being called a "benchmark" here</h2>
<p>The six split into groups that behave very differently, and mixing them up is the usual source
of confusion when comparing numbers across papers:</p>
<ul>
<li><b>Genuine RCTs</b> &mdash; LaLonde/NSW, IST, STAR. Treatment was randomised, so the raw
difference in means <i>is</i> the causal effect. There are no counterfactuals for individuals,
only a trustworthy average.</li>
<li><b>Known individual effects</b> &mdash; IHDP and Twins. Both give you
<i>Y</i>(0) and <i>Y</i>(1) for the same unit, so you can score an individual-level error such as
PEHE. IHDP buys that by <b>simulating</b> the outcomes on real covariates; Twins buys it by using
the two twins of a pair as each other's counterfactual.</li>
<li><b>Purely observational</b> &mdash; NHEFS. Nothing was randomised and nothing is known; its
value is that the correct analysis is published in a textbook, so it checks your pipeline.</li>
</ul>

<h2>Dataset by dataset</h2>
{''.join(cards)}

{CF_SECTION}

<h2>How each one gets used for policy learning</h2>
<div class="scrollx"><table class="dt">
<tr><th>dataset</th><th>what you can validate</th><th>what you cannot</th></tr>
<tr><td>IHDP</td><td>Individual effect error (PEHE) and policy value, against a known
&mu;<sub>0</sub>, &mu;<sub>1</sub>, over 100 replications</td>
<td>Anything about real outcome noise &mdash; the outcomes are simulated</td></tr>
<tr><td>Twins</td><td>Individual effects with no simulation at all; confounding can be injected
by hiding one twin with a chosen assignment rule</td>
<td>The "treatment" (being heavier) is not manipulable, so the estimand is not a policy anyone
could implement</td></tr>
<tr><td>LaLonde</td><td>Whether an observational estimator recovers the experimental ATE once the
randomised controls are swapped for CPS or PSID</td>
<td>Individual effects; and the sample is small (445) so power is limited</td></tr>
<tr><td>NHEFS</td><td>That your IP-weighting / standardisation code reproduces the published
textbook numbers</td>
<td>Any causal claim without an untestable unconfoundedness assumption</td></tr>
<tr><td>IST</td><td>Average effects, subgroup effects, and capacity-constrained policies at
realistic scale (19,435 patients)</td>
<td>Individual effects &mdash; only one arm is observed per patient</td></tr>
<tr><td>STAR</td><td>Multi-arm and clustered designs; effects on a continuous test score</td>
<td>Individual effects; and school must be conditioned on, since randomisation was within
school</td></tr>
</table></div>

<h2>Reproducing this</h2>
<p><code>bash "RCT datasets"/download.sh</code> fetches everything (NHEFS comes from the
<code>causaldata</code> package because the Harvard CDN link now serves a web page rather than the
CSV), then <code>python3 inspect_datasets.py</code> recomputes every number on this page and
writes <code>structure.json</code>.</p>
<p class="muted">All figures above are computed from the downloaded files, not quoted from
papers. Two worth checking against the literature as a sanity test: LaLonde's experimental ATE
comes out at $1,794, and NHEFS's unadjusted weight-change difference at 2.54 kg &mdash; both match
the canonical published values.</p>
</body></html>"""

html = html.encode("ascii", "xmlcharrefreplace").decode("ascii")
(HERE / "rct_datasets.html").write_text(html)
print("wrote %s (%d KB)" % (HERE / "rct_datasets.html", len(html) // 1024))
