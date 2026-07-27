#!/usr/bin/env python3
"""Build 'semi experiments/semi_options.html' -- the survey of UCI semi-synthetic dataset
options for the OWGAP paper. Reads smoke_results.json (real measured numbers, no placeholders).
Pure ASCII output; family color code n/a here (no method curves) but house style matches the
main report. Rerun: python3 "semi experiments/semi_build.py"
"""
import json, os
from pathlib import Path

HERE = Path(__file__).resolve().parent
R = json.load(open(HERE / "smoke_results.json"))
OUT = HERE / "semi_options.html"

def esc(s): return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

# ---------------- the diagnostic map (corr vs Lambda) ----------------
PTS = [  # name, corr_XS, lambda (real; None -> designer-chosen), verdict color
    ("Mushroom", R["mushroom"]["corr_XS"], None, "#0a7d33", "C"),
    ("SUPPORT2", R["support2"]["corr_XS"], R["support2"]["lambda"], "#0a7d33", "B/A"),
    ("Heart", R["heart"]["corr_XS"], None, "#b45309", "A"),
    ("Adult", R["adult"]["corr_XS"], None, "#b45309", "A"),
    ("Student", R["student"]["corr_XS"], R["student"]["lambda"], "#b45309", "B"),
    ("Diabetes-130", R["diabetes130_reference"]["corr_XS"], R["diabetes130_reference"]["lambda"], "#64748b", "done"),
    ("Bank Mktg", R["bank_marketing"]["corr_XS"], R["bank_marketing"]["lambda"], "#b91c1c", "-"),
    ("Credit Default", R["credit_default"]["corr_XS"], R["credit_default"]["lambda"], "#0a7d33", "B-diag"),
]

def diagmap():
    W, H, pL, pR, pT, pB = 720, 360, 60, 20, 30, 46
    x0, x1, y0, y1 = 0.0, 1.0, 0.9, 3.2
    X = lambda v: pL + (v - x0) / (x1 - x0) * (W - pL - pR)
    Y = lambda v: H - pB - (v - y0) / (y1 - y0) * (H - pT - pB)
    p = [f'<svg viewBox="0 0 {W} {H}" class="chart">']
    p.append(f'<text x="{pL}" y="16" class="ct">The diagnostic map: measured coupling vs realized selection strength</text>')
    # alpha-sweep regime shading on the coupling axis (corr >= ~0.75 in-regime, 0.55-0.75 boundary)
    p.append(f'<rect x="{X(0.75):.0f}" y="{pT}" width="{X(1.0)-X(0.75):.0f}" height="{H-pT-pB}" fill="#0a7d33" opacity="0.10"/>')
    p.append(f'<rect x="{X(0.55):.0f}" y="{pT}" width="{X(0.75)-X(0.55):.0f}" height="{H-pT-pB}" fill="#b45309" opacity="0.10"/>')
    p.append(f'<text x="{X(0.87):.0f}" y="{pT+14}" class="tk" text-anchor="middle">O-W wins (alpha-sweep)</text>')
    p.append(f'<text x="{X(0.65):.0f}" y="{pT+14}" class="tk" text-anchor="middle">boundary</text>')
    p.append(f'<text x="{X(0.27):.0f}" y="{pT+14}" class="tk" text-anchor="middle">real coupling too weak: synthesize S (recipe A) or use as diagnostic</text>')
    for t in (0, 0.2, 0.4, 0.6, 0.8, 1.0):
        p.append(f'<line x1="{X(t):.0f}" y1="{pT}" x2="{X(t):.0f}" y2="{H-pB}" class="grid"/>')
        p.append(f'<text x="{X(t):.0f}" y="{H-pB+16}" class="tk" text-anchor="middle">{t:g}</text>')
    for t in (1.0, 1.5, 2.0, 2.5, 3.0):
        p.append(f'<line x1="{pL}" y1="{Y(t):.0f}" x2="{W-pR}" y2="{Y(t):.0f}" class="grid"/>')
        p.append(f'<text x="{pL-6}" y="{Y(t)+3.5:.0f}" class="tk" text-anchor="end">{t:g}</text>')
    for name, c, lam, col, _ in PTS:
        yv = lam if lam else 1.0
        mk = 'r="6"' if lam else 'r="6" stroke-dasharray="2 2"'
        if lam:
            p.append(f'<circle cx="{X(c):.0f}" cy="{Y(yv):.0f}" r="6" fill="{col}"/>')
        else:
            p.append(f'<circle cx="{X(c):.0f}" cy="{Y(yv):.0f}" r="6" fill="none" stroke="{col}" stroke-width="2.4"/>')
        dy = -10 if name not in ("Diabetes-130",) else 16
        p.append(f'<text x="{X(c):.0f}" y="{Y(yv)+dy:.0f}" class="pl" text-anchor="middle" fill="{col}">{esc(name)}</text>')
    p.append(f'<line x1="{pL}" y1="{H-pB}" x2="{W-pR}" y2="{H-pB}" class="ax"/>')
    p.append(f'<line x1="{pL}" y1="{pT}" x2="{pL}" y2="{H-pB}" class="ax"/>')
    p.append(f'<text x="{(pL+W-pR)/2:.0f}" y="{H-8}" class="al" text-anchor="middle">measured corr(X_obs proxy, S)  --  the alpha-sweep operating-regime axis</text>')
    p.append(f'<text x="14" y="{(pT+H-pB)/2:.0f}" class="al" text-anchor="middle" transform="rotate(-90 14 {(pT+H-pB)/2:.0f})">realized Lambda (real T)</text>')
    p.append('</svg>')
    return f'<figure class="fig">{"".join(p)}</figure><p class="muted figcap">Filled dot: real treatment exists, Lambda measured by logistic T ~ X_obs + S (S coded &plusmn;1, Lambda = e<sup>2|&beta;_S|</sup>). Open dot: no natural T (Lambda is a design knob; drawn at 1). Shading: the coupling regimes from the alpha sweep (corr &gtrsim; 0.75 in-regime, &lesssim; 0.55 out).</p>'

def card(title, r, story, design, verdict, vclass):
    lam = f"{r['lambda']:.2f}" if "lambda" in r else "&mdash; (design knob)"
    pt = f"{r['P_T']:.2f}" if "P_T" in r else "&mdash;"
    return f"""<div class="card {vclass}">
<h3>{esc(title)}</h3>
<p class="muted mono">n = {r['n']:,} &nbsp;|&nbsp; corr(X,S) = {r['corr_XS']:.2f} (AUC {r['auc_XS']:.2f})
&nbsp;|&nbsp; realized &Lambda; = {lam} &nbsp;|&nbsp; P(T) = {pt} &nbsp;|&nbsp; S&rarr;Y shift = {r['s_to_y']:+.2f}</p>
<p><b>Hidden confounder:</b> {esc(r['S_def'])}. <b>Story:</b> {story}</p>
<p><b>Proposed design:</b> {design}</p>
<p><b>Verdict:</b> {verdict}</p>
</div>"""

html = ["""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Semi-synthetic dataset options (UCI)</title>
<style>
:root{--bg:#f5f6fb;--surface:#ffffff;--fg:#0f172a;--muted:#667085;--border:#e9ebf3;
 --accent:#334155;--accent-soft:#f1f5f9;--good:#0a7d33;--warn:#b45309;--bad:#b91c1c;--sh:0 6px 18px rgba(16,24,40,.09);}
@media (prefers-color-scheme: dark){:root{--bg:#0e1420;--surface:#161d2b;--fg:#e6eaf2;--muted:#93a0b4;
 --border:#263043;--accent:#9fb2c9;--accent-soft:#1d2636;--good:#4ade80;--warn:#fbbf24;--bad:#f87171;--sh:0 6px 18px rgba(0,0,0,.35);}}
:root[data-theme="dark"]{--bg:#0e1420;--surface:#161d2b;--fg:#e6eaf2;--muted:#93a0b4;--border:#263043;
 --accent:#9fb2c9;--accent-soft:#1d2636;--good:#4ade80;--warn:#fbbf24;--bad:#f87171;--sh:0 6px 18px rgba(0,0,0,.35);}
:root[data-theme="light"]{--bg:#f5f6fb;--surface:#ffffff;--fg:#0f172a;--muted:#667085;--border:#e9ebf3;
 --accent:#334155;--accent-soft:#f1f5f9;--good:#0a7d33;--warn:#b45309;--bad:#b91c1c;--sh:0 6px 18px rgba(16,24,40,.09);}
*{box-sizing:border-box}
body{font-family:Inter,-apple-system,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;background:var(--bg);
 color:var(--fg);margin:0 auto;max-width:980px;padding:0 22px 80px;line-height:1.6;}
code,.mono{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;font-size:.85em;}
.hero{color:#fff;border-radius:20px;padding:26px 30px;margin:18px 0 20px;
 background:radial-gradient(130% 150% at 0% 0%,#3b5f47 0%,#28513a 46%,#0f2a1c 100%);box-shadow:var(--sh);}
.hero h1{margin:0;font-weight:800;letter-spacing:-.03em;font-size:1.7rem;}
.hero p{margin:.5rem 0 0;color:#d9e8de;max-width:820px;}
h2{font-size:1.3rem;font-weight:750;margin:36px 0 6px;} h3{font-size:1.02rem;font-weight:700;margin:0 0 4px;}
p{margin:.5rem 0;} .muted{color:var(--muted);}
.card{background:var(--surface);border:1px solid var(--border);border-radius:16px;padding:14px 18px;margin:12px 0;box-shadow:var(--sh);}
.good{border-left:4px solid var(--good);} .warn{border-left:4px solid var(--warn);} .bad{border-left:4px solid var(--bad);}
table{border-collapse:collapse;width:100%;margin:10px 0;font-variant-numeric:tabular-nums;font-size:.88rem;}
.tw{overflow-x:auto;}
th{font-size:.74rem;letter-spacing:.04em;text-transform:uppercase;color:var(--muted);text-align:right;padding:6px 9px;border-bottom:2px solid var(--border);}
th:first-child,td:first-child{text-align:left;} td{padding:6px 9px;border-bottom:1px solid var(--border);text-align:right;}
td.g{color:var(--good);font-weight:700;} td.b{color:var(--bad);font-weight:700;}
.fig{margin:14px 0;background:var(--surface);border:1px solid var(--border);border-radius:14px;padding:10px 10px 4px;}
svg.chart{width:100%;height:auto;display:block;}
.ct{font-size:12.5px;font-weight:700;fill:var(--fg);} .tk{font-size:10px;fill:var(--muted);}
.al{font-size:11px;fill:var(--muted);font-weight:600;} .pl{font-size:10.5px;font-weight:700;}
.grid{stroke:var(--border);stroke-width:1;} .ax{stroke:var(--muted);stroke-width:1.2;}
.figcap{font-size:.8rem;line-height:1.5;margin:4px 4px 14px;}
</style></head><body>
<div class="hero"><h1>Semi-synthetic dataset options from UCI</h1>
<p>Survey + smoke tests for the paper's real-data section. Every number below is measured on the
actual downloaded data (script: smoke_tests.py). Selection principle: a dataset earns a slot only
if it either (i) sits in the O-W operating regime by the measured coupling diagnostic, or
(ii) deliberately probes the regime boundary the alpha sweep predicts.</p></div>"""]

html.append("""
<h2>1. What makes the real-data section unrejectable</h2>
<p>The alpha sweep gives a falsifiable prediction: O-W beats naive when the observed covariates
proxy the hidden confounder well (corr &gtrsim; 0.75), ties around 0.55&ndash;0.75, and loses
below. So the strategy is NOT "find a dataset where we win" &mdash; it is to populate the
coupling axis with real datasets and show the prediction holds at every point. A reviewer can
reject a cherry-picked win; a validated diagnostic that says where the method should and should
not be used is much harder to dismiss.</p>
<h2>2. Three construction recipes</h2>
<div class="card"><b>A &mdash; real covariates, synthetic confounding (ACIC-style).</b> Bootstrap
real X for realistic geometry; synthesize S with designed coupling, T with designed selection
(matched-&Gamma; protocol intact), and Y(0)/Y(1) for ground truth. Any tabular dataset works;
the dataset contributes covariate realism. Already built once (Diabetes-A).</div>
<div class="card"><b>B &mdash; real X, S, T; synthetic outcomes only.</b> Keep a REAL treatment
decision and a REAL hidden variable; synthesize only Y(0)/Y(1) (needed for ground truth,
calibrated to the real outcome). The coupling and Lambda are facts about the world, not knobs
&mdash; strongest realism claim, but the regime is whatever the data gives. Already built once
(Diabetes-B, out-of-regime by design).</div>
<div class="card"><b>C &mdash; classification-to-policy with a hidden informative feature.</b>
Standard bandit-literature conversion (Dudik et al.; POEM): actions = decisions, reward =
correctness; confounding is created by HIDING an informative feature and letting the logging
policy use it. The X&ndash;S coupling is real (how well the remaining features predict the
hidden one); the selection strength is a knob, so the matched-&Gamma; protocol applies exactly.</div>""")

html.append("<h2>3. The measured landscape</h2>" + diagmap())

rows = []
for key, label in [("mushroom", "Mushroom"), ("support2", "SUPPORT2"), ("credit_default", "Credit Default"),
                   ("adult", "Adult"), ("student", "Student Perf."), ("heart", "Heart (Cleveland)"),
                   ("bank_marketing", "Bank Marketing"), ("diabetes130_reference", "Diabetes-130 (done)")]:
    r = R[key]
    if r.get("status") != "ok": continue
    lam = f"{r['lambda']:.2f}" if "lambda" in r else "knob"
    sy = f"{r['s_to_y']:+.2f}" if "s_to_y" in r else "&mdash;"
    rows.append(f"<tr><td>{label}</td><td>{r['n']:,}</td><td>{r['corr_XS']:.2f}</td>"
                f"<td>{r.get('auc_XS', float('nan')):.2f}</td><td>{lam}</td><td>{sy}</td>"
                f"<td>{esc(r.get('T_real') or 'synthetic')}</td><td>{esc(r['design'])}</td></tr>")
html.append('<div class="tw"><table><tr><th>dataset</th><th>n</th><th>corr(X,S)</th><th>AUC</th>'
            '<th>real &Lambda;</th><th>S&rarr;Y</th><th>treatment</th><th>recipe</th></tr>'
            + "".join(rows) + "</table></div>")

html.append("<h2>4. The candidates, ranked</h2>")
html.append(card("1. Mushroom (UCI 73) — the near-in-regime showcase", R["mushroom"],
    "a forager decides whether to eat a mushroom by SMELL (odor is nearly deterministic for "
    "edibility: S&rarr;Y shift +0.97); the dataset we learn from records everything EXCEPT odor. "
    "Eating outcomes look great for the forager's picks &mdash; vitality-style selection, "
    "perfectly natural.",
    "Recipe C. X = 20 morphological features, S = bad odor (hidden), logging policy eats with "
    "P(eat|S) set to hit &Lambda;=5 (matched-&Gamma; protocol identical to owgap). Y(eat) with "
    "asymmetric penalty for poisonous, Y(pass)=0; capped variant = 'eat at most 30%' basket "
    "budget. Measured corr(X,S)=0.71 (AUC 0.90) &mdash; just below the 0.75 in-regime line: "
    "quote as the near-boundary point, and (honestly disclosed) an X-augmented variant "
    "(+spore-print) can nudge coupling up.",
    "STRONG ADD. Real coupling near the regime boundary, huge S&rarr;Y, natural story, n=8,124 "
    "(LP-friendly), zero licensing issues. Complements owgap rather than repeating it.", "good"))
html.append(card("2. SUPPORT2 (UCI 880) — the medical headline", R["support2"],
    "seriously ill hospitalized adults; the DNR order is a REAL clinical decision driven by "
    "physiologic severity (APS) that we hide; DNR patients die far more often &rarr; a naive "
    "analysis concludes DNR is lethal. The classic confounding story (same family as the famous "
    "SUPPORT right-heart-cath analyses), on public data.",
    "Two experiments, mirroring the Diabetes pair but STRONGER: (B) fully real (X, S=APS, "
    "T=DNR): measured &Lambda;=2.59 &mdash; twice Diabetes-B's 1.26 &mdash; with coupling 0.47, "
    "squarely in the PREDICTED-BOUNDARY band: the diagnostic forecasts a small/unstable O-W "
    "edge, a genuinely risky (= credible) test. (A) amplified variant on the real covariates "
    "with synthetic S/T at matched &Gamma;=5, like Diabetes-A.",
    "STRONG ADD. Best realized selection strength of any candidate, publication-grade clinical "
    "narrative, n=9,075. Fills the empty middle of the diagnostic map.", "good"))
html.append(card("3. Credit-Card Default (UCI 350) — the falsification point", R["credit_default"],
    "the bank sets credit limits using the applicant's delinquency file (hidden to us); "
    "delinquency also drives default. Demographics alone carry almost NO information about the "
    "file: corr(X,S)=0.03.",
    "Recipe B: real limit decision (&Lambda;=2.45!) with essentially zero X&ndash;S coupling "
    "&mdash; the extreme out-of-regime point. Prediction: robust methods at matched &Gamma; "
    "over-hedge and lose to naive; the diagnostic says DON'T USE the method here, and the "
    "experiment shows the diagnostic is right.",
    "ADD AS DIAGNOSTIC. Cheap (one L x Gamma sweep), and it turns 'our method can lose' into "
    "evidence FOR the paper. Pairs with Diabetes-B (0.20) to bracket the low end.", "good"))
html.append(card("4. Adult / Census Income (UCI 2) — the recipe-A platform", R["adult"],
    "rich, familiar socio-economic covariates (n=47,876); no natural treatment, so it "
    "contributes realism to a fully-designed experiment.",
    "Recipe A at matched &Gamma;=5 (like Diabetes-A) if a second in-regime real-X experiment "
    "is wanted; multivariate X comes for free, which would double as the multivariate "
    "robustness check the synthetic section lacks.",
    "OPTIONAL. Use only if reviewers demand a second ACIC-style experiment or multivariate X; "
    "otherwise redundant with Diabetes-A.", "warn"))
html.append(card("5. Student Performance (UCI 320) — natural but underpowered", R["student"],
    "real paid-tutoring decision (&Lambda;=2.19 vs hidden parental education), real grades.",
    "Recipe B, but n=649 and P(T)=0.06: ~39 treated students. The Wasserstein LP is fine with "
    "n but the treated arm is too thin for stable weights.",
    "PASS. Charming story, insufficient power.", "warn"))
html.append(card("6. Bank Marketing (UCI 222) — ruled out by measurement", R["bank_marketing"],
    "re-contact decisions barely load on hidden wealth (&Lambda;=1.11) and demographics do not "
    "proxy balance (corr 0.14).",
    "None viable: no confounding to correct and no coupling to exploit.",
    "REJECT. Kept in the table as evidence the screening is real.", "bad"))
html.append(card("7. Heart Disease Cleveland (UCI 45) — too small", R["heart"],
    "coupling 0.47 is interesting but n=303 with ~20 solver seeds would be all noise.",
    "None at this size; the SUPPORT2 clinical story dominates it anyway.",
    "REJECT (size).", "bad"))

html.append("""
<h2>5. Recommended portfolio and cost</h2>
<div class="card good"><b>Proposal.</b> Real-data section = <b>Diabetes pair (done) + Mushroom
(recipe C) + SUPPORT2 pair (B real + A amplified) + Credit-Default (diagnostic point)</b>. The
diagnostic map then contains SIX measured couplings (0.03, 0.20, 0.47, 0.71, plus the two
amplified in-regime designs at &Gamma;=5), each with the outcome the alpha-sweep predicts
&mdash; over-treatment corrected (owgap-style), under-treatment corrected (Diabetes-A-style),
boundary behavior (SUPPORT2-B), and out-of-regime honesty (Diabetes-B, Credit-Default). That is
the unrejectable shape: the paper ships a decision rule for practitioners, validated at every
point of its range, not a collection of wins.</p>
<p><b>Compute:</b> each new experiment is one L&times;&Gamma; sweep + Kallus baseline &asymp;
2&ndash;3 h on the 2-worker license (Mushroom is discrete-ish/categorical X &mdash; cheaper).
Total &asymp; 4 chained jobs &asymp; one overnight run. All runners and the report pipeline
already exist; each dataset needs one dgp.py adapter (prepare script + generate() contract),
&asymp; a day of work each including calibration.</b></div>
<h2>6. Risks and honesty notes</h2>
<div class="card warn"><ul>
<li><b>Mushroom coupling sits at 0.71</b>, just under the 0.75 in-regime line &mdash; margins may
be modest; that is scientifically fine (it is the boundary point) but it is not a guaranteed
blowout. The X-augmented variant is the fallback and must be disclosed as such.</li>
<li><b>SUPPORT2-B at coupling 0.47 is predicted-boundary</b>: the honest expectation is a small
or zero O-W edge with &Gamma;-stability, not a win. If it loses badly, that WEAKENS the
diagnostic story &mdash; this is a genuine test, which is exactly why it is credible.</li>
<li><b>Synthetic-outcome disclosure</b> applies to every recipe-B experiment (Y(0)/Y(1) must be
synthesized for ground truth; calibrate to real outcome rates and disclose).</li>
<li><b>Leakage checks</b> needed per dataset (e.g. Diabetes 'change'/'diabetesMed' lesson: no
treatment-derived columns in X_obs).</li>
<li>All smoke numbers are point estimates from one 3-fold CV logistic; rerun with the final
X_obs definitions before quoting in the paper.</li>
</ul></div>
<p class="muted">Materials: smoke_tests.py (measurement script), smoke_results.json (raw
numbers), semi_build.py (this page). Raw UCI downloads cached in /scratch1/haghim/uci_cache
(not in the repo).</p>
</body></html>""")

page = "\n".join(html)
page = page.encode("ascii", "xmlcharrefreplace").decode("ascii")
OUT.write_text(page)
print(f"wrote {OUT} ({len(page)/1024:.0f} KB)")
