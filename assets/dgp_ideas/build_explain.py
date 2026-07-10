"""Build a self-contained HTML page that EXPLAINS the 10 candidate DGPs (math in LaTeX + a truth plot
each). No method results, just the data-generating processes. Output: dgp_ideas_explained.html.

Run:  python3 assets/dgp_ideas/build_explain.py
"""
import importlib.util, glob, os, io, base64
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
BIGN = 240000

# control / treated / effect / assignment / hidden colours
C0, C1, CT, CE, CH = "#2563eb", "#dc2626", "#111827", "#16a34a", "#7c3aed"

def _load(path):
    n = os.path.splitext(os.path.basename(path))[0]
    s = importlib.util.spec_from_file_location(n, path); m = importlib.util.module_from_spec(s); s.loader.exec_module(m)
    return n, m

# title, one-line natural story, hidden-confounder label, and the LaTeX block (MathJax \[..\])
INFO = {
"dgp01_binaryS_selection": ("1. Binary hidden type (selection + outcome)",
    "A hidden vitality type S that is more common at high fitness X dominates outcomes in both arms and "
    "also drives who is treated; the treatment helps the vital slightly more.", r"\mathbb{E}[S\mid X]",
    r"""\[ S\in\{-1,+1\},\qquad \Pr(S{=}{+}1\mid X)=\sigma(8X), \]
\[ e(X,S)=\sigma(0.8\,S-2X),\qquad \mu_0=5S,\quad \mu_1=6S+X, \]
\[ \tau(X)=\mathbb{E}[\mu_1-\mu_0\mid X]=\big(2\sigma(8X)-1\big)+X. \]"""),
"dgp02_gaussianU_outcome": ("2. Continuous outcome confounder (Kallus style)",
    "An unrecorded continuous prognosis U, correlated with X, raises outcomes in both arms and drives "
    "treatment; the confounder sits on the outcome as in Kallus and Zhou.", r"\mathbb{E}[U\mid X]",
    r"""\[ U\mid X\sim\mathcal{N}(X,\,1),\qquad e=\sigma(U-1.5X), \]
\[ \mu_0=4U,\qquad \mu_1=0.3\,X+4.5\,U, \]
\[ \tau(X)=0.3X+(4.5-4)\,\mathbb{E}[U\mid X]=0.8\,X. \]"""),
"dgp03_strong_proxy": ("3. X is a near-perfect proxy of U",
    "The observed covariate X is a high-quality noisy measurement of the latent driver U, so balancing X "
    "almost fully balances the hidden U.", r"\mathbb{E}[U\mid X]",
    r"""\[ U\mid X\sim\mathcal{N}(X,\,0.4^2),\qquad e=\sigma(1.2\,U-X), \]
\[ \mu_0=4U,\qquad \mu_1=0.2\,X+4.6\,U,\qquad \tau(X)\approx 0.8\,X. \]"""),
"dgp04_signflip_indication": ("4. Sign-flip effect modification (confounding by indication)",
    "A hidden frailty flips the sign of the effect: the treatment helps the robust but harms the frail, "
    "and the frail are treated more, so the naive contrast points the wrong way (Simpson reversal).",
    r"\Pr(F{=}1\mid X)",
    r"""\[ F\in\{0,1\},\qquad \Pr(F{=}1\mid X)=\sigma(-5X),\qquad e=\sigma(1.2\,F-X), \]
\[ \mu_0=-3F,\qquad \mu_1=\mu_0+\big(\mathbf{1}[F{=}0]\,(+2)+\mathbf{1}[F{=}1]\,(-2)\big), \]
\[ \tau(X)=2\big(1-\Pr(F{=}1\mid X)\big)-2\,\Pr(F{=}1\mid X). \]"""),
"dgp05_multiplicative_effect": ("5. Multiplicative confounding (hidden U scales the effect)",
    "A hidden positive vitality factor U multiplies the treatment benefit, so the same intervention helps "
    "high-U patients far more; U also confounds the baseline.", r"\mathbb{E}[U\mid X]",
    r"""\[ \log U\mid X\sim\mathcal{N}(0.8X,\,0.5^2),\qquad e=\sigma(\log U-X), \]
\[ \mu_0=U,\qquad \mu_1=\mu_0+(-0.2+X)\,U, \]
\[ \tau(X)=(-0.2+X)\,\mathbb{E}[U\mid X]. \]"""),
"dgp06_saturation_doseresponse": ("6. Saturating dose-response with a treatment cost",
    "Outcomes are a saturating function of a latent health index; the treatment shifts the index but costs "
    "a fixed amount, so it is only worth it on the steep part of the curve.", r"\mathbb{E}[U\mid X]",
    r"""\[ U\mid X\sim\mathcal{N}(X,\,1),\qquad \eta=X+U,\qquad e=\sigma(U-0.5X), \]
\[ \mu_0=4\,\sigma(2\eta),\qquad \mu_1=4\,\sigma\!\big(2(\eta+1.2)\big)-0.5, \]
\[ \tau(X)=\mathbb{E}\big[\mu_1-\mu_0\mid X\big]. \]"""),
"dgp07_latent_class_mixture": ("7. Latent class mixture (responders vs non-responders)",
    "A hidden subgroup determines the benefit: responders (more likely at high X) gain a lot, "
    "non-responders little or nothing, and responders also have a better baseline.", r"\Pr(R{=}1\mid X)",
    r"""\[ R\in\{0,1\},\qquad \Pr(R{=}1\mid X)=\sigma(4X),\qquad e=\sigma(1.2\,R), \]
\[ \mu_0=2R,\qquad \mu_1=\mu_0+\big(\mathbf{1}[R{=}1]\,(2.5)+\mathbf{1}[R{=}0]\,(-0.5)\big), \]
\[ \tau(X)=2.5\,\Pr(R{=}1\mid X)-0.5\big(1-\Pr(R{=}1\mid X)\big). \]"""),
"dgp08_cancelling_baseline_CONTRAST": ("8. Cancelling baseline (negative control)",
    "The hidden U shifts the baseline equally in both arms, so it CANCELS in the treat-minus-control "
    "contrast and the true effect is a clean function of X. Included to show where O-W has nothing to add.",
    r"\mathbb{E}[U\mid X]",
    r"""\[ U\mid X\sim\mathcal{N}(X,\,1),\qquad e=\sigma(U-X), \]
\[ \mu_0=4U,\qquad \mu_1=\mu_0+0.8\,X, \]
\[ \tau(X)=0.8\,X\quad(\text{U cancels in the contrast}). \]"""),
"dgp09_two_confounders": ("9. Two correlated hidden confounders",
    "Selection is driven by one latent variable while outcomes are driven by another, correlated one, so "
    "no single observed proxy explains assignment, yet the prognosis confounder still tracks X.",
    r"\mathbb{E}[U_2\mid X]",
    r"""\[ U_1=0.8X+Z_1,\quad U_2=0.8X+0.7\,Z_1+\sqrt{1-0.7^2}\,Z_2,\quad Z_1,Z_2\sim\mathcal{N}(0,1), \]
\[ e=\sigma(U_1-X),\qquad \mu_0=4\,U_2,\qquad \mu_1=0.3X+4.4\,U_2, \]
\[ \tau(X)=0.3X+(4.4-4)\,\mathbb{E}[U_2\mid X]. \]"""),
"dgp10_interaction_treated_only": ("10. Interaction in the treated arm only (clean control)",
    "The control outcome depends only on observed X, but the treatment effect depends on a hidden fitness U "
    "that also drives treatment, so the treated arm alone is confounded.", r"\mathbb{E}[U\mid X]",
    r"""\[ U\mid X\sim\mathcal{N}(0.5X,\,1),\qquad e=\sigma(U-X), \]
\[ \mu_0=0,\qquad \mu_1=(-0.3+0.3X)+4\,U, \]
\[ \tau(X)=(-0.3+0.3X)+4\cdot 0.5\,X. \]"""),
}

def fig_for(m):
    g = m.grid_truth(); X = np.asarray(g["X"], float)
    _, full = m.generate(BIGN, seed=0); U = np.asarray(full["U"], float); Xs = full["X"].ravel()
    hid = np.array([U[np.isclose(Xs, x)].mean() for x in X])     # E[hidden|X] (Monte Carlo)
    fig, ax = plt.subplots(1, 3, figsize=(12.6, 3.5))
    # panel 1: assignment + hidden confounder
    a = ax[0]; a.plot(X, g["e_marg"], "-o", color=CT, lw=2, ms=4, label=r"$\Pr(T{=}1\mid X)$")
    a.set_ylim(-0.05, 1.05); a.set_xlabel("X"); a.set_ylabel("propensity", color=CT)
    a2 = a.twinx(); a2.plot(X, hid, "--s", color=CH, lw=2, ms=4, label="hidden mean")
    a2.set_ylabel("E[hidden | X]", color=CH)
    a.set_title("Assignment & hidden confounder", fontsize=10)
    a.legend(loc="upper left", fontsize=8); a2.legend(loc="lower right", fontsize=8)
    # panel 2: potential-outcome means + oracle region
    b = ax[1]; b.plot(X, g["eY0"], "-o", color=C0, lw=2, ms=4, label=r"$\mathrm{E}[\mu_0\mid X]$ (control)")
    b.plot(X, g["eY1"], "-o", color=C1, lw=2, ms=4, label=r"$\mathrm{E}[\mu_1\mid X]$ (treated)")
    for x, o in zip(X, g["oracle"]):
        if o > 0.5: b.axvspan(x - 0.13, x + 0.13, color=C1, alpha=0.07)
    b.set_xlabel("X"); b.set_ylabel("mean outcome"); b.set_title("Potential-outcome means", fontsize=10)
    b.legend(fontsize=8)
    # panel 3: true effect vs confounded contrast
    c = ax[2]; c.axhline(0, color="#9ca3af", lw=1)
    c.plot(X, g["cate"], "-o", color=CE, lw=2, ms=4, label=r"true effect $\tau(X)$")
    c.plot(X, g["obs_contrast"], "--D", color="#6b7280", lw=2, ms=4, label="observed contrast")
    c.set_xlabel("X"); c.set_ylabel("treated - control"); c.set_title("True effect vs confounded contrast", fontsize=10)
    c.legend(fontsize=8)
    fig.tight_layout()
    buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=115); plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")

secs = []
for path in sorted(glob.glob(os.path.join(HERE, "dgp[0-9]*.py"))):
    name, m = _load(path)
    title, story, _, latex = INFO[name]
    png = fig_for(m)
    note = ("The shaded band marks the X levels the oracle treats. Where the grey dashed observed "
            "contrast departs from the black true effect, unobserved confounding is biasing the naive estimate.")
    secs.append(f"""<section class="dgp">
  <h2>{title}</h2>
  <p class="story">{story}</p>
  <div class="math">{latex}</div>
  <img alt="{name}" src="data:image/png;base64,{png}"/>
  <p class="cap">{note}</p>
</section>""")

html = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Candidate DGPs - unobserved confounding</title>
<script>window.MathJax = {{ tex: {{ inlineMath: [['\\\\(','\\\\)']], displayMath: [['\\\\[','\\\\]']] }} }};</script>
<script async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
<style>
  body{{font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;max-width:1040px;margin:0 auto;
    padding:32px 22px 80px;color:#111827;line-height:1.5;background:#fff}}
  h1{{font-size:26px;margin:0 0 6px}} .sub{{color:#6b7280;margin:0 0 26px}}
  .intro{{background:#f9fafb;border:1px solid #e5e7eb;border-radius:10px;padding:14px 18px;margin-bottom:30px}}
  section.dgp{{border:1px solid #e5e7eb;border-radius:12px;padding:20px 22px;margin:0 0 26px;
    box-shadow:0 1px 2px rgba(0,0,0,.04)}}
  h2{{font-size:18px;margin:0 0 8px;color:#0f172a}}
  .story{{margin:0 0 12px;color:#374151}}
  .math{{background:#f8fafc;border:1px solid #eef2f7;border-radius:8px;padding:6px 14px;overflow-x:auto;font-size:15px}}
  img{{width:100%;height:auto;margin:14px 0 6px;border:1px solid #f1f5f9;border-radius:8px}}
  .cap{{font-size:13px;color:#6b7280;margin:0}}
  code{{background:#f1f5f9;padding:1px 5px;border-radius:4px;font-size:13px}}
</style></head><body>
<h1>Candidate data-generating processes</h1>
<p class="sub">Unobserved-confounding policy-optimization study &middot; discrete X (7 levels on [-1,1]),
binary treatment. Explanations only: the model and its truth curves, no method results.</p>
<div class="intro">Each process draws a hidden confounder differently and induces a gap between the
<b>true treatment effect</b> \\(\\tau(X)=\\mathrm{{E}}[\\mu_1-\\mu_0\\mid X]\\) and the
<b>observed (confounded) contrast</b> \\(\\mathrm{{E}}[Y\\mid T{{=}}1,X]-\\mathrm{{E}}[Y\\mid T{{=}}0,X]\\).
The oracle treats where \\(\\tau(X)\\gt 0\\). Here \\(\\sigma\\) is the logistic function. Idea 8 is a
deliberate negative control (the confounder cancels in the contrast).</div>
{''.join(secs)}
</body></html>"""

out = os.path.join(HERE, "dgp_ideas_explained.html")
open(out, "w").write(html)
print("wrote", out, "(%d KB)" % (len(html) // 1024))
