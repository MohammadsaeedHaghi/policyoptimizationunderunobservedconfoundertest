#!/usr/bin/env python3
"""Build 'assets/exp_gstar/gstar_report.html' -- the Gamma-star showcase campaign report.

Three tabs (DGP / Discrete / Continuous), auto-filling from the wave's JSONs as they land.
House style: family color code (estimator = color, uncertainty set = dash + marker),
math-definition captions, MathML via latex2mathml, pure-ASCII output.
MANDATORY plots (user): average test-set outcome E[Y] and learned-policy-vs-X, both tabs.
Rerun: python3 assets/exp_gstar/build_gstar_report.py
"""
import json, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT / "semi experiments"))
from report_common import CSS, MC, mdash, mmark, ser, esc, linechart, heatmap, bars, _ticks
from latex2mathml.converter import convert as l2m
import importlib.util

def _load(p, n):
    sp = importlib.util.spec_from_file_location(n, p); m = importlib.util.module_from_spec(sp)
    sys.modules[n] = m; sp.loader.exec_module(m); return m
d = _load(HERE / "dgp.py", "gstar_dgp_r")
dc = _load(HERE / "dgp_cont.py", "gstar_cont_r")

def J(fn):
    try: return json.load(open(HERE / fn))
    except Exception: return None

R = {ce: J(f"gstar_ce{ce}.json") for ce in ("1.0", "1.5", "2.0")}
RCAP = {c: J(f"gstar_cap{c}_ce1.0.json") for c in ("40", "50")}
RB = {b: J(f"gstar_beta{b}_ce1.0.json") for b in ("1", "2", "4", "6")}
KAL = J("gstar_kallus.json")
C = {ce: J(f"gstar_cont_uncap_ce{ce}.json") for ce in ("1.0", "1.5", "2.0")}
CCAP = J("gstar_cont_cap30_ce1.0.json")
KALC = J("gstar_kallus_cont.json")
GSTAR = 5.0
MORDER = ["IPW-O-W", "DoublyRobust-O-W", "DoublyRobust-X-X", "DoublyRobust-O-X",
          "IPW-X-X", "IPW-O-X", "Hajek-O-X", "Direct-X-X"]

def figcap(fig, cap): return f'<div>{fig}<p class="muted figcap">{cap}</p></div>'
def M(tex): return f'<div style="text-align:center;overflow-x:auto;margin:10px 0">{l2m(tex)}</div>'

def gstar_vline_chart(series, title, xlab, ylab, hlines=None, W=620, xticks=None):
    """linechart + a vertical Gamma* = 5 marker drawn as an extra 2-point series."""
    ys = [y for it in series for y in it[3] if y == y] + [h[2] for h in (hlines or [])]
    marker = ("Gamma* = 5", "#0a7d33", [GSTAR, GSTAR], [min(ys), max(ys)], "3 3")
    return linechart(series + [marker], title=title, xlab=xlab, ylab=ylab, hlines=hlines,
                     W=W, xticks=xticks)

# ================================ TAB 1: DGP ================================
xg = np.linspace(-1, 1, 241); sig = lambda z: 1 / (1 + np.exp(-z))
gt = d.grid_truth()

EQ_X = M(r"X \sim \mathrm{Unif}\{-1,\ldots,1\}\ \text{(7 levels; continuous: } X\sim\mathrm{Unif}[-1,1]\text{)},\quad S\mid X \in\{\pm1\},\ P(S{=}{+}1\mid X)=\sigma(10X)")
EQ_E = M(r"e(X,S) = \sigma\!\big(-1.5X + \tfrac{1}{2}\ln(5)\, S\big)"
         r"\;\Rightarrow\; \frac{\mathrm{odds}(T{=}1\mid X, S{=}{+}1)}{\mathrm{odds}(T{=}1\mid X, S{=}{-}1)} = e^{\ln 5} = 5 \;=\; \Gamma^{\star}\quad\forall X")
EQ_Y = M(r"\mu_0 = 8S,\quad \mu_1 = 9S + 3X - 1,\quad Y(t) = \mu_t + \mathcal{N}(0, 0.6^2)")
EQ_C = M(r"\mathrm{CATE}(X) = (2\sigma(10X)-1) + 3X - 1 \;\Rightarrow\; \pi^{*}(X) = 1\{X > 0\},\;\; \mathrm{CATE}(0) = -1")

cate = (2 * sig(10 * xg) - 1) + 3 * xg - 1
ep = sig(0.5 * np.log(5) - 1.5 * xg); em = sig(-0.5 * np.log(5) - 1.5 * xg)
_obs, _full = d.generate(600, 0)
_rj = np.random.default_rng(1).uniform(-0.06, 0.06, 600)
_Xj = _obs["X"].ravel() + _rj

dgp_figs1 = '<div class="figrow">' + figcap(
    linechart([("P(S=+1|X)", "#334155", list(xg), list(sig(10 * xg)), "")],
              title="Hidden-vitality coupling", xlab="X", ylab="P(S=+1|X)", legend=False),
    "P(S=+1 | X) = sigma(10X); Var(S | X) maximal at X = 0.") + figcap(
    linechart([("P(T=1 | X, S=+1)", "#2ca02c", list(xg), list(ep), ""),
               ("P(T=1 | X, S=-1)", "#9467bd", list(xg), list(em), "")],
              title="Propensity: the S-odds ratio is pinned to Gamma* = 5", xlab="X", ylab="P(T=1 | X,S)"),
    "e(X,S) = sigma(-1.5X + 0.5 ln(5) S), range [0.091, 0.909] -- NO clipping, so "
    "odds(S=+1)/odds(S=-1) = 5 exactly at every X.") + figcap(
    linechart([("CATE(X)", "#d62728", list(xg), list(cate), "")],
              title="True CATE: treat iff X > 0 (CATE(0) = -1)", xlab="X", ylab="CATE",
              hlines=[("0", "#888", 0.0, "4 3")], legend=False),
    "CATE(X) = E[Y(1)-Y(0) | X] = (2 sigma(10X)-1) + 3X - 1; oracle pi*(X) = 1{CATE > 0}.") + '</div>'

# naive bias: infinite-data limit per level + a finite draw
lv = list(gt["X"])
ps1v = gt["p_s1"]
naive_inf = []
for j, x in enumerate(gt["X"]):
    p = ps1v[j]; e1, e0 = d.propensity(x, 1.0), d.propensity(x, -1.0)
    pt = p * e1 + (1 - p) * e0
    pS_T1 = p * e1 / pt; pS_T0 = p * (1 - e1) / (1 - pt)
    m1 = d.D1 * (2 * pS_T1 - 1) + d.B1 * x - d.THETA
    m0 = d.D0 * (2 * pS_T0 - 1)
    naive_inf.append(m1 - m0)
dgp_figs2 = '<div class="figrow">' + figcap(
    linechart([("true CATE", "#111", lv, list(gt["cate"]), ""),
               ("naive CATE-hat (infinite data)", MC["DoublyRobust-X-X"], lv, naive_inf, "2 3", "t")],
              title="The ranking inversion at X = 0", xlab="X level", ylab="CATE",
              hlines=[("0", "#888", 0.0, "4 3")]),
    "Blue: the infinite-data naive limit E[Y|T=1,X] - E[Y|T=0,X]; the gap is pure confounding "
    "bias, concentrated at X = 0 where Var(S|X) peaks (-1 -> ~+5.5).") + figcap(
    linechart([("mu1(X,S=+1)", "#d62728", list(xg), list(9 + 3 * xg - 1), ""),
               ("mu0(X,S=+1)", "#1f77b4", list(xg), [8.0] * 241, ""),
               ("mu1(X,S=-1)", "#d62728", list(xg), list(-9 + 3 * xg - 1), "5 4"),
               ("mu0(X,S=-1)", "#1f77b4", list(xg), [-8.0] * 241, "5 4")],
              title="Mean potential outcomes mu_t(X,S)", xlab="X", ylab="mu_t"),
    "mu0 = 8S, mu1 = 9S + 3X - 1: the confounder moves outcomes by +-8-9 while the effect is "
    "a few units -- treated patients look good because they are vital.") + '</div>'

T1 = f"""
<h2>1. The data-generating process</h2>
<p>Aggressive therapy under hidden vitality with a fixed burden; the design follows the
synthetic conventions of the MSM literature (logistic-in-hidden-confounder propensity and a
level-shift confounder as in Kallus-Mao-Zhou 2019 and Hess/Frauen et al. ICLR 2026; a
uniform observed covariate; a budget/burden term from the budgeted policy-learning tradition).
N = 600 (discrete) / 400 (continuous) per seed; 5 seeds.</p>
{EQ_X}{EQ_E}{EQ_Y}{EQ_C}
<div class="card good"><b>&Gamma;&#9733; by inspection (the design's central property).</b>
The propensity is logistic in the hidden S with coefficient &frac12;ln(5), and its range
[0.091, 0.909] means no clipping ever occurs &mdash; so the selection odds ratio between the
two S-arms is <b>exactly &Gamma;&#9733; = 5 at every X, by algebra</b>. Every method below is
run at the single matched &Gamma; = &Gamma;&#9733; = 5 (flagged in every figure); the full
&Gamma;-grid is reported only to show misspecification behavior. Because the fitted marginal
propensity lies between the two S-extremals, the &Gamma;&#9733; odds-box around it covers every
unit's true weight: the sensitivity model is correctly specified with a KNOWN parameter &mdash;
nothing is tuned.</div>
{dgp_figs1}
{dgp_figs2}
<p>Reference values (exact): never-treat 0, all-treat -1, oracle 0.847 uncapped / 0.727 under
the 30% cap (cap &lt; 43% oracle-treat mass: genuinely scarce). Capped-regime ablations at
40%/50%; coupling ablation replaces ALPHA=10 by {{1,2,4,6}} leaving &Gamma;&#9733; = 5 untouched.</p>
"""

# ============================ TAB 2: DISCRETE ============================
T2_parts = []
R0 = R.get("1.0")
if R0:
    gam = R0["gammas"]; gi = gam.index(5.0)
    parts_fig = []
    for reg, lab, orc in (("uncap", "Uncapped (oracle 0.847)", 0.847), ("cap", "Capped 30% (capped oracle 0.727)", 0.727)):
        mu = R0["regimes"][reg]["mean"]
        s = [ser(m, gam, mu[m]) for m in MORDER if m in mu]
        if KAL:
            s.append(ser("Kallus", KAL["gammas"], KAL["regimes"]["uncap"]["mean"]["Kallus"])) if reg == "uncap" else None
        parts_fig.append(gstar_vline_chart(s, f"Average test outcome vs Gamma -- {lab}", "Gamma",
                                           "exact E[Y]", hlines=[("oracle", "#111", orc, "5 4"),
                                                                 ("never-treat", "#888", 0.0, "2 3")], xticks=gam))
    # matched-Gamma table across ceps
    rows = []
    for ce, Rce in R.items():
        if not Rce: continue
        gce = Rce["gammas"].index(5.0)
        for reg in ("uncap", "cap"):
            mu = Rce["regimes"][reg]["mean"]
            nv = max(mu["DoublyRobust-X-X"][0], mu["IPW-X-X"][0], mu["Direct-X-X"][0])
            rows.append(f"<tr><td>c<sub>&epsilon;</sub>={ce}, {reg}</td><td>{nv:.3f}</td>"
                        f"<td>{mu['IPW-O-W'][gce]:.3f}</td><td>{mu['DoublyRobust-O-W'][gce]:.3f}</td>"
                        f"<td class='g'>{max(mu['IPW-O-W'][gce], mu['DoublyRobust-O-W'][gce]) - nv:+.3f}</td></tr>")
    cap_rows = []
    mu_c = R0["regimes"]["cap"]["mean"]
    nvc = max(mu_c["DoublyRobust-X-X"][0], mu_c["IPW-X-X"][0], mu_c["Direct-X-X"][0])
    cap_rows.append(f"<tr><td>30% (main)</td><td>{nvc:.3f}</td><td>{mu_c['IPW-O-W'][gi]:.3f}</td>"
                    f"<td class='g'>{mu_c['IPW-O-W'][gi] - nvc:+.3f}</td></tr>")
    for c, Rc in RCAP.items():
        if not Rc: continue
        gc = Rc["gammas"].index(5.0); mc = Rc["regimes"]["cap"]["mean"]
        nvx = max(mc["DoublyRobust-X-X"][0], mc["IPW-X-X"][0], mc["Direct-X-X"][0])
        cap_rows.append(f"<tr><td>{c}%</td><td>{nvx:.3f}</td><td>{mc['IPW-O-W'][gc]:.3f}</td>"
                        f"<td class='g'>{mc['IPW-O-W'][gc] - nvx:+.3f}</td></tr>")
    # coupling sweep
    beta_rows, bx, bm = [], [], []
    for b in ("1", "2", "4", "6"):
        Rb = RB.get(b)
        if not Rb: continue
        gb = Rb["gammas"].index(5.0); mb = Rb["regimes"]["uncap"]["mean"]
        nvb = max(mb["DoublyRobust-X-X"][0], mb["IPW-X-X"][0], mb["Direct-X-X"][0])
        marg = mb["IPW-O-W"][gb] - nvb
        bx.append(float(b)); bm.append(marg)
        beta_rows.append(f"<tr><td>&alpha;={b}</td><td>{Rb['oracle']:.3f}</td><td>{nvb:.3f}</td>"
                         f"<td>{mb['IPW-O-W'][gb]:.3f}</td>"
                         f"<td class='{'g' if marg > 0 else 'b'}'>{marg:+.3f}</td></tr>")
    mu_u = R0["regimes"]["uncap"]["mean"]
    nvu = max(mu_u["DoublyRobust-X-X"][0], mu_u["IPW-X-X"][0], mu_u["Direct-X-X"][0])
    bx.append(10.0); bm.append(mu_u["IPW-O-W"][gi] - nvu)
    beta_rows.append(f"<tr><td>&alpha;=10 (main)</td><td>{R0['oracle']:.3f}</td><td>{nvu:.3f}</td>"
                     f"<td>{mu_u['IPW-O-W'][gi]:.3f}</td><td class='g'>{mu_u['IPW-O-W'][gi] - nvu:+.3f}</td></tr>")
    beta_fig = linechart([ser("IPW-O-W", bx, bm, lab="O-W margin over best naive at Gamma*")],
                         title="Coupling ablation: when does the W-term pay?", xlab="ALPHA (coupling)",
                         ylab="margin at Gamma* = 5", hlines=[("0", "#888", 0.0, "4 3")],
                         xticks=[1, 2, 4, 6, 10], legend=False)
    T2_parts.append(f"""
<h2>1. Average test outcome: E[Y] vs &Gamma; ({len(R0['seeds'])} seeds, N=600, exact evaluation)</h2>
<div class="figrow">{parts_fig[0]}{parts_fig[1]}</div>
<h3>Values at the flagged &Gamma;&#9733; = 5 (no tuning), all transport budgets</h3>
<div class="tw"><table>
<tr><th>setting</th><th>best naive</th><th>IPW-O-W</th><th>DR-O-W</th><th>O-W margin</th></tr>
{''.join(rows)}
</table></div>
<h3>Cap-budget robustness (at &Gamma;&#9733;)</h3>
<div class="tw"><table>
<tr><th>budget</th><th>best naive</th><th>IPW-O-W</th><th>margin</th></tr>
{''.join(cap_rows)}
</table></div>
<h3>Coupling ablation (&alpha; sweep; &Gamma;&#9733; = 5 fixed throughout)</h3>
<div class="tw"><table>
<tr><th>coupling</th><th>oracle</th><th>best naive</th><th>IPW-O-W at &Gamma;&#9733;</th><th>margin</th></tr>
{''.join(beta_rows)}
</table></div>
{beta_fig}
""")
else:
    T2_parts.append('<h2>Discrete results</h2><p class="muted">Wave running; rerun this builder when jobs land.</p>')

# ============================ TAB 3: CONTINUOUS ============================
T3_parts = []
C0 = C.get("1.0")
if C0:
    Gk, Lk = C0["gammas"], C0["Lgrid"]
    hm_u = heatmap(["G=" + g for g in Gk], Lk, [[C0["surface"]["IPW-O-W"][g][l] for l in Lk] for g in Gk],
                   "UNCAPPED IPW-O-W: test E[Y] over Gamma x L (oracle %.2f)" % C0["oracle"],
                   "Gamma", "Lipschitz L", min(C0["never_treat"], -0.5), C0["oracle"])
    blocks = [hm_u]
    if CCAP:
        blocks.append(heatmap(["G=" + g for g in CCAP["gammas"]], CCAP["Lgrid"],
                              [[CCAP["surface"]["IPW-O-W"][g][l] for l in CCAP["Lgrid"]] for g in CCAP["gammas"]],
                              "CAPPED 30% IPW-O-W: test E[Y] over Gamma x L",
                              "Gamma", "Lipschitz L", min(CCAP["never_treat"], -0.5), C0["oracle"]))
    # matched-Gamma* slice table + best cells
    ct_rows = []
    for lab, Cx in [("uncap ce1.0", C0), ("uncap ce1.5", C.get("1.5")), ("uncap ce2.0", C.get("2.0")),
                    ("cap30 ce1.0", CCAP)]:
        if not Cx: continue
        g5 = "5"
        row = {m: Cx["surface"][m][g5]["3"] for m in Cx["methods"]}
        bo = Cx["best_overall"]
        ct_rows.append(f"<tr><td>{lab}</td><td>{Cx['naive_dr']:.3f}</td>"
                       f"<td>{row['IPW-O-X']:.3f}</td><td>{row['IPW-O-W']:.3f}</td>"
                       f"<td>{row['DoublyRobust-O-X']:.3f}</td><td>{row['DoublyRobust-O-W']:.3f}</td>"
                       f"<td class='g'>{max(row['IPW-O-W'],row['DoublyRobust-O-W']) - max(row['IPW-O-X'],row['DoublyRobust-O-X']):+.3f}</td>"
                       f"<td>{bo['method']}@G{bo['gamma']},L{bo['L']}={bo['value']:.3f}</td></tr>")
    T3_parts.append(f"""
<h2>1. The L &times; &Gamma; surfaces ({len(C0['seeds'])} seeds, N=400/4000, Shapley deployment)</h2>
<div class="figrow">{''.join(blocks)}</div>
<h3>At the flagged &Gamma;&#9733; = 5, L = 3 (the sweet spot); O-W minus box-only</h3>
<div class="tw"><table>
<tr><th>setting</th><th>naive DR</th><th>IPW-O-X</th><th>IPW-O-W</th><th>DR-O-X</th><th>DR-O-W</th>
<th>O-W gap</th><th>best overall</th></tr>
{''.join(ct_rows)}
</table></div>
""")
else:
    T3_parts.append('<h2>Continuous results</h2><p class="muted">Wave running; rerun this builder when jobs land.</p>')

# ---------------- assembly (interactive widgets injected post-wave; see PD/SURFD note) ----------------
hero = """<div class="hero" style="background:radial-gradient(130% 150% at 0% 0%,#14532d 0%,#166534 46%,#052e16 100%)">
<h1>The &Gamma;&#9733; showcase &mdash; discrete + continuous, capped + uncapped</h1>
<p>One DGP, every experiment: the sensitivity parameter &Gamma;&#9733; = 5 is readable off the
propensity by algebra (no clipping, logistic-in-S), so all methods run at the single flagged
matched &Gamma; with nothing tuned. 5 seeds; all raw per-seed policies persisted; the whole
wave ran as ~15 simultaneous jobs on the CARC cluster Gurobi license.</p></div>"""

TABS = f"""
<div class="tabs" role="tablist" style="position:sticky;top:0;z-index:9;background:var(--bg);border-bottom:1px solid var(--border);display:flex;gap:6px;padding:10px 0;margin:0 0 6px">
<button id="tb-dgp" class="on" onclick="showTab('dgp')" style="font:inherit;font-weight:700;border-radius:99px;padding:7px 18px;cursor:pointer">DGP explanation</button>
<button id="tb-disc" onclick="showTab('disc')" style="font:inherit;font-weight:700;border-radius:99px;padding:7px 18px;cursor:pointer">Discrete experiments</button>
<button id="tb-cont" onclick="showTab('cont')" style="font:inherit;font-weight:700;border-radius:99px;padding:7px 18px;cursor:pointer">Continuous experiments</button>
</div>
<div id="tab-dgp" class="tabpane on">{T1}</div>
<div id="tab-disc" class="tabpane">{''.join(T2_parts)}</div>
<div id="tab-cont" class="tabpane">{''.join(T3_parts)}</div>
<style>.tabpane{{display:none}}.tabpane.on{{display:block}}
.tabs button{{color:var(--muted);background:var(--surface);border:1px solid var(--border)}}
.tabs button.on{{color:#fff;background:var(--accent);border-color:var(--accent)}}</style>
<script>
function showTab(id){{
  for (const t of ['dgp','disc','cont']){{
    document.getElementById('tab-'+t).classList.toggle('on', t===id);
    document.getElementById('tb-'+t).classList.toggle('on', t===id);
  }}
  window.scrollTo(0,0);
}}
</script>"""

page = ('<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        f'<title>Gamma-star showcase</title><style>{CSS}</style></head><body>'
        f'{hero}{TABS}</body></html>')
page = page.encode("ascii", "xmlcharrefreplace").decode("ascii")
(HERE / "gstar_report.html").write_text(page)
print("wrote", HERE / "gstar_report.html", "(%d KB)" % (len(page) // 1024))
