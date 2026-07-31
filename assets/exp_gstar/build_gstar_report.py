#!/usr/bin/env python3
"""Build 'assets/exp_gstar/gstar_report.html' -- the Gamma-star showcase campaign report.
Three tabs (DGP / Discrete / Continuous) from the completed 15-job wave.
House style: family color code, math captions, MathML, ASCII. Interactive: pi(X)-vs-X viewers
(method chips) on both results tabs; E[Y]-vs-Gamma-with-L-selector widgets on the continuous tab.
Rerun: python3 assets/exp_gstar/build_gstar_report.py
"""
import json, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT / "semi experiments"))
from report_common import CSS, MC, mdash, mmark, ser, esc, linechart, heatmap, bars, ev_widget_html, EV_JS
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
SHARP = J("gstar_sharp.json")
SHARPC = J("gstar_sharp_cont.json")
MC["SharpIPW-O-X"] = "#9467bd"   # sharp box-only: purple, box-set dash/marker via the O-X suffix
GS = 5.0
MORDER = ["IPW-O-W", "DoublyRobust-O-W", "DoublyRobust-X-X", "DoublyRobust-O-X",
          "IPW-X-X", "IPW-O-X", "Hajek-O-X", "Direct-X-X"]

def figcap(fig, cap): return f'<div>{fig}<p class="muted figcap">{cap}</p></div>'
def M(tex): return f'<div style="text-align:center;overflow-x:auto;margin:10px 0">{l2m(tex)}</div>'

def vline_chart(series, title, ylab, hlines=None, W=620, xticks=None, xlab="Gamma"):
    ys = [y for it in series for y in it[3] if y == y] + [h[2] for h in (hlines or [])]
    marker = ("Gamma* = 5", "#0a7d33", [GS, GS], [min(ys), max(ys)], "3 3")
    return linechart(series + [marker], title=title, xlab=xlab, ylab=ylab, hlines=hlines, W=W, xticks=xticks)

# ---- analytic capped references for the continuous tab (infinite data, exact) ----
_xg = np.linspace(-1, 1, 20001); _sig = lambda z: 1 / (1 + np.exp(-z))
_ps = _sig(10 * _xg); _ES = 2 * _ps - 1
_eY0 = 8 * _ES; _eY1 = 9 * _ES + 3 * _xg - 1
_cate = _eY1 - _eY0
_po = (_cate >= max(float(np.quantile(_cate, 0.70)), 0.0)).astype(float)
CAP_ORACLE = float(np.mean(_po * _eY1 + (1 - _po) * _eY0))            # 0.628, treats x > 0.40
_e1 = dc.propensity(_xg, 1.0); _e0 = dc.propensity(_xg, -1.0)
_pt = _ps * _e1 + (1 - _ps) * _e0
_hat = (9 * (2 * (_ps * _e1 / _pt) - 1) + 3 * _xg - 1) - 8 * (2 * (_ps * (1 - _e1) / (1 - _pt)) - 1)
_pn = (_hat >= float(np.quantile(_hat, 0.70))).astype(float)
CAP_NAIVE = float(np.mean(_pn * _eY1 + (1 - _pn) * _eY0))             # 0.171

# ================================ TAB 1: DGP ================================
xg = np.linspace(-1, 1, 241); sig = _sig
gt = d.grid_truth()
EQ_X = M(r"X \sim \mathrm{Unif}\{-1,\ldots,1\}\ \text{(7 levels; continuous: } X\sim\mathrm{Unif}[-1,1]\text{)},\quad S\mid X \in\{\pm1\},\ P(S{=}{+}1\mid X)=\sigma(10X)")
EQ_E = M(r"e(X,S) = \sigma\!\big(-1.5X + \tfrac{1}{2}\ln(5)\, S\big)"
         r"\;\Rightarrow\; \frac{\mathrm{odds}(T{=}1\mid X, S{=}{+}1)}{\mathrm{odds}(T{=}1\mid X, S{=}{-}1)} = e^{\ln 5} = 5 \;=\; \Gamma^{\star}\quad\forall X")
EQ_Y = M(r"\mu_0 = 8S,\quad \mu_1 = 9S + 3X - 1,\quad Y(t) = \mu_t + \mathcal{N}(0, 0.6^2)")
EQ_C = M(r"\mathrm{CATE}(X) = (2\sigma(10X)-1) + 3X - 1 \;\Rightarrow\; \pi^{*}(X) = 1\{X > 0\},\;\; \mathrm{CATE}(0) = -1")
cate = (2 * sig(10 * xg) - 1) + 3 * xg - 1
ep = sig(0.5 * np.log(5) - 1.5 * xg); em = sig(-0.5 * np.log(5) - 1.5 * xg)
lv = list(gt["X"]); naive_inf = []
for j, x in enumerate(gt["X"]):
    p = gt["p_s1"][j]; e1, e0 = d.propensity(x, 1.0), d.propensity(x, -1.0)
    pt = p * e1 + (1 - p) * e0
    naive_inf.append((9 * (2 * (p * e1 / pt) - 1) + 3 * x - 1) - 8 * (2 * (p * (1 - e1) / (1 - pt)) - 1))

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
dgp_figs2 = '<div class="figrow">' + figcap(
    linechart([("true CATE", "#111", lv, list(gt["cate"]), ""),
               ("naive CATE-hat (infinite data)", MC["DoublyRobust-X-X"], lv, naive_inf, "2 3", "t")],
              title="The ranking inversion at X = 0", xlab="X level", ylab="CATE",
              hlines=[("0", "#888", 0.0, "4 3")]),
    "Blue: the infinite-data naive limit E[Y|T=1,X] - E[Y|T=0,X]; the gap is pure confounding "
    "bias, concentrated at X = 0 where Var(S|X) peaks (-1 -> ~+5.4).") + figcap(
    linechart([("mu1(X,S=+1)", "#d62728", list(xg), list(9 + 3 * xg - 1), ""),
               ("mu0(X,S=+1)", "#1f77b4", list(xg), [8.0] * 241, ""),
               ("mu1(X,S=-1)", "#d62728", list(xg), list(-9 + 3 * xg - 1), "5 4"),
               ("mu0(X,S=-1)", "#1f77b4", list(xg), [-8.0] * 241, "5 4")],
              title="Mean potential outcomes mu_t(X,S)", xlab="X", ylab="mu_t"),
    "mu0 = 8S, mu1 = 9S + 3X - 1: the confounder moves outcomes by +-8-9 while the treatment "
    "differential is a few units.") + '</div>'
T1 = f"""
<h2>1. The data-generating process</h2>
<p>Aggressive therapy under hidden vitality with a fixed burden; the design follows the
synthetic conventions of the MSM literature (logistic-in-hidden-confounder propensity and a
level-shift confounder as in Kallus-Mao-Zhou 2019 and Hess/Frauen et al. ICLR 2026; uniform
observed covariate; burden per the budgeted policy-learning tradition). N = 600 (discrete) /
400 train + 4000 test (continuous) per seed; 5 seeds; every raw per-seed policy persisted.</p>
{EQ_X}{EQ_E}{EQ_Y}{EQ_C}
<div class="card good"><b>&Gamma;&#9733; by inspection (the design's central property).</b>
The propensity is logistic in the hidden S with coefficient &frac12;ln(5) and range
[0.091, 0.909] &mdash; no clipping ever occurs &mdash; so the selection odds ratio between the
S-arms is <b>exactly &Gamma;&#9733; = 5 at every X, by algebra</b> (verified numerically to
1e-12, and empirically: fitted per-level odds ratios 4.5&ndash;5.5 at n = 20,000). Every method
runs at the single flagged matched &Gamma; = 5 with nothing tuned; the &Gamma;-grid appears
only to show misspecification behavior. The fitted marginal propensity lies between the two
S-extremals, so the &Gamma;&#9733; odds-box around it covers every unit's true weight: a
correctly specified sensitivity model with a KNOWN parameter.</div>
{dgp_figs1}
{dgp_figs2}
<p>Reference values (exact): never-treat 0, all-treat -1; discrete oracle 0.847 uncapped /
0.727 capped(30%); continuous oracle 0.732 (realized, 5x4000 draws) / capped-oracle
{CAP_ORACLE:.3f} (treats x &gt; 0.40). The 30% cap &lt; 43% oracle mass: genuinely scarce.</p>
"""

# ============================ shared: PD viewer data ============================
PD = {}
def pol_widget_html(wid, dta, defaults=None):
    df = defaults or {}
    dm = df.get("m", [dta["methods"][0]]); dm = [dm] if isinstance(dm, str) else dm
    def opts(vals, dv):
        return "".join('<option value="%s"%s>%s</option>' % (v, " selected" if str(v) == str(dv) else "", v) for v in vals)
    chips = "".join('<label class="mchip"><input type="checkbox" data-m="%s"%s>'
                    '<span class="sw" style="background:%s"></span>%s</label>'
                    % (m, " checked" if m in dm else "", MC.get(m, "#7f7f7f"), m) for m in dta["methods"])
    c = [f'<div class="polw" id="pw-{wid}">']
    c.append(f'<div class="ctl"><span class="ctt">overlay methods:</span>'
             f'<span class="mck" id="pw-{wid}-m">{chips}</span>'
             f'<button type="button" class="mbtn" data-sel="all">all</button>'
             f'<button type="button" class="mbtn" data-sel="none">none</button></div>')
    c.append('<div class="ctl">')
    c.append(f'<label>&Gamma; <select id="pw-{wid}-g">{opts(dta["gammas"], df.get("g", "5"))}</select></label>')
    if dta["kind"] == "2d":
        c.append(f'<label>L <select id="pw-{wid}-l">{opts(dta["Ls"], df.get("l", "3"))}</select></label>')
    if dta["kind"] == "disc":
        c.append(f'<label>regime <select id="pw-{wid}-r">{opts(dta["regimes"], df.get("r", "uncap"))}</select></label>')
    c.append(f'</div><div id="pw-{wid}-plot" class="fig"></div></div>')
    return "".join(c)

# ============================ TAB 2: DISCRETE ============================
T2 = []
R0 = R.get("1.0")
if R0:
    gam = R0["gammas"]; gi = gam.index(5.0)
    figs = []
    for reg, lab, orc in (("uncap", "Uncapped (oracle 0.847)", 0.847), ("cap", "Capped 30% (capped oracle 0.727)", 0.727)):
        mu = R0["regimes"][reg]["mean"]
        s = [ser(m, gam, mu[m]) for m in MORDER if m in mu]
        if SHARP:
            s.append(ser("SharpIPW-O-X", SHARP["gammas"], SHARP["regimes"][reg]["mean"]["SharpIPW-O-X"],
                         lab="Sharp-O-X"))
        if reg == "uncap" and KAL:
            s.append(ser("Kallus", KAL["gammas"], KAL["regimes"]["uncap"]["mean"]["Kallus"]))
        figs.append(vline_chart(s, "Average test outcome vs Gamma -- " + lab, "exact E[Y]",
                                hlines=[("oracle", "#111", orc, "5 4"), ("never-treat", "#888", 0.0, "2 3")],
                                xticks=gam))
    rows = []
    for ce, Rce in R.items():
        if not Rce: continue
        gce = Rce["gammas"].index(5.0)
        for reg in ("uncap", "cap"):
            mu = Rce["regimes"][reg]["mean"]
            nv = max(mu["DoublyRobust-X-X"][0], mu["IPW-X-X"][0], mu["Direct-X-X"][0])
            ow = max(mu["IPW-O-W"][gce], mu["DoublyRobust-O-W"][gce])
            sh = SHARP["regimes"][reg]["mean"]["SharpIPW-O-X"][SHARP["gammas"].index(5.0)] if SHARP else float("nan")
            rows.append(f"<tr><td>c<sub>&epsilon;</sub>={ce}, {reg}</td><td>{nv:.3f}</td><td>{sh:.3f}</td>"
                        f"<td>{mu['IPW-O-W'][gce]:.3f}</td><td>{mu['DoublyRobust-O-W'][gce]:.3f}</td>"
                        f"<td class='g'>{ow - nv:+.3f}</td>"
                        f"<td class='{'g' if ow - sh > 0 else 'b'}'>{ow - sh:+.3f}</td></tr>")
    mu_c = R0["regimes"]["cap"]["mean"]
    nvc = max(mu_c["DoublyRobust-X-X"][0], mu_c["IPW-X-X"][0], mu_c["Direct-X-X"][0])
    cap_rows = [f"<tr><td>30% (main)</td><td>{nvc:.3f}</td><td>{mu_c['IPW-O-W'][gi]:.3f}</td>"
                f"<td class='g'>{mu_c['IPW-O-W'][gi] - nvc:+.3f}</td></tr>"]
    for c, Rc in RCAP.items():
        if not Rc: continue
        gc = Rc["gammas"].index(5.0); mc = Rc["regimes"]["cap"]["mean"]
        nvx = max(mc["DoublyRobust-X-X"][0], mc["IPW-X-X"][0], mc["Direct-X-X"][0])
        cap_rows.append(f"<tr><td>{c}%</td><td>{nvx:.3f}</td><td>{mc['IPW-O-W'][gc]:.3f}</td>"
                        f"<td class='g'>{mc['IPW-O-W'][gc] - nvx:+.3f}</td></tr>")
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
                         title="Coupling ablation: when does the W-term pay?", xlab="ALPHA (X-S coupling)",
                         ylab="margin at Gamma* = 5", hlines=[("0", "#888", 0.0, "4 3")],
                         xticks=[1, 2, 4, 6, 10], legend=False)
    # PD: discrete policy viewer (mean over the 5 seeds; skip the "avg" pseudo-key)
    disc_pol = {}
    for reg in ("uncap", "cap"):
        disc_pol[reg] = {}
        pbs = R0["regimes"][reg]["policy_by_seed"]
        for m in MORDER:
            if m not in pbs: continue
            seeds = [k for k in pbs[m] if k.isdigit()]
            gks = [k for k in pbs[m][seeds[0]]]
            disc_pol[reg][m] = {gk: [float(np.mean([pbs[m][s][gk][j] for s in seeds]))
                                     for j in range(len(R0["grid"]))] for gk in gks}
    if SHARP:
        for reg in ("uncap", "cap"):
            pbss = SHARP["regimes"][reg]["policy_by_seed"]["SharpIPW-O-X"]
            seeds = [k for k in pbss if k.isdigit()]
            disc_pol[reg]["SharpIPW-O-X"] = {gk: [float(np.mean([pbss[s][gk][j] for s in seeds]))
                                                  for j in range(len(R0["grid"]))]
                                             for gk in pbss[seeds[0]]}
    if KAL:
        pbsk = KAL["regimes"]["uncap"]["policy_by_seed"]["Kallus"]
        seeds = [k for k in pbsk if k.isdigit()]
        disc_pol["uncap"]["Kallus"] = {gk: [float(np.mean([pbsk[s][gk][j] for s in seeds]))
                                            for j in range(len(R0["grid"]))] for gk in pbsk[seeds[0]]}
    cap_orc = [0.0] * 7; rem = 0.3
    for j in sorted(range(7), key=lambda i: -gt["cate"][i]):
        if gt["cate"][j] <= 0 or rem <= 1e-9: break
        take = min(1.0, rem * 7); cap_orc[j] = take; rem -= take / 7
    PD["disc"] = {"kind": "disc", "grid": [float(x) for x in R0["grid"]],
                  "gammas": [k for k in disc_pol["uncap"]["IPW-O-W"]],
                  "regimes": ["uncap", "cap"],
                  "methods": MORDER + (["SharpIPW-O-X"] if SHARP else []) + (["Kallus"] if KAL else []), "pol": disc_pol,
                  "refs": {"oracle_uncap": [0.0 if c_ <= 0 else 1.0 for c_ in gt["cate"]],
                           "oracle_cap": cap_orc}}
    T2.append(f"""
<h2>1. Average test outcome: E[Y] vs &Gamma; (5 seeds, N=600, exact evaluation)</h2>
<div class="figrow">{figs[0]}{figs[1]}</div>
<div class="card finding"><b>At the flagged &Gamma;&#9733; = 5 (nothing tuned):</b> uncapped
IPW-O-W {mu_u['IPW-O-W'][gi]:.3f} = 91% of oracle, margin <b>+{mu_u['IPW-O-W'][gi] - nvu:.3f}</b>
over the best naive; capped margin <b>+{mu_c['IPW-O-W'][gi] - nvc:.3f}</b>. Box-only methods
peak far below naive (uncapped best {max(mu_u['IPW-O-X'][gi], mu_u['DoublyRobust-O-X'][gi], mu_u['Hajek-O-X'][gi]):.3f});
Kallus collapses to never-treat by &Gamma; &asymp; 2.5. The O-W curves are FLAT across
&Gamma; = 2&ndash;8: robustness to misspecifying the sensitivity level, on top of winning at
the true one.</div>
<div class="card warn"><b>The sharp-box test (the strongest available box-only baseline;
Dorn-Guo-style sharp MSM bounds, closed-form, added deliberately as the hardest referee
question).</b> Two-sided result, reported in full. UNCAPPED: the sharp score never flips any
level's treatment sign, so Sharp-O-X is FLAT at 0.704 for every &Gamma; &mdash; exactly the
infinite-data naive value: <b>sharpness removes the box's pessimism but inherits naive's bias;
the ranking inversion at X=0 survives sharpening</b>, and only the W constraint fixes it
(IPW-O-W 0.772, +0.068 over sharp, +0.152 over realized naive). CAPPED at the flagged
&Gamma;&#9733;=5: the sharp greedy ranking is genuinely strong (0.479) and BEATS IPW-O-W
(0.424) there; O-W overtakes from &Gamma; &ge; 6 (0.527 at &Gamma;=8) and wins every other
cell of the campaign. We report this openly: it says the productive comparison is not
sharp-vs-W but sharp-AND-W &mdash; the sharpness constraints are linear in the adversary's
weights and can be added to the O-W program; we flag Sharp-O-W as the natural extension.</div>
<h3>Values at &Gamma;&#9733; = 5, all transport budgets (Sharp-O-X is &epsilon;-free; its
column repeats across budgets)</h3>
<div class="tw"><table>
<tr><th>setting</th><th>best naive</th><th>Sharp-O-X</th><th>IPW-O-W</th><th>DR-O-W</th>
<th>O-W vs naive</th><th>O-W vs Sharp</th></tr>
{''.join(rows)}
</table></div>
<h3>Cap-budget robustness (at &Gamma;&#9733;)</h3>
<div class="tw"><table>
<tr><th>budget</th><th>best naive</th><th>IPW-O-W</th><th>margin</th></tr>
{''.join(cap_rows)}
</table></div>
<h3>Coupling ablation (&Gamma;&#9733; = 5 fixed; only the X&ndash;S coupling varies)</h3>
<div class="tw"><table>
<tr><th>coupling</th><th>oracle</th><th>best naive</th><th>IPW-O-W at &Gamma;&#9733;</th><th>margin</th></tr>
{''.join(beta_rows)}
</table></div>
{beta_fig}
<h2>2. The learned policy vs X (every method, every &Gamma;; mean over 5 seeds)</h2>
<p>Tick any set of methods; pick &Gamma; and the regime. Oracle for the regime shown dashed.
With one method ticked the exact &pi; vector is printed under the plot.</p>
{pol_widget_html("disc", PD["disc"], defaults={"m": ["IPW-O-W", "DoublyRobust-X-X"], "g": "5", "r": "uncap"})}
""")
else:
    T2.append('<p class="muted">Discrete results pending.</p>')

# ============================ TAB 3: CONTINUOUS ============================
T3 = []
C0 = C.get("1.0")
def policy_2d_dataset(RJ):
    ps = RJ["policies_seed0"]
    dta = {"kind": "2d", "grid": [float(x) for x in RJ["policy_grid"]],
           "gammas": RJ["gammas"], "Ls": RJ["Lgrid"], "methods": RJ["methods"],
           "pol": {m: ps[m] for m in RJ["methods"]},
           "refs": {"oracle": ps["_refs"]["oracle"], "naive": ps["_refs"]["naive_dr"]}}
    sup = RJ.get("policies_support_seed0") or (RJ.get("policies_support_by_seed") or {}).get("0")
    if sup and "_X" in sup:
        dta["supX"] = [round(float(x), 3) for x in sup["_X"]]
        dta["sup"] = {m: {g: {l: [int(round(float(v) * 100)) for v in sup[m][g][l]]
                              for l in RJ["Lgrid"] if l in sup[m][g]} for g in RJ["gammas"]}
                      for m in RJ["methods"]}
        if "_naive_dr" in sup:
            dta["supNaive"] = [int(round(float(v) * 100)) for v in sup["_naive_dr"]]
    return dta

SURFDS = {}
EVD = {}
if C0:
    Gk, Lk = C0["gammas"], C0["Lgrid"]
    gl = [float(g) for g in Gk]
    hm_u = heatmap(["G=" + g for g in Gk], Lk, [[C0["surface"]["IPW-O-W"][g][l] for l in Lk] for g in Gk],
                   "UNCAPPED IPW-O-W: test E[Y] over Gamma x L (oracle %.2f)" % C0["oracle"],
                   "Gamma", "Lipschitz L", min(C0["never_treat"], -0.5), C0["oracle"], W=560, H=280)
    blocks = [hm_u]
    if CCAP:
        blocks.append(heatmap(["G=" + g for g in CCAP["gammas"]], CCAP["Lgrid"],
                              [[CCAP["surface"]["IPW-O-W"][g][l] for l in CCAP["Lgrid"]] for g in CCAP["gammas"]],
                              "CAPPED 30%% IPW-O-W: test E[Y] over Gamma x L (capped oracle %.2f)" % CAP_ORACLE,
                              "Gamma", "Lipschitz L", min(CCAP["never_treat"], -0.5), CAP_ORACLE, W=560, H=280))
    sl_u = [ser(m, gl, [C0["surface"][m][g]["3"] for g in Gk]) for m in C0["methods"]]
    if SHARPC:
        sl_u.append(ser("SharpIPW-O-X", [float(g) for g in SHARPC["gammas"]],
                        SHARPC["regimes"]["uncap"]["mean"]["SharpIPW-O-X"], lab="Sharp-O-X (binned)"))
    if KALC:
        sl_u.append(ser("Kallus", [float(g) for g in KALC["gammas"]], KALC["regimes"]["uncap"]["mean"]["Kallus"]))
    ch_u = vline_chart(sl_u, "UNCAPPED: average test outcome vs Gamma at L = 3", "test E[Y]",
                       hlines=[("oracle", "#111", C0["oracle"], "5 4"),
                               ("naive DR", MC["DoublyRobust-X-X"], C0["naive_dr"], "2 3"),
                               ("never-treat", "#888", C0["never_treat"], "2 3")], xticks=gl)
    parts = [ch_u]
    if CCAP:
        sl_c = [ser(m, gl, [CCAP["surface"][m][g]["3"] for g in CCAP["gammas"]]) for m in CCAP["methods"]]
        if SHARPC:
            sl_c.append(ser("SharpIPW-O-X", [float(g) for g in SHARPC["gammas"]],
                            SHARPC["regimes"]["cap"]["mean"]["SharpIPW-O-X"], lab="Sharp-O-X (binned)"))
        parts.append(vline_chart(sl_c, "CAPPED 30%: average test outcome vs Gamma at L = 3", "test E[Y]",
                                 hlines=[("capped oracle", "#111", CAP_ORACLE, "5 4"),
                                         ("capped naive (analytic)", MC["DoublyRobust-X-X"], CAP_NAIVE, "2 3")],
                                 xticks=gl))
    ct_rows = []
    for lab, Cx in [("uncap ce1.0", C0), ("uncap ce1.5", C.get("1.5")), ("uncap ce2.0", C.get("2.0")),
                    ("cap30 ce1.0", CCAP)]:
        if not Cx: continue
        row = {m: Cx["surface"][m]["5"]["3"] for m in Cx["methods"]}
        bo = Cx["best_overall"]
        ow = max(row["IPW-O-W"], row["DoublyRobust-O-W"]); box = max(row["IPW-O-X"], row["DoublyRobust-O-X"])
        nv = Cx["naive_dr"] if "uncap" in lab else CAP_NAIVE
        shreg = "uncap" if "uncap" in lab else "cap"
        shv = SHARPC["regimes"][shreg]["mean"]["SharpIPW-O-X"][SHARPC["gammas"].index(5.0)] if SHARPC else float("nan")
        ct_rows.append(f"<tr><td>{lab}</td><td>{nv:.3f}</td><td>{shv:.3f}</td>"
                       f"<td>{row['IPW-O-X']:.3f}</td><td>{row['IPW-O-W']:.3f}</td>"
                       f"<td>{row['DoublyRobust-O-X']:.3f}</td><td>{row['DoublyRobust-O-W']:.3f}</td>"
                       f"<td class='g'>{ow - nv:+.3f}</td>"
                       f"<td>{bo['method']}@G{bo['gamma']},L{bo['L']}={bo['value']:.3f}</td></tr>")
    PD["cont"] = policy_2d_dataset(C0)
    if CCAP: PD["contcap"] = policy_2d_dataset(CCAP)
    SURFDS["uncap"] = {"gammas": Gk, "Ls": Lk, "methods": C0["methods"], "surface": C0["surface"],
                       "oracle": C0["oracle"], "naive": C0["naive_dr"], "never": C0["never_treat"]}
    if CCAP:
        SURFDS["cap"] = {"gammas": CCAP["gammas"], "Ls": CCAP["Lgrid"], "methods": CCAP["methods"],
                         "surface": CCAP["surface"], "oracle": CAP_ORACLE, "naive": CAP_NAIVE,
                         "never": CCAP["never_treat"]}
    sv_widgets = ""
    for wid, lab in (("uncap", "uncapped"), ("cap", "capped 30%")):
        if wid not in SURFDS: continue
        D0 = SURFDS[wid]
        EVD[wid] = {"gammas": D0["gammas"], "Ls": D0["Ls"], "methods": D0["methods"],
                    "surface": D0["surface"], "gstar": 5.0,
                    "hlines": {"oracle": D0["oracle"], "naive": D0["naive"], "never-treat": D0["never"]},
                    "extra": {}}
        sv_widgets += ev_widget_html(wid, EVD[wid],
                                     defaults={"m": ["IPW-O-W", "IPW-O-X"], "l": "3"},
                                     title="E[Y] vs Gamma -- tick methods, choose L (%s)" % lab)
    T3.append(f"""
<h2>1. Average test outcome (5 seeds, N=400 train / 4000 test, Shapley deployment)</h2>
<div class="figrow">{''.join(blocks)}</div>
<div class="figrow">{''.join(parts)}</div>
{sv_widgets}
<h3>At &Gamma;&#9733; = 5, L = 3; margins vs the honest reference per regime</h3>
<div class="tw"><table>
<tr><th>setting</th><th>naive ref</th><th>Sharp-O-X</th><th>IPW-O-X</th><th>IPW-O-W</th><th>DR-O-X</th><th>DR-O-W</th>
<th>O-W margin</th><th>best overall</th></tr>
{''.join(ct_rows)}
</table></div>
<div class="card finding"><b>Continuous verdict.</b> Uncapped at &Gamma;&#9733; = 5, L = 3:
O-W 0.549&ndash;0.553 vs naive 0.311 (<b>+0.24</b>) and vs box-only 0.383 (+0.17), stable
across all three transport budgets. Capped 30% (the new capped-continuous cell, enabled by the
Lipschitz port to the capped solvers): against the HONEST capped references &mdash; capped
naive {CAP_NAIVE:.3f}, capped oracle {CAP_ORACLE:.3f} &mdash; the best capped O-W cell reaches
{(CCAP or {}).get('best_overall', {}).get('value', float('nan')):.3f}
(IPW-O-W @ &Gamma;8, L1), i.e. +{(CCAP or {}).get('best_overall', {}).get('value', 0) - CAP_NAIVE:.2f}
over budgeted naive and 67% of the capped oracle. Against the SHARP box baseline (binned, closed-form): O-W leads in both continuous regimes
&mdash; uncapped 0.549 vs sharp 0.502 at &Gamma;&#9733;, capped 0.420 (best cell) vs sharp
0.238 (best) &mdash; the smooth-policy setting is where the W term is clearly indispensable
even against sharp bounds. L behaves as in every experiment: L = &infin; is &Gamma;-inert and
poor, L &asymp; 2&ndash;3 is the sweet spot, L &le; 1 over-smooths.</div>
<h2>2. The learned policy vs X (uncapped)</h2>
<p>First panel: the solver's RAW per-unit policy at the 400 support points (seed 0). Second:
the Shapley-deployed &pi;(x). Exact at support &mdash; the curve must thread the dots.</p>
{pol_widget_html("cont", PD["cont"], defaults={"m": ["IPW-O-W", "DoublyRobust-O-W"], "g": "5", "l": "3"})}
""" + (f"""
<h2>3. The learned policy vs X (capped 30%)</h2>
{pol_widget_html("contcap", PD["contcap"], defaults={"m": ["IPW-O-W"], "g": "8", "l": "1"})}
""" if CCAP else ""))
else:
    T3.append('<p class="muted">Continuous results pending.</p>')

# ============================ assembly + JS ============================
hero = """<div class="hero" style="background:radial-gradient(130% 150% at 0% 0%,#14532d 0%,#166534 46%,#052e16 100%)">
<h1>The &Gamma;&#9733; showcase &mdash; discrete + continuous, capped + uncapped</h1>
<p>One DGP, every experiment: the sensitivity parameter &Gamma;&#9733; = 5 is readable off the
propensity by algebra (logistic-in-S, never clipped), so every method runs at the single
flagged matched &Gamma; with nothing tuned. 5 seeds; every raw per-seed policy persisted; the
15-job wave ran simultaneously on the CARC cluster Gurobi license in ~75 minutes.</p></div>"""

WIDGET_CSS = """
.tabpane{display:none}.tabpane.on{display:block}
.tabs button{font:inherit;font-weight:700;border-radius:99px;padding:7px 18px;cursor:pointer;color:var(--muted);background:var(--surface);border:1px solid var(--border)}
.tabs button.on{color:#fff;background:var(--accent);border-color:var(--accent)}
.ctl{display:flex;gap:10px 18px;flex-wrap:wrap;align-items:center;margin:12px 0 4px}
.ctl label{font-size:.84rem;font-weight:600;color:var(--muted);display:inline-flex;align-items:center;gap:7px}
.ctl select{font:inherit;font-size:.86rem;padding:4px 10px;border-radius:9px;border:1px solid var(--border);background:var(--surface);color:var(--fg)}
.ctt{font-size:.84rem;font-weight:600;color:var(--muted)}
.mck{display:inline-flex;flex-wrap:wrap;gap:4px 8px}
.mchip{display:inline-flex;align-items:center;gap:5px;font-size:.78rem;font-weight:600;color:var(--muted);border:1px solid var(--border);border-radius:99px;padding:2px 10px;cursor:pointer;background:var(--surface)}
.mchip input{accent-color:var(--accent);width:13px;height:13px;margin:0;cursor:pointer}
.mchip:has(input:checked){color:var(--fg);border-color:var(--accent);background:var(--accent-soft)}
.mbtn{font:inherit;font-size:.75rem;font-weight:700;color:var(--muted);background:var(--surface);border:1px solid var(--border);border-radius:8px;padding:2px 10px;cursor:pointer}
"""

def _r3(o):
    if isinstance(o, list): return [_r3(x) for x in o]
    if isinstance(o, dict): return {k: _r3(v) for k, v in o.items()}
    if isinstance(o, float): return round(o, 3)
    return o

JS = r"""
const MCJS = %(MC)s; const DASHJS = %(DASH)s; const MARKJS = %(MARK)s;
function pwMark(cx, cy, col, mk, r){
  if (mk==='s') return '<rect x="'+(cx-r).toFixed(1)+'" y="'+(cy-r).toFixed(1)+'" width="'+(2*r)+'" height="'+(2*r)+'" style="fill:'+col+'"/>';
  if (mk==='t') return '<polygon points="'+cx.toFixed(1)+','+(cy-1.3*r).toFixed(1)+' '+(cx-1.2*r).toFixed(1)+','+(cy+r).toFixed(1)+' '+(cx+1.2*r).toFixed(1)+','+(cy+r).toFixed(1)+'" style="fill:'+col+'"/>';
  if (mk==='d') return '<polygon points="'+cx.toFixed(1)+','+(cy-1.4*r).toFixed(1)+' '+(cx-1.4*r).toFixed(1)+','+cy.toFixed(1)+' '+cx.toFixed(1)+','+(cy+1.4*r).toFixed(1)+' '+(cx+1.4*r).toFixed(1)+','+cy.toFixed(1)+'" style="fill:'+col+'"/>';
  return '<circle cx="'+cx.toFixed(1)+'" cy="'+cy.toFixed(1)+'" r="'+r+'" style="fill:'+col+'"/>';
}
function pwSvg(xs, series){
  const W=760,H=340,pL=52,pR=14,pT=24,pB=42;
  const x0=xs[0],x1=xs[xs.length-1],ylo=-0.06,yhi=1.06;
  const X=v=>pL+(v-x0)/(x1-x0)*(W-pL-pR);
  const Y=v=>H-pB-(v-ylo)/(yhi-ylo)*(H-pT-pB);
  let s='<svg viewBox="0 0 '+W+' '+H+'" class="chart">';
  for (const t of [0,0.25,0.5,0.75,1]){
    s+='<line x1="'+pL+'" y1="'+Y(t).toFixed(1)+'" x2="'+(W-pR)+'" y2="'+Y(t).toFixed(1)+'" class="grid"/>'
      +'<text x="'+(pL-6)+'" y="'+(Y(t)+3.5).toFixed(1)+'" class="tk" text-anchor="end">'+t+'</text>';
  }
  for (const t of [-1,-0.5,0,0.5,1])
    s+='<text x="'+X(t).toFixed(1)+'" y="'+(H-pB+16)+'" class="tk" text-anchor="middle">'+t+'</text>';
  for (const se of series){
    const pts=xs.map((x,i)=>X(x).toFixed(1)+','+Y(se.ys[i]).toFixed(1)).join(' ');
    s+='<polyline points="'+pts+'" fill="none" style="stroke:'+se.col+'" stroke-width="2.2"'+(se.dash?' stroke-dasharray="'+se.dash+'"':'')+'/>';
    if (xs.length<=9) for (let i=0;i<xs.length;i++) s+=pwMark(X(xs[i]),Y(se.ys[i]),se.col,se.mk||'c',3.2);
  }
  s+='<line x1="'+pL+'" y1="'+(H-pB)+'" x2="'+(W-pR)+'" y2="'+(H-pB)+'" class="ax"/><line x1="'+pL+'" y1="'+pT+'" x2="'+pL+'" y2="'+(H-pB)+'" class="ax"/>';
  s+='<text x="'+((pL+W-pR)/2)+'" y="'+(H-8)+'" class="al" text-anchor="middle">X</text>';
  s+='<text x="14" y="'+((pT+H-pB)/2)+'" class="al" text-anchor="middle" transform="rotate(-90 14 '+((pT+H-pB)/2)+')">pi(X)</text>';
  return s+'</svg>';
}
function pwScat(xs, series, naive){
  const W=760,H=300,pL=52,pR=14,pT=24,pB=42;
  const x0=Math.min(...xs),x1=Math.max(...xs),ylo=-0.06,yhi=1.06;
  const X=v=>pL+(v-x0)/(x1-x0)*(W-pL-pR);
  const Y=v=>H-pB-(v-ylo)/(yhi-ylo)*(H-pT-pB);
  let s='<svg viewBox="0 0 '+W+' '+H+'" class="chart">';
  s+='<text x="'+pL+'" y="14" class="ct">Raw pointwise policy pi(X_i) at the '+xs.length+' support points (seed 0)</text>';
  for (const t of [0,0.5,1]) s+='<line x1="'+pL+'" y1="'+Y(t).toFixed(1)+'" x2="'+(W-pR)+'" y2="'+Y(t).toFixed(1)+'" class="grid"/>';
  if (naive) for (let i=0;i<xs.length;i++) s+='<g opacity="0.25">'+pwMark(X(xs[i]),Y(naive[i]/100),MCJS['DoublyRobust-X-X'],'t',1.7)+'</g>';
  for (const se of series) for (let i=0;i<xs.length;i++) s+='<g opacity="0.8">'+pwMark(X(xs[i]),Y(se.ys[i]/100),se.col,se.mk||'c',1.9)+'</g>';
  s+='<line x1="'+pL+'" y1="'+(H-pB)+'" x2="'+(W-pR)+'" y2="'+(H-pB)+'" class="ax"/>';
  return s+'</svg>';
}
function pwDraw(wid){
  const d=PD[wid]; if(!d) return;
  const gv=(s)=>{const e=document.getElementById('pw-'+wid+'-'+s); return (e&&e.tagName==='SELECT')?e.value:null;};
  const g=gv('g'), l=gv('l'), reg=gv('r')||'uncap';
  const ms=Array.from(document.querySelectorAll('#pw-'+wid+'-m input:checked')).map(e=>e.dataset.m);
  let series=[], note='';
  if (d.kind==='2d'){
    series.push({lab:'oracle', col:'var(--fg)', ys:d.refs.oracle, dash:'6 4'});
    series.push({lab:'naive DR', col:MCJS['DoublyRobust-X-X'], ys:d.refs.naive, dash:'2 3'});
    for (const mm of ms) series.push({lab:mm+' (G='+g+', L='+l+')', col:MCJS[mm]||'#7f7f7f', ys:d.pol[mm][g][l], dash:DASHJS[mm]||'', mk:MARKJS[mm]||'c'});
  } else {
    series.push({lab:reg==='uncap'?'oracle':'capped oracle', col:'var(--fg)',
                 ys:reg==='uncap'?d.refs.oracle_uncap:d.refs.oracle_cap, dash:'6 4'});
    for (const mm of ms){
      const c=d.pol[reg][mm]&&d.pol[reg][mm][g];
      if (c) series.push({lab:mm+' (G='+g+', '+reg+')', col:MCJS[mm]||'#7f7f7f', ys:c, dash:DASHJS[mm]||'', mk:MARKJS[mm]||'c'});
      else note='<p class="muted" style="padding:0 8px 8px">'+mm+' has no '+reg+' variant.</p>';
    }
  }
  if (!ms.length) note='<p class="muted" style="padding:0 8px 8px">Select at least one method above.</p>';
  let leg='<div class="leg">'+series.map(se=>'<span class="li"><span class="sw" style="background:'+se.col+'"></span>'+se.lab+'</span>').join('')+'</div>';
  let vals='';
  if (d.kind==='disc' && ms.length===1 && series.length>1){
    const ys=series[series.length-1].ys;
    vals='<div class="muted mono" style="padding:0 8px 8px">pi = ['+ys.map(v=>Math.round(v*100)/100).join(', ')+']</div>';
  }
  let head='';
  if (d.kind==='2d' && d.sup){
    const ss=[];
    for (const mm of ms) if (d.sup[mm]&&d.sup[mm][g]&&d.sup[mm][g][l]) ss.push({col:MCJS[mm]||'#7f7f7f', ys:d.sup[mm][g][l], mk:MARKJS[mm]||'c'});
    head=pwScat(d.supX, ss, d.supNaive||null);
  }
  document.getElementById('pw-'+wid+'-plot').innerHTML=head+pwSvg(d.grid,series)+leg+vals+note;
}
function svSvg(xs, series, hls){
  const W=760,H=340,pL=56,pR=14,pT=24,pB=42;
  let ys=[]; for(const s of series) ys=ys.concat(s.ys.filter(v=>v===v));
  for(const h of hls) ys.push(h[1]);
  let ylo=Math.min(...ys), yhi=Math.max(...ys);
  const pad=0.08*(yhi-ylo+1e-9); ylo-=pad; yhi+=pad;
  const x0=xs[0], x1=xs[xs.length-1];
  const X=v=>pL+(v-x0)/(x1-x0+1e-12)*(W-pL-pR);
  const Y=v=>H-pB-(v-ylo)/(yhi-ylo+1e-12)*(H-pT-pB);
  let s='<svg viewBox="0 0 '+W+' '+H+'" class="chart">';
  for(let i=0;i<=5;i++){const t=ylo+pad+i*(yhi-ylo-2*pad)/5;
    s+='<line x1="'+pL+'" y1="'+Y(t).toFixed(1)+'" x2="'+(W-pR)+'" y2="'+Y(t).toFixed(1)+'" class="grid"/>'
      +'<text x="'+(pL-6)+'" y="'+(Y(t)+3.5).toFixed(1)+'" class="tk" text-anchor="end">'+t.toFixed(2)+'</text>';}
  for(const g of xs) s+='<text x="'+X(g).toFixed(1)+'" y="'+(H-pB+16)+'" class="tk" text-anchor="middle">'+g+'</text>';
  s+='<line x1="'+X(5).toFixed(1)+'" y1="'+pT+'" x2="'+X(5).toFixed(1)+'" y2="'+(H-pB)+'" style="stroke:#0a7d33" stroke-width="1.6" stroke-dasharray="3 3"/>';
  s+='<text x="'+X(5).toFixed(1)+'" y="'+(pT+10)+'" class="tk" text-anchor="middle" style="fill:#0a7d33">Gamma*</text>';
  const hcol={'oracle':'var(--fg)','naive':MCJS['DoublyRobust-X-X'],'never':'#888888'};
  const hdash={'oracle':'5 4','naive':'2 3','never':'2 3'};
  for(const [lab,v] of hls){
    s+='<line x1="'+pL+'" y1="'+Y(v).toFixed(1)+'" x2="'+(W-pR)+'" y2="'+Y(v).toFixed(1)+'" style="stroke:'+hcol[lab]+'" stroke-width="1.4" stroke-dasharray="'+hdash[lab]+'"/>'
      +'<text x="'+(W-pR-2)+'" y="'+(Y(v)-4).toFixed(1)+'" class="tk" text-anchor="end" style="fill:'+hcol[lab]+'">'+lab+'</text>';}
  for(const se of series){
    const pts=xs.map((x,i)=>X(x).toFixed(1)+','+Y(se.ys[i]).toFixed(1)).join(' ');
    s+='<polyline points="'+pts+'" fill="none" style="stroke:'+se.col+'" stroke-width="2.2"'+(se.dash?' stroke-dasharray="'+se.dash+'"':'')+'/>';
    for(let i=0;i<xs.length;i++) s+=pwMark(X(xs[i]),Y(se.ys[i]),se.col,se.mk||'c',3.0);
  }
  s+='<line x1="'+pL+'" y1="'+(H-pB)+'" x2="'+(W-pR)+'" y2="'+(H-pB)+'" class="ax"/><line x1="'+pL+'" y1="'+pT+'" x2="'+pL+'" y2="'+(H-pB)+'" class="ax"/>';
  s+='<text x="'+((pL+W-pR)/2)+'" y="'+(H-8)+'" class="al" text-anchor="middle">Gamma</text>';
  s+='<text x="14" y="'+((pT+H-pB)/2)+'" class="al" text-anchor="middle" transform="rotate(-90 14 '+((pT+H-pB)/2)+')">test E[Y]</text>';
  return s+'</svg>';
}
function svDraw(wid){
  const d=SURFDS[wid]; const e=document.getElementById('sv-'+wid+'-l'); if(!e||!d) return;
  const l=e.value; const xs=d.gammas.map(Number);
  const series=d.methods.map(m=>({lab:m+' (L='+l+')', col:MCJS[m]||'#7f7f7f',
    ys:d.gammas.map(g=>d.surface[m][g][l]), dash:DASHJS[m]||'', mk:MARKJS[m]||'c'}));
  const leg='<div class="leg">'+series.map(se=>'<span class="li"><span class="sw" style="background:'+se.col+'"></span>'+se.lab+'</span>').join('')+'</div>';
  document.getElementById('sv-'+wid+'-plot').innerHTML=svSvg(xs,series,[['oracle',d.oracle],['naive',d.naive],['never',d.never]])+leg;
}
document.addEventListener('change',e=>{
  const w=e.target.closest('.polw'); if(w){ pwDraw(w.id.slice(3)); return; }
  if (e.target && e.target.dataset && e.target.dataset.sv) svDraw(e.target.dataset.sv);
});
document.addEventListener('click',e=>{
  const b=e.target.closest('.mbtn'); if(!b) return;
  const w=b.closest('.polw');
  w.querySelectorAll('.mck input').forEach(i=>{i.checked=(b.dataset.sel==='all');});
  pwDraw(w.id.slice(3));
});
for (const k of Object.keys(PD)) pwDraw(k);
for (const k of Object.keys(SURFDS)) svDraw(k);
"""

TABS = f"""
<div class="tabs" role="tablist" style="position:sticky;top:0;z-index:9;background:var(--bg);border-bottom:1px solid var(--border);display:flex;gap:6px;padding:10px 0;margin:0 0 6px">
<button id="tb-dgp" class="on" onclick="showTab('dgp')">DGP explanation</button>
<button id="tb-disc" onclick="showTab('disc')">Discrete experiments</button>
<button id="tb-cont" onclick="showTab('cont')">Continuous experiments</button>
</div>
<div id="tab-dgp" class="tabpane on">{T1}</div>
<div id="tab-disc" class="tabpane">{''.join(T2)}</div>
<div id="tab-cont" class="tabpane">{''.join(T3)}</div>
<script>
function showTab(id){{
  for (const t of ['dgp','disc','cont']){{
    document.getElementById('tab-'+t).classList.toggle('on', t===id);
    document.getElementById('tb-'+t).classList.toggle('on', t===id);
  }}
  window.scrollTo(0,0);
}}
</script>"""

import json as _json
data_js = ("<script>\nconst PD = " + _json.dumps(_r3(PD), separators=(",", ":")) + ";\n"
           + "const EVD = " + _json.dumps(_r3(EVD), separators=(",", ":")) + ";\n"
           + "const SURFDS = " + _json.dumps(_r3(SURFDS), separators=(",", ":")) + ";\n"
           + (JS % {"MC": _json.dumps(MC),
                    "DASH": _json.dumps({m: mdash(m) for m in MORDER + ["Kallus"]}),
                    "MARK": _json.dumps({m: mmark(m) for m in MORDER + ["Kallus"]})})
           + EV_JS
           + "\n</script>")

page = ('<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        f'<title>Gamma-star showcase</title><style>{CSS}{WIDGET_CSS}</style></head><body>'
        f'{hero}{TABS}{data_js}</body></html>')
page = page.encode("ascii", "xmlcharrefreplace").decode("ascii")
(HERE / "gstar_report.html").write_text(page)
print("wrote", HERE / "gstar_report.html", "(%d KB)" % (len(page) // 1024))
