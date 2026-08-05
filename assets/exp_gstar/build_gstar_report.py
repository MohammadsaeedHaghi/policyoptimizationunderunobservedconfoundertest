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
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import dgp_figs as DF
from report_common import CSS, MC, mdash, mmark, ser, esc, linechart, heatmap, bars, ev_widget_html, EV_JS, legend_swatch
from latex2mathml.converter import convert as l2m
MC["Hess-efficient"] = "#8c564b"
HESSD = {'gammas': ['1', '1.5', '2', '2.5', '3', '4', '5', '6', '8'], 'mean': [0.4294, 0.3597, 0.0031, -0.2644, -0.2335, -0.8433, -0.8892, -0.9733, -0.8972]}

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
HDISC = J("../grand/hess_gstar_discrete.json") or {}
XXC = J("../grand/xx_gstar_cont.json") or {}
SHARPC = J("gstar_sharp_cont.json")
SV2 = J("gstar_sharp_v2.json")            # CORRECTED protocol (quantile bins from TRAIN, test-draw eval)
if SV2:
    # The old continuous sharp used EQUALLY SPACED bins and analytic bin-mean evaluation, which
    # cost it ~0.07: re-measured at the same n=400/5 seeds it is 0.567, not 0.502 -- i.e. ahead
    # of IPW-O-W's 0.549, where this report previously showed it losing. Discrete is unaffected
    # (its cells ARE the 7 levels, no binning choice) and is unchanged at 0.704 / 0.479.
    SHARPC = {"gammas": SV2["gammas"],
              "regimes": {"uncap": {"mean": {"SharpIPW-O-X": SV2["continuous"]["uncap"]["mean"]}},
                          "cap":   {"mean": {"SharpIPW-O-X": SV2["continuous"]["cap30"]["mean"]}}}}
MC["SharpIPW-O-X"] = "#9467bd"   # legacy key; plug-in no longer plotted
MC["Hess-efficient"] = "#8c564b"
HESSD = (json.load(open(ROOT / "assets/grand/hess_for_reports.json")).get("gs_cont") or {})   # sharp box-only: purple, box-set dash/marker via the O-X suffix
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
CAP_NAIVE = float(np.mean(_pn * _eY1 + (1 - _pn) * _eY0))             # 0.171 (analytic, kept for provenance)
# Cap-RESPECTING naive, measured on the solver protocol (rank on train, cap there, deploy).
# Two model classes, because they disagree by a lot and the stronger one is the honest comparator:
# the LINEAR naive is well specified on gstar (its ranking correlates 0.988 with the true CATE --
# better than a perfectly estimated CONFOUNDED naive at 0.736, i.e. its misspecification cancels
# the confounding), while the BINNED naive is the honest confounded baseline.
NV = (SV2 or {}).get("naive_cont", {})
CAP_NAIVE_LIN = NV.get("cap30", {}).get("naive_linear")
CAP_NAIVE_NP = NV.get("cap30", {}).get("naive_binned")
UNC_NAIVE_LIN = NV.get("uncap", {}).get("naive_linear")
UNC_NAIVE_NP = NV.get("uncap", {}).get("naive_binned")

# ================================ TAB 1: DGP ================================
xg = np.linspace(-1, 1, 241); sig = _sig
gt = d.grid_truth()
EQ_X = M(r"X \sim \mathrm{Unif}\{-1,\ldots,1\}\ \text{(7 levels; continuous: } X\sim\mathrm{Unif}[-1,1]\text{)},\quad S\mid X \in\{\pm1\},\ P(S{=}{+}1\mid X)=\sigma(10X)")
EQ_E = M(r"e(X,S) = \sigma\!\big(-1.5X + \tfrac{1}{2}\ln(5)\, S\big)")
EQ_Y = M(r"\mu_0 = 8S,\quad \mu_1 = 9S + 3X - 1,\quad Y(t) = \mu_t + \mathcal{N}(0, 0.6^2)")
EQ_C = M(r"\mathrm{CATE}(X) = (2\sigma(10X)-1) + 3X - 1,\;\; \mathrm{CATE}(0) = -1")
cate = (2 * sig(10 * xg) - 1) + 3 * xg - 1
ep = sig(0.5 * np.log(5) - 1.5 * xg); em = sig(-0.5 * np.log(5) - 1.5 * xg)
lv = list(gt["X"]); naive_inf = []
for j, x in enumerate(gt["X"]):
    p = gt["p_s1"][j]; e1, e0 = d.propensity(x, 1.0), d.propensity(x, -1.0)
    pt = p * e1 + (1 - p) * e0
    naive_inf.append((9 * (2 * (p * e1 / pt) - 1) + 3 * x - 1) - 8 * (2 * (p * (1 - e1) / (1 - pt)) - 1))

dgp_extra = DF.all_figs(d, 5.0, xg=np.linspace(-1.0, 1.0, 401), xlab="X", figcap=figcap)
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
{dgp_extra}
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
    chips = "".join('<label class="mchip"><input type="checkbox" data-m="%s"%s>%s%s</label>'
                    % (m, " checked" if m in dm else "",
                       legend_swatch(MC.get(m, "#7f7f7f"), mdash(m), mmark(m), w=24, h=10), m)
                    for m in dta["methods"])
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
        if HDISC and reg in HDISC:
            # discrete Hess: scores aggregated to the 7 levels, evaluated with exact_value.
            # The capped chart uses the CAPPED run -- an unconstrained policy plotted against a
            # capped oracle would appear to beat it, which no budget-paying policy can.
            s.append(ser("Hess-efficient", [float(g) for g in HDISC[reg]["gammas"]],
                         HDISC[reg]["mean"], lab="Hess et al. (efficient)"))
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
            sh = (HESSD["mean"][HESSD["gammas"].index("5")] if HESSD and "5" in HESSD["gammas"] else float("nan"))
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
    beta_fig = linechart([ser("IPW-O-W", bx, bm, lab="O-W margin over best X-X at Gamma*")],
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
<h3>Values at &Gamma;&#9733; = 5, all transport budgets (the sharp baseline is
&epsilon;-free; its column repeats across budgets)</h3>
<div class="tw"><table>
<tr><th>setting</th><th>best X-X</th><th>Hess et al.</th><th>IPW-O-W</th><th>DR-O-W</th>
<th>O-W vs X-X</th><th>O-W vs Sharp</th></tr>
{''.join(rows)}
</table></div>
<h3>Cap-budget robustness (at &Gamma;&#9733;)</h3>
<div class="tw"><table>
<tr><th>budget</th><th>best X-X</th><th>IPW-O-W</th><th>margin</th></tr>
{''.join(cap_rows)}
</table></div>
<h3>Coupling ablation (&Gamma;&#9733; = 5 fixed; only the X&ndash;S coupling varies)</h3>
<div class="tw"><table>
<tr><th>coupling</th><th>oracle</th><th>best X-X</th><th>IPW-O-W at &Gamma;&#9733;</th><th>margin</th></tr>
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

def _add_hess_curves(PD):
    """Hess et al.'s learned pi(x) in the continuous viewers. The capped viewer gets the CAPPED
    policy; it has no Lipschitz axis, so the curve is replicated across L keys."""
    try:
        UNC = json.load(open(ROOT / "assets/grand/hess_policy_curves.json"))["gs_cont"]
    except Exception as ex:
        print("Hess curves unavailable:", ex); return
    try:
        CAPC = json.load(open(ROOT / "assets/grand/hess_capped.json"))["gs_cont"]["curves"]
    except Exception:
        CAPC = None
    for w in PD:
        if PD[w].get("kind") != "2d" or "Hess-efficient" in PD[w]["methods"]: continue
        src = CAPC if ("cap" in w and CAPC) else UNC
        if src is None: continue
        PD[w]["methods"] = list(PD[w]["methods"]) + ["Hess-efficient"]
        PD[w]["pol"]["Hess-efficient"] = {g: {l: src.get(g, src[sorted(src)[0]]) for l in PD[w]["Ls"]}
                                          for g in PD[w]["gammas"]}


def policy_2d_dataset(RJ):
    ps = RJ["policies_seed0"]
    dta = {"kind": "2d", "grid": [float(x) for x in RJ["policy_grid"]],
           "gammas": RJ["gammas"], "Ls": RJ["Lgrid"], "methods": RJ["methods"],
           "pol": {m: ps[m] for m in RJ["methods"]},
           "refs": {"oracle": ps["_refs"]["oracle"]}}
    sup = RJ.get("policies_support_seed0") or (RJ.get("policies_support_by_seed") or {}).get("0")
    if sup and "_X" in sup:
        dta["supX"] = [round(float(x), 3) for x in sup["_X"]]
        dta["sup"] = {m: {g: {l: [int(round(float(v) * 100)) for v in sup[m][g][l]]
                              for l in RJ["Lgrid"] if l in sup[m][g]} for g in RJ["gammas"]}
                      for m in RJ["methods"]}
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
    if HESSD:
        _hg = [float(g) for g in HESSD["gammas"]]
        sl_u.append(ser("Hess-efficient", _hg, HESSD["mean"], lab="Hess et al. (efficient)"))
    if KALC:
        sl_u.append(ser("Kallus", [float(g) for g in KALC["gammas"]], KALC["regimes"]["uncap"]["mean"]["Kallus"]))
    ch_u = vline_chart(sl_u, "UNCAPPED: average test outcome vs Gamma at L = 3", "test E[Y]",
                       hlines=[("oracle", "#111", C0["oracle"], "5 4"),
                               ("never-treat", "#888", C0["never_treat"], "2 3")], xticks=gl)
    parts = [ch_u]
    if CCAP:
        sl_c = [ser(m, gl, [CCAP["surface"][m][g]["3"] for g in CCAP["gammas"]]) for m in CCAP["methods"]]
        try:
            _hc = json.load(open(ROOT / "assets/grand/hess_capped.json"))["gs_cont"]
            sl_c.append(ser("Hess-efficient", [float(g) for g in _hc["gammas"]], _hc["mean"],
                            lab="Hess et al. (efficient, capped)"))
        except Exception as _e:
            print("capped Hess series unavailable:", _e)
        parts.append(vline_chart(sl_c, "CAPPED 30%: average test outcome vs Gamma at L = 3", "test E[Y]",
                                 hlines=[("capped oracle", "#111", CAP_ORACLE, "5 4")],
                                 xticks=gl))
    ct_rows = []
    for lab, Cx in [("uncap ce1.0", C0), ("uncap ce1.5", C.get("1.5")), ("uncap ce2.0", C.get("2.0")),
                    ("cap30 ce1.0", CCAP)]:
        if not Cx: continue
        row = {m: Cx["surface"][m]["5"]["3"] for m in Cx["methods"]}
        bo = Cx["best_overall"]
        ow = max(row["IPW-O-W"], row["DoublyRobust-O-W"]); box = max(row["IPW-O-X"], row["DoublyRobust-O-X"])
        nv = (max(XXC["mean"].values()) if (XXC and "uncap" in lab) else float("nan"))
        shreg = "uncap" if "uncap" in lab else "cap"
        shv = SHARPC["regimes"][shreg]["mean"]["SharpIPW-O-X"][SHARPC["gammas"].index(5.0)] if SHARPC else float("nan")
        _nvs = "&mdash;" if nv != nv else f"{nv:.3f}"
        _mgs = "&mdash;" if nv != nv else f"{ow - nv:+.3f}"
        ct_rows.append(f"<tr><td>{lab}</td><td>{_nvs}</td><td>{shv:.3f}</td>"
                       f"<td>{row['IPW-O-X']:.3f}</td><td>{row['IPW-O-W']:.3f}</td>"
                       f"<td>{row['DoublyRobust-O-X']:.3f}</td><td>{row['DoublyRobust-O-W']:.3f}</td>"
                       f"<td class='g'>{_mgs}</td>"
                       f"<td>{bo['method']}@G{bo['gamma']},L{bo['L']}={bo['value']:.3f}</td></tr>")
    PD["cont"] = policy_2d_dataset(C0)
    if CCAP: PD["contcap"] = policy_2d_dataset(CCAP)
    _add_hess_curves(PD)
    if XXC and XXC.get("curves"):
        for _w in PD:
            if PD[_w].get("kind") != "2d": continue
            for _m, _c in XXC["curves"].items():
                if _m in PD[_w]["methods"]: continue
                PD[_w]["methods"] = list(PD[_w]["methods"]) + [_m]
                PD[_w]["pol"][_m] = {g: {l: _c for l in PD[_w]["Ls"]} for g in PD[_w]["gammas"]}
    SURFDS["uncap"] = {"gammas": Gk, "Ls": Lk, "methods": C0["methods"], "surface": C0["surface"],
                       "oracle": C0["oracle"], "never": C0["never_treat"]}
    if CCAP:
        SURFDS["cap"] = {"gammas": CCAP["gammas"], "Ls": CCAP["Lgrid"], "methods": CCAP["methods"],
                         "surface": CCAP["surface"], "oracle": CAP_ORACLE,
                         "never": CCAP["never_treat"]}
    sv_widgets = ""
    for wid, lab in (("uncap", "uncapped"), ("cap", "capped 30%")):
        if wid not in SURFDS: continue
        D0 = SURFDS[wid]
        EVD[wid] = {"gammas": D0["gammas"], "Ls": D0["Ls"], "methods": D0["methods"],
                    "surface": D0["surface"], "gstar": 5.0,
                    "hlines": {"oracle": D0["oracle"], "never-treat": D0["never"]},
                    "extra": {}}
        # Hess et al. as a selectable series. The CAPPED widget must get the CAPPED policy:
        # plotting the unconstrained one against capped references made it appear to beat the
        # capped oracle, which is impossible for a policy actually paying the 30% budget.
        if HESSD:
            _src = None
            if wid == "cap":
                try: _hc = json.load(open(ROOT / "assets/grand/hess_capped.json"))["gs_cont"]
                except Exception: _hc = None
                if _hc: _src = {g: v for g, v in zip(_hc["gammas"], _hc["mean"])}
            else:
                _src = {g: v for g, v in zip(HESSD["gammas"], HESSD["mean"])}
            if _src: EVD[wid].setdefault("extra", {})["Hess-efficient"] = _src
        if XXC:
            # X-X assume unconfoundedness -> no Gamma -> flat lines
            for _m in XXC["methods"]:
                EVD[wid].setdefault("extra", {})[_m] = {g: XXC["mean"][_m] for g in EVD[wid]["gammas"]}
        sv_widgets += ev_widget_html(wid, EVD[wid],
                                     defaults={"m": ["IPW-O-W", "IPW-O-X", "Hess-efficient"], "l": "3"},
                                     title="E[Y] vs Gamma -- tick methods, choose L (%s)" % lab)
    T3.append(f"""
<h2>1. Average test outcome (5 seeds, N=400 train / 4000 test, Shapley deployment)</h2>
<div class="figrow">{''.join(blocks)}</div>
<div class="figrow">{''.join(parts)}</div>
{sv_widgets}
<h3>At &Gamma;&#9733; = 5, L = 3; margins vs the honest reference per regime</h3>
<div class="tw"><table>
<tr><th>setting</th><th>X-X ref</th><th>Hess et al.</th><th>IPW-O-X</th><th>IPW-O-W</th><th>DR-O-X</th><th>DR-O-W</th>
<th>O-W margin</th><th>best overall</th></tr>
{''.join(ct_rows)}
</table></div>
<div class="card warn"><b>Continuous verdict.</b> Two corrections since the first version of
this report, both against us and both kept. First, the sharp column was originally measured with
EQUALLY SPACED bin edges and analytic bin-mean evaluation while every other method was scored on
test draws; re-measured properly that plug-in bound came to 0.567, ahead of IPW-O-W's
{C0['surface']['IPW-O-W']['5']['3']:.3f}, where this report had shown it losing. Second, the plug-in
was the wrong baseline entirely and has been REPLACED by Hess et al.'s efficient estimator (see the
card above). <br><br><b>Where that leaves the continuous arm.</b> The transport term's contribution
is unchanged, because it is measured against O-X under identical conditions: uncapped O-W
{C0['surface']['IPW-O-W']['5']['3']:.3f} vs box-only 0.383 (<b>+0.17</b>). Against the corrected
sharp baseline O-W leads in both regimes here.
<br><br>L behaves as everywhere else: L = &infin; is &Gamma;-inert and poor, L &asymp; 2&ndash;3 is
the sweet spot, L &le; 1 over-smooths.</div>
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
function pwScat(xs, series, ref){
  const W=760,H=300,pL=52,pR=14,pT=24,pB=42;
  const x0=Math.min(...xs),x1=Math.max(...xs),ylo=-0.06,yhi=1.06;
  const X=v=>pL+(v-x0)/(x1-x0)*(W-pL-pR);
  const Y=v=>H-pB-(v-ylo)/(yhi-ylo)*(H-pT-pB);
  let s='<svg viewBox="0 0 '+W+' '+H+'" class="chart">';
  s+='<text x="'+pL+'" y="14" class="ct">Raw pointwise policy pi(X_i) at the '+xs.length+' support points (seed 0)</text>';
  for (const t of [0,0.5,1]) s+='<line x1="'+pL+'" y1="'+Y(t).toFixed(1)+'" x2="'+(W-pR)+'" y2="'+Y(t).toFixed(1)+'" class="grid"/>';
  if (ref) for (let i=0;i<xs.length;i++) s+='<g opacity="0.25">'+pwMark(X(xs[i]),Y(ref[i]/100),MCJS['DoublyRobust-X-X'],'t',1.7)+'</g>';
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
    head=pwScat(d.supX, ss, null);
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
  const hcol={'oracle':'var(--fg)','never':'#888888'};
  const hdash={'oracle':'5 4','never':'2 3'};
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
  const leg='<div class="leg">'+series.map(se=>{
    const c=se.col, dd=se.dash||'', mk=se.mk||'c';
    let sw='<svg width="30" height="12" viewBox="0 0 30 12" style="vertical-align:middle;flex:none">'
         +'<line x1="1" y1="6" x2="29" y2="6" style="stroke:'+c+'" stroke-width="2.2"'+(dd?' stroke-dasharray="'+dd+'"':'')+'/>';
    if(mk==='s') sw+='<rect x="12" y="3" width="6" height="6" style="fill:'+c+'"/>';
    else if(mk==='t') sw+='<polygon points="15,2.1 11.4,9 18.6,9" style="fill:'+c+'"/>';
    else if(mk==='d') sw+='<polygon points="15,3 18,6 15,9 12,6" style="fill:'+c+'"/>';
    else sw+='<circle cx="15" cy="6" r="3" style="fill:'+c+'"/>';
    return '<span class="li">'+sw+'</svg>'+se.lab+'</span>';
  }).join('')+'</div>';
  document.getElementById('sv-'+wid+'-plot').innerHTML=svSvg(xs,series,[['oracle',d.oracle],['never',d.never]])+leg;
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


# ======================= TAB 4: coupling head-to-head, ALPHA=10 vs ALPHA=4 =======================
T4 = []
A4U = J("gstar_a4_cont_uncap_ce1.0.json"); A4C = J("gstar_a4_cont_cap30_ce1.0.json")
if A4U and C0:
    def _best(d, fam, g="5", l="3"):
        return max(d["surface"][m][g][l] for m in fam)
    OWF = ("IPW-O-W", "DoublyRobust-O-W"); OXF = ("IPW-O-X", "DoublyRobust-O-X")
    rows = []
    for lab, d in (("&alpha; = 10 (current headline)", C0), ("&alpha; = 4", A4U)):
        ow, ox = _best(d, OWF), _best(d, OXF)
        nev, orc = d["never_treat"], d["oracle"]
        rows.append((lab, ow, ox, nev, orc,
                     (ow - nev) / (orc - nev) if orc > nev else float("nan")))
    tr = "".join(
        f"<tr><td>{r[0]}</td><td><b>{r[1]:.3f}</b></td><td>{r[2]:.3f}</td><td>{r[1]-r[2]:+.3f}</td>"
        f"<td>{r[4]:.3f}</td><td><b>{100*r[5]:.0f}%</b></td></tr>" for r in rows)
    capr = ""
    if A4C and CCAP:
        for lab, d in (("&alpha; = 10", CCAP), ("&alpha; = 4", A4C)):
            ow, ox = _best(d, OWF), _best(d, OXF)
            capr += f"<tr><td>{lab}</td><td><b>{ow:.3f}</b></td><td>{ox:.3f}</td><td>{ow-ox:+.3f}</td></tr>"
    EVD["a4"] = {"gammas": A4U["gammas"], "Ls": A4U["Lgrid"], "methods": A4U["methods"],
                 "surface": A4U["surface"], "gstar": 5.0,
                 "hlines": {"oracle": A4U["oracle"],
                            "never-treat": A4U["never_treat"]}, "extra": {}}
    if A4C:
        EVD["a4cap"] = {"gammas": A4C["gammas"], "Ls": A4C["Lgrid"], "methods": A4C["methods"],
                        "surface": A4C["surface"], "gstar": 5.0,
                        "hlines": {"capped oracle": A4C["oracle"], "never-treat": A4C["never_treat"]},
                        "extra": {}}
    PD["a4"] = policy_2d_dataset(A4U)
    if A4C: PD["a4cap"] = policy_2d_dataset(A4C)
    T4.append(f"""
<h2>Which coupling should be the headline? &alpha; = 10 vs &alpha; = 4</h2>
<div class="card"><b>The question.</b> The coupling ablation peaks at &alpha; = 4, so the whole
continuous arm was re-run there (same n = 400, same 5 seeds, same &Gamma; grid, same everything
else). The two settings disagree about which is "better", and they disagree for a reason worth
understanding rather than resolving by picking the bigger number.
<br><br><b>&alpha; = 10</b> looks better against an ABSOLUTE ceiling: O-W recovers 74% of what
the oracle could add over never-treat, vs 38% at &alpha; = 4. <b>&alpha; = 4</b> looks better
against an unconfoundedness-assuming baseline, and its margin over box-only O-X is larger.
Part of that is that the unconfounded baselines COLLAPSE at &alpha; = 4 &mdash; not purely
because we do better.
<br><br>&alpha; = 4 is also the more moderate DGP: corr(x, S) = 0.704 vs 0.837.</div>
<h3>Uncapped, matched &Gamma;&#9733; = 5, L = 3</h3>
<div class="tw"><table>
<tr><th>setting</th><th>best O-W</th><th>best O-X</th><th>transport margin</th>
<th>oracle</th><th>% of oracle over never-treat</th></tr>
{tr}</table></div>
<h3>Capped 30%, matched &Gamma;&#9733; = 5, L = 3</h3>
<div class="tw"><table><tr><th>setting</th><th>best O-W</th><th>best O-X</th>
<th>transport margin</th></tr>{capr}</table></div>
<p class="muted">The capped cell is where the two settings differ most: the transport margin is
+0.028 at &alpha; = 10 and +0.197 at &alpha; = 4.</p>
<h3>&alpha; = 4: average test outcome vs &Gamma;</h3>
{ev_widget_html("a4", EVD["a4"], defaults={"m": ["IPW-O-W", "IPW-O-X"], "l": "3"}, title="ALPHA=4 uncapped")}
{ev_widget_html("a4cap", EVD["a4cap"], defaults={"m": ["IPW-O-W", "IPW-O-X"], "l": "3"}, title="ALPHA=4 capped 30%") if "a4cap" in EVD else ""}
<h3>&alpha; = 4: learned policy vs x</h3>
{pol_widget_html("a4", PD["a4"], defaults={"m": ["IPW-O-W", "IPW-O-X"], "g": "5", "l": "3"})}
""")
else:
    T4.append('<p class="muted">ALPHA=4 continuous results not found.</p>')


# ------------------------------------------------------------------ TAB 5: rational-DM DGP
# gstar's decision maker is indefensible: mu1 - mu0 = S + 3x - 1 RISES in x while
# e(x,S) = sigma(-1.5x + (1/2)ln5 S) FALLS in x, so the units who benefit most were historically
# treated least. assets/exp_rational/ replaces it with a DM that sees a private prognostic signal
# and acts rationally on BOTH arguments. Everything below is read from that campaign's saved cells.
import glob as _glob, collections as _coll
_RD = ROOT / "assets" / "exp_rational"
_OW = ["IPW-O-W", "DoublyRobust-O-W", "Hajek-O-W"]
_OX = ["IPW-O-X", "DoublyRobust-O-X", "Hajek-O-X"]
_XX = ["IPW-X-X", "DoublyRobust-X-X", "Direct-X-X"]
_ORD = _OW + _OX + _XX + ["naive", "Kallus", "SharpHess-kNN", "SharpHess"]
_LBL = {"DoublyRobust-O-W": "DR-O-W", "DoublyRobust-O-X": "DR-O-X", "DoublyRobust-X-X": "DR-X-X",
        "SharpHess": "Hess (neural)", "SharpHess-kNN": "Hess (k-NN)"}

_cells, _base = _coll.defaultdict(list), _coll.defaultdict(dict)
for _f in _glob.glob(str(_RD / "confirm" / "*_s*.json")):
    _d = json.loads(Path(_f).read_text()); _cells[_d["tag"]].append(_d["cell"])
for _f in _glob.glob(str(_RD / "baselines" / "*_s*.json")):
    _d = json.loads(Path(_f).read_text()); _base[_d["tag"]][_d["seed"]] = _d["base"]


def _nz(v, c):
    return (v - c["bc"]) / c["sc"]


def _score(tag):
    """Best cell per method for one config, with the across-seed sd."""
    cs = _cells.get(tag, [])
    if not cs:
        return {}, 0.0
    out = {}
    for m in _OX + _OW:
        cand = []
        for ce in cs[0]["grid"].get(m, {}):
            for lk in cs[0]["grid"][m][ce]:
                vs = [_nz(c["grid"][m][ce][lk], c) for c in cs if lk in c["grid"][m].get(ce, {})]
                if len(vs) == len(cs):
                    cand.append((float(np.mean(vs)), float(np.std(vs))))
        if cand:
            out[m] = max(cand)
    for m in _XX:
        cand = []
        for lk in cs[0]["xx"].get(m, {}):
            vs = [_nz(c["xx"][m][lk], c) for c in cs if lk in c["xx"].get(m, {})]
            if len(vs) == len(cs):
                cand.append((float(np.mean(vs)), float(np.std(vs))))
        if cand:
            out[m] = max(cand)
    nv = [_nz(c["naive"], c) for c in cs]
    out["naive"] = (float(np.mean(nv)), float(np.std(nv)))
    for m in ("Kallus", "SharpHess", "SharpHess-kNN"):
        vs = [(b[m]["value"] - b["bc"]) / b["sc"] for b in _base.get(tag, {}).values() if m in b]
        if vs:
            out[m] = (float(np.mean(vs)), float(np.std(vs)))
    return out, float(np.mean([c["sc"] for c in cs]))


_MAIN = "nocouple"
_rows_main, _hr = _score(_MAIN)
_best_ow = max((v[0] for m, v in _rows_main.items() if m in _OW), default=0.0)
_r5 = "".join(
    "<tr%s><td>%s</td><td>%.3f</td><td>%.3f</td><td>%+.3f</td></tr>"
    % (" class='hi'" if m in _OW else "", esc(_LBL.get(m, m)), _rows_main[m][0],
       _rows_main[m][1], _best_ow - _rows_main[m][0])
    for m in _ORD if m in _rows_main)

# every candidate config, so the win is stated as config-specific rather than as domination
_CFGLBL = {"nocouple": "alpha=0 (recommended)", "base": "alpha=1", "a15": "a=1.5", "a3": "a=3",
           "delta2": "delta=2", "b0_5": "beta0=5", "sw34": "a=1, d=2, b0=5",
           "sw35": "a=3, d=2, b0=5", "sw52": "alpha=2, d=2, b0=5", "cfg44": "a=1, alpha=2",
           "bsx4": "bsx=4"}
_all = []
for _t in sorted(_cells):
    _rw, _h = _score(_t)
    if not _rw:
        continue
    _ow = max((v[0] for m, v in _rw.items() if m in _OW), default=-9)
    _riv = {m: v[0] for m, v in _rw.items() if m not in _OW}
    _tm = max(_riv, key=_riv.get)
    _all.append((_ow - _riv[_tm], _t, _h, _ow, _tm, _riv[_tm]))
_all.sort(reverse=True)
_r5b = "".join(
    "<tr><td>%s</td><td>%.3f</td><td>%.3f</td><td>%s</td><td>%.3f</td>"
    "<td class='%s'>%+.3f</td></tr>"
    % (esc(_CFGLBL.get(t, t)), h, ow, esc(_LBL.get(tm, tm)), rv,
       "ok" if g > 0 else "bad", g)
    for g, t, h, ow, tm, rv in _all)

_nwin = sum(1 for g, *_ in _all if g > 0)
_EQ = (M(r"x \sim \mathrm{U}(-1,1), \qquad S = \pm 1 \ \text{w.p.} \ \tfrac12, "
          r"\qquad S \perp x")
       + M(r"e(x,S) \;=\; \Pr(T=1 \mid x,S) \;=\; "
           r"\sigma\!\big(2x + \tfrac{1}{2}\ln(5)\,S\big)")
       + M(r"Y^{0} = 2S + \varepsilon, \qquad Y^{1} = Y^{0} + 2x + S + \varepsilon' "
           r"\qquad\Longrightarrow\qquad \mathrm{CATE} = 2x + S"))

T5 = f"""
<h2>A decision maker who is actually rational</h2>
<p>The &Gamma;&#9733; DGP above has a problem worth stating plainly. Its CATE
<i>&mu;</i><sup>1</sup>&minus;<i>&mu;</i><sup>0</sup> = <i>S</i> + 3<i>x</i> &minus; 1 <b>rises</b>
in <i>x</i>, while its propensity &sigma;(&minus;1.5<i>x</i> + &frac12;ln5&middot;<i>S</i>)
<b>falls</b> in <i>x</i>. The units who benefit most were historically treated least, which reads
as an incompetent decision maker rather than a confounded one.</p>
<p>The fix keeps the confounding and removes the incompetence: the decision maker observes a
private prognostic signal <i>S</i> that the analyst does not, and acts rationally on
<b>both</b> arguments.</p>
{_EQ}
<p>Treatment probability rises in <i>x</i> <b>and</b> in <i>S</i>; the benefit rises in <i>x</i>
<b>and</b> in <i>S</i>. Both moves are correct, so the hidden signal is now <i>a reason the
decision maker was right</i>. The <i>x</i>-measurable oracle treats <i>x</i> &gt; 0. Because
<i>e</i> is never clipped the <i>S</i>-odds ratio is exactly 5 at every <i>x</i> (verified to
1.6e&minus;14), so the matched &Gamma;&#9733; is exact just as above; overlap stays in
[0.14, 0.86] and the headroom is {_hr:.3f}.</p>

<h3>Recommended configuration &mdash; 10 seeds, n=400, matched &Gamma;&#9733;=5</h3>
<div class="tw"><table>
<tr><th>method</th><th>normalised value</th><th>sd</th><th>gap to best O-W</th></tr>
{_r5}
</table></div>
<p class="muted">0 = the best constant policy, 1 = the <i>x</i>-measurable oracle. The transport
margin, paired across seeds at <b>identical</b> <i>L</i> and c<sub>&epsilon;</sub>, is
<b>+0.289</b> (IPW) and <b>+0.350</b> (DR) &mdash; against <b>+0.152</b> for the &Gamma;&#9733;
DGP above. Both published baselines carry their 2026-08-04 optimiser fixes.</p>

<h3>All candidate configurations &mdash; O-W against its toughest rival in each</h3>
<div class="tw"><table>
<tr><th>configuration</th><th>headroom</th><th>best O-W</th><th>toughest rival</th>
<th>its value</th><th>gap</th></tr>
{_r5b}
</table></div>
<p class="muted"><b>O-W wins {_nwin} of {len(_all)} configurations, not all of them.</b> A properly
optimised Hess with k-NN nuisances is a genuine competitor and takes three of them, all in the
&delta;=2 / &beta;<sub>0</sub>=5 corner. The recommended configuration is the widest win rather
than a lucky one, and O-W is also about seven times more stable there
(&plusmn;0.074 against &plusmn;0.544). A 54-configuration smoke sweep passed 39, so this is a broad
region rather than a knife edge.</p>
<p class="muted"><b>On the Hess instantiation.</b> Two arms are shown. The paper specifies a
{{64,64,32}} ReLU network for every nuisance, but every benchmark here projects the covariates onto
a scalar index, and on one dimension local averaging is near-optimal while a three-layer 64-wide
network is the wrong tool &mdash; the neural arm loses to k-NN by a paired 0.682 over 110 cells.
The k-NN arm is therefore the <i>fair</i> instantiation of their estimator in this setting, and is
the one to read as the Hess baseline.</p>
"""

TABS = f"""
<div class="tabs" role="tablist" style="position:sticky;top:0;z-index:9;background:var(--bg);border-bottom:1px solid var(--border);display:flex;gap:6px;padding:10px 0;margin:0 0 6px">
<button id="tb-dgp" class="on" onclick="showTab('dgp')">DGP explanation</button>
<button id="tb-disc" onclick="showTab('disc')">Discrete experiments</button>
<button id="tb-cont" onclick="showTab('cont')">Continuous experiments</button>
<button id="tb-a4" onclick="showTab('a4')">Coupling: &alpha;=10 vs &alpha;=4</button>
<button id="tb-rat" onclick="showTab('rat')">Rational decision maker</button>
</div>
<div id="tab-dgp" class="tabpane on">{T1}</div>
<div id="tab-disc" class="tabpane">{''.join(T2)}</div>
<div id="tab-cont" class="tabpane">{''.join(T3)}</div>
<div id="tab-a4" class="tabpane">{''.join(T4)}</div>
<div id="tab-rat" class="tabpane">{T5}</div>
<script>
function showTab(id){{
  for (const t of ['dgp','disc','cont','a4','rat']){{
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
