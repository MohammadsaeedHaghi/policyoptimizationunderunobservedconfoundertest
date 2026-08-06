#!/usr/bin/env python3
"""Build 'assets/exp_msmbench/kmz_report.html' -- the KMZ'19 (arXiv 1810.02894) DGP campaign.

The literature's standard MSM synthetic, UNMODIFIED, run through our full pipeline: the paper's
own Gamma*-strength sweep (log Gamma* in {0.5, 1.0, 1.5}, each at its matched Gamma -- known by
construction), transport-budget and sample-size ablations, the capped 30% variant (novel vs the
literature), the L x Gamma surface, Kallus + Hess et al. baselines, and analytic references.
Three tabs: DGP / Results / Policies. Rerun: python3 assets/exp_msmbench/build_kmz_report.py
"""
import json, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT / "semi experiments"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import dgp_figs as DF
from report_common import CSS, MC, mdash, mmark, ser, esc, linechart, heatmap, ev_widget_html, EV_JS, legend_swatch
from latex2mathml.converter import convert as l2m
MC["Hess-efficient"] = "#8c564b"
HESSD = {'gammas': ['1', '2', '3', '4.4817', '6', '8'], 'mean': [0.1343, 0.4396, 0.6705, 0.9599, 0.9813, 1.0177]}

import importlib.util

def _load(p, n):
    sp = importlib.util.spec_from_file_location(n, p); m = importlib.util.module_from_spec(sp)
    sys.modules[n] = m; sp.loader.exec_module(m); return m
d = _load(HERE / "dgp_g15.py", "kmz_rep_d")
MC["Hess-efficient"] = "#8c564b"
HESSD = (json.load(open(ROOT / "assets/grand/hess_paper_for_reports.json")).get("km") or {})
# Hess at THEIR OWN n=5000 (the paper's operating point) -- quoted as a caveat wherever the
# n=400 number appears, because their five-network pipeline is data-hungry.
try:
    HESS5K = json.load(open(ROOT / "assets/grand/hess_paper_km5000.json"))["4.4817"]
except Exception:
    HESS5K = None

def J(fn):
    try: return json.load(open(HERE / fn))
    except Exception: return None
EVD = {}
# X-X at MATCHED L (xxL_kmz.json: L in {inf, 3, 1}); the old xx_kmz.json carried the
# no-lipschitz solve only (~0.35), which overstated the O-W-vs-X-X gap by ~5x.
_XXL = J("../grand/xxL_kmz.json") or {}
if _XXL:
    _best = {m: max(d, key=d.get) for m, d in _XXL["mean"].items()}      # best L per method
    XX = {"methods": list(_XXL["mean"]),
          "mean": {m: _XXL["mean"][m][_best[m]] for m in _XXL["mean"]},
          "sd": {m: _XXL["sd"][m][_best[m]] for m in _XXL["mean"]},
          "bestL": _best, "byL": _XXL["mean"]}
else:
    XX = J("../grand/xx_kmz.json") or {}
CM = {ce: J(f"kmz_main_ce{ce}.json") for ce in ("1.0", "1.5", "2.0")}
CC = J("kmz_cap30_ce1.0.json")
GST = {"05": J("kmz_g05.json"), "10": J("kmz_g10.json")}
N1K = J("kmz_n1000.json")
KAL = J("kmz_kallus.json")
REFS = J("kmz_refs.json")
GSTAR = 4.4817
GKEY = "4.4817"

def figcap(fig, cap): return f'<div>{fig}<p class="muted figcap">{cap}</p></div>'
def M(tex): return f'<div style="text-align:center;overflow-x:auto;margin:10px 0">{l2m(tex)}</div>'

def vline_chart(series, title, ylab, hlines=None, W=680, xticks=None):
    ys = [y for it in series for y in it[3] if y == y] + [h[2] for h in (hlines or [])]
    marker = ("Gamma* = e^1.5", "#0a7d33", [GSTAR, GSTAR], [min(ys), max(ys)], "3 3")
    return linechart(series + [marker], title=title, xlab="Gamma", ylab=ylab, hlines=hlines, W=W, xticks=xticks)

# ================================ TAB 1: DGP ================================
xg = np.linspace(-1, 1, 241); Xg = 2 * xg
EQ = M(r"Y(a) = (2a{-}1)X + (2a{-}1) - 2\sin(2(2a{-}1)X) - 2S(1+0.5X) + \mathcal{N}(0,1),"
       r"\quad X \sim \mathrm{Unif}[-2,2],\; S \in \{\pm 1\},\; P(S{=}{+}1) = \tfrac12,\; S \perp X")
EQ_E = M(r"e(x,S) = \frac{1{+}S}{2\,\rho(x, 1/\Gamma^{\!*})} + \frac{1{-}S}{2\,\rho(x, \Gamma^{\!*})},"
         r"\quad \rho(x,\gamma) = 1 + \Big(\frac{1}{e(x)}-1\Big)\gamma,\quad e(x) = \sigma(0.75X + 0.5)")
cate = 2 * Xg + 2 - 4 * np.sin(2 * Xg)
e1 = d.propensity(xg, np.ones_like(xg)); e0 = d.propensity(xg, -np.ones_like(xg))
# Four more DGP figures, shared with the gstar report (assets/dgp_figs.py).
dgp_extra = DF.all_figs(d, d.GSTAR, xg=np.linspace(-1.0, 1.0, 401),
                        xlab="x = X/2", figcap=figcap)
T1 = f"""
<h2>1. The DGP (Kallus-Mao-Zhou 2019, arXiv 1810.02894 -- unmodified)</h2>
{EQ}{EQ_E}
<p>We run the paper's own confounding-strength sweep, log &Gamma;* &isin; {{0.5, 1.0, 1.5}}
(&Gamma;* &asymp; 1.65 / 2.72 / 4.48), each experiment at its MATCHED &Gamma; = &Gamma;* --
known by construction, nothing tuned. Pipeline units: x = X/2 &isin; [-1, 1]. 5 seeds,
N = 400 train / 4,000 test draws, Shapley deployment; every raw
per-seed policy persisted.</p>
<div class="card warn"><b>Declared expectation (the honest frame, written before the wave).</b>
This DGP has S &perp; X: measured corr(x, S) &asymp; 0.01 -- the ZERO-COUPLING anchor of our
diagnostic map. No X-balance device can constrain a confounder that is independent of X, so
the prediction is <b>O-W &asymp; O-X (do no harm), with box-family methods on their
correctly-specified home turf performing well</b>. The campaign's value: our methods evaluated
on the literature's own benchmark with its own strength axis, the do-no-harm property
demonstrated, and the capped 30% variant (absent from the literature) added.</div>
{dgp_extra}
<p>Analytic references: oracle {REFS['oracle_uncap']:.3f} uncapped / {REFS['oracle_cap30']:.3f}
capped(30%); never-treat {REFS['never']:.3f}; all-treat {REFS['all']:.3f}.</p>
"""

# ================================ TAB 2: RESULTS ================================
T2 = []
C0 = CM.get("1.0")
if C0:
    Gk, Lk = C0["gammas"], C0["Lgrid"]; gl = [float(g) for g in Gk]
    sl = [ser(m, gl, [C0["surface"][m][g]["3"] for g in Gk]) for m in C0["methods"]]
    if KAL: sl.append(ser("Kallus", [float(g) for g in KAL["gammas"]], KAL["regimes"]["uncap"]["mean"]["Kallus"]))
    ch = vline_chart(sl, "UNCAPPED: average test outcome vs Gamma at L = 3 (Gamma* = e^1.5)", "test E[Y]",
                     hlines=[("oracle", "#111", REFS["oracle_uncap"], "5 4"),
                             ("never-treat", "#888", REFS["never"], "2 3")], xticks=gl)
    EVD["kuncap"] = {"gammas": Gk, "Ls": Lk, "methods": C0["methods"], "surface": C0["surface"],
                     "gstar": GSTAR,
                     "hlines": {"oracle": REFS["oracle_uncap"],
                                "never-treat": REFS["never"]},
                     "extra": {}}
    if KAL: EVD["kuncap"]["extra"]["Kallus"] = {("%g" % g): v for g, v in zip(KAL["gammas"], KAL["regimes"]["uncap"]["mean"]["Kallus"])}
    parts = [ch]
    if CC:
        sl_c = [ser(m, gl, [CC["surface"][m][g]["3"] for g in CC["gammas"]]) for m in CC["methods"]]
        # No capped Hess overlay: the only capped series came from the superseded custom
        # trainer; a paper-exact capped run for KMZ does not exist yet.
        parts.append(vline_chart(sl_c, "CAPPED 30%: average test outcome vs Gamma at L = 3", "test E[Y]",
                                 hlines=[("capped oracle", "#111", REFS["oracle_cap30"], "5 4")],
                                 xticks=gl))
        EVD["kcap"] = {"gammas": CC["gammas"], "Ls": CC["Lgrid"], "methods": CC["methods"],
                       "surface": CC["surface"], "gstar": GSTAR,
                       "hlines": {"capped oracle": REFS["oracle_cap30"]},
                       "extra": {}}
    if XX:
        # X-X methods assume unconfoundedness, so they carry no Gamma: each is a FLAT line.
        for _m in XX["methods"]:
            EVD["kuncap"].setdefault("extra", {})[_m] = {g: XX["mean"][_m] for g in Gk}
    if HESSD:
        _hx = {g: v for g, v in zip(HESSD["gammas"], HESSD["mean"])}
        # UNCAPPED numbers go ONLY in the uncapped widget. Applying them to the capped panel
        # plotted an unconstrained policy against capped references, which is why it appeared to
        # beat the capped oracle: it was not paying the 30% budget.
        # capped overlay REMOVED: it came from the superseded custom trainer, and a paper-exact
        # capped Hess run for KMZ does not exist yet -- better absent than wrong.
        _hxc = None
        for _w in EVD:
            _is_cap = "cap" in _w
            if _is_cap and _hxc: EVD[_w].setdefault("extra", {})["Hess-efficient"] = _hxc
            elif not _is_cap:    EVD[_w].setdefault("extra", {})["Hess-efficient"] = _hx
    hms = [heatmap(["G=" + g for g in Gk], Lk, [[C0["surface"]["IPW-O-W"][g][l] for l in Lk] for g in Gk],
                   "UNCAPPED IPW-O-W surface (oracle %.2f)" % C0["oracle"], "Gamma", "Lipschitz L",
                   C0["never_treat"], C0["oracle"], W=560, H=280)]
    if CC:
        hms.append(heatmap(["G=" + g for g in CC["gammas"]], CC["Lgrid"],
                           [[CC["surface"]["IPW-O-W"][g][l] for l in CC["Lgrid"]] for g in CC["gammas"]],
                           "CAPPED 30%% IPW-O-W surface (capped oracle %.2f)" % REFS["oracle_cap30"],
                           "Gamma", "Lipschitz L", CC["never_treat"], REFS["oracle_cap30"], W=560, H=280))
    # strength ablation table (the paper's axis): each Gamma* at its matched Gamma
    st_rows = []
    for tag, lab in (("05", "e^0.5 = 1.65"), ("10", "e^1.0 = 2.72")):
        Rg = GST.get(tag)
        if not Rg: continue
        g0 = Rg["gammas"][0]
        row = {m: Rg["surface"][m][g0]["3"] for m in Rg["methods"]}
        shv = float("nan")   # strength ladder: Hess not run per strength arm
        st_rows.append(f"<tr><td>&Gamma;* = {lab}</td><td>{shv:.3f}</td>"
                       f"<td>{row['IPW-O-X']:.3f}</td><td>{row['IPW-O-W']:.3f}</td>"
                       f"<td>{row['DoublyRobust-O-X']:.3f}</td><td>{row['DoublyRobust-O-W']:.3f}</td></tr>")
    row15 = {m: C0["surface"][m][GKEY]["3"] for m in C0["methods"]}
    sh15 = (HESSD["mean"][HESSD["gammas"].index("4.4817")] if HESSD and "4.4817" in HESSD["gammas"] else float("nan"))
    st_rows.append(f"<tr><td>&Gamma;* = e^1.5 = 4.48 (main)</td>"
                   f"<td>{sh15:.3f}</td>"
                   f"<td>{row15['IPW-O-X']:.3f}</td><td>{row15['IPW-O-W']:.3f}</td>"
                   f"<td>{row15['DoublyRobust-O-X']:.3f}</td><td>{row15['DoublyRobust-O-W']:.3f}</td></tr>")
    # ceps + n ablations at matched Gamma
    ab_rows = []
    for ce, Cx in CM.items():
        if not Cx: continue
        r = {m: Cx["surface"][m][GKEY]["3"] for m in Cx["methods"]}
        ab_rows.append(f"<tr><td>c<sub>&epsilon;</sub> = {ce}, n = 400</td>"
                       f"<td>{r['IPW-O-W']:.3f}</td><td>{r['DoublyRobust-O-W']:.3f}</td>"
                       f"<td>{r['IPW-O-X']:.3f}</td><td>{r['DoublyRobust-O-X']:.3f}</td></tr>")
    if N1K:
        g0 = N1K["gammas"][0]
        r = {m: N1K["surface"][m][g0]["3"] for m in N1K["methods"]}
        ab_rows.append(f"<tr><td>c<sub>&epsilon;</sub> = 1.0, n = 1000</td>"
                       f"<td>{r['IPW-O-W']:.3f}</td><td>{r['DoublyRobust-O-W']:.3f}</td>"
                       f"<td>{r['IPW-O-X']:.3f}</td><td>{r['DoublyRobust-O-X']:.3f}</td></tr>")
    xx_rows = []
    for _m in (XX.get("methods") or []):
        _lb = XX.get("bestL", {}).get(_m, "")
        xx_rows.append(f"<tr><td>{_m}{' (L=' + _lb + ')' if _lb else ''}</td>"
                       f"<td>{XX['mean'][_m]:.3f}</td><td>{XX['sd'][_m]:.3f}</td></tr>")
    T2.append(f"""
<h2>1. Average test outcome (5 seeds; &Gamma; grid with the matched &Gamma;* flagged)</h2>
<p class="muted">Tick methods to overlay; the static pair below shows every series at L = 3.</p>
{ev_widget_html("kuncap", EVD["kuncap"], defaults={"m": ["IPW-O-W", "IPW-O-X", "Hess-efficient"], "l": "3"},
                title="UNCAPPED &mdash; pick methods and L")}
{ev_widget_html("kcap", EVD["kcap"], defaults={"m": ["IPW-O-W"], "l": "3"},
                title="CAPPED 30% &mdash; pick methods and L") if "kcap" in EVD else ""}
<div class="figrow">{parts[0]}{parts[1] if len(parts) > 1 else ''}</div>
<div class="figrow">{''.join(hms)}</div>
<h2>2. The paper's own axis: confounding strength &Gamma;* (each at its matched &Gamma;, L = 3)</h2>
<div class="tw"><table>
<tr><th>strength</th><th>Hess et al.</th><th>IPW-O-X</th><th>IPW-O-W</th>
<th>DR-O-X</th><th>DR-O-W</th></tr>
{''.join(st_rows)}
</table></div>
<p class="muted">Hess et al. runs the authors' own pipeline verbatim (their repository:
{{64,32}} ReLU networks for every nuisance and the policy, Adam lr 10<sup>&minus;3</sup>,
batch 64, &le;300 epochs, patience 10, disjoint 50/50 split). It is data-hungry: at this
report's shared n&nbsp;=&nbsp;400 it scores {HESSD["mean"][HESSD["gammas"].index("4.4817")]:.3f}
at the matched &Gamma;&#9733;, but {HESS5K["mean"]:.3f}&nbsp;&plusmn;&nbsp;{HESS5K["sd"]:.3f}
at its own paper's n&nbsp;=&nbsp;5000 &mdash; a size the LP methods cannot run at
(n&sup2; transport variables). Both numbers belong in any citation.</p>
<h2>3. Ablations at the matched &Gamma;* = 4.48, L = 3</h2>
<div class="tw"><table>
<tr><th>setting</th><th>IPW-O-W</th><th>DR-O-W</th><th>IPW-O-X</th><th>DR-O-X</th></tr>
{''.join(ab_rows)}
</table></div>
<h3>The unconfoundedness-assuming (X&ndash;X) methods</h3>
<p class="muted">These assume no unobserved confounding, so they have no &Gamma; and are a single
number each &mdash; the baselines the whole premise is about. Same protocol, same seeds, same
Shapley deployment.</p>
<div class="tw"><table><tr><th>method</th><th>test E[Y]</th><th>sd over seeds</th></tr>
{''.join(xx_rows)}</table></div>

""")
else:
    T2.append('<p class="muted">Wave running; rerun this builder when jobs land.</p>')

# ================================ TAB 3: POLICIES ================================
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
    c.append(f'<label>&Gamma; <select id="pw-{wid}-g">{opts(dta["gammas"], df.get("g", GKEY))}</select></label>')
    c.append(f'<label>L <select id="pw-{wid}-l">{opts(dta["Ls"], df.get("l", "3"))}</select></label>')
    c.append(f'</div><div id="pw-{wid}-plot" class="fig"></div></div>')
    return "".join(c)


def _add_hess_curves(PD, key):
    """Add Hess et al.'s learned pi(x) to the 2-D policy viewers (1-D covariate, binary A here).
    It has no Lipschitz axis, so the same curve is replicated across L keys."""
    try:
        HC = json.load(open(ROOT / "assets/grand/hess_policy_curves.json"))[key]
    except Exception as ex:
        print("Hess curves unavailable:", ex); return
    for w in PD:
        if PD[w].get("kind") != "2d" or "Hess-efficient" in PD[w]["methods"]: continue
        PD[w]["methods"] = list(PD[w]["methods"]) + ["Hess-efficient"]
        _src = HC
        if "cap" in w:
            try: _src = json.load(open(ROOT / "assets/grand/hess_capped.json"))[key]["curves"]
            except Exception: _src = None
        if _src is None: continue
        PD[w]["pol"]["Hess-efficient"] = {g: {l: _src.get(g, _src[sorted(_src)[0]]) for l in PD[w]["Ls"]}
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

T3 = []
if C0:
    PD["kuncap"] = policy_2d_dataset(C0)
    T3.append("""
<h2>1. Learned policy vs x -- uncapped</h2>
<p>Raw per-unit support policy (seed 0) above; Shapley-deployed pi(x) below. The oracle's
oscillating interior structure is the test: an unconfounded fit under-treats the positive regions.</p>""")
    # _add_hess_curves(PD, "km")  -- old-learner curves; superseded by the paper pipeline
    if XX and XX.get("curves"):
        for _w in PD:
            if PD[_w].get("kind") != "2d": continue
            for _m, _c in XX["curves"].items():
                if _m in PD[_w]["methods"]: continue
                PD[_w]["methods"] = list(PD[_w]["methods"]) + [_m]
                PD[_w]["pol"][_m] = {g: {l: _c for l in PD[_w]["Ls"]} for g in PD[_w]["gammas"]}
    T3.append(pol_widget_html("kuncap", PD["kuncap"], defaults={"m": ["IPW-O-W", "IPW-O-X"], "g": GKEY, "l": "3"}))
    if CC:
        PD["kcap"] = policy_2d_dataset(CC)
        T3.append("<h2>2. Learned policy vs x -- capped 30%</h2>")
        T3.append(pol_widget_html("kcap", PD["kcap"], defaults={"m": ["IPW-O-W"], "g": GKEY, "l": "3"}))
else:
    T3.append('<p class="muted">Wave running.</p>')


# ================================ assembly ================================
hero = """<div class="hero" style="background:radial-gradient(130% 150% at 0% 0%,#4a4a2d 0%,#3d3d1f 46%,#1a1a0c 100%)">
<h1>The KMZ'19 benchmark campaign &mdash; the literature's synthetic, full pipeline</h1>
<p>Kallus-Mao-Zhou (2019) DGP, unmodified, with the paper's own confounding-strength sweep
(log &Gamma;* &isin; {0.5, 1.0, 1.5}, each at its matched &Gamma; -- known by construction) plus
our ablations: transport budgets, the L &times; &Gamma; surface, the capped 30% variant,
and Kallus + Hess et al. baselines. 5 seeds; parallel CARC-license wave; all raw
policies persisted.</p></div>"""

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
  return '<circle cx="'+cx.toFixed(1)+'" cy="'+cy.toFixed(1)+'" r="'+r+'" style="fill:'+col+'"/>';
}
function pwSvg(xs, series){
  const W=760,H=340,pL=52,pR=14,pT=24,pB=42;
  const x0=xs[0],x1=xs[xs.length-1],ylo=-0.06,yhi=1.06;
  const X=v=>pL+(v-x0)/(x1-x0)*(W-pL-pR);
  const Y=v=>H-pB-(v-ylo)/(yhi-ylo)*(H-pT-pB);
  let s='<svg viewBox="0 0 '+W+' '+H+'" class="chart">';
  for (const t of [0,0.25,0.5,0.75,1]) s+='<line x1="'+pL+'" y1="'+Y(t).toFixed(1)+'" x2="'+(W-pR)+'" y2="'+Y(t).toFixed(1)+'" class="grid"/>';
  for (const t of [-1,-0.5,0,0.5,1]) s+='<text x="'+X(t).toFixed(1)+'" y="'+(H-pB+16)+'" class="tk" text-anchor="middle">'+t+'</text>';
  for (const se of series){
    const pts=xs.map((x,i)=>X(x).toFixed(1)+','+Y(se.ys[i]).toFixed(1)).join(' ');
    s+='<polyline points="'+pts+'" fill="none" style="stroke:'+se.col+'" stroke-width="2.2"'+(se.dash?' stroke-dasharray="'+se.dash+'"':'')+'/>';
  }
  s+='<line x1="'+pL+'" y1="'+(H-pB)+'" x2="'+(W-pR)+'" y2="'+(H-pB)+'" class="ax"/><line x1="'+pL+'" y1="'+pT+'" x2="'+pL+'" y2="'+(H-pB)+'" class="ax"/>';
  s+='<text x="'+((pL+W-pR)/2)+'" y="'+(H-8)+'" class="al" text-anchor="middle">x</text>';
  s+='<text x="14" y="'+((pT+H-pB)/2)+'" class="al" text-anchor="middle" transform="rotate(-90 14 '+((pT+H-pB)/2)+')">pi(x)</text>';
  return s+'</svg>';
}
function pwScat(xs, series, ref){
  const W=760,H=300,pL=52,pR=14,pT=24,pB=42;
  const x0=Math.min(...xs),x1=Math.max(...xs),ylo=-0.06,yhi=1.06;
  const X=v=>pL+(v-x0)/(x1-x0)*(W-pL-pR);
  const Y=v=>H-pB-(v-ylo)/(yhi-ylo)*(H-pT-pB);
  let s='<svg viewBox="0 0 '+W+' '+H+'" class="chart">';
  s+='<text x="'+pL+'" y="14" class="ct">Raw pointwise policy at the support points (seed 0)</text>';
  if (ref) for (let i=0;i<xs.length;i++) s+='<g opacity="0.25">'+pwMark(X(xs[i]),Y(ref[i]/100),MCJS['DoublyRobust-X-X'],'t',1.7)+'</g>';
  for (const se of series) for (let i=0;i<xs.length;i++) s+='<g opacity="0.8">'+pwMark(X(xs[i]),Y(se.ys[i]/100),se.col,se.mk||'c',1.9)+'</g>';
  s+='<line x1="'+pL+'" y1="'+(H-pB)+'" x2="'+(W-pR)+'" y2="'+(H-pB)+'" class="ax"/>';
  return s+'</svg>';
}
function pwDraw(wid){
  const d=PD[wid]; if(!d) return;
  const gv=(s)=>{const e=document.getElementById('pw-'+wid+'-'+s); return (e&&e.tagName==='SELECT')?e.value:null;};
  const g=gv('g'), l=gv('l');
  const ms=Array.from(document.querySelectorAll('#pw-'+wid+'-m input:checked')).map(e=>e.dataset.m);
  let series=[{lab:'oracle', col:'var(--fg)', ys:d.refs.oracle, dash:'6 4'},
              ];
  for (const mm of ms) series.push({lab:mm+' (G='+g+', L='+l+')', col:MCJS[mm]||'#7f7f7f', ys:d.pol[mm][g][l], dash:DASHJS[mm]||'', mk:MARKJS[mm]||'c'});
  let head='';
  if (d.sup){
    const ss=[];
    for (const mm of ms) if (d.sup[mm]&&d.sup[mm][g]&&d.sup[mm][g][l]) ss.push({col:MCJS[mm]||'#7f7f7f', ys:d.sup[mm][g][l], mk:MARKJS[mm]||'c'});
    head=pwScat(d.supX, ss, null);
  }
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
  document.getElementById('pw-'+wid+'-plot').innerHTML=head+pwSvg(d.grid,series)+leg;
}
document.addEventListener('change',e=>{const w=e.target.closest('.polw'); if(w) pwDraw(w.id.slice(3));});
document.addEventListener('click',e=>{
  const b=e.target.closest('.mbtn'); if(!b) return;
  const w=b.closest('.polw');
  w.querySelectorAll('.mck input').forEach(i=>{i.checked=(b.dataset.sel==='all');});
  pwDraw(w.id.slice(3));
});
for (const k of Object.keys(PD)) pwDraw(k);
"""

TABS = f"""
<div class="tabs" role="tablist" style="position:sticky;top:0;z-index:9;background:var(--bg);border-bottom:1px solid var(--border);display:flex;gap:6px;padding:10px 0;margin:0 0 6px">
<button id="tb-dgp" class="on" onclick="showTab('dgp')">DGP</button>
<button id="tb-res" onclick="showTab('res')">Results &amp; ablations</button>
</div>
<div id="tab-dgp" class="tabpane on">{T1}</div>
<div id="tab-res" class="tabpane">{''.join(T2)}{''.join(T3)}</div>
<script>
function showTab(id){{
  for (const t of ['dgp','res']){{
    document.getElementById('tab-'+t).classList.toggle('on', t===id);
    document.getElementById('tb-'+t).classList.toggle('on', t===id);
  }}
  window.scrollTo(0,0);
}}
</script>"""

import json as _json
data_js = ("<script>\nconst PD = " + _json.dumps(_r3(PD), separators=(",", ":")) + ";\n"
           + "const EVD = " + _json.dumps(_r3(EVD), separators=(",", ":")) + ";\n"
           + (JS % {"MC": _json.dumps(MC),
                    "DASH": _json.dumps({m: mdash(m) for m in list(MC)}),
                    "MARK": _json.dumps({m: mmark(m) for m in list(MC)})})
           + EV_JS
           + "\n</script>")

page = ('<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        f'<title>KMZ benchmark campaign</title><style>{CSS}{WIDGET_CSS}</style></head><body>'
        f'{hero}{TABS}{data_js}</body></html>')
page = page.encode("ascii", "xmlcharrefreplace").decode("ascii")
(HERE / "kmz_report.html").write_text(page)
print("wrote", HERE / "kmz_report.html", "(%d KB)" % (len(page) // 1024))
