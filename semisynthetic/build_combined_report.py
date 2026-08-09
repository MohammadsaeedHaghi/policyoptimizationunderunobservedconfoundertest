#!/usr/bin/env python3
"""Two-tab HTML report: the risk-score-paper construction ("semi") and the RCT-derived one ("RCT").

Each tab carries its own flowchart, its construction facts, and its results. Both are built from
saved summaries only -- nothing is recomputed here.
"""
import json, sys
from pathlib import Path
import numpy as np

GAMMA_ALL = (0.0, 1.0, 1.5, 2.0, 3.0, 4.0)   # 3.0 and 4.0 added 2026-08-04
GAM = [1.0, 1.5, 2.0, 3.0, 4.0]        # confounded cells only; gamma=0 is the correctness check
GCOL = {1.0: "#0a7d33", 1.5: "#5b8c1a", 2.0: "#b07105", 3.0: "#c0392b", 4.0: "#7d1f6a"}

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "semi experiments"))
from report_common import CSS, esc
from latex2mathml.converter import convert as l2m


def M(tex):
    """Render LaTeX as MathML, as the other reports do -- real symbols, not mojibake."""
    return '<div class="eq">%s</div>' % l2m(tex)

# ------------------------------------------------------------------ data
SEMI_DS = ["bank_marketing", "wine_quality", "german_credit", "credit_default", "adult"]
SEMI = {d: json.loads((HERE / ("summary_%s.json" % d)).read_text())
        for d in SEMI_DS if (HERE / ("summary_%s.json" % d)).exists()}
RCT = json.loads((ROOT / "RCT datasets" / "summary_rct.json").read_text())
RCT_TITLE = {"ihdp": "IHDP", "twins": "Twins", "ist": "IST"}

OWF = ["IPW-O-W", "DoublyRobust-O-W", "Hajek-O-W"]
RCT_ORDER = ["IPW-O-X", "DoublyRobust-O-X", "Hajek-O-X", "IPW-O-W", "DoublyRobust-O-W",
             "IPW-X-X", "DoublyRobust-X-X", "Direct-X-X", "SharpHess", "Kallus"]
SHORT = {"DoublyRobust-O-W": "DR-O-W", "DoublyRobust-O-X": "DR-O-X",
         "DoublyRobust-X-X": "DR-X-X", "SharpHess": "Hess", "Hajek-O-W": "Hajek-O-W"}

EXTRA = """
table.dt{border-collapse:collapse;font-size:12.5px;margin:10px 0;width:100%}
table.dt th,table.dt td{border:1px solid #e3e6ea;padding:5px 8px;text-align:right;
  white-space:nowrap;font-variant-numeric:tabular-nums}
table.dt th{background:#f4f6f8;font-weight:600}
table.dt td.l,table.dt th.l{text-align:left}
table.dt td.ref{color:#111}
table.dt .sd{font-size:10.5px;opacity:.65;font-weight:400}
table.dt tr.dsrow td{border-top:2px solid #b9c0c8;background:#eef1f5;font-weight:700;text-align:left}
.scrollx{overflow-x:auto;max-width:100%}
.hsub{font-weight:400;font-size:.82rem;color:#667085}
.tabbar{display:flex;gap:6px;margin:18px 0 4px}
.tb{padding:8px 22px;border:1px solid #d8dde3;border-bottom:none;border-radius:8px 8px 0 0;
    background:#f2f4f7;cursor:pointer;font-size:15px}
.tb.on{background:#fff;font-weight:700}
.tabpane{display:none}.tabpane.on{display:block}
"""
RAMP = [(0.0, (0xC0, 0x39, 0x2B)), (0.5, (0xB0, 0x71, 0x05)), (1.0, (0x0A, 0x7D, 0x33))]


def tint(t):
    t = 0.0 if t != t else min(max(t, 0.0), 1.0)
    for (a, ca), (b, cb) in zip(RAMP, RAMP[1:]):
        if t <= b:
            f = (t - a) / (b - a)
            return "#%02x%02x%02x" % tuple(int(round(ca[i] + f * (cb[i] - ca[i]))) for i in range(3))
    return "#0a7d33"


def cells(vals, order, sds=None):
    vs = [vals[m] for m in order if m in vals]
    lo, hi = (min(vs), max(vs)) if vs else (0.0, 1.0)
    out = []
    for m in order:
        if m not in vals: out.append("<td>&mdash;</td>"); continue
        v = vals[m]; t = 1.0 if hi <= lo else (v - lo) / (hi - lo)
        bold = ";font-weight:700" if abs(v - hi) < 1e-12 else ""
        txt = "%.4f" % v
        if sds and m in sds: txt += " <span class='sd'>&plusmn;%.3f</span>" % sds[m]
        out.append("<td style='color:%s%s'>%s</td>" % (tint(t), bold, txt))
    return "".join(out)


def norm(v, R):
    bc = max(R["never_treat"], R["all_treat"])
    return (v - bc) / (R["oracle"] - bc)


# ------------------------------------------------------------------ the DGPs, in math
SEMI_MATH = "".join([
    M(r"X_i \in \mathbb{R}^{d}\ \text{(real UCI covariates, standardised)},\qquad "
      r"y_i^{\mathrm{UCI}} \in \{-1,+1\}"),
    M(r"U_i \;=\; Y_i^{0} \;=\; y_i^{\mathrm{UCI}} \qquad\text{(the label plays both roles)}"),
    M(r"\tau(x) \;=\; a\Big(\sigma(b^{\top}x - c) - \tfrac{1}{2}\Big),\qquad "
      r"a \sim \mathrm{U}(0.5,\,2),\;\; b \sim \mathcal{N}(0, I_d),\;\; c \sim \mathcal{N}(0,1)"),
    M(r"Y_i^{1} \;=\; Y_i^{0} + \tau(X_i) + \varepsilon_i,\qquad "
      r"\varepsilon_i \sim \mathcal{N}(0,\sigma^{2}),\;\; \sigma = 0.1"),
    M(r"\operatorname{logit}\pi^{0}(x,u) \;=\; \lambda^{\top}x + \gamma u,\qquad "
      r"\lambda_j \sim \mathrm{U}(-0.1,\,0.1),\qquad \pi^{0} = \Pr(T=0 \mid x,u)"),
    M(r"\pi^{0}(x,u) \;=\; \sigma\big(\lambda^{\top}x + \gamma u\big) \;=\; "
      r"\frac{1}{1 + e^{-(\lambda^{\top}x + \gamma u)}},\qquad "
      r"e(x,u) \;=\; \Pr(T=1\mid x,u) \;=\; 1 - \pi^{0}(x,u)"),
    M(r"\text{since } u \in \{-1,+1\} \text{ and } \Gamma^{\!\star} = e^{2\gamma}:\quad "
      r"\pi^{0}(x,+1) = \frac{\sqrt{\Gamma^{\!\star}}}"
      r"{\sqrt{\Gamma^{\!\star}} + e^{-\lambda^{\top}x}},\qquad "
      r"\pi^{0}(x,-1) = \frac{1}{1 + \sqrt{\Gamma^{\!\star}}\,e^{-\lambda^{\top}x}}"),
    M(r"T_i \sim \mathrm{Bernoulli}\big(1-\pi^{0}(X_i,U_i)\big),\qquad "
      r"Y_i \;=\; T_i Y_i^{1} + (1-T_i) Y_i^{0}"),
    M(r"\frac{\pi^{0}(x,+1)}{1-\pi^{0}(x,+1)} \Big/ \frac{\pi^{0}(x,-1)}{1-\pi^{0}(x,-1)}"
      r" \;=\; e^{2\gamma} \quad \forall x \qquad\Longrightarrow\qquad "
      r"\Gamma^{\!\star} = e^{2\gamma}"),
])

# Overlap summary, precomputed by precompute_overlap.py. PERCENTILES, not min/max: the extremes
# are taken over 25 populations x 20k rows, and at gamma=0 -- where there is no confounding at all
# and e must sit near 1/2 -- the minimum is already 0.0014. That is one outlier row in the
# standardised UCI covariates, not a property of the design.
_OVP = HERE / "prepared" / "_overlap.json"
overlap_tbl = ""
if _OVP.exists():
    _ov = json.loads(_OVP.read_text())
    _rows = "".join(
        "<tr><td class='l'>&gamma; = %g</td><td class='ref'>%.1f</td><td>%.3f</td><td>%.3f</td>"
        "<td>%.3f</td><td>%.4f</td><td>%.4f</td></tr>"
        % (r["gamma"], r["Gamma"], r["p01"], r["p50"], r["p99"],
           r["frac_below_01"], r["frac_above_99"])
        for r in sorted(_ov.values(), key=lambda r: r["gamma"]))
    overlap_tbl = ("<div class='scrollx'><table class='dt'>"
                   "<tr><th class='l'>&nbsp;</th><th>&Gamma;&#9733;</th><th>e, 1st pct</th>"
                   "<th>median</th><th>99th pct</th><th>frac e&lt;0.01</th>"
                   "<th>frac e&gt;0.99</th></tr>" + _rows + "</table></div>")

RCT_MATH = "".join([
    M(r"\big(Y_i^{0},\,Y_i^{1}\big)\ \text{taken from the source dataset; neither is resimulated}"),
    M(r"x_i \;=\; \mathrm{clip}\!\left(\frac{z_i^{A}}{q_{99}(|z^{A}|)},\,-1,\,1\right)"
      r"\in[-1,1], \qquad u_i \;=\; z_i^{B}"),
    M(r"\text{groups } A,B \text{ chosen to maximise } \big|\mathrm{corr}(x,u)\big|, \qquad "
      r"S_i \;=\; \operatorname{sign}\!\big(u_i - \operatorname{median}(u)\big) \in \{-1,+1\}"),
    M(r"e(x,S) \;=\; \sigma\!\Big(1.2\,x + \tfrac{1}{2}\ln(\Lambda)\,S\Big),\qquad "
      r"\Lambda = 4, \qquad \text{no clipping}"),
    M(r"T_i \sim \mathrm{Bernoulli}\big(e(x_i,S_i)\big), \qquad "
      r"Y_i \;=\; T_i Y_i^{1} + (1-T_i) Y_i^{0}"),
    M(r"\frac{e(x,+1)}{1-e(x,+1)} \Big/ \frac{e(x,-1)}{1-e(x,-1)} \;=\; \Lambda \quad \forall x"
      r" \qquad\Longrightarrow\qquad \Gamma^{\!\star} = \Lambda = 4"),
])

# ------------------------------------------------------------------ flowcharts
PAL = {"real": ("#dbeafe", "#1e40af"), "dual": ("#fef3c7", "#92400e"),
       "ours": ("#dcfce7", "#166534"), "synth": ("#ede9fe", "#5b21b6"),
       "out": ("#f1f5f9", "#334155")}


def flow(items, arrows, W, H, note=""):
    p = ['<defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" '
         'markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="#94a3b8"/></marker></defs>']
    for (x, y, w, h, kind, title, lines, mono) in items:
        bg, fg = PAL[kind]
        p.append('<rect x="%d" y="%d" width="%d" height="%d" rx="8" fill="%s" stroke="%s" '
                 'stroke-width="1.5"/>' % (x, y, w, h, bg, fg))
        p.append('<text x="%d" y="%d" font-size="13" font-weight="700" fill="%s">%s</text>'
                 % (x + 12, y + 20, fg, esc(title)))
        for i, ln in enumerate(lines):
            p.append('<text x="%d" y="%d" font-size="11.3" fill="#334155"%s>%s</text>'
                     % (x + 12, y + 38 + 14.5 * i, ' font-family="monospace"' if mono else '',
                        esc(ln)))
    for (x1, y1, x2, y2, lab) in arrows:
        p.append('<line x1="%g" y1="%g" x2="%g" y2="%g" stroke="#94a3b8" stroke-width="1.8" '
                 'marker-end="url(#ah)"/>' % (x1, y1, x2, y2))
        if lab:
            p.append('<text x="%g" y="%g" font-size="10.5" fill="#64748b" text-anchor="middle">'
                     '%s</text>' % ((x1 + x2) / 2 + 60, (y1 + y2) / 2 + 4, esc(lab)))
    fig = ('<figure class="fig"><svg viewBox="0 0 %d %d" class="chart" style="max-width:%dpx">%s'
           '</svg></figure>' % (W, H, W, "".join(p)))
    return fig + ("<p class='muted figcap'>%s</p>" % note if note else "")


SEMI_FLOW = flow(
    items=[
        (40, 26, 900, 62, "real", "UCI classification dataset", [
            "screened by the paper's rule: cross-validated logistic log-loss in [0.35, ln 2]",
            "-- not trivially separable, not pure noise. 9 of 15 candidates pass."], False),
        (40, 128, 430, 78, "dual", "The label is used TWICE  (their construction)", [
            "U     = y_UCI   the unobserved confounder",
            "Y0    = y_UCI   the untreated potential outcome,  in {-1,+1}"], True),
        (520, 128, 420, 78, "ours", "CATE, our addition", [
            "tau(x) = a * (sigmoid(b'x - c) - 1/2)",
            "Y1 = Y0 + tau(x) + eps,   a ~ U(0.5, 2)"], True),
        (40, 240, 900, 74, "synth", "Confounded assignment  (their Eq. 6)", [
            "logit pi0(x,u) = lam'x + gamma*u,   lam_j ~ U(-0.1, 0.1),   NO clipping",
            "T ~ Bern(1 - pi0)     =>   odds ratio between u=+1 and u=-1  ==  e^{2 gamma}  EXACTLY"], True),
        (40, 348, 430, 60, "out", "Observed", ["D_obs = (x, T, Y),  Y = Y_T"], True),
        (520, 348, 420, 60, "out", "Oracle (evaluation only)",
         ["U, Y0, Y1, tau, pi0, e*"], True),
    ],
    arrows=[(255, 88, 255, 126, ""), (730, 88, 730, 126, ""),
            (255, 206, 255, 238, ""), (730, 206, 730, 238, ""),
            (255, 314, 255, 346, ""), (730, 314, 730, 346, "")],
    W=980, H=420,
    note="Amber is the one place their construction is unusual: the same column is both the "
         "hidden confounder and the untreated outcome, which makes U maximally informative about "
         "Y0. Green is our only addition. The <b>&minus;&frac12;</b> in tau is load-bearing: "
         "without it tau &gt; 0 everywhere, the CATE never changes sign, and the oracle policy is "
         "&quot;treat everyone&quot; &mdash; measured headroom exactly +0.00000.")

RCT_FLOW = flow(
    items=[
        (40, 26, 900, 62, "real", "A dataset that already has BOTH potential outcomes", [
            "IHDP: yf + ycf (NPCI surface)   Twins: the two twins' survival (observed)",
            "IST: T-learner imputation (AUC 0.79/0.80 on the factual arm)"], False),
        (40, 128, 430, 78, "real", "Covariates split into two groups", [
            "X-group -> observed scalar index x in [-1,1]",
            "U-group -> hidden index u;  split searched to maximise |corr(x,u)|"], False),
        (520, 128, 420, 78, "real", "Hidden binary state", [
            "S = sign(u - median u)  in  {-1,+1}",
            "never shown to the learner"], True),
        (40, 240, 900, 74, "synth", "The assignment is REPLACED (the sources are randomised)", [
            "e(x,S) = sigmoid(1.2 x + 0.5 ln(Lambda) S),   NO clipping,   Lambda = 4",
            "T ~ Bern(e);  Y0, Y1 are the dataset's OWN, never resimulated"], True),
        (40, 348, 430, 60, "out", "Observed", ["D_obs = (x, T, Y),  Y = Y_T"], True),
        (520, 348, 420, 60, "out", "Oracle (evaluation only)", ["u, S, e, Y0, Y1"], True),
    ],
    arrows=[(255, 88, 255, 126, ""), (730, 88, 730, 126, ""),
            (255, 206, 255, 238, ""), (730, 206, 730, 238, ""),
            (255, 314, 255, 346, ""), (730, 314, 730, 346, "")],
    W=980, H=420,
    note="These three sources are randomised (or, for Twins, a matched pair), so they contain no "
         "hidden confounding at all. Only the assignment is replaced; both potential outcomes are "
         "the datasets' own. Purple marks the synthetic part.")

# ------------------------------------------------------------------ SEMI tab content
COLS = ["IPW-O-X", "DoublyRobust-O-X", "Hajek-O-X",
        "IPW-O-W", "DoublyRobust-O-W", "Hajek-O-W",
        "IPW-X-X", "DoublyRobust-X-X", "Direct-X-X", "SharpHess", "Kallus"]
CHEAD = ["IPW-O-X", "DR-O-X", "Hajek-O-X", "IPW-O-W", "DR-O-W", "Hajek-O-W",
         "IPW-X-X", "DR-X-X", "Direct-X-X", "Hess", "Kallus"]

def _hd(d):
    h = list(SEMI[d][0].get("per_dgp_headroom", {}).values())
    return (float(np.mean(h)), float(np.std(h))) if h else (float("nan"), float("nan"))


cons = "".join(
    "<tr><td class='l'>%s</td><td>%d</td><td>%d</td><td>%.4f</td>"
    "<td>%.3f <span class='sd'>&plusmn;%.3f</span></td><td>%+.3f</td><td>%.3f</td>"
    "<td class='ref'>%d</td></tr>"
    % (esc(d), SEMI[d][0]["meta"]["n_pool"], SEMI[d][0]["meta"]["d"],
       SEMI[d][0]["meta"]["cv_logloss"], _hd(d)[0], _hd(d)[1],
       SEMI[d][0]["meta"]["corr_x_u"], SEMI[d][0]["meta"]["frac_tau_pos"],
       SEMI[d][0].get("n_dgp_draws", 1)) for d in SEMI)

def reps(s_):
    """(DGP draws, split seeds per draw) behind one summary entry. n_seeds counts result files,
    which is draws x seeds, so the per-draw figure is the quotient."""
    nd = int(s_.get("n_dgp_draws") or 0)
    ns = int(s_.get("n_seeds") or 0)
    return nd, (ns // nd if nd else 0)


def repcell(s_):
    nd, ps = reps(s_)
    return "<td class='reps'>%d &times; %d</td>" % (nd, ps) if nd else "<td class='reps'>&mdash;</td>"


rows = []
for d in SEMI:
    _nd, _ps = reps(SEMI[d][0]) if SEMI[d] else (0, 0)
    rows.append("<tr class='dsrow'><td colspan='%d'>%s <span class='dsn'>&mdash; %d DGP draws "
                "&times; %d split seeds = %d replications per &gamma;</span></td></tr>"
                % (len(COLS) + 4, esc(d), _nd, _ps, _nd * _ps))
    for s in SEMI[d]:
        r, rr = s["rows"], s["rows_rat"]
        vals, sds = {}, {}
        for m in COLS:
            if m in r: vals[m] = r[m]["mean"]; sds[m] = r[m]["sd"]
        rows.append("<tr><td class='l'>&gamma; = %.1f</td><td class='ref'>%.2f</td>%s%s</tr>"
                    % (s["gamma"], s["Gamma"], repcell(s), cells(vals, COLS, sds)))
semi_tbl = ("<div class='scrollx'><table class='dt'><tr><th class='l'>&nbsp;</th><th>&Gamma;</th>"
            "<th>draws &times; seeds</th>"
            + "".join("<th>%s</th>" % esc(c) for c in CHEAD) + "</tr>" + "".join(rows)
            + "</table></div>")

POOL, PHEAD = COLS, CHEAD
pool = []
for g in GAMMA_ALL:
    agg = {}
    for c in POOL:
        vs = []
        for d in SEMI:
            for s in SEMI[d]:
                if abs(s["gamma"] - g) > 1e-12: continue
                base = c.replace("+C4", "")
                src = s["rows_rat"] if c.endswith("+C4") else s["rows"]
                if base in src: vs.append(norm(src[base]["mean"], s["refs"]))
        if vs: agg[c] = float(np.mean(vs))
    _nd = sum(reps(s_)[0] for d in SEMI for s_ in SEMI[d] if abs(s_["gamma"] - g) < 1e-12)
    _ps = max([reps(s_)[1] for d in SEMI for s_ in SEMI[d] if abs(s_["gamma"] - g) < 1e-12] or [0])
    pool.append("<tr><td class='l'>&gamma; = %.1f</td><td class='ref'>%.2f</td>"
                "<td class='reps'>%d &times; %d</td>%s</tr>"
                % (g, float(np.exp(2 * g)), _nd, _ps, cells(agg, POOL)))
pool_tbl = ("<div class='scrollx'><table class='dt'><tr><th class='l'>&nbsp;</th><th>&Gamma;</th>"
            "<th>draws &times; seeds</th>"
            + "".join("<th>%s</th>" % esc(c) for c in PHEAD) + "</tr>" + "".join(pool)
            + "</table></div>")

tm = []
for d in SEMI:
    for s_ in SEMI[d]:
        m_ = s_["transport_margin"]
        def gg(k, m_=m_):
            if k not in m_: return "<td>&mdash;</td>"
            v = m_[k]
            return "<td style='color:%s'>%+.4f</td>" % ("#0a7d33" if v > 0 else "#c0392b", v)
        tm.append("<tr><td class='l'>%s</td><td>&gamma; = %.1f</td><td class='ref'>%.2f</td>%s%s</tr>"
                  % (esc(d), s_["gamma"], s_["Gamma"],
                     gg("IPW-O-W|rat0"), gg("DoublyRobust-O-W|rat0")))
tm_tbl = ("<div class='scrollx'><table class='dt'>"
          "<tr><th class='l'>dataset</th><th>&nbsp;</th><th>&Gamma;</th>"
          "<th>IPW-O-W &minus; IPW-O-X</th><th>DR-O-W &minus; DR-O-X</th></tr>"
          + "".join(tm) + "</table></div>")

SPREAD = []
for d in SEMI:
    row = ["<tr><td class='l'>%s</td>" % esc(d)]
    for g in GAM:
        s_ = [x for x in SEMI[d] if abs(x["gamma"] - g) < 1e-12]
        v = ([q["IPW-O-W"]["mean"] for q in s_[0].get("per_dgp", {}).values()
              if "IPW-O-W" in q] if s_ else [])
        row.append("<td>%.2f <span class='sd'>&plusmn;%.2f</span></td>" % (np.mean(v), np.std(v))
                   if v else "<td>&mdash;</td>")
    SPREAD.append("".join(row) + "</tr>")
spread_tbl = ("<div class='scrollx'><table class='dt'><tr><th class='l'>dataset</th>"
              + "".join("<th>&gamma; = %g</th>" % g for g in GAM) + "</tr>"
              + "".join(SPREAD) + "</table></div>")


# ------------------------------------------------------------------ grouped bar charts
# One group per method, one bar per gamma, whisker = +-1 sd ACROSS THE 5 DGP DRAWS (the spread
# that actually dominates here). Values are normalised: 0 = best constant policy, 1 = oracle.
BAR_METHODS = ["IPW-O-W", "DoublyRobust-O-W", "Hajek-O-W",
               "IPW-O-X", "DoublyRobust-O-X", "Hajek-O-X",
               "IPW-X-X", "DoublyRobust-X-X", "Direct-X-X", "SharpHess", "Kallus"]
BAR_SHORT = {"DoublyRobust-O-W": "DR-O-W", "DoublyRobust-O-X": "DR-O-X",
             "DoublyRobust-X-X": "DR-X-X", "SharpHess": "Hess", "Hajek-O-W": "Haj-O-W",
             "Hajek-O-X": "Haj-O-X", "Direct-X-X": "Dir-X-X"}


def barchart(stats, title, methods=None, series=None, colors=None, labels=None,
             note="whisker = &plusmn;1 sd across the 5 DGP draws", W=1240, H=430):
    """stats[(method, gamma)] = (mean, sd). Grouped bars with sd whiskers.

    Geometry is derived from len(GAM) rather than assumed: with the gamma grid extended to five
    values the old fixed bar width (0.24 of a slot) spanned 1.2 slots and neighbouring method
    groups ran into each other. Bars now fill a fixed FRACTION of each slot and are centred on it,
    so adding or removing a gamma re-flows cleanly.
    """
    methods = methods or BAR_METHODS
    series = series or GAM
    colors = colors or GCOL
    labels = labels or (lambda k: "&gamma; = %g" % k)
    pL, pR, pT, pB = 58, 16, 34, 74
    vals = ([m for (m, sd) in stats.values()] + [m + sd for (m, sd) in stats.values()]
            + [m - sd for (m, sd) in stats.values()])
    lo, hi = min(vals + [0.0]), max(vals + [0.0])
    pad = 0.10 * (hi - lo + 1e-9); lo -= pad; hi += pad
    Y = lambda v: H - pB - (v - lo) / (hi - lo + 1e-12) * (H - pT - pB)
    n, ng = len(methods), len(series)
    slot = (W - pL - pR) / n
    fill = 0.76                                    # of the slot occupied by the bar group
    bw = slot * fill / ng
    gw = bw * ng
    p = ['<svg viewBox="0 0 %d %d" class="chart" preserveAspectRatio="xMinYMin meet">' % (W, H),
         '<text x="%d" y="18" class="ct">%s</text>' % (pL, esc(title))]
    # alternating band per method group -- the main thing that stops 55 bars reading as one mass
    for i in range(n):
        if i % 2:
            p.append('<rect x="%.1f" y="%d" width="%.1f" height="%d" class="band"/>'
                     % (pL + slot * i, pT, slot, H - pT - pB))
    for t in (-0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0):
        if lo <= t <= hi:
            p.append('<line x1="%d" y1="%.1f" x2="%d" y2="%.1f" class="grid"/>'
                     % (pL, Y(t), W - pR, Y(t)))
            p.append('<text x="%d" y="%.1f" class="tk" text-anchor="end">%.1f</text>'
                     % (pL - 7, Y(t) + 3.5, t))
    y0 = Y(0.0)
    for i, m in enumerate(methods):
        cx = pL + slot * (i + 0.5)
        for j, g in enumerate(series):
            if (m, g) not in stats: continue
            mu, sd = stats[(m, g)]
            x = cx - gw / 2 + j * bw
            yt, hgt = Y(max(mu, 0.0)), abs(y0 - Y(mu))
            p.append('<rect x="%.1f" y="%.1f" width="%.1f" height="%.1f" fill="%s" rx="1"/>'
                     % (x + 0.6, yt, max(bw - 1.2, 1.0), max(hgt, 0.8), colors[g]))
            if sd > 0:
                xm, cap = x + bw / 2, min(bw * 0.30, 3.2)
                p.append('<line x1="%.1f" y1="%.1f" x2="%.1f" y2="%.1f" class="whisk"/>'
                         % (xm, Y(mu - sd), xm, Y(mu + sd)))
                for yy in (Y(mu - sd), Y(mu + sd)):
                    p.append('<line x1="%.1f" y1="%.1f" x2="%.1f" y2="%.1f" class="whisk"/>'
                             % (xm - cap, yy, xm + cap, yy))
        p.append('<text x="%.1f" y="%d" class="tk" text-anchor="end" '
                 'transform="rotate(-35 %.1f %d)">%s</text>'
                 % (cx, H - pB + 17, cx, H - pB + 17, esc(BAR_SHORT.get(m, SHORT.get(m, m)))))
    p.append('<line x1="%d" y1="%.1f" x2="%d" y2="%.1f" class="ax"/>' % (pL, y0, W - pR, y0))
    p.append('<line x1="%d" y1="%d" x2="%d" y2="%d" class="ax"/>' % (pL, pT, pL, H - pB))
    ymid = (pT + H - pB) // 2
    p.append('<text x="15" y="%d" class="al" text-anchor="middle" transform="rotate(-90 15 %d)">'
             'normalised value</text>' % (ymid, ymid))
    p.append('</svg>')
    leg = "".join('<span class="li"><span class="sw" style="background:%s"></span>%s</span>'
                  % (colors[g], labels(g)) for g in series)
    return ('<figure class="fig"><div class="scrollx chartbox">%s</div><div class="leg">%s'
            '<span class="li">%s</span></div></figure>' % ("".join(p), leg, note))


def _stats_for(dsets):
    """Bar height is EXACTLY the number in the tables -- pooled mean normalised by pooled
    references, averaged over the datasets in `dsets`. The whisker is the sd of the same
    quantity computed per DGP draw (draw-level normalisation), which is the honest spread but
    a different estimand: equal-weighting draws de-emphasises the high-headroom draws that
    dominate the pooled figure, so its centre sits lower. Height and whisker are therefore
    reported from the matching table value and the draw-level dispersion respectively."""
    st = {}
    for m in BAR_METHODS:
        for g in GAM:
            mu, sd = [], []
            for d in dsets:
                for x in SEMI[d]:
                    if abs(x["gamma"] - g) > 1e-12: continue
                    if m in x["rows"]: mu.append(norm(x["rows"][m]["mean"], x["refs"]))
                    sd += [byM[m]["mean"] for byM in x["per_dgp"].values() if m in byM]
            if mu: st[(m, g)] = (float(np.mean(mu)), float(np.std(sd)) if len(sd) > 1 else 0.0)
    return st



# --------------------------------------------- SharpHess under the two nuisance estimators
HROWS = []
for g in GAMMA_ALL:
    cells_ = []
    for key in ("SharpHess", "SharpHess-kNN"):
        vs = [norm(x["rows"][key]["mean"], x["refs"]) for d in SEMI for x in SEMI[d]
              if abs(x["gamma"] - g) < 1e-12 and key in x["rows"]]
        cells_.append("<td>%.3f</td>" % np.mean(vs) if vs else "<td>&mdash;</td>")
    HROWS.append("<tr><td class='l'>&gamma; = %g</td><td class='ref'>%.2f</td>%s</tr>"
                 % (g, float(np.exp(2 * g)), "".join(cells_)))
hess_tbl = ("<div class='scrollx'><table class='dt'><tr><th class='l'>&nbsp;</th><th>&Gamma;</th>"
            "<th>neural {64,64,32} (paper)</th><th>k-NN k=15</th></tr>"
            + "".join(HROWS) + "</table></div>")

RCOL = {"ihdp": "#1e40af", "twins": "#0a7d33", "ist": "#c0392b"}


def rct_chart():
    """Same grammar as the semi charts, with the three benchmarks in place of the gamma grid --
    the RCT tab has a single matched Gamma per benchmark, so there is no gamma axis to group on.
    The whisker is the across-seed sd, put on the normalised scale by the same divisor."""
    st = {}
    for s_ in RCT:
        R = s_["refs"]; bc = max(R["never_treat"], R["all_treat"]); sc = R["oracle"] - bc
        if sc <= 0: continue
        for m in RCT_ORDER:
            cell = s_["rows"].get(m, {}).get("matched_best_L")
            if not cell: continue
            st[(m, s_["dataset"])] = ((cell["mean"] - bc) / sc, abs(cell.get("sd", 0.0)) / abs(sc))
    order = [d for d in ("ihdp", "twins", "ist") if any(k[1] == d for k in st)]
    if not st: return ""
    _ns = max([int(s_.get("n_seeds") or 0) for s_ in RCT] or [0])
    # NB: chart titles go through esc() into SVG text, so an HTML entity here would be
    # double-escaped and render as "&Gamma;" literally. Plain ASCII only.
    return barchart(st, "RCT benchmarks -- normalised value at the matched Gamma, by method  "
                    "(%d seeds per benchmark)" % _ns,
                    methods=RCT_ORDER, series=order, colors=RCOL,
                    labels=lambda k: esc(RCT_TITLE.get(k, k)),
                    note="whisker = &plusmn;1 sd across seeds")


RCT_CHART = rct_chart()

def _cap(dsets):
    """Chart subtitle: how many DGP draws and split seeds stand behind these bars."""
    nd = sum(reps(s_)[0] for d in dsets for s_ in SEMI[d] if abs(s_["gamma"] - GAM[0]) < 1e-12)
    ps = max([reps(s_)[1] for d in dsets for s_ in SEMI[d]] or [0])
    return "%d DGP draws x %d split seeds = %d reps per bar" % (nd, ps, nd * ps)


CHARTS = barchart(_stats_for(list(SEMI)),
                  "ALL FIVE DATASETS POOLED -- normalised value by method and gamma  (%s)"
                  % _cap(list(SEMI)))
CHARTS += "".join(barchart(_stats_for([d]),
                           "%s -- normalised value by method and gamma  (%s)" % (d, _cap([d])))
                  for d in SEMI)

from collections import Counter
wins = Counter()
for d in SEMI:
    for s in SEMI[d]:
        if s["gamma"] == 0: continue
        # COLS only: SharpHess-kNN is the same method under a different nuisance estimator,
        # not a rival, and counting both arms would double-count Hess in this ledger.
        v = {m: s["rows"][m]["mean"] for m in s["rows"] if m in COLS}
        wins[max(v, key=v.get)] += 1
win_tbl = ("<div class='scrollx'><table class='dt'><tr><th class='l'>method</th>"
           "<th>cells won (of %d)</th></tr>" % sum(wins.values())
           + "".join("<tr><td class='l'>%s</td><td>%d</td></tr>" % (esc(k), c)
                     for k, c in wins.most_common()) + "</table></div>")

# ------------------------------------------------------------------ RCT tab content
rct_cons = "".join(
    "<tr><td class='l'>%s</td><td>%d</td><td>%s</td><td>%+.2f</td><td>%.3f</td>"
    "<td class='ref'>%.4f</td><td class='ref'>%.4f</td><td>%.4f</td></tr>"
    % (esc(RCT_TITLE[s["dataset"]]), s["meta"]["n_total"], s["meta"]["outcome_kind"],
       s["meta"]["corr_x_S"], s["meta"]["LAM_realised"],
       max(s["refs"]["never_treat"], s["refs"]["all_treat"]), s["refs"]["oracle"],
       s["refs"]["oracle"] - max(s["refs"]["never_treat"], s["refs"]["all_treat"]))
    for s in RCT)

rct_rows = []
for s in RCT:
    vals = {m: s["rows"][m]["matched_best_L"]["mean"] for m in RCT_ORDER
            if m in s["rows"] and s["rows"][m].get("matched_best_L")}
    sds = {m: s["rows"][m]["matched_best_L"]["sd"] for m in RCT_ORDER
           if m in s["rows"] and s["rows"][m].get("matched_best_L")}
    R = s["refs"]
    rct_rows.append("<tr><td class='l'>%s</td>%s<td class='ref'>%.4f</td>"
                    "<td class='ref'>%.4f</td></tr>"
                    % (esc(RCT_TITLE[s["dataset"]]), cells(vals, RCT_ORDER, sds),
                       max(R["never_treat"], R["all_treat"]), R["oracle"]))
agg = {}
for m in RCT_ORDER:
    vs = [norm(s["rows"][m]["matched_best_L"]["mean"], s["refs"]) for s in RCT
          if m in s["rows"] and s["rows"][m].get("matched_best_L")]
    if vs: agg[m] = float(np.mean(vs))
allt = float(np.mean([norm(s["refs"]["all_treat"], s["refs"]) for s in RCT]))
rct_rows.append("<tr class='dsrow'><td>mean normalized</td>%s<td class='ref'>%.3f</td>"
                "<td class='ref'>1.000</td></tr>" % (cells(agg, RCT_ORDER), allt))
rct_tbl = ("<div class='scrollx'><table class='dt'><tr><th class='l'>benchmark</th>"
           + "".join("<th>%s</th>" % esc(SHORT.get(m, m)) for m in RCT_ORDER)
           + "<th>best const</th><th>oracle</th></tr>" + "".join(rct_rows) + "</table></div>")

EQCSS = (".eq{text-align:center;overflow-x:auto;margin:18px 0;font-size:1.3em}.eq math{font-size:1.06em}.constr p{font-size:1.06rem;line-height:1.65;margin:.9rem 0}.constr p.muted{font-size:1.0rem}@media (max-width:720px){.eq{font-size:1.1em}.constr p{font-size:1rem}}")
REPSCSS = ("table.dt td.reps{font-variant-numeric:tabular-nums;opacity:.75;font-size:11.5px}.dsn{font-weight:400;opacity:.72;font-size:11.5px}")
BOXCSS = (".box{border:1px solid rgba(128,128,128,.35);border-radius:6px;padding:.85rem 1rem;margin:1rem 0;font-size:.9rem;line-height:1.55}.box>b{display:block;margin-bottom:.4rem;font-size:.95rem}.box ol{margin:.5rem 0 .5rem 1.1rem;padding:0}.box li{margin:.3rem 0}.box p{margin:.5rem 0}.box p:last-child{margin-bottom:0}.mono{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:.92em}")
CHARTCSS = (".chartbox{overflow-x:auto}.chartbox .chart{min-width:1000px;width:100%;height:auto}.band{fill:currentColor;opacity:.035}.whisk{stroke:currentColor;stroke-width:1.1;opacity:.75}")
STAMPCSS = (".stamp{font-size:.78rem;opacity:.65;margin:.2rem 0 1rem;font-variant-numeric:tabular-nums}")
NOTECSS = (".note{border-left:3px solid #b07105;background:rgba(176,113,5,.07);"
           "padding:.7rem .9rem;margin:1rem 0;font-size:.88rem;line-height:1.5;"
           "border-radius:0 4px 4px 0}"
           "@media (prefers-color-scheme:dark){.note{background:rgba(176,113,5,.14)}}"
           ":root[data-theme=\"dark\"] .note{background:rgba(176,113,5,.14)}"
           ":root[data-theme=\"light\"] .note{background:rgba(176,113,5,.07)}")
import datetime as _dt
_ncells = sum(x["n_seeds"] for d in SEMI for x in SEMI[d])
_gammas = sorted({x["gamma"] for d in SEMI for x in SEMI[d]})
STAMP = ("<div class='stamp'>build %s &middot; %d cells &middot; &gamma; &isin; {%s} &middot; "
         "&Gamma; up to %.0f</div>"
         % (_dt.date.today().isoformat(), _ncells,
            ", ".join("%g" % g for g in _gammas), float(np.exp(2 * max(_gammas)))))

# ------------------------------------------------------------------ KMZ tab
# The Kallus-Mao-Zhou (2019) benchmark, read entirely from assets/exp_msmbench + assets/grand at
# build time so no number here can drift from the campaign files. Baselines are the authors' own
# code for both Hess and Kallus (the *_paper ports).
_MSM = ROOT / "assets" / "exp_msmbench"
_GRD = ROOT / "assets" / "grand"


def _J(path):
    try:
        return json.loads(Path(path).read_text())
    except Exception:
        return None


KM = _J(_MSM / "kmz_main_ce1.0.json")
KM_CE = {ce: _J(_MSM / ("kmz_main_ce%s.json" % ce)) for ce in ("1.0", "1.5", "2.0")}
KM_G = {"05": _J(_MSM / "kmz_g05.json"), "10": _J(_MSM / "kmz_g10.json")}
KM_CAP = _J(_MSM / "kmz_cap30_ce1.0.json")
KM_REFS = _J(_MSM / "kmz_refs.json")
KM_XX = _J(_GRD / "xxL_kmz.json")
KM_HESS = (_J(_GRD / "hess_paper_for_reports.json") or {}).get("km")
KM_H5K = _J(_GRD / "hess_paper_km5000.json")
KM_KAL = _J(_MSM / "kmz_kallus.json")
KM_KALB = {b: _J(_MSM / ("kmz_kallus_base_%s.json" % b)) for b in ("all", "nominal")}
KM_H15 = _J(_GRD / "hess_paper_km_g15.json")
KM_K15 = _J(_MSM / "kmz_kallus_g15.json")
KM_H50 = _J(_GRD / "hess_paper_km_g50.json")
KM_K50 = _J(_MSM / "kmz_kallus_g50.json")
KMZ_SWEEP_G = ["1", "2", "3", "4.4817", "6", "8", "15", "50"]   # the full computed grid
KMGK = "4.4817"

KMZ_MATH = "".join([
    M(r"Y(a) = (2a{-}1)X + (2a{-}1) - 2\sin(2(2a{-}1)X) - 2S(1+0.5X) + \mathcal{N}(0,1)"),
    M(r"X \sim \mathrm{Unif}[-2,2], \qquad S \in \{\pm 1\}, \quad P(S{=}{+}1)=\tfrac12, "
      r"\qquad S \perp X"),
    M(r"e(x) = \sigma(0.75X + 0.5) \quad\text{(nominal)}, \qquad "
      r"e(x,S) = \frac{1{+}S}{2\,\rho(x, 1/\Gamma^{\!*})} + "
      r"\frac{1{-}S}{2\,\rho(x, \Gamma^{\!*})}, \qquad "
      r"\rho(x,\gamma) = 1 + \Big(\frac{1}{e(x)} - 1\Big)\gamma"),
    M(r"\Rightarrow\ \text{the MSM holds exactly with odds ratio } \Gamma^{\!*} "
      r"\text{ at every } x; \quad \log\Gamma^{\!*} \in \{0.5, 1.0, 1.5\} "
      r"\ (\Gamma^{\!*} \approx 1.65 / 2.72 / 4.48)"),
    M(r"\mathrm{CATE}(X) = 2X + 2 - 4\sin(2X), \qquad x = X/2 \in [-1,1] "
      r"\ \text{(pipeline units)}"),
])


def _kmz_tab():
    if not KM:
        return None, "<p>KMZ campaign files not found.</p>", "", "", "", "", "", ""
    R = {"oracle": KM["oracle"], "never": KM["never_treat"], "all": KM["all_treat"],
         "naive": KM["naive_dr"]}
    bc = max(R["never"], R["all"]); hr = R["oracle"] - bc

    def nz(v):
        return (v - bc) / hr

    Lgrid = KM["Lgrid"]
    surf = KM["surface"]
    LBL2 = {"IPW-O-X": "IPW-O-X", "DoublyRobust-O-X": "DR-O-X", "Hajek-O-X": "Hajek-O-X",
            "IPW-O-W": "IPW-O-W", "DoublyRobust-O-W": "DR-O-W"}

    rows = []
    stats = {}
    for m in ["IPW-O-W", "DoublyRobust-O-W", "IPW-O-X", "DoublyRobust-O-X", "Hajek-O-X"]:
        best = max((surf[m][KMGK][L], L) for L in Lgrid)
        stats[(LBL2[m], "kmz")] = (nz(best[0]), 0.0)
        det = "  ".join("L=%s %+.3f" % (L, surf[m][KMGK][L]) for L in Lgrid)
        rows.append((best[0], "<tr%s><td class='l'>%s</td><td>%.3f</td><td>%+.3f</td>"
                     "<td>L=%s</td><td class='l' style='font-size:10.5px'>%s</td></tr>"
                     % (" class='hl'" if m.endswith("O-W") else "", LBL2[m], best[0],
                        nz(best[0]), best[1], det)))
    if KM_XX:
        for m, d in KM_XX["mean"].items():
            bl = max(d, key=d.get)
            lbl = m.replace("DoublyRobust", "DR")
            stats[(lbl, "kmz")] = (nz(d[bl]), abs(nz(d[bl] + KM_XX["sd"][m][bl]) - nz(d[bl])))
            det = "  ".join("L=%s %+.3f" % (L, d[L]) for L in ("inf", "3", "1"))
            rows.append((d[bl], "<tr><td class='l'>%s <span class='hsub'>(&Gamma;-free)</span>"
                         "</td><td>%.3f</td><td>%+.3f</td><td>L=%s</td>"
                         "<td class='l' style='font-size:10.5px'>%s</td></tr>"
                         % (lbl, d[bl], nz(d[bl]), bl, det)))
    if KM_HESS:
        i = KM_HESS["gammas"].index(KMGK)
        v = KM_HESS["mean"][i]
        stats[("Hess (paper)", "kmz")] = (nz(v), abs(nz(v + KM_HESS["sd"][i]) - nz(v)))
        n5 = ("; <b>%.3f &plusmn; %.3f at their own n = 5000</b>"
              % (KM_H5K[KMGK]["mean"], KM_H5K[KMGK]["sd"])) if KM_H5K and KMGK in KM_H5K else ""
        rows.append((v, "<tr><td class='l'>Hess (paper)</td><td>%.3f &plusmn; %.3f</td>"
                     "<td>%+.3f</td><td>&mdash;</td><td class='l' style='font-size:10.5px'>"
                     "authors' code at n=400%s</td></tr>" % (v, KM_HESS["sd"][i], nz(v), n5)))
    rows.append((R["naive"], "<tr><td class='l'>naive (DR plug-in)</td><td>%.3f</td>"
                 "<td>%+.3f</td><td>&mdash;</td><td class='l'>under-treats badly</td></tr>"
                 % (R["naive"], nz(R["naive"]))))
    if KM_KAL:
        i = KM_KAL["gammas"].index(4.4817)
        v = KM_KAL["regimes"]["uncap"]["mean"]["Kallus"][i]
        stats[("Kallus (paper)", "kmz")] = (nz(v), 0.0)
        rows.append((v, "<tr><td class='l'>Kallus (paper, ref = control)</td><td>%.3f</td>"
                     "<td>%+.3f</td><td>&mdash;</td><td class='l' style='font-size:10.5px'>"
                     "authors' code; = never-treat exactly (see the reference-policy note)"
                     "</td></tr>" % (v, nz(v))))
    rows.sort(key=lambda t: -t[0])
    main_tbl = ("<div class='scrollx'><table class='dt'><tr><th class='l'>method</th>"
                "<th>E[Y]</th><th>normalised</th><th>best L</th><th class='l'>per-L detail "
                "(mean over 5 seeds)</th></tr>" + "".join(r[1] for r in rows) + "</table></div>")

    tm = "".join("<td>%+.3f</td>" % (surf["IPW-O-W"][KMGK][L] - surf["IPW-O-X"][KMGK][L])
                 for L in Lgrid)
    tm_tbl = ("<div class='scrollx'><table class='dt'><tr><th class='l'>&nbsp;</th>"
              + "".join("<th>L=%s</th>" % L for L in Lgrid)
              + "</tr><tr><td class='l'>IPW-O-W &minus; IPW-O-X</td>" + tm + "</tr></table></div>")

    st = []
    for tag, lab in (("05", "e<sup>0.5</sup> = 1.65"), ("10", "e<sup>1.0</sup> = 2.72")):
        Rg = KM_G.get(tag)
        if not Rg:
            continue
        g0 = Rg["gammas"][0]
        r0 = {m: Rg["surface"][m][g0]["3"] for m in Rg["methods"]}
        st.append("<tr><td class='l'>&Gamma;* = %s</td><td>%.3f</td><td>%.3f</td><td>%.3f</td>"
                  "<td>%.3f</td><td>%.3f</td></tr>"
                  % (lab, r0["IPW-O-W"], r0["DoublyRobust-O-W"], r0["IPW-O-X"],
                     r0["DoublyRobust-O-X"], r0["Hajek-O-X"]))
    r15 = {m: surf[m][KMGK]["3"] for m in KM["methods"]}
    st.append("<tr><td class='l'>&Gamma;* = e<sup>1.5</sup> = 4.48 (main)</td><td>%.3f</td>"
              "<td>%.3f</td><td>%.3f</td><td>%.3f</td><td>%.3f</td></tr>"
              % (r15["IPW-O-W"], r15["DoublyRobust-O-W"], r15["IPW-O-X"],
                 r15["DoublyRobust-O-X"], r15["Hajek-O-X"]))
    st_tbl = ("<div class='scrollx'><table class='dt'><tr><th class='l'>strength (matched "
              "&Gamma;, L=3)</th><th>IPW-O-W</th><th>DR-O-W</th><th>IPW-O-X</th><th>DR-O-X</th>"
              "<th>Hajek-O-X</th></tr>" + "".join(st) + "</table></div>")

    ce_rows = "".join("<tr><td class='l'>c<sub>&epsilon;</sub> = %s</td><td>%.3f</td></tr>"
                      % (ce, max(KM_CE[ce]["surface"]["IPW-O-W"][KMGK][L]
                                 for L in KM_CE[ce]["Lgrid"]))
                      for ce in ("1.0", "1.5", "2.0") if KM_CE.get(ce))
    ce_tbl = ("<div class='scrollx'><table class='dt'><tr><th class='l'>transport budget</th>"
              "<th>IPW-O-W (best L)</th></tr>" + ce_rows + "</table></div>")

    cap_tbl = ""
    if KM_CAP and KM_REFS:
        cr = []
        for m in KM_CAP["methods"]:
            b = max((KM_CAP["surface"][m][KMGK][L], L) for L in KM_CAP["Lgrid"])
            cr.append((b[0], "<tr%s><td class='l'>%s</td><td>%.3f</td><td>L=%s</td></tr>"
                       % (" class='hl'" if m.endswith("O-W") else "", LBL2[m], b[0], b[1])))
        cr.sort(key=lambda t: -t[0])
        cap_tbl = ("<div class='scrollx'><table class='dt'><tr><th class='l'>method</th>"
                   "<th>E[Y], capped 30%</th><th>best L</th></tr>"
                   + "".join(r[1] for r in cr)
                   + "</table></div><p class='muted'>Capped oracle "
                   + "%.3f" % KM_REFS["oracle_cap30"]
                   + "; the budget compresses the ordering but O-W stays on top.</p>")

    kb_tbl = ""
    if any(KM_KALB.values()):
        kb = []
        if KM_KAL:
            i = KM_KAL["gammas"].index(4.4817)
            kb.append("<tr><td class='l'>control (their driver)</td><td>%.3f</td>"
                      "<td class='l'>= never-treat exactly</td></tr>"
                      % KM_KAL["regimes"]["uncap"]["mean"]["Kallus"][i])
        for b, lab, note in (("all", "treat-all (their tmnt_p_1)", "= all-treat exactly"),
                             ("nominal", "logging policy e(x)", "restart-0 init caveat")):
            d = KM_KALB.get(b)
            if d:
                i = [j for j, g in enumerate(d["gammas"]) if abs(float(g) - 4.4817) < .01][0]
                kb.append("<tr><td class='l'>%s</td><td>%.3f</td><td class='l'>%s</td></tr>"
                          % (lab, d["mean"][i], note))
        kb_tbl = ("<div class='scrollx'><table class='dt'><tr><th class='l'>Kallus reference "
                  "policy</th><th>value at matched &Gamma;</th><th class='l'>&nbsp;</th></tr>"
                  + "".join(kb) + "</table></div>")

    # ---- Gamma sweep: the requested grid, best L per (method, Gamma) ----
    gks = [g for g in KMZ_SWEEP_G if g in surf["IPW-O-W"]]
    sw_stats = {}
    sw_rows = []
    for m in ["IPW-O-W", "DoublyRobust-O-W", "IPW-O-X", "DoublyRobust-O-X", "Hajek-O-X"]:
        cells_ = []
        for g in gks:
            v = max(surf[m][g].values())
            sw_stats[(LBL2[m], g)] = (nz(v), 0.0)
            cells_.append("<td>%.3f</td>" % v)
        sw_rows.append((max(surf[m][gks[0]].values()),
                        "<tr%s><td class='l'>%s</td>%s</tr>"
                        % (" class='hl'" if m.endswith("O-W") else "", LBL2[m], "".join(cells_))))
    if KM_HESS:
        hx = {g: v for g, v in zip(KM_HESS["gammas"], KM_HESS["mean"])}
        if KM_H15:
            hx["15"] = KM_H15["mean"]
        if KM_H50:
            hx["50"] = KM_H50["mean"]
        cells_ = []
        for g in gks:
            if g in hx:
                sw_stats[("Hess (paper)", g)] = (nz(hx[g]), 0.0)
                cells_.append("<td>%.3f</td>" % hx[g])
            else:
                cells_.append("<td>&mdash;</td>")
        sw_rows.append((hx.get(gks[0], -9),
                        "<tr><td class='l'>Hess (paper)</td>%s</tr>" % "".join(cells_)))
    if KM_KAL:
        kx = {("%g" % g): v for g, v in zip(KM_KAL["gammas"],
                                            KM_KAL["regimes"]["uncap"]["mean"]["Kallus"])}
        if KM_K15:
            kx["15"] = KM_K15["regimes"]["uncap"]["mean"]["Kallus"][0]
        if KM_K50:
            kx["50"] = KM_K50["regimes"]["uncap"]["mean"]["Kallus"][0]
        cells_ = []
        for g in gks:
            if g in kx:
                sw_stats[("Kallus (paper)", g)] = (nz(kx[g]), 0.0)
                cells_.append("<td>%.3f</td>" % kx[g])
            else:
                cells_.append("<td>&mdash;</td>")
        sw_rows.append((kx.get(gks[0], -9),
                        "<tr><td class='l'>Kallus (paper)</td>%s</tr>" % "".join(cells_)))
    sw_rows.sort(key=lambda t: -t[0])
    hdrs = "".join("<th>&Gamma; = %s%s</th>"
                   % (g, " (matched)" if g == KMGK else "") for g in gks)
    sweep_tbl = ("<div class='scrollx'><table class='dt'><tr><th class='l'>method "
                 "(best L per cell)</th>" + hdrs + "</tr>"
                 + "".join(r[1] for r in sw_rows) + "</table></div>")
    GC15 = {"1": "#0a7d33", "2": "#5b8c1a", "4.4817": "#b07105", "8": "#c0392b",
            "15": "#7d1f6a", "50": "#334155"}
    sw_methods = [m for m in ["IPW-O-W", "DR-O-W", "IPW-O-X", "DR-O-X", "Hajek-O-X",
                              "Hess (paper)", "Kallus (paper)"]
                  if any((m, g) in sw_stats for g in gks)]
    sweep_chart = barchart(sw_stats, "KMZ across Gamma -- normalised value, best L per cell",
                           methods=sw_methods, series=gks,
                           colors={g: GC15.get(g, "#64748b") for g in gks},
                           labels=lambda g: ("Gamma = %s%s"
                                             % (g, " (matched)" if g == KMGK else "")),
                           note="0 = best constant policy, 1 = oracle; only the "
                                "Gamma-dependent methods (O-W, O-X, Hess, Kallus) are shown")

    # ---- per-Gamma LINE chart, every method ----
    def _lines():
        W, H = 1080, 480
        pL, pR, pT, pB = 58, 120, 34, 56
        xs = list(range(len(gks)))
        X_ = lambda i: pL + i * (W - pL - pR) / (len(gks) - 1)
        ylo, yhi = -1.12, 1.55
        Y_ = lambda v: pT + (yhi - v) / (yhi - ylo) * (H - pT - pB)
        series = []
        DSH = {"IPW-O-W": ("#0a7d33", ""), "DR-O-W": ("#0a7d33", "7 4"),
               "IPW-O-X": ("#b07105", ""), "DR-O-X": ("#b07105", "7 4"),
               "Hajek-O-X": ("#b07105", "2 3"),
               "Hess (paper)": ("#7d1f6a", ""), "Kallus (paper)": ("#8c564b", "")}
        for m in ["IPW-O-W", "DoublyRobust-O-W", "IPW-O-X", "DoublyRobust-O-X", "Hajek-O-X"]:
            series.append((LBL2[m], [max(surf[m][g].values()) for g in gks], True))
        hx2 = {g: v for g, v in zip(KM_HESS["gammas"], KM_HESS["mean"])} if KM_HESS else {}
        if KM_H15: hx2["15"] = KM_H15["mean"]
        if KM_H50: hx2["50"] = KM_H50["mean"]
        if hx2 and all(g in hx2 for g in gks):
            series.append(("Hess (paper)", [hx2[g] for g in gks], True))
        kx2 = ({("%g" % g): v for g, v in zip(KM_KAL["gammas"],
                KM_KAL["regimes"]["uncap"]["mean"]["Kallus"])} if KM_KAL else {})
        if KM_K15: kx2["15"] = KM_K15["regimes"]["uncap"]["mean"]["Kallus"][0]
        if KM_K50: kx2["50"] = KM_K50["regimes"]["uncap"]["mean"]["Kallus"][0]
        if kx2 and all(g in kx2 for g in gks):
            series.append(("Kallus (paper)", [kx2[g] for g in gks], True))

        pp = ['<svg viewBox="0 0 %d %d" class="chart">' % (W, H),
              '<text x="%d" y="18" class="ct">KMZ: O-W, O-X, Hess, Kallus across the solver '
              'Gamma (best L per point; n=400, 5 seeds)</text>' % pL]
        for lab, col, yv, dsh in (("oracle", "#111", R["oracle"], "5 4"),
                                  ("all-treat", "#555", R["all"], "3 3"),
                                  ("never-treat", "#555", R["never"], "3 3")):
            pp.append('<line x1="%d" y1="%.1f" x2="%d" y2="%.1f" stroke="%s" stroke-width="1.2" '
                      'stroke-dasharray="%s" opacity="0.65"/>' % (pL, Y_(yv), W - pR, Y_(yv), col, dsh))
            pp.append('<text x="%d" y="%.1f" class="tk" text-anchor="start">%s %.2f</text>'
                      % (W - pR + 6, Y_(yv) + 3.5, lab, yv))
        for t in (-1.0, -0.5, 0.0, 0.5, 1.0):
            pp.append('<line x1="%d" y1="%.1f" x2="%d" y2="%.1f" class="grid"/>' % (pL, Y_(t), W - pR, Y_(t)))
            pp.append('<text x="%d" y="%.1f" class="tk" text-anchor="end">%g</text>' % (pL - 6, Y_(t) + 3.5, t))
        gm = gks.index(KMGK)
        pp.append('<line x1="%.1f" y1="%d" x2="%.1f" y2="%d" stroke="#0a7d33" stroke-width="1.2" '
                  'stroke-dasharray="3 3" opacity="0.7"/>' % (X_(gm), pT, X_(gm), H - pB))
        for i, g in enumerate(gks):
            pp.append('<text x="%.1f" y="%d" class="tk" text-anchor="middle">%s%s</text>'
                      % (X_(i), H - pB + 16, g if g != KMGK else "4.48",
                         "*" if g == KMGK else ""))
        cbs = []
        for si, (lbl, vals, mark) in enumerate(series):
            col, dsh = DSH.get(lbl, ("#000", ""))
            pts = " ".join("%.1f,%.1f" % (X_(i), Y_(v)) for i, v in enumerate(vals))
            g = ['<g id="swg%d">' % si,
                 '<polyline points="%s" fill="none" stroke="%s" stroke-width="2"%s/>'
                 % (pts, col, (' stroke-dasharray="%s"' % dsh) if dsh else "")]
            if mark:
                for i, v in enumerate(vals):
                    g.append('<circle cx="%.1f" cy="%.1f" r="2.6" fill="%s"/>' % (X_(i), Y_(v), col))
            g.append('</g>')
            pp.append("".join(g))
            sw = ('<svg width="30" height="12" viewBox="0 0 30 12" style="flex:none">'
                  '<line x1="1" y1="6" x2="29" y2="6" stroke="%s" stroke-width="2.4"%s/>'
                  '<circle cx="15" cy="6" r="2.6" fill="%s"/></svg>'
                  % (col, (' stroke-dasharray="%s"' % dsh) if dsh else "", col))
            cbs.append('<label style="display:inline-flex;align-items:center;gap:5px;'
                       'margin:0 14px 4px 0;cursor:pointer;font-size:13px">'
                       '<input type="checkbox" checked onchange="document.getElementById'
                       "('swg%d').style.display=this.checked?'':'none'\">%s %s</label>"
                       % (si, sw, lbl))
        ctl = ('<div style="margin:6px 0 2px 58px"><span class="hsub" style="margin-right:10px">'
               'show:</span>%s</div>' % "".join(cbs))
        pp.append('<line x1="%d" y1="%d" x2="%d" y2="%d" class="ax"/>' % (pL, H - pB, W - pR, H - pB))
        pp.append('<line x1="%d" y1="%d" x2="%d" y2="%d" class="ax"/>' % (pL, pT, pL, H - pB))
        pp.append('<text x="%d" y="%d" class="al" text-anchor="middle">Gamma assumed by the solver '
                  '(* = matched Gamma) -- ordinal spacing</text>'
                  % ((pL + W - pR) // 2, H - 8))
        ym = (pT + H - pB) // 2
        pp.append('<text x="15" y="%d" class="al" text-anchor="middle" '
                  'transform="rotate(-90 15 %d)">average test outcome E[Y]</text>' % (ym, ym))
        pp.append('</svg>')
        return ('<figure class="fig">%s<div class="scrollx chartbox">%s</div></figure>'
                % (ctl, "".join(pp)))

    sweep_lines = _lines()

    ordered = [k[0] for k in sorted(stats, key=lambda k: -stats[k][0])]
    chart = barchart(stats, "KMZ at the matched Gamma* = 4.4817, n=400 -- normalised value",
                     methods=ordered, series=["kmz"], colors={"kmz": "#4a4a2d"},
                     labels=lambda k: "matched Gamma*",
                     note="0 = best constant policy (all-treat), 1 = oracle; whisker = "
                          "across-seed sd where stored")
    return R, main_tbl, tm_tbl, st_tbl, ce_tbl, cap_tbl, kb_tbl, chart, sweep_tbl, sweep_lines


(_KMZR, KMZ_MAIN, KMZ_TM, KMZ_ST, KMZ_CE, KMZ_CAP, KMZ_KB, KMZ_CHART,
 KMZ_SWEEP, KMZ_SWEEPCH) = _kmz_tab()


# ------------------------------------------------ semi tab: Gamma mis-specification sweep
# DGP held at gamma = 1 (true Gamma* = 7.389); the solvers' Gamma swept over
# {1, 2, matched, 15, 50}. 125 cells (5 datasets x 5 draws x 5 seeds), c_eps = 1,
# L in {inf, 3, 1}, authors'-code baselines per Gamma. Read from gsweep/ at build time.
import glob as _gl


def _gsweep():
    fs = sorted(_gl.glob(str(HERE / "gsweep" / "*.json")))
    if not fs:
        return "", ""
    cells = [json.loads(Path(f).read_text()) for f in fs]
    gg = cells[0]["ggrid"]; matched = cells[0]["matched"]
    meths = ["IPW-O-W", "DoublyRobust-O-W", "Hajek-O-W",
             "IPW-O-X", "DoublyRobust-O-X", "Hajek-O-X"]
    SH2 = {"DoublyRobust-O-W": "DR-O-W", "DoublyRobust-O-X": "DR-O-X"}

    def nz(v, c):
        R = c["refs"]; bc = max(R["never_treat"], R["all_treat"])
        return (v - bc) / (R["oracle"] - bc)

    st, rows = {}, []
    for m in meths:
        cells_html = []
        for g in gg:
            vs = [nz(max(c["grid"][g][m].values()), c) for c in cells
                  if m in c["grid"].get(g, {}) and c["grid"][g][m]]
            if vs:
                st[(SH2.get(m, m), g)] = (float(np.mean(vs)), float(np.std(vs)))
                cells_html.append("<td>%.3f <span class='sd'>&plusmn;%.2f</span></td>"
                                  % (np.mean(vs), np.std(vs)))
            else:
                cells_html.append("<td>&mdash;</td>")
        rows.append((st.get((SH2.get(m, m), matched), (0,))[0],
                     "<tr%s><td class='l'>%s</td>%s</tr>"
                     % (" class='hl'" if m.endswith("O-W") else "", SH2.get(m, m),
                        "".join(cells_html))))
    for bm in ("SharpHess", "Kallus"):
        lbl = "Hess (paper)" if bm == "SharpHess" else "Kallus (paper)"
        cells_html = []
        for g in gg:
            vs = [nz(c["baselines"][g][bm], c) for c in cells
                  if bm in c.get("baselines", {}).get(g, {})]
            if vs:
                st[(lbl, g)] = (float(np.mean(vs)), float(np.std(vs)))
                cells_html.append("<td>%.3f <span class='sd'>&plusmn;%.2f</span></td>"
                                  % (np.mean(vs), np.std(vs)))
            else:
                cells_html.append("<td>&mdash;</td>")
        rows.append((st.get((lbl, matched), (-9,))[0],
                     "<tr><td class='l'>%s</td>%s</tr>" % (lbl, "".join(cells_html))))
    rows.sort(key=lambda t: -t[0])
    hdr = "".join("<th>&Gamma; = %s%s</th>" % (g, " (matched)" if g == matched else "")
                  for g in gg)
    tbl = ("<div class='scrollx'><table class='dt'><tr><th class='l'>method (best L, "
           "c<sub>&epsilon;</sub>=1)</th>" + hdr + "</tr>"
           + "".join(r[1] for r in rows) + "</table></div>")
    GC = {"1": "#0a7d33", "2": "#5b8c1a", matched: "#b07105", "15": "#c0392b", "50": "#334155"}
    ms = [m for m in ["IPW-O-W", "DR-O-W", "Hajek-O-W", "IPW-O-X", "DR-O-X", "Hajek-O-X",
                      "Hess (paper)", "Kallus (paper)"] if any((m, g) in st for g in gg)]
    ch = barchart(st, "Gamma mis-specification at DGP gamma = 1 -- pooled normalised value, "
                  "125 cells", methods=ms, series=gg,
                  colors={g: GC.get(g, "#64748b") for g in gg},
                  labels=lambda g: "Gamma = %s%s" % (g, " (matched)" if g == matched else ""),
                  note="whisker = sd across the 25 (dataset, draw) x 5 seed cells")
    return tbl, ch


GSWEEP_TBL, GSWEEP_CH = _gsweep()


nseeds = SEMI[SEMI_DS[0]][0]["n_seeds"] if SEMI else 0
html = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Semi-synthetic policy-learning experiments</title>
<style>{CSS}{EXTRA}{NOTECSS}{STAMPCSS}{CHARTCSS}{BOXCSS}{EQCSS}{REPSCSS}</style></head><body>
<div class="hero" style="background:linear-gradient(135deg,#3b0764,#7e22ce)">
<h1>Semi-synthetic policy-learning experiments</h1>
<p>Two constructions, both giving real covariates a synthetic confounded assignment with a KNOWN
sensitivity parameter. <b>Higher is better</b> throughout.</p></div>

<div class="tabbar">
<div class="tb on" id="tb-semi" onclick="showTab('semi')">semi</div>
<div class="tb" id="tb-rct" onclick="showTab('rct')">RCT</div>
<div class="tb" id="tb-kmz" onclick="showTab('kmz')">KMZ</div>
</div>

<div id="tab-semi" class="tabpane on">
{STAMP}
<h2>The construction</h2>
<section class="constr">
<p>Follows the data-generating procedure of <i>"Learning Risk Scores Robust to Unobserved
Confounders"</i> exactly for <i>X</i>, <i>U</i>, <i>Y</i><sup>0</sup> and <i>T</i>, and adds the
one thing policy learning needs and a risk score does not: a second potential outcome.</p>
{SEMI_MATH}
<p class="muted">The last line is the identity that makes the matched &Gamma; <b>exact</b> rather
than the interval [e<sup>&gamma;</sup>, e<sup>2&gamma;</sup>] the paper settles for: because
&pi;<sup>0</sup> is never clipped, the odds ratio between the two hidden states is
e<sup>2&gamma;</sup> at <i>every</i> x. Verified to ~1e&minus;13 on all five datasets.
Only &tau; and <i>Y</i><sup>1</sup> are ours; everything else is theirs.</p>
<p class="muted">Both closed forms were checked against the generated populations and agree to
1.1e&minus;16. Note the identity holds <i>conditional on x</i>: the marginal difference in
logit&nbsp;&pi;<sup>0</sup> between the two hidden states is slightly under 2&gamma; because
<i>u</i>&nbsp;=&nbsp;<i>Y</i><sup>0</sup> is weakly correlated with <i>x</i>. Since
&pi;<sup>0</sup> is never clipped, overlap degrades with &gamma; but never formally fails. The
percentiles below are over all 25 populations at each &gamma; (500k units); the raw min and max
are not shown because they are single outlier rows &mdash; at &gamma;&nbsp;=&nbsp;0, where there is
no confounding at all, the minimum e is already 0.0014 while the median is exactly 0.500.</p>
{overlap_tbl}
</section>
{SEMI_FLOW}
<div class="box"><b>How the inverse propensity scores are determined</b>
<p>Every solver is handed weights built from an <b>estimated</b> propensity. The true
&pi;<sup>0</sup>(x,u) depends on the hidden state and is never passed to any method &mdash; it
exists in the population file only to draw <i>T</i> and to certify that the matched &Gamma; equals
e<sup>2&gamma;</sup>.</p>
<ol>
<li><b>Fit</b> a logistic model <i>T</i> ~ <i>x</i> on the n&nbsp;=&nbsp;300 training rows
(scikit-learn <span class="mono">LogisticRegression</span>, lbfgs, max_iter 2000, default
L2). The covariate is the scalar CATE index, so this is a one-dimensional sigmoid in <i>x</i>.</li>
<li><b>Clip and renormalise</b>: probabilities are clipped to
[10<sup>&minus;3</sup>, 1&minus;10<sup>&minus;3</sup>] and each row rescaled to sum to 1 &mdash; a
positivity guard that keeps the inverse weights finite.</li>
<li><b>Invert</b> at the observed arm: <i>w<sub>i</sub></i> =
1&nbsp;/&nbsp;&ecirc;(<i>T<sub>i</sub></i>&nbsp;|&nbsp;<i>x<sub>i</sub></i>).</li>
<li><b>Two flavours are produced and used deliberately.</b> The <i>Hajek</i> form rescales each
arm's weights to sum to <i>n</i> and is what the IPW, doubly-robust and X-X solvers receive. The
<i>raw</i> Horvitz&ndash;Thompson form (no rescaling, so <i>w</i>&nbsp;&ge;&nbsp;1) goes to the
Hajek-* solvers and to Kallus, because the MSM box is built by multiplying and dividing a raw
inverse weight by &Gamma; &mdash; feeding it rescaled weights would silently move the box, and the
code raises rather than allow it.</li>
</ol>
<p>The propensity is fit once on the training split; it is <i>not</i> cross-fitted. SharpHess is the
exception &mdash; it estimates its own propensity inside its cross-fitted nuisance step (a
{{64,64,32}} ReLU net, per the paper), so it never sees the weights described here.</p>
<p class="muted"><b>The gap this leaves is the point of the experiment.</b> Assignment truly depends
on both <i>x</i> and <i>u</i>, while &ecirc; can only capture the <i>x</i> part; the residual
<i>u</i>-dependence is exactly what &Gamma; is meant to cover. One wrinkle to keep in mind: the
assignment uses &lambda;&prime;<i>X</i> over the full covariate vector, which is <b>not</b> a
function of the scalar index <i>b</i>&prime;<i>X</i> the solvers see. So &ecirc;(<i>x</i>) carries
unexplained variation beyond <i>u</i>, and the <i>operative</i> &Gamma; a fitted pipeline would need
can exceed the declared e<sup>2&gamma;</sup>.</p></div>

<p>Because &pi;<sup>0</sup> is never clipped, the odds ratio between the two hidden states equals
e<sup>2&gamma;</sup> to ~1e&minus;13 at every <i>x</i>, so the matched &Gamma; is <b>exact</b> here
rather than the interval the paper settles for. Every dataset below passes the paper's screen.</p>
<div class="scrollx"><table class="dt">
<tr><th class="l">dataset</th><th>N</th><th>d</th><th>CV log-loss</th>
<th>headroom (&plusmn; across draws)</th><th>corr(x,u)</th><th>frac &tau;&gt;0</th>
<th>DGP draws</th></tr>
{cons}</table></div>

<div class="note"><b>Baselines re-optimised, nuisances made neural, &gamma; extended to 4
&mdash; 2026-08-04.</b> Both parametric baselines were mis-optimised: their policy learners used a
raw &nabla;&pi; step whose <i>p</i>(1&minus;<i>p</i>) factor collapses as the policy sharpens, so
both returned near-uniform policies, losing up to 0.96 normalised value against a brute-force grid
over their own two-parameter class. The estimators were <b>not</b> touched and were correct
throughout &mdash; Hess's Eq.&nbsp;15 reproduces the AIPW score at &Gamma;&nbsp;=&nbsp;1 to
9e&minus;16 on both nuisance paths, and Kallus's sign convention is exact
(|&Delta;&theta;|&nbsp;=&nbsp;0).
<br><br>
SharpHess is now the authors' repository pipeline <b>verbatim</b>
(<span class="mono">konstantinhess/Efficient_sharp_policy_learning</span>): disjoint 50/50
nuisance/policy split, {{64,32}} ReLU networks for every nuisance and the policy &mdash; their code
differs from their own Table&nbsp;5 &mdash; Adam lr&nbsp;1e&minus;3, batch&nbsp;64, &le;300 epochs,
early stopping patience&nbsp;10, loss = the estimated bound with nuisances frozen. The k-NN row is
kept only as a nuisance-sensitivity diagnostic. Earlier arms are preserved in the result files
(<span class="mono">hess_nn_customopt</span>, <span class="mono">hess_knn</span>,
<span class="mono">hess_k50</span>, <span class="mono">hess_legacy</span>).
<br><br>
<b>Caveat on the high-&gamma; rows:</b> &alpha;<sup>+</sup>&nbsp;=&nbsp;&Gamma;/(1+&Gamma;) reaches
0.982 at &gamma;&nbsp;=&nbsp;2 and 0.9997 at &gamma;&nbsp;=&nbsp;4, so the conditional quantile
there is an extreme tail that n&nbsp;=&nbsp;300 cannot resolve on any estimator. Read SharpHess's
&gamma;&nbsp;&ge;&nbsp;2 entries as sample-limited rather than as a verdict on the method.</div>

<h2>O-W family against the published baselines
<span class="hsub">&mdash; matched &Gamma; = e<sup>2&gamma;</sup>, best (c<sub>&epsilon;</sub>, L)
cell, mean &plusmn; sd over {nseeds} replications = 5 DGP draws &times; 10 splits</span></h2>
{semi_tbl}

<h2>Normalised and pooled across the five datasets</h2>
<p>(V &minus; best constant) / (oracle &minus; best constant). 0 = no better than not learning a
policy, 1 = oracle, negative = worse than the best constant policy.</p>
{pool_tbl}
<p class="muted">The &gamma; = 0 row is a correctness check: with no confounding the box
collapses to a point, so every non-sharp method must be solving the same LP. They agree exactly on
all five datasets.</p>

<h2>How much does the CATE draw matter?
<span class="hsub">&mdash; IPW-O-W normalised, mean &plusmn; spread across the 5 DGP draws</span></h2>
{spread_tbl}
<div class="card warn"><b>The draw matters more than the split.</b> Each cell above averages 10
treatment/split seeds; the &plusmn; is the spread across the five independent
(<i>a</i>, <i>b</i>, <i>c</i>, &lambda;) draws. That spread is <b>5&ndash;10&times; larger</b> than
the split-seed spread, because <i>a</i> scales the whole CATE and so sets how much signal exists at
all: across the 25 draws in this campaign the headroom ranges from <b>+0.036 to +0.333</b>. The
pooled ordering is robust &mdash; bank_marketing is tight at &plusmn;0.03&ndash;0.06 and clearly
positive &mdash; but any single per-dataset number should be quoted with the across-draw bar, not
the across-split one.</div>

<h2>By dataset and pooled
<span class="hsub">&mdash; normalised value, whisker = &plusmn;1 sd across the 5 DGP draws</span></h2>
<p>0 on the vertical axis is the best constant policy and 1 is the oracle, so a bar below the axis
means the method is worse than not learning a policy at all. The whisker is the spread across the
five independent CATE draws, which is the dominant source of uncertainty here.</p>
{CHARTS}

<h2>SharpHess instantiations
<span class="hsub">&mdash; the authors' own pipeline vs the k-NN diagnostic</span></h2>
{hess_tbl}

<h2>&Gamma; mis-specification <span class="hsub">&mdash; DGP fixed at &gamma; = 1
(true &Gamma;* = 7.39); the solvers' &Gamma; swept over 1, 2, matched, 15, 50</span></h2>
{GSWEEP_CH}
{GSWEEP_TBL}
<p class="muted">The robustness hyperparameter swept off its matched value: &Gamma; = 1 is the
plain point estimate, &Gamma; = 50 assumes ~7&times; the true confounding. X-X and naive are
&Gamma;-free and unchanged from the tables above. 125 cells: 5 datasets &times; 5 DGP draws
&times; 5 split seeds, c<sub>&epsilon;</sub> = 1, best L per cell. <b>Normalisation note:</b>
this table averages per-cell ratios (each cell normalised by its own references), while the
pooled tables above divide pooled means &mdash; the two differ by design (about 0.12 at the
matched point), so compare within this table, not across to the pooled one.</p>

<h2>Which method wins <span class="hsub">&mdash; confounded cells only (&gamma; &gt; 0)</span></h2>
{win_tbl}

<h2>Transport margin</h2>
<p>O-W minus its own O-X twin at identical <i>L</i> &mdash; same estimator, same box, same policy
class, so the Wasserstein ball is the only difference. This is the quantity the paper's
Robust-vs-Robust-Baseline comparison measures, and the one the O-W claim rests on.</p>
{tm_tbl}
</div>

<div id="tab-rct" class="tabpane">
<h2>The construction</h2>
<section class="constr">
<p>Three datasets that already carry both potential outcomes, given the same treatment of a
synthetic confounded assignment with a known &Lambda; = 4.</p>
{RCT_MATH}
<p class="muted">Here <i>z<sup>A</sup></i> and <i>z<sup>B</sup></i> are the standardised
principal-direction indices of the two covariate groups, and <i>q</i><sub>99</sub> is the 99th
percentile, so about 1% of the mass is clipped to the boundary. As above, not clipping
<i>e</i> makes &Gamma;<sup>&#9733;</sup> = &Lambda; exact; the realised value measured back out of
the data is 4.000 on all three benchmarks.</p>
</section>
{RCT_FLOW}
<div class="scrollx"><table class="dt">
<tr><th class="l">benchmark</th><th>N</th><th>outcome</th><th>corr(x,S)</th><th>&Lambda; realised</th>
<th>best const</th><th>oracle</th><th>headroom</th></tr>
{rct_cons}</table></div>

<h2>Average test outcome at the matched &Gamma; = 4
<span class="hsub">&mdash; best (L, c<sub>&epsilon;</sub>) cell, mean &plusmn; sd over 5 seeds</span></h2>
{rct_tbl}

<h2>By benchmark <span class="hsub">&mdash; normalised value, 0 = best constant
policy, 1 = oracle</span></h2>
<p class="muted"><b>Read this chart with the headroom column above in hand.</b> Normalising divides
by (oracle &minus; best constant), which is only <b>0.0099</b> on IHDP and <b>0.0098</b> on Twins
&mdash; so on those two the divisor is near zero and the bars magnify noise rather than measure
skill. All-treat beats every method on all three benchmarks, which is why the whole field sits
below the zero line. IST, with headroom 0.127, is the only one of the three where the normalised
scale carries much meaning.</p>
{RCT_CHART}

<div class="card bad"><b>These three benchmarks are degenerate for policy learning.</b> Normalised
as (V &minus; never)/(oracle &minus; never), the constant <b>all-treat</b> policy scores
0.998 / 0.443 / 0.070 on IHDP / Twins / IST, while the best method manages 0.997 / 0.294 / 0.046.
In raw units the oracle beats the better constant policy by only 0.010, 0.010 and 0.127. When the
optimal policy is almost constant there is nothing for a policy learner to find, and every method
lands in the same place &mdash; which is exactly the flat ordering above.
<br><br>The cause is that the treatment effect in all three is a <b>level shift</b> rather than a
heterogeneous sign flip: IHDP's response surface adds a near-constant benefit, the heavier twin
survives more nearly everywhere, and IST's aspirin effect is 0.0099 on 6-month mortality. A policy
can only exploit a sign flip. The knob ranking is <b>L &gt;&gt; &Gamma; &asymp;
c<sub>&epsilon;</sub> &asymp; 0</b>, and the transport margin is ~0 on all three
(&minus;0.0000 / +0.0001 / +0.0002).
<br><br>The IST row is the one informative case: it has real headroom (0.127) and every method
recovers under 5% of it, so the heterogeneity exists and nothing finds it.</div>
</div>

<div id="tab-kmz" class="tabpane">
<h2>The construction</h2>
<section class="constr">
<p>The synthetic benchmark of <b>Kallus, Mao &amp; Zhou (2019), "Interval Estimation of
Individual-Level Causal Effects Under Unobserved Confounding," AISTATS 2019
(arXiv:1810.02894)</b> &mdash; run unmodified. It has become the community-standard MSM
synthetic: Dorn&ndash;Guo use it, and Hess et&nbsp;al. (ICLR 2026, arXiv:2502.13022) build their
own synthetic experiment on it. The label plays no role here; the confounder <i>S</i> is a fair
coin independent of <i>X</i>, and the true propensity is the <b>MSM extremal</b> around the
nominal one, so the sensitivity model holds exactly with a known odds ratio.</p>
{KMZ_MATH}
<p>Because the true propensity is the extremal itself, the matched &Gamma; is known by
construction &mdash; nothing is tuned. We run the paper's own strength sweep, each experiment at
its matched &Gamma;. Protocol: n&nbsp;=&nbsp;400 train / 4,000 test, 5 seeds, Shapley deployment,
x&nbsp;=&nbsp;X/2. <b>Structural role:</b> S&nbsp;&perp;&nbsp;X makes this the
corr(x,&nbsp;S)&nbsp;=&nbsp;0 anchor of the coupling diagnostic &mdash; the covariate carries no
information about the confounder, so the declared prediction (written before the runs) is
O-W&nbsp;&asymp;&nbsp;O-X.</p>
</section>

<h2>All methods at the matched &Gamma;* = 4.4817
<span class="hsub">&mdash; n = 400, 5 seeds; references: oracle {_KMZR["oracle"]:.3f}, all-treat {_KMZR["all"]:.3f},
never-treat {_KMZR["never"]:.3f}</span></h2>
{KMZ_CHART}
{KMZ_MAIN}
<p class="muted">Both published baselines run <b>the authors' own code</b>: Hess via their
repository's {{64,32}}/Adam pipeline (data-hungry &mdash; 0.205 at this shared n=400 vs 1.192 at
their own n=5000, a size the LP methods cannot reach because of the n&sup2; transport variables);
Kallus via their <span class="mono">grad_descent_sharp</span>/Armijo/15-restart protocol,
bit-validated against their repository.</p>

<h2>Across &Gamma; <span class="hsub">&mdash; the full grid 1 &hellip; 50; the solver's &Gamma;
varies, the DGP stays fixed; O-W, O-X, Hess, Kallus; best L per point</span></h2>
{KMZ_SWEEPCH}
{KMZ_SWEEP}
<p class="muted">&Gamma; = 1 collapses every box to the point estimate; &Gamma; = 15 is
&asymp;&nbsp;3&times; the matched value, i.e. deliberate over-robustness. Mild over-statement of
&Gamma; costs little here (the surface peaks slightly ABOVE the matched value), while
under-statement costs more &mdash; and the Kallus row pins to its reference at every
&Gamma;&nbsp;&ge;&nbsp;2 regardless.</p>

<h2>Transport margin by L <span class="hsub">&mdash; the zero-coupling prediction,
verified</span></h2>
{KMZ_TM}
<p class="muted">O-W minus its O-X twin at identical <i>L</i> is &asymp;&nbsp;+0.03 at best: with
S&nbsp;&perp;&nbsp;X the Wasserstein term has provably nothing to grab, and the measured
near-tie is the <i>confirmation</i> of the coupling diagnostic, not a failure. Smoothness is
load-bearing for every method (all collapse to &asymp;0.35 at L=&infin;, peak at L=5&ndash;10).</p>

<h2>The paper's own strength axis</h2>
{KMZ_ST}
<p class="muted">Graceful degradation as confounding strengthens, the O-W/O-X tie holding at every
strength; Hajek-O-X is the one collapse.</p>

<h2>Ablations</h2>
<div class="figrow">
<div>{KMZ_CE}<p class="muted figcap">The transport budget is inert here &mdash; consistent with the
zero-coupling structure.</p></div>
<div>{KMZ_CAP}</div>
</div>

<h2>Kallus and the reference policy
<span class="hsub">&mdash; why its row is the anchor, not the estimator</span></h2>
{KMZ_KB}
<p class="muted">Their method minimises worst-case regret against a reference whose own regret is
zero by construction, so under real confounding the certified-safe optimum <i>is</i> the
reference: at every &Gamma;&nbsp;&ge;&nbsp;2 the deployed policy equals the anchor digit for
digit. Their synthetic driver anchors to control &mdash; the worst constant policy on this
benchmark (&minus;1.010) while all-treat is near-oracle (+1.035) &mdash; a 2.05 swing from the
reference choice alone, with the estimator unchanged. Any Kallus number should be quoted with its
reference policy.</p>
</div>

<script>
function showTab(id){{
  for (const t of ['semi','rct','kmz']){{
    document.getElementById('tab-'+t).classList.toggle('on', t===id);
    document.getElementById('tb-'+t).classList.toggle('on', t===id);
  }}
  window.scrollTo(0,0);
}}
</script>
</body></html>"""

html = html.encode("ascii", "xmlcharrefreplace").decode("ascii")
(HERE / "semisynth_experiments.html").write_text(html)
print("wrote %s (%d KB)" % (HERE / "semisynth_experiments.html", len(html) // 1024))
