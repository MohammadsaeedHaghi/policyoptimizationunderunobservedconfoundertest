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


# ---------------------------------------------------------------- shared interactive charts
# colour = estimator; dash + marker shape = uncertainty set (documented in the "read me" tab)
DSHMAP = {"IPW-O-W": ("#1f77b4", "", "circle"), "DR-O-W": ("#d62728", "", "circle"),
          "Hajek-O-W": ("#0a7d33", "", "circle"),
          "IPW-O-X": ("#1f77b4", "6 4", "square"), "DR-O-X": ("#d62728", "6 4", "square"),
          "Hajek-O-X": ("#0a7d33", "6 4", "square"),
          "IPW-X-X": ("#1f77b4", "2 3", "otri"), "DR-X-X": ("#d62728", "2 3", "otri"),
          "Direct-X-X": ("#64748b", "2 3", "otri"),
          "Hess (paper)": ("#7d1f6a", "", "diamond"),
          "Kallus (paper)": ("#8c564b", "", "tri")}
SELSTY = ('font-size:13px;padding:2px 6px;margin-right:8px;border:1px solid #bbb;'
          'border-radius:4px;background:#fff;cursor:pointer')


def hxpack(vals):
    """Hex-pack a policy vector (2 chars per unit, 0..100 scale) for compact embedding."""
    v = np.clip(np.asarray(vals, float).ravel(), 0.0, 1.0)
    return "".join("%02x" % int(round(x * 100)) for x in v)


def mkshape(x, y, col, shape):
    if shape == "circle":
        return '<circle cx="%.1f" cy="%.1f" r="3" fill="%s"/>' % (x, y, col)
    if shape == "square":
        return ('<rect x="%.1f" y="%.1f" width="5.6" height="5.6" fill="#fff" '
                'stroke="%s" stroke-width="1.6"/>' % (x - 2.8, y - 2.8, col))
    if shape == "diamond":
        return ('<rect x="%.1f" y="%.1f" width="5.2" height="5.2" fill="%s" '
                'transform="rotate(45 %.1f %.1f)"/>' % (x - 2.6, y - 2.6, col, x, y))
    if shape == "otri":
        return ('<path d="M %.1f %.1f L %.1f %.1f L %.1f %.1f Z" fill="#fff" '
                'stroke="%s" stroke-width="1.6"/>'
                % (x, y - 3.4, x - 3.2, y + 2.6, x + 3.2, y + 2.6, col))
    return ('<path d="M %.1f %.1f L %.1f %.1f L %.1f %.1f Z" fill="%s"/>'
            % (x, y - 3.4, x - 3.2, y + 2.6, x + 3.2, y + 2.6, col))


_ILINES_JS = """
<script>
var PFXSW=__DATA__, PFXGK=__GKS__, PFXR=__REFS__;
var PFXX0=__X0__, PFXXS=__XS__, PFXY0=__Y0__, PFXYHI=__YHI__, PFXYS=__YS__;
var PFXW=__W__, PFXPR=__PR__;
function PFXX(k){return PFXX0+k*PFXXS;}
function PFXY(v){return PFXY0+(PFXYHI-v)*PFXYS;}
function PFXMode(){
 var e=document.querySelector('input[name=PFXmode]:checked');
 return e?e.value:'test';
}
function PFXMk(x,y,c,shp){
 x=+x; y=+y;
 if(shp=='circle')return '<circle cx="'+x.toFixed(1)+'" cy="'+y.toFixed(1)+'" r="3" fill="'+c+'"/>';
 if(shp=='square')return '<rect x="'+(x-2.8).toFixed(1)+'" y="'+(y-2.8).toFixed(1)+'" width="5.6" height="5.6" fill="#fff" stroke="'+c+'" stroke-width="1.6"/>';
 if(shp=='diamond')return '<rect x="'+(x-2.6).toFixed(1)+'" y="'+(y-2.6).toFixed(1)+'" width="5.2" height="5.2" fill="'+c+'" transform="rotate(45 '+x.toFixed(1)+' '+y.toFixed(1)+')"/>';
 var f=(shp=='otri')?'#fff':c;
 var st=(shp=='otri')?' stroke="'+c+'" stroke-width="1.6"':'';
 return '<path d="M '+x.toFixed(1)+' '+(y-3.4).toFixed(1)+' L '+(x-3.2).toFixed(1)+' '+(y+2.6).toFixed(1)+' L '+(x+3.2).toFixed(1)+' '+(y+2.6).toFixed(1)+' Z" fill="'+f+'"'+st+'/>';
}
function PFXTogM(i,cb){
 document.getElementById('PFXg'+i).style.display=cb.checked?'':'none';
 PFXCntM();
}
function PFXAllM(v){
 var cbs=document.querySelectorAll('#PFXmdd input[type=checkbox]');
 for(var i=0;i<cbs.length;i++){
  cbs[i].checked=v;
  document.getElementById('PFXg'+cbs[i].getAttribute('data-i')).style.display=v?'':'none';
 }
 PFXCntM();
}
function PFXCntM(){
 var cbs=document.querySelectorAll('#PFXmdd input[type=checkbox]'),n=0;
 for(var i=0;i<cbs.length;i++)if(cbs[i].checked)n++;
 document.getElementById('PFXmcnt').textContent=n;
}
function PFXBestV(row){var b=null;for(var k in row){if(b===null||row[k]>b)b=row[k];}return b;}
function PFXVal(s,g,L,ce,md){
 var d=(md=='train')?s.dtr:s.data;
 if(!d)return null;
 if(s.kind=='fixed')return (g in d)?d[g]:null;
 if(s.kind=='xx'){if(L=='best')return PFXBestV(d);return (L in d)?d[L]:null;}
 var srf=(s.kind=='ow')?d[ce]:d;
 if(!srf||!(g in srf))return null;
 if(L=='best')return PFXBestV(srf[g]);
 return (L in srf[g])?srf[g][L]:null;
}
function PFXDraw(){
 var L=document.getElementById('PFXLsel').value;
 var ce=document.getElementById('PFXcesel').value;
 var md=PFXMode();
 var rg=document.getElementById('PFXrefs');
 if(rg&&PFXR[md]){
  var rp=[];
  for(var r=0;r<PFXR[md].length;r++){
   var rr=PFXR[md][r];
   rp.push('<line x1="'+PFXX0.toFixed(1)+'" y1="'+PFXY(rr[2]).toFixed(1)+'" x2="'+(PFXW-PFXPR)+'" y2="'+PFXY(rr[2]).toFixed(1)+'" stroke="'+rr[1]+'" stroke-width="1.2" stroke-dasharray="'+rr[3]+'" opacity="0.65"/>');
   rp.push('<text x="'+(PFXW-PFXPR+6)+'" y="'+(PFXY(rr[2])+3.5).toFixed(1)+'" class="tk" text-anchor="start">'+rr[0]+' '+rr[2].toFixed(2)+'</text>');
  }
  rg.innerHTML=rp.join('');
 }
 for(var i=0;i<PFXSW.length;i++){
  var s=PFXSW[i],parts=[],pts=[];
  for(var k=0;k<PFXGK.length;k++){
   var v=PFXVal(s,PFXGK[k],L,ce,md);
   if(v!==null&&v!==undefined)pts.push([PFXX(k),PFXY(v)]);
  }
  if(pts.length){
   var str='';
   for(var j=0;j<pts.length;j++)str+=(j?' ':'')+pts[j][0].toFixed(1)+','+pts[j][1].toFixed(1);
   parts.push('<polyline points="'+str+'" fill="none" stroke="'+s.col+'" stroke-width="2"'+(s.dsh?' stroke-dasharray="'+s.dsh+'"':'')+'/>');
   for(var j2=0;j2<pts.length;j2++)parts.push(PFXMk(pts[j2][0],pts[j2][1],s.col,s.shape));
  }
  document.getElementById('PFXg'+i).innerHTML=parts.join('');
 }
}
PFXDraw();
</script>"""


def ilines(pfx, gks, gmatch, mticklab, series, refs3, Lgrid, ceopts, title, ctlnote,
           ylo, yhi, xaxis_lab, W=1080, H=480, refs3_train=None):
    """Interactive per-Gamma line chart. series: [{lbl, kind, data, col, dsh, shape}] where
    kind 'ow' -> data[ce][g][L]; 'ox' -> data[g][L]; 'xx' -> data[L]; 'fixed' -> data[g].
    Any series may carry "dtr" (a train-set mirror of data); when refs3_train is given a
    test/train toggle is rendered and the reference lines switch with it.
    refs3: [(label, colour, value, dash)] horizontal reference lines.
    ceopts: [(value, label)] for the c_eps dropdown. pfx must be a unique JS-safe prefix."""
    pL, pR, pT, pB = 58, 120, 34, 56
    X_ = lambda i: pL + i * (W - pL - pR) / (len(gks) - 1)
    Y_ = lambda v: pT + (yhi - v) / (yhi - ylo) * (H - pT - pB)
    pp = ['<svg viewBox="0 0 %d %d" class="chart">' % (W, H),
          '<text x="%d" y="18" class="ct">%s</text>' % (pL, title),
          '<g id="%srefs"></g>' % pfx]
    t = np.ceil(ylo * 2) / 2
    while t <= yhi:
        pp.append('<line x1="%d" y1="%.1f" x2="%d" y2="%.1f" class="grid"/>' % (pL, Y_(t), W - pR, Y_(t)))
        pp.append('<text x="%d" y="%.1f" class="tk" text-anchor="end">%g</text>' % (pL - 6, Y_(t) + 3.5, t))
        t += 0.5
    gm = gks.index(gmatch)
    pp.append('<line x1="%.1f" y1="%d" x2="%.1f" y2="%d" stroke="#888" stroke-width="1.2" '
              'stroke-dasharray="3 3" opacity="0.8"/>' % (X_(gm), pT, X_(gm), H - pB))
    for i, g in enumerate(gks):
        pp.append('<text x="%.1f" y="%d" class="tk" text-anchor="middle">%s</text>'
                  % (X_(i), H - pB + 16, mticklab if g == gmatch else g))
    cbs = []
    for si, s in enumerate(series):
        pp.append('<g id="%sg%d"></g>' % (pfx, si))
        sw = ('<svg width="30" height="14" viewBox="0 0 30 14" style="flex:none">'
              '<line x1="1" y1="7" x2="29" y2="7" stroke="%s" stroke-width="2.4"%s/>'
              '%s</svg>'
              % (s["col"], (' stroke-dasharray="%s"' % s["dsh"]) if s["dsh"] else "",
                 mkshape(15, 7, s["col"], s["shape"])))
        cbs.append('<label style="display:flex;align-items:center;gap:6px;'
                   'padding:2px 0;cursor:pointer;font-size:13px">'
                   '<input type="checkbox" data-i="%d" checked onchange="%sTogM(%d,this)">'
                   '%s %s</label>' % (si, pfx, si, sw, s["lbl"]))
    LOPTS = ["best"] + list(Lgrid)
    lop = ('<select id="%sLsel" onchange="%sDraw()" style="%s">%s</select>'
           % (pfx, pfx, SELSTY,
              "".join('<option value="%s"%s>%s</option>'
                      % (L, " selected" if L == "best" else "",
                         "best per point" if L == "best" else
                         ("L = inf" if L == "inf" else "L = " + L))
                      for L in LOPTS)))
    cop = ('<select id="%scesel" onchange="%sDraw()" style="%s">%s</select>'
           % (pfx, pfx, SELSTY,
              "".join('<option value="%s"%s>%s</option>'
                      % (v, " selected" if i == 0 else "", lab)
                      for i, (v, lab) in enumerate(ceopts))))
    mdd = ('<details id="%smdd" style="display:inline-block;position:relative;font-size:13px">'
           '<summary style="list-style:none;cursor:pointer;border:1px solid #bbb;'
           'border-radius:4px;padding:2px 10px;background:#fff;user-select:none">'
           'methods (<span id="%smcnt">%d</span>/%d) &#9662;</summary>'
           '<div style="position:absolute;z-index:30;top:calc(100%% + 4px);left:0;'
           'background:#fff;border:1px solid #bbb;border-radius:6px;'
           'box-shadow:0 4px 14px rgba(0,0,0,.18);padding:8px 14px;white-space:nowrap">'
           '<div style="margin-bottom:4px"><a href="javascript:%sAllM(true)">all</a>'
           ' &middot; <a href="javascript:%sAllM(false)">none</a></div>%s</div></details>'
           % (pfx, pfx, len(series), len(series), pfx, pfx, "".join(cbs)))
    tgl = ""
    if refs3_train is not None:
        tgl = ('<span class="hsub" style="margin-right:4px;margin-left:12px">evaluate on:'
               '</span>'
               '<label style="cursor:pointer;font-size:13px;margin-right:6px">'
               '<input type="radio" name="%smode" value="test" checked '
               'onchange="%sDraw()"> test</label>'
               '<label style="cursor:pointer;font-size:13px">'
               '<input type="radio" name="%smode" value="train" onchange="%sDraw()"> '
               'train (in-sample)</label>' % (pfx, pfx, pfx, pfx))
    ctl = ('<div style="margin:6px 0 2px 58px;display:flex;align-items:center;'
           'flex-wrap:wrap;gap:4px">%s'
           '<span class="hsub" style="margin-right:6px;margin-left:12px">Lipschitz L:</span>'
           '%s<span class="hsub" style="margin-right:6px;margin-left:6px">'
           'c<sub>&epsilon;</sub>:</span>%s%s'
           '<span class="hsub">%s</span></div>' % (mdd, lop, cop, tgl, ctlnote))
    pp.append('<line x1="%d" y1="%d" x2="%d" y2="%d" class="ax"/>' % (pL, H - pB, W - pR, H - pB))
    pp.append('<line x1="%d" y1="%d" x2="%d" y2="%d" class="ax"/>' % (pL, pT, pL, H - pB))
    pp.append('<text x="%d" y="%d" class="al" text-anchor="middle">%s</text>'
              % ((pL + W - pR) // 2, H - 8, xaxis_lab))
    ym = (pT + H - pB) // 2
    pp.append('<text x="15" y="%d" class="al" text-anchor="middle" '
              'transform="rotate(-90 15 %d)">average outcome E[Y]</text>' % (ym, ym))
    pp.append('</svg>')
    refsjs = {"test": [[a_, b_, c_, d_] for a_, b_, c_, d_ in refs3],
              "train": [[a_, b_, c_, d_] for a_, b_, c_, d_ in (refs3_train or refs3)]}
    js = (_ILINES_JS.replace("PFX", pfx)
          .replace("__DATA__", json.dumps(series)).replace("__GKS__", json.dumps(gks))
          .replace("__REFS__", json.dumps(refsjs))
          .replace("__X0__", "%.4f" % float(pL))
          .replace("__XS__", "%.4f" % ((W - pL - pR) / (len(gks) - 1)))
          .replace("__Y0__", "%.4f" % float(pT)).replace("__YHI__", "%.4f" % yhi)
          .replace("__YS__", "%.4f" % ((H - pT - pB) / (yhi - ylo)))
          .replace("__W__", "%d" % W).replace("__PR__", "%d" % pR))
    return ('<figure class="fig">%s<div class="scrollx chartbox">%s</div>%s</figure>'
            % (ctl, "".join(pp), js))


_ISURF_JS = """
<script>
var PFXS=__S3__, PFXStr=__S3TR__, PFXL=__LS__, PFXZLO=__ZLO__, PFXZHI=__ZHI__;
var PFXyaw=0.65, PFXpit=0.42, PFXdrag=null;
function PFXel(i){return document.getElementById(i);}
function PFXmd(){
 var e=document.querySelector('input[name=PFXsmode]:checked');
 return (e&&e.value=='train'&&PFXStr)?PFXStr:PFXS;
}
function PFXrgb(t){
 t=Math.max(0,Math.min(1,t));var r,g,b,u;
 if(t<0.5){u=t*2;r=59+u*162;g=76+u*145;b=192+u*29;}
 else{u=(t-0.5)*2;r=221-u*41;g=221-u*217;b=221-u*183;}
 return 'rgb('+Math.round(r)+','+Math.round(g)+','+Math.round(b)+')';
}
function PFXdraw(){
 var D=PFXmd()[PFXel('PFXce').value][PFXel('PFXm').value];
 var G=D.g, Z=D.z, nL=PFXL.length, ng=G.length;
 var cy=Math.cos(PFXyaw), sy=Math.sin(PFXyaw), cp=Math.cos(PFXpit), sp=Math.sin(PFXpit);
 var W=980,H=560,CX=W/2,CY=H/2+14,SC=230;
 function P(x,y,z){
  var zn=((z-PFXZLO)/(PFXZHI-PFXZLO)*2-1)*0.62;
  var xr=x*cy-y*sy, yr=x*sy+y*cy;
  return [CX+SC*xr, CY-SC*(zn*cp+yr*sp), yr*cp-zn*sp];
 }
 function xs(i){return ng>1?-1+2*i/(ng-1):0;}
 function ys(j){return nL>1?-1+2*j/(nL-1):0;}
 var out=[];
 var base=[P(-1,-1,PFXZLO),P(1,-1,PFXZLO),P(1,1,PFXZLO),P(-1,1,PFXZLO)];
 out.push('<path d="M'+base.map(function(p){return p[0].toFixed(1)+' '+p[1].toFixed(1);}).join('L')+'Z" fill="#f4f4f2" stroke="#999" stroke-width="0.8"/>');
 var zt;
 for(zt=Math.ceil(PFXZLO*2)/2; zt<=PFXZHI; zt+=0.5){
  var a=P(-1,-1,zt);
  out.push('<text x="'+(a[0]-6).toFixed(1)+'" y="'+(a[1]+3).toFixed(1)+'" class="tk" text-anchor="end">'+zt.toFixed(1)+'</text>');
 }
 var zx0=P(-1,-1,PFXZLO), zx1=P(-1,-1,PFXZHI);
 out.push('<line x1="'+zx0[0].toFixed(1)+'" y1="'+zx0[1].toFixed(1)+'" x2="'+zx1[0].toFixed(1)+'" y2="'+zx1[1].toFixed(1)+'" stroke="#666" stroke-width="1"/>');
 var quads=[];
 for(var i=0;i+1<ng;i++)for(var j=0;j+1<nL;j++){
  var p=[P(xs(i),ys(j),Z[i][j]),P(xs(i+1),ys(j),Z[i+1][j]),
         P(xs(i+1),ys(j+1),Z[i+1][j+1]),P(xs(i),ys(j+1),Z[i][j+1])];
  var za=(Z[i][j]+Z[i+1][j]+Z[i+1][j+1]+Z[i][j+1])/4;
  quads.push([(p[0][2]+p[1][2]+p[2][2]+p[3][2])/4,p,za]);
 }
 quads.sort(function(a,b){return b[0]-a[0];});
 for(var q=0;q<quads.length;q++){
  var pp=quads[q][1];
  out.push('<path d="M'+pp.map(function(p){return p[0].toFixed(1)+' '+p[1].toFixed(1);}).join('L')+'Z" fill="'+PFXrgb((quads[q][2]-PFXZLO)/(PFXZHI-PFXZLO))+'" fill-opacity="0.93" stroke="#444" stroke-width="0.6"/>');
 }
 for(var i2=0;i2<ng;i2++){
  var t=P(xs(i2),-1.16,PFXZLO);
  out.push('<text x="'+t[0].toFixed(1)+'" y="'+t[1].toFixed(1)+'" class="tk" text-anchor="middle">'+(G[i2]=='__MG__'?'__MT__':G[i2])+'</text>');
 }
 for(var j2=0;j2<nL;j2++){
  var t2=P(1.14,ys(j2),PFXZLO);
  out.push('<text x="'+t2[0].toFixed(1)+'" y="'+t2[1].toFixed(1)+'" class="tk" text-anchor="start">'+PFXL[j2]+'</text>');
 }
 var tg=P(0,-1.42,PFXZLO), tl=P(1.45,0,PFXZLO);
 out.push('<text x="'+tg[0].toFixed(1)+'" y="'+tg[1].toFixed(1)+'" class="al" text-anchor="middle">Gamma (* = matched)</text>');
 out.push('<text x="'+tl[0].toFixed(1)+'" y="'+tl[1].toFixed(1)+'" class="al" text-anchor="middle">L</text>');
 out.push('<text x="'+(zx1[0]).toFixed(1)+'" y="'+(zx1[1]-8).toFixed(1)+'" class="al" text-anchor="middle">E[Y]</text>');
 PFXel('PFXsvg').innerHTML=out.join('');
}
function PFXdown(e){PFXdrag=[e.clientX,e.clientY];e.preventDefault();}
function PFXmove(e){
 if(!PFXdrag)return;
 PFXyaw+=(e.clientX-PFXdrag[0])*0.008;
 PFXpit=Math.max(0.05,Math.min(1.35,PFXpit+(e.clientY-PFXdrag[1])*0.006));
 PFXdrag=[e.clientX,e.clientY];PFXdraw();
}
document.addEventListener('mouseup',function(){PFXdrag=null;});
document.addEventListener('mousemove',PFXmove);
PFXel('PFXsvg').addEventListener('mousedown',PFXdown);
PFXel('PFXsvg').addEventListener('touchstart',function(e){var t=e.touches[0];PFXdown({clientX:t.clientX,clientY:t.clientY,preventDefault:function(){e.preventDefault();}});},{passive:false});
PFXel('PFXsvg').addEventListener('touchmove',function(e){var t=e.touches[0];PFXmove({clientX:t.clientX,clientY:t.clientY});e.preventDefault();},{passive:false});
document.addEventListener('touchend',function(){PFXdrag=null;});
PFXdraw();
</script>"""


def isurf(pfx, S3, Ls, zlo, zhi, ceopts, methods, gmatch, mticklab, ctlnote,
          s3_train=None):
    """Interactive drag-to-rotate 3-D surface. S3[ce][method] = {"g": [...], "z": [[...]]}
    (rows indexed by gamma, columns by Ls). s3_train, when given, adds a test/train toggle.
    pfx must be a unique JS-safe prefix."""
    msel = ('<select id="%sm" onchange="%sdraw()" style="%s">%s</select>'
            % (pfx, pfx, SELSTY,
               "".join('<option value="%s"%s>%s</option>'
                       % (m, " selected" if i == 0 else "", m)
                       for i, m in enumerate(methods))))
    cesel = ('<select id="%sce" onchange="%sdraw()" style="%s">%s</select>'
             % (pfx, pfx, SELSTY,
                "".join('<option value="%s"%s>%s</option>'
                        % (v, " selected" if i == 0 else "", lab)
                        for i, (v, lab) in enumerate(ceopts))))
    tgl = ""
    if s3_train is not None:
        tgl = ('<span class="hsub" style="margin-left:8px">evaluate on:</span>'
               '<label style="cursor:pointer;font-size:13px">'
               '<input type="radio" name="%ssmode" value="test" checked '
               'onchange="%sdraw()"> test</label>'
               '<label style="cursor:pointer;font-size:13px">'
               '<input type="radio" name="%ssmode" value="train" onchange="%sdraw()"> '
               'train</label>' % (pfx, pfx, pfx, pfx))
    ctl = ('<div style="margin:6px 0 2px 58px;display:flex;align-items:center;'
           'flex-wrap:wrap;gap:6px"><span class="hsub">method:</span>%s'
           '<span class="hsub">c<sub>&epsilon;</sub>:</span>%s%s'
           '<span class="hsub">%s</span></div>' % (msel, cesel, tgl, ctlnote))
    js = (_ISURF_JS.replace("PFX", pfx)
          .replace("__S3__", json.dumps(S3))
          .replace("__S3TR__", json.dumps(s3_train) if s3_train is not None else "null")
          .replace("__LS__", json.dumps(Ls))
          .replace("__ZLO__", "%f" % zlo).replace("__ZHI__", "%f" % zhi)
          .replace("__MG__", gmatch).replace("__MT__", mticklab))
    return ('<figure class="fig">%s<div class="scrollx chartbox">'
            '<svg id="%ssvg" viewBox="0 0 980 560" class="chart" '
            'style="cursor:grab;touch-action:none"></svg></div>%s</figure>' % (ctl, pfx, js))


_IPOL_JS = """
<script>
var PFXD=__DATA__;
function PFXel(i){return document.getElementById(i);}
function PFXnav(d,kind,g,L,ce){
 if(!d)return null;
 if(kind=='flat')return d;
 if(kind=='fixed')return (g in d)?d[g]:null;
 if(kind=='xx')return (L in d)?d[L]:null;
 if(kind=='ow'){d=d[ce]; if(!d)return null;}
 if(!(g in d))return null;
 return (L in d[g])?d[g][L]:null;
}
function PFXcurves(m,g,L,ce){
 var s=PFXD.methods[m]; if(!s)return null;
 return PFXnav(s.data,s.kind,g,L,ce);
}
function PFXhx(s){
 var a=[];
 for(var i=0;i<s.length;i+=2)a.push(parseInt(s.substr(i,2),16)/100);
 return a;
}
function PFXpath(xg,c,X_,Y_){
 var p='';
 for(var i=0;i<c.length;i++)p+=(i?' ':'')+X_(xg[i]).toFixed(1)+','+Y_(c[i]).toFixed(1);
 return p;
}
function PFXdraw(){
 var m=PFXel('PFXm').value, g=PFXel('PFXg').value, L=PFXel('PFXL').value,
     ce=PFXel('PFXce').value, sd=PFXel('PFXsd').value;
 var pme=document.querySelector('input[name=PFXpm]:checked');
 var pm=pme?pme.value:'deployed';
 var W=1080,H=430,pL=58,pR=30,pT=30,pB=52;
 var xlo=PFXD.xlo,xhi=PFXD.xhi;
 var X_=function(x){return pL+(x-xlo)/(xhi-xlo)*(W-pL-pR);};
 var Y_=function(v){return pT+(1.04-v)/1.08*(H-pT-pB);};
 var out=[];
 for(var t=0;t<=1;t+=0.25){
  out.push('<line x1="'+pL+'" y1="'+Y_(t).toFixed(1)+'" x2="'+(W-pR)+'" y2="'+Y_(t).toFixed(1)+'" class="grid"/>');
  out.push('<text x="'+(pL-6)+'" y="'+(Y_(t)+3.5).toFixed(1)+'" class="tk" text-anchor="end">'+t.toFixed(2)+'</text>');
 }
 for(var x=Math.ceil(xlo*2)/2;x<=xhi+1e-9;x+=0.5){
  out.push('<text x="'+X_(x).toFixed(1)+'" y="'+(H-pB+16)+'" class="tk" text-anchor="middle">'+x+'</text>');
 }
 var s=PFXD.methods[m], col=s?s.col:'#000';
 var xg=(s&&s.xg)?s.xg:PFXD.grid;
 if(PFXD.oracle){
  var oc=PFXD.oracle;
  out.push('<polyline points="'+PFXpath(PFXD.grid,oc,X_,Y_)+'" fill="none" stroke="#111" stroke-width="1.4" stroke-dasharray="5 4" opacity="0.7"/>');
 }
 if(pm=='learned'){
  var s2=PFXD.methods[m];
  var le=(s2&&s2.lrn)?PFXnav(s2.lrn,s2.kind,g,L,ce):null;
  if(le&&Object.keys(le).length==0)le=null;
  if(!s2||!s2.lrn){
   out.push('<text x="'+((pL+W-pR)/2)+'" y="'+((pT+H-pB)/2)+'" class="al" text-anchor="middle">'+(s2&&s2.par?'parametric policy: the deployed curve IS the learned function (no LP extension)':'learned (training-point) policy not stored for this method')+'</text>');
  }else if(!le){
   out.push('<text x="'+((pL+W-pR)/2)+'" y="'+((pT+H-pB)/2)+'" class="al" text-anchor="middle">no stored solution for this (method, Gamma, L, c_eps) combination</text>');
  }else{
   var sds=(sd=='mean')?Object.keys(le):[sd];
   var ndots=0;
   for(var q=0;q<sds.length;q++){
    var hxs=le[sds[q]], lx=PFXD.lx[sds[q]];
    if(!hxs||!lx)continue;
    var pv=PFXhx(hxs);
    var op=(sds.length>1)?0.3:0.85;
    for(var i3=0;i3<pv.length;i3++){
     out.push('<circle cx="'+X_(lx[i3]).toFixed(1)+'" cy="'+Y_(pv[i3]).toFixed(1)+'" r="2" fill="'+col+'" fill-opacity="'+op+'"/>');
     ndots++;
    }
   }
   out.push('<text x="'+(W-pR-4)+'" y="'+(pT+12)+'" class="tk" text-anchor="end">raw LP solution pi_i on the training points ('+ndots+' points'+(sds.length>1?', all seeds':', seed '+sd)+')</text>');
  }
 }else{
 var cs=PFXcurves(m,g,L,ce);
 if(cs&&Object.keys(cs).length==0)cs=null;
 if(!cs){
  out.push('<text x="'+((pL+W-pR)/2)+'" y="'+((pT+H-pB)/2)+'" class="al" text-anchor="middle">no stored policy for this (method, Gamma, L, c_eps) combination</text>');
 }else{
  var keys=Object.keys(cs);
  if(sd=='mean'){
   var n=PFXD.grid.length, acc=null, cnt=0;
   for(var k=0;k<keys.length;k++){
    var c=cs[keys[k]]; if(!c)continue;
    out.push('<polyline points="'+PFXpath(xg,c,X_,Y_)+'" fill="none" stroke="'+col+'" stroke-width="1" opacity="0.25"/>');
    if(!acc){acc=c.slice();}else{for(var i=0;i<c.length;i++)acc[i]+=c[i];}
    cnt++;
   }
   if(acc){
    for(var i2=0;i2<acc.length;i2++)acc[i2]/=cnt;
    out.push('<polyline points="'+PFXpath(xg,acc,X_,Y_)+'" fill="none" stroke="'+col+'" stroke-width="2.6"/>');
   }
   out.push('<text x="'+(W-pR-4)+'" y="'+(pT+12)+'" class="tk" text-anchor="end">'+cnt+' seed'+(cnt>1?'s':'')+' (thin) + mean (bold)</text>');
  }else{
   var c2=cs[sd];
   if(!c2){
    out.push('<text x="'+((pL+W-pR)/2)+'" y="'+((pT+H-pB)/2)+'" class="al" text-anchor="middle">seed '+sd+' has no stored curve here</text>');
   }else{
    out.push('<polyline points="'+PFXpath(xg,c2,X_,Y_)+'" fill="none" stroke="'+col+'" stroke-width="2.4"/>');
   }
  }
 }
 }
 out.push('<line x1="'+pL+'" y1="'+(H-pB)+'" x2="'+(W-pR)+'" y2="'+(H-pB)+'" class="ax"/>');
 out.push('<line x1="'+pL+'" y1="'+pT+'" x2="'+pL+'" y2="'+(H-pB)+'" class="ax"/>');
 out.push('<text x="'+((pL+W-pR)/2)+'" y="'+(H-8)+'" class="al" text-anchor="middle">x (standardised index); dashed black = oracle policy</text>');
 out.push('<text x="15" y="'+((pT+H-pB)/2)+'" class="al" text-anchor="middle" transform="rotate(-90 15 '+((pT+H-pB)/2)+')">pi(x) = treatment probability</text>');
 PFXel('PFXsvg').innerHTML=out.join('');
}
PFXdraw();
</script>"""


def ipol(pfx, data, methods_order, gks, gmatch, Lgrid, ceopts, seedkeys, title, ctlnote):
    """Interactive policy plot pi(x) vs x. data = {"grid": [...], "xlo","xhi", "oracle": [...]
    or None, "methods": {label: {kind, data, col, xg?}}} where kind decides the lookup:
    ow -> data[ce][g][L][seed]; ox -> data[g][L][seed]; xx -> data[L][seed];
    fixed -> data[g][seed]; flat -> data[seed]. Every leaf is a curve on grid (or xg)."""
    sel = lambda i, opts: ('<select id="%s%s" onchange="%sdraw()" style="%s">%s</select>'
                           % (pfx, i, pfx, SELSTY,
                              "".join('<option value="%s"%s>%s</option>'
                                      % (v, " selected" if j == 0 else "", lab)
                                      for j, (v, lab) in enumerate(opts))))
    msel = sel("m", [(m, m) for m in methods_order])
    gsel = sel("g", [(g, "Gamma = %s%s" % (g, " (matched)" if g == gmatch else ""))
                     for g in ([gmatch] + [g for g in gks if g != gmatch])])
    lsel = sel("L", [(L, "L = inf" if L == "inf" else "L = " + L)
                     for L in (["3"] + [L for L in Lgrid if L != "3"])])
    cesel = sel("ce", ceopts)
    sdsel = sel("sd", [("mean", "all seeds + mean")] + [(s, "seed " + s) for s in seedkeys])
    pmr = ""
    if data.get("lx"):
        pmr = ('<span class="hsub" style="margin-left:8px">view:</span>'
               '<label style="cursor:pointer;font-size:13px;margin-right:6px">'
               '<input type="radio" name="%spm" value="deployed" checked '
               'onchange="%sdraw()"> deployed &pi;(x)</label>'
               '<label style="cursor:pointer;font-size:13px">'
               '<input type="radio" name="%spm" value="learned" onchange="%sdraw()"> '
               'learned (raw LP output)</label>' % (pfx, pfx, pfx, pfx))
    ctl = ('<div style="margin:6px 0 2px 58px;display:flex;align-items:center;flex-wrap:wrap;'
           'gap:6px"><span class="hsub">method:</span>%s<span class="hsub">&Gamma;:</span>%s'
           '<span class="hsub">L:</span>%s<span class="hsub">c<sub>&epsilon;</sub>:</span>%s'
           '<span class="hsub">seed:</span>%s%s</div>'
           '<div style="margin:0 0 2px 58px"><span class="hsub">%s</span></div>'
           % (msel, gsel, lsel, cesel, sdsel, pmr, ctlnote))
    js = (_IPOL_JS.replace("PFX", pfx).replace("__DATA__", json.dumps(data)))
    return ('<figure class="fig"><figcaption class="ct" style="margin-left:58px">%s'
            '</figcaption>%s<div class="scrollx chartbox">'
            '<svg id="%ssvg" viewBox="0 0 1080 430" class="chart"></svg></div>%s</figure>'
            % (title, ctl, pfx, js))


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
KM_TR = _J(_MSM / "kmz_train_values.json")          # in-sample values, from stored policies
KM_XXTR = _J(_GRD / "xx_km_train.json")             # X-X re-solve with train+test
KM_HCURV = {}                                        # Hess curves + train values per gamma
for _i in range(8):
    _d = _J(_GRD / ("hess_km_curves_%d.json" % _i))
    if _d: KM_HCURV[_d["gamma"]] = _d
KM_KALTR = {}                                        # Kallus refits with train values
for _i in range(8):
    _d = _J(_MSM / ("kallus_km_train_%d.json" % _i))
    if _d: KM_KALTR[_d["gamma"]] = _d
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
    # X-X series: the 2026-08-10 re-solve is authoritative when present -- it stores
    # per-seed TEST and TRAIN values from the SAME solves (required for the toggle).
    # The old xxL run's tight-L cells sit on massively degenerate LP optima, so vertex
    # picks (and deployed values) are environment-dependent and cannot be paired with
    # freshly computed train values.
    if KM_XXTR:
        XXM = KM_XXTR["test_mean"]
        XXS = {m: {L: float(np.std(list(KM_XXTR["test"][m][L].values())))
                   for L in KM_XXTR["test"][m]} for m in KM_XXTR["test"]}
    elif KM_XX:
        XXM, XXS = KM_XX["mean"], KM_XX["sd"]
    else:
        XXM = XXS = None

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
    if XXM:
        for m, d in XXM.items():
            bl = max(d, key=d.get)
            lbl = m.replace("DoublyRobust", "DR")
            stats[(lbl, "kmz")] = (nz(d[bl]), abs(nz(d[bl] + XXS[m][bl]) - nz(d[bl])))
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

    # ---- per-Gamma LINE chart (shared helper ilines, JS prefix "sw") ----
    # series kinds: ow -> data[ce][gamma][L]; ox -> data[gamma][L] (epsilon-invariant,
    # verified equal across the ce files); xx -> data[L] (Gamma-free); fixed -> data[gamma]
    CES = [ce for ce in ("1.0", "1.5", "2.0") if KM_CE.get(ce)]
    CEOPT = [(ce, "c_eps = %s%s" % (ce, " (tight)" if ce == "1.0" else "")) for ce in CES]
    series = []
    for m in ["IPW-O-W", "DoublyRobust-O-W", "IPW-O-X", "DoublyRobust-O-X", "Hajek-O-X"]:
        if m.endswith("O-W"):
            series.append({"lbl": LBL2[m], "kind": "ow",
                           "data": {ce: KM_CE[ce]["surface"][m] for ce in CES}})
        else:
            series.append({"lbl": LBL2[m], "kind": "ox", "data": surf[m]})
    if XXM:
        for m, dd in XXM.items():
            series.append({"lbl": m.replace("DoublyRobust", "DR"), "kind": "xx", "data": dd})
    hx2 = {g: v for g, v in zip(KM_HESS["gammas"], KM_HESS["mean"])} if KM_HESS else {}
    if KM_H15: hx2["15"] = KM_H15["mean"]
    if KM_H50: hx2["50"] = KM_H50["mean"]
    if hx2:
        series.append({"lbl": "Hess (paper)", "kind": "fixed", "data": hx2})
    kx2 = ({("%g" % g): v for g, v in zip(KM_KAL["gammas"],
            KM_KAL["regimes"]["uncap"]["mean"]["Kallus"])} if KM_KAL else {})
    if KM_K15: kx2["15"] = KM_K15["regimes"]["uncap"]["mean"]["Kallus"][0]
    if KM_K50: kx2["50"] = KM_K50["regimes"]["uncap"]["mean"]["Kallus"][0]
    if kx2:
        series.append({"lbl": "Kallus (paper)", "kind": "fixed", "data": kx2})
    for s in series:
        s["col"], s["dsh"], s["shape"] = DSHMAP.get(s["lbl"], ("#000", "", "circle"))
    # in-sample mirrors for the test/train toggle
    refs3_train = None
    if KM_TR:
        for s in series:
            if s["kind"] == "ow":
                s["dtr"] = {ce: KM_TR["surface"][ce][ [m for m in LBL2 if LBL2[m] == s["lbl"]][0] ]
                            for ce in CES if ce in KM_TR["surface"]}
            elif s["kind"] == "ox":
                s["dtr"] = KM_TR["surface"]["1.0"][[m for m in LBL2 if LBL2[m] == s["lbl"]][0]]
            elif s["kind"] == "xx" and KM_XXTR:
                s["dtr"] = KM_XXTR["train_mean"][s["lbl"].replace("DR", "DoublyRobust")]
            elif s["lbl"] == "Hess (paper)" and KM_HCURV and \
                    all("mean_train" in d for d in KM_HCURV.values()):
                s["dtr"] = {g: d["mean_train"] for g, d in KM_HCURV.items()}
            elif s["lbl"] == "Kallus (paper)" and KM_KALTR:
                s["dtr"] = {g: d["train_mean"] for g, d in KM_KALTR.items()}
        rtr = {k: float(np.mean([KM_TR["refs_by_seed"][s_][k]
                                 for s_ in KM_TR["refs_by_seed"]]))
               for k in ("oracle", "all", "never")}
        refs3_train = [("oracle", "#111", rtr["oracle"], "5 4"),
                       ("all-treat", "#555", rtr["all"], "3 3"),
                       ("never-treat", "#555", rtr["never"], "3 3")]
    allv = [-1.12, 1.55]
    for s in series:
        for d in (s.get("dtr"),):
            if not d: continue
            def _flat(o):
                if isinstance(o, dict):
                    for v in o.values(): yield from _flat(v)
                elif isinstance(o, (int, float)): yield o
            allv.extend(_flat(d))
    ylo_, yhi_ = min(allv) - 0.05, max(allv) + 0.05
    sweep_lines = ilines(
        "sw", gks, KMGK, "4.48*", series,
        [("oracle", "#111", R["oracle"], "5 4"), ("all-treat", "#555", R["all"], "3 3"),
         ("never-treat", "#555", R["never"], "3 3")],
        KM["Lgrid"], CEOPT,
        "KMZ: all methods across the solver Gamma (n=400, 5 seeds; pick L and c_eps below)",
        "(X-X was solved only at L = inf, 3, 1 &mdash; its lines hide at other L; "
        "c<sub>&epsilon;</sub> scales the Wasserstein radius, so O-W only, and "
        "&Gamma; = 15, 50 exist only at c<sub>&epsilon;</sub> = 1.0; Hess/Kallus have "
        "neither knob; train = value of the same policy on its own training draw)",
        ylo_, yhi_,
        "Gamma assumed by the solver (* = matched Gamma) -- ordinal spacing",
        refs3_train=refs3_train)

    # ---- interactive 3-D surface (shared helper isurf, JS prefix "s3") ----
    Ls = list(KM["Lgrid"])
    S3 = {}
    S3tr = {} if KM_TR else None
    zlo, zhi = 1e9, -1e9
    for ce in CES:
        S3[ce] = {}
        if S3tr is not None: S3tr[ce] = {}
        for m in ["IPW-O-W", "DoublyRobust-O-W", "IPW-O-X", "DoublyRobust-O-X", "Hajek-O-X"]:
            src = KM_CE[ce]["surface"][m] if m.endswith("O-W") else surf[m]
            gam = [g for g in gks if g in src]
            z = [[src[g][L] for L in Ls] for g in gam]
            S3[ce][LBL2[m]] = {"g": gam, "z": z}
            for row in z:
                zlo = min(zlo, min(row)); zhi = max(zhi, max(row))
            if S3tr is not None:
                tsrc = KM_TR["surface"][ce if m.endswith("O-W") else "1.0"][m]
                tgam = [g for g in gam if g in tsrc]
                tz = [[tsrc[g][L] for L in Ls] for g in tgam]
                S3tr[ce][LBL2[m]] = {"g": tgam, "z": tz}
                for row in tz:
                    zlo = min(zlo, min(row)); zhi = max(zhi, max(row))
    zlo, zhi = zlo - 0.05, zhi + 0.05
    surf3d = isurf(
        "s3", S3, Ls, zlo, zhi, CEOPT,
        [LBL2[m] for m in ["IPW-O-W", "DoublyRobust-O-W", "IPW-O-X", "DoublyRobust-O-X",
                           "Hajek-O-X"]],
        "4.4817", "4.48*",
        "drag the plot to rotate; colour = height; X-X / Hess / Kallus have no "
        "(&Gamma;, L) surface; O-W at c<sub>&epsilon;</sub> &gt; 1 stops at &Gamma; = 8",
        s3_train=S3tr)

    # ---- interactive policy plot pi(x) vs x (shared helper ipol, JS prefix "kp") ----
    def _polfig():
        seeds = sorted(KM["policies_by_seed"].keys())
        P0 = KM["policies_by_seed"]
        SUP = {ce: KM_CE[ce].get("policies_support_by_seed") for ce in CES}
        methods = {}
        for m in ["IPW-O-W", "DoublyRobust-O-W"]:
            data, lrn = {}, {}
            for ce in CES:
                Pce = KM_CE[ce]["policies_by_seed"]
                dce, lce = {}, {}
                for g in gks:
                    gd, lg = {}, {}
                    for L in KM["Lgrid"]:
                        sd_ = {s: Pce[s][m][g][L] for s in seeds
                               if g in Pce[s][m] and L in Pce[s][m][g]}
                        if sd_: gd[L] = sd_
                        if SUP[ce]:
                            ld = {s: hxpack(SUP[ce][s][m][g][L]) for s in seeds
                                  if g in SUP[ce][s][m] and L in SUP[ce][s][m][g]}
                            if ld: lg[L] = ld
                    if gd: dce[g] = gd
                    if lg: lce[g] = lg
                data[ce] = dce; lrn[ce] = lce
            methods[LBL2[m]] = {"kind": "ow", "data": data, "lrn": lrn}
        for m in ["IPW-O-X", "DoublyRobust-O-X", "Hajek-O-X"]:
            data, lrn = {}, {}
            for g in gks:
                gd, lg = {}, {}
                for L in KM["Lgrid"]:
                    sd_ = {s: P0[s][m][g][L] for s in seeds
                           if g in P0[s][m] and L in P0[s][m][g]}
                    if sd_: gd[L] = sd_
                    if SUP["1.0"]:
                        ld = {s: hxpack(SUP["1.0"][s][m][g][L]) for s in seeds
                              if g in SUP["1.0"][s][m] and L in SUP["1.0"][s][m][g]}
                        if ld: lg[L] = ld
                if gd: data[g] = gd
                if lg: lrn[g] = lg
            methods[LBL2[m]] = {"kind": "ox", "data": data, "lrn": lrn}
        if KM_XXTR and "curves" in KM_XXTR:
            for m, dd in KM_XXTR["curves"].items():          # per-seed curves, all 5 seeds
                e = {"kind": "xx", "data": dd}
                if "learned" in KM_XXTR:
                    e["lrn"] = KM_XXTR["learned"][m]
                methods[m.replace("DoublyRobust", "DR")] = e
        elif KM_XX and "curves" in KM_XX:
            for m, dd in KM_XX["curves"].items():
                methods[m.replace("DoublyRobust", "DR")] = {
                    "kind": "xx", "data": {L: {"0": c} for L, c in dd.items()}}
        kd = {}
        for src in (KM_KAL, KM_K15, KM_K50):
            if src and "policy_by_seed" in src.get("regimes", {}).get("uncap", {}):
                pb = src["regimes"]["uncap"]["policy_by_seed"]["Kallus"]
                for s in seeds:
                    for g, c in pb.get(s, {}).items():
                        kd.setdefault(g, {})[s] = c
        if kd:
            methods["Kallus (paper)"] = {"kind": "fixed", "data": kd,
                                         "xg": KM_KAL["grid"]}
        import glob as _g2
        hd = {}
        for f in sorted(_g2.glob(str(_GRD / "hess_km_curves_*.json"))):
            d = json.loads(Path(f).read_text())
            hd[d["gamma"]] = d["curves"]
        if hd:
            methods["Hess (paper)"] = {"kind": "fixed", "data": hd, "par": True}
        if "Kallus (paper)" in methods:
            methods["Kallus (paper)"]["par"] = True
        methods["naive (DR)"] = {"kind": "flat",
                                 "data": {s: P0[s]["_refs"]["naive_dr"] for s in seeds}}
        if SUP["1.0"]:
            methods["naive (DR)"]["lrn"] = {s: hxpack(SUP["1.0"][s]["_naive_dr"])
                                            for s in seeds}
        for lbl in methods:
            methods[lbl]["col"] = DSHMAP.get(lbl, ("#334155",))[0]
        orc = np.mean([np.array(P0[s]["_refs"]["oracle"], float) for s in seeds], axis=0)
        data = {"grid": KM["policy_grid"], "xlo": -1.0, "xhi": 1.0,
                "oracle": [round(float(v), 3) for v in orc],
                "methods": methods}
        if SUP["1.0"]:
            data["lx"] = {s: SUP["1.0"][s]["_X"] for s in seeds}
        order = [m for m in ["IPW-O-W", "DR-O-W", "IPW-O-X", "DR-O-X", "Hajek-O-X",
                             "IPW-X-X", "DR-X-X", "Direct-X-X", "Hess (paper)",
                             "Kallus (paper)", "naive (DR)"] if m in methods]
        return ipol(
            "kp", data, order, gks, KMGK, KM["Lgrid"], CEOPT, seeds,
            "KMZ: deployed policy pi(x) on the standardised index (n=400)",
            "&Gamma; applies to O-X/O-W/Hess/Kallus only; L to the LP methods only; "
            "c<sub>&epsilon;</sub> moves O-W only; Hess/Kallus/naive ignore L and "
            "c<sub>&epsilon;</sub>. Kallus curves are stored on a coarser 7-point grid. "
            "<b>At L = inf the LP is per-point separable and positive reweighting cannot "
            "flip a sign, so the whole IPW/Hajek family returns the identical policy "
            "1{Y_i &gt; 0} at every &Gamma;</b> (only DR differs) &mdash; pick L &le; 10 "
            "to see the methods and &Gamma; separate.")

    polfig = _polfig()

    ordered = [k[0] for k in sorted(stats, key=lambda k: -stats[k][0])]
    chart = barchart(stats, "KMZ at the matched Gamma* = 4.4817, n=400 -- normalised value",
                     methods=ordered, series=["kmz"], colors={"kmz": "#4a4a2d"},
                     labels=lambda k: "matched Gamma*",
                     note="0 = best constant policy (all-treat), 1 = oracle; whisker = "
                          "across-seed sd where stored")
    # train twin of the matched bar chart: SAME cells (the test-best L per method),
    # valued on each policy's own training draw and normalised by the TRAIN references
    if (KM_TR and KM_XXTR and KM_KALTR and KM_HCURV
            and all("mean_train" in d for d in KM_HCURV.values())):
        rtr2 = {k: float(np.mean([KM_TR["refs_by_seed"][s_][k]
                                  for s_ in KM_TR["refs_by_seed"]]))
                for k in ("oracle", "all", "never")}
        bctr = max(rtr2["all"], rtr2["never"]); hrtr = rtr2["oracle"] - bctr
        nzt = lambda v: (v - bctr) / hrtr
        stats_tr = {}
        for m in ["IPW-O-W", "DoublyRobust-O-W", "IPW-O-X", "DoublyRobust-O-X",
                  "Hajek-O-X"]:
            bestL = max((surf[m][KMGK][L], L) for L in Lgrid)[1]
            stats_tr[(LBL2[m], "kmz")] = (nzt(KM_TR["surface"]["1.0"][m][KMGK][bestL]), 0.0)
        for m, d_ in XXM.items():
            bl = max(d_, key=d_.get)
            pv = [nzt(v) for v in KM_XXTR["train"][m][bl].values()]
            stats_tr[(m.replace("DoublyRobust", "DR"), "kmz")] = (
                float(np.mean(pv)), float(np.std(pv)))
        hv = [nzt(v) for v in KM_HCURV[KMGK]["values_train"]]
        stats_tr[("Hess (paper)", "kmz")] = (float(np.mean(hv)), float(np.std(hv)))
        stats_tr[("Kallus (paper)", "kmz")] = (nzt(KM_KALTR[KMGK]["train_mean"]), 0.0)
        chart_tr = barchart(
            stats_tr, "KMZ at the matched Gamma* = 4.4817 -- TRAIN (in-sample) value, "
            "same cells", methods=[m for m in ordered if any(k[0] == m for k in stats_tr)],
            series=["kmz"], colors={"kmz": "#6b4a2d"}, labels=lambda k: "matched Gamma*",
            note="same policies as the test chart (test-best L per method), valued on "
                 "their own training draws; 0 = best constant, 1 = oracle, both on train")
        _tg = ('onchange="document.getElementById(\'kbc_test\').style.display='
               "this.value=='test'?'':'none';document.getElementById('kbc_train')"
               ".style.display=this.value=='train'?'':'none';\"")
        chart = ('<div><div style="margin:4px 0 0 58px;font-size:13px">'
                 '<span class="hsub" style="margin-right:6px">evaluate on:</span>'
                 '<label style="cursor:pointer;margin-right:8px"><input type="radio" '
                 'name="kbcmode" value="test" checked %s> test</label>'
                 '<label style="cursor:pointer"><input type="radio" name="kbcmode" '
                 'value="train" %s> train (in-sample)</label></div>'
                 '<div id="kbc_test">%s</div>'
                 '<div id="kbc_train" style="display:none">%s</div></div>'
                 % (_tg, _tg, chart, chart_tr))
    return (R, main_tbl, tm_tbl, st_tbl, ce_tbl, cap_tbl, kb_tbl, chart, sweep_tbl,
            sweep_lines, surf3d, polfig)


(_KMZR, KMZ_MAIN, KMZ_TM, KMZ_ST, KMZ_CE, KMZ_CAP, KMZ_KB, KMZ_CHART,
 KMZ_SWEEP, KMZ_SWEEPCH, KMZ_3D, KMZ_POL) = _kmz_tab()


# ------------------------------------------------ rational-DM DGP tab
# nocouple config (a=2, alpha=0, delta=1, beta0=2, Gamma*=5), n=400, 5 seeds; the solver's
# Gamma swept over {1,2,3,5*,8,15,50} x L {inf..0.5} x c_eps {1,1.5,2}; X-X on the full L
# grid (Gamma-free); paper Kallus + paper Hess per Gamma. Hess-kNN deliberately EXCLUDED.
RAT_GK = ["1", "2", "3", "5", "8", "15", "50"]
RAT_MATCH = "5"
RAT_CES = ["1", "1.5", "2"]
RAT_CEOPT = [("1", "c_eps = 1.0 (tight)"), ("1.5", "c_eps = 1.5"), ("2", "c_eps = 2.0")]
RAT_OX = ["IPW-O-X", "DoublyRobust-O-X", "Hajek-O-X"]
RAT_OW = ["IPW-O-W", "DoublyRobust-O-W", "Hajek-O-W"]
RAT_LBL = {"IPW-O-X": "IPW-O-X", "DoublyRobust-O-X": "DR-O-X", "Hajek-O-X": "Hajek-O-X",
           "IPW-O-W": "IPW-O-W", "DoublyRobust-O-W": "DR-O-W", "Hajek-O-W": "Hajek-O-W"}

RAT_MATH = "".join([
    M(r"x \sim \mathrm{Unif}[-1,1], \qquad S = \pm 1 \text{ with prob } \tfrac12, "
      r"\quad S \perp x \quad\text{(the DM's private prognostic signal)}"),
    M(r"e(x,S) = \Pr(T{=}1 \mid x, S) = \sigma\!\big(2x + \tfrac12 \ln(5)\, S\big) "
      r"\quad\text{-- treatment probability RISES in } x \text{ and } S"),
    M(r"Y_0 = 2S + \varepsilon_0, \qquad Y_1 = Y_0 + 2x + S + \varepsilon_1, \qquad "
      r"\varepsilon \sim \mathcal{N}(0, 0.6^2) \ \Rightarrow\ \tau(x,S) = 2x + S"),
    M(r"\Rightarrow\ \frac{e(x,+1)/(1-e(x,+1))}{e(x,-1)/(1-e(x,-1))} "
      r"= \frac{\exp(2x + \tfrac12\ln 5)}{\exp(2x - \tfrac12\ln 5)} "
      r"= e^{\ln 5} = 5 \quad\text{at every } x \text{ (the } 2x \text{ cancels): } "
      r"\Gamma^{\!*} = 5 \text{ exactly, no clipping}"),
    M(r"S \perp x,\ E[S \mid x] = 0 \ \Rightarrow\ E[\tau \mid x] = 2x: "
      r"\text{ the } x\text{-measurable oracle treats } x > 0"),
])


def linefig(title, series, xlab, ylab, W=880, H=340, xlo=-1.0, xhi=1.0, note="",
            vline=None, hline=None):
    """Small static multi-line SVG. series: [{x, y, col, dsh, lbl, w}]."""
    pL, pR, pT, pB = 56, 168, 30, 46
    ys = [v for s in series for v in s["y"]]
    ylo, yhi = min(ys), max(ys)
    pad = 0.08 * (yhi - ylo or 1.0); ylo -= pad; yhi += pad
    X_ = lambda x: pL + (x - xlo) / (xhi - xlo) * (W - pL - pR)
    Y_ = lambda v: pT + (yhi - v) / (yhi - ylo) * (H - pT - pB)
    pp = ['<svg viewBox="0 0 %d %d" class="chart">' % (W, H),
          '<text x="%d" y="17" class="ct">%s</text>' % (pL, title)]
    step = max(0.25, round((yhi - ylo) / 5 / 0.25) * 0.25)
    t = np.ceil(ylo / step) * step
    while t <= yhi:
        pp.append('<line x1="%d" y1="%.1f" x2="%d" y2="%.1f" class="grid"/>'
                  % (pL, Y_(t), W - pR, Y_(t)))
        pp.append('<text x="%d" y="%.1f" class="tk" text-anchor="end">%g</text>'
                  % (pL - 5, Y_(t) + 3.5, round(t, 2)))
        t += step
    for x in np.arange(np.ceil(xlo * 2) / 2, xhi + 1e-9, 0.5):
        pp.append('<text x="%.1f" y="%d" class="tk" text-anchor="middle">%g</text>'
                  % (X_(x), H - pB + 15, x))
    if vline is not None:
        pp.append('<line x1="%.1f" y1="%d" x2="%.1f" y2="%d" stroke="#888" '
                  'stroke-width="1.1" stroke-dasharray="3 3"/>'
                  % (X_(vline), pT, X_(vline), H - pB))
    if hline is not None:
        pp.append('<line x1="%d" y1="%.1f" x2="%d" y2="%.1f" stroke="#888" '
                  'stroke-width="1.1" stroke-dasharray="3 3"/>'
                  % (pL, Y_(hline), W - pR, Y_(hline)))
    ly = pT + 8
    for s in series:
        pts = " ".join("%.1f,%.1f" % (X_(a), Y_(b)) for a, b in zip(s["x"], s["y"]))
        pp.append('<polyline points="%s" fill="none" stroke="%s" stroke-width="%.1f"%s/>'
                  % (pts, s["col"], s.get("w", 2.0),
                     (' stroke-dasharray="%s"' % s["dsh"]) if s.get("dsh") else ""))
        lx = W - pR + 8
        pp.append('<line x1="%d" y1="%.1f" x2="%d" y2="%.1f" stroke="%s" '
                  'stroke-width="%.1f"%s/>'
                  % (lx, ly - 3, lx + 26, ly - 3, s["col"], s.get("w", 2.0),
                     (' stroke-dasharray="%s"' % s["dsh"]) if s.get("dsh") else ""))
        pp.append('<text x="%d" y="%.1f" class="tk" text-anchor="start">%s</text>'
                  % (lx + 30, ly, s["lbl"]))
        ly += 15
    pp.append('<line x1="%d" y1="%d" x2="%d" y2="%d" class="ax"/>' % (pL, H - pB, W - pR, H - pB))
    pp.append('<line x1="%d" y1="%d" x2="%d" y2="%d" class="ax"/>' % (pL, pT, pL, H - pB))
    pp.append('<text x="%d" y="%d" class="al" text-anchor="middle">%s</text>'
              % ((pL + W - pR) // 2, H - 6, xlab))
    ym = (pT + H - pB) // 2
    pp.append('<text x="14" y="%d" class="al" text-anchor="middle" '
              'transform="rotate(-90 14 %d)">%s</text>' % (ym, ym, ylab))
    pp.append('</svg>')
    cap = ('<figcaption class="muted" style="margin-left:56px">%s</figcaption>' % note
           if note else "")
    return '<figure class="fig"><div class="scrollx chartbox">%s</div>%s</figure>' % (
        "".join(pp), cap)


def _rat_dgp_figs():
    """Static explainer figures for the nocouple DGP, straight from its closed forms
    (plus one numeric panel: the APPARENT cate a naive analyst measures)."""
    xs = np.linspace(-1, 1, 201)
    sig = lambda z: 1.0 / (1.0 + np.exp(-z))
    SP, SM, GREY = "#b0620b", "#2b6cb0", "#666"
    ep, em = sig(2 * xs + 0.5 * np.log(5)), sig(2 * xs - 0.5 * np.log(5))
    f1 = linefig(
        "Historical treatment probability  e(x, S) = sigma(2x + (1/2)ln(5) S)",
        [{"x": xs, "y": ep, "col": SP, "lbl": "S = +1"},
         {"x": xs, "y": em, "col": SM, "lbl": "S = -1"},
         {"x": xs, "y": (ep + em) / 2, "col": GREY, "dsh": "5 4",
          "lbl": "marginal e(x) (what the analyst can estimate)"}],
        "x", "P(T = 1 | x, S)", hline=0.5,
        note="Treatment probability RISES in x and in S -- the decision maker treats "
             "more where treating helps more (rational). The S-odds ratio is exactly 5 "
             "at every x, and overlap holds without clipping (e in [0.14, 0.86]).")
    f2 = linefig(
        "Expected outcome under each arm  (Y0 = 2S,  Y1 = Y0 + 2x + S)",
        [{"x": xs, "y": 2 * xs + 3, "col": SP, "lbl": "E[Y1 | x, S=+1] = 2x + 3"},
         {"x": xs, "y": 0 * xs + 2, "col": SP, "dsh": "6 4", "lbl": "E[Y0 | x, S=+1] = 2"},
         {"x": xs, "y": 2 * xs - 3, "col": SM, "lbl": "E[Y1 | x, S=-1] = 2x - 3"},
         {"x": xs, "y": 0 * xs - 2, "col": SM, "dsh": "6 4", "lbl": "E[Y0 | x, S=-1] = -2"}],
        "x", "E[Y_a | x, S]",
        note="Solid = treated arm, dashed = control; colour = the hidden signal S. The "
             "baseline outcome moves ONLY with S (prognostic level shift beta0 = 2), while "
             "the benefit of treating rises in x and in S.")
    f3 = linefig(
        "CATE  tau(x, S) = 2x + S,   E[tau | x] = 2x",
        [{"x": xs, "y": 2 * xs + 1, "col": SP, "lbl": "tau(x, S=+1) = 2x + 1"},
         {"x": xs, "y": 2 * xs - 1, "col": SM, "lbl": "tau(x, S=-1) = 2x - 1"},
         {"x": xs, "y": 2 * xs, "col": "#111", "w": 2.8, "lbl": "E[tau | x] = 2x (oracle)"}],
        "x", "treatment effect", vline=0.0, hline=0.0,
        note="The x-measurable oracle treats x > 0 (grey vertical line). The hidden S "
             "shifts the individual effect by +-1 around the 2x trend.")
    # numeric panel: what a naive analyst SEES vs the truth
    import importlib.util as _iu
    _sp = _iu.spec_from_file_location(
        "dgp_rat_fig", str(ROOT / "assets" / "exp_rational" / "dgp_rational.py"))
    _dm = _iu.module_from_spec(_sp); sys.modules["dgp_rat_fig"] = _dm
    _sp.loader.exec_module(_dm)
    dd = _dm.draw(400000, 7, _dm.Cfg(a=2.0, alpha=0.0, delta=1.0, beta0=2.0))
    edges = np.linspace(-1, 1, 41); mids = (edges[:-1] + edges[1:]) / 2
    app = np.full(len(mids), np.nan)
    for j in range(len(mids)):
        inb = (dd["x"] >= edges[j]) & (dd["x"] < edges[j + 1])
        t1, t0 = inb & (dd["T"] == 1), inb & (dd["T"] == 0)
        if t1.sum() > 30 and t0.sum() > 30:
            app[j] = dd["Y"][t1].mean() - dd["Y"][t0].mean()
    ok = ~np.isnan(app)
    f4 = linefig(
        "What the naive analyst measures:  E[Y | x, T=1] - E[Y | x, T=0]  vs the truth",
        [{"x": mids[ok], "y": app[ok], "col": "#c0392b",
          "lbl": "apparent CATE (naive, 400k draws)"},
         {"x": xs, "y": 2 * xs, "col": "#111", "dsh": "6 4", "lbl": "true E[tau | x] = 2x"}],
        "x", "difference in means", vline=0.0, hline=0.0,
        note="Selection on S biases the comparison UPWARD everywhere: treated units are "
             "S-rich (high Y1), controls are S-poor (low Y0). The apparent zero-crossing "
             "sits left of the true one, so a naive analyst over-treats -- this gap is "
             "exactly what the robust methods must survive, and its size is what "
             "Gamma* = 5 licenses the adversary to exploit.")
    return f1 + f2 + f3 + f4


RAT_DGPFIGS = _rat_dgp_figs()


def _rat_tab():
    import glob
    fs = sorted(glob.glob(str(ROOT / "assets" / "exp_rational" / "gsweep"
                              / "nocouple_s*_g*.json")))
    if len(fs) < 35:
        return None
    cells = [json.loads(Path(f).read_text()) for f in fs]
    seeds = sorted({c["seed"] for c in cells})
    Ls = cells[0]["Lgrid"]
    by = {(c["seed"], c["gamma"]): c for c in cells}
    g1 = [by[(s, "1")] for s in seeds]

    refs = {k: float(np.mean([c["refs"][k] for c in g1])) for k in ("oracle", "all", "never")}
    bc = max(refs["all"], refs["never"]); sc = refs["oracle"] - bc
    nz = lambda v: (v - bc) / sc

    # seed-averaged surfaces
    surf = {}                        # O-W: [m][ce][g][L]; O-X: [m][g][L]  (raw E[Y] means)
    for m in RAT_OX:
        surf[m] = {g: {L: float(np.mean([by[(s, g)]["grid"][m]["1"][L] for s in seeds]))
                       for L in Ls} for g in RAT_GK}
    for m in RAT_OW:
        surf[m] = {ce: {g: {L: float(np.mean([by[(s, g)]["grid"][m][ce][L] for s in seeds]))
                            for L in Ls} for g in RAT_GK} for ce in RAT_CES}
    xx = {m.replace("DoublyRobust", "DR"):
          {L: float(np.mean([c["xx"][m][L] for c in g1])) for L in Ls}
          for m in ("IPW-X-X", "DoublyRobust-X-X", "Direct-X-X")}
    naive = float(np.mean([c["naive"] for c in g1]))
    hess = {g: float(np.mean([by[(s, g)]["baselines"]["SharpHess"] for s in seeds]))
            for g in RAT_GK}
    kal = {g: float(np.mean([by[(s, g)]["baselines"]["Kallus"] for s in seeds]))
           for g in RAT_GK}

    # in-sample mirrors (present once the train-values rerun landed)
    has_tr = all("grid_train" in by[(s, g)] for s in seeds for g in RAT_GK)
    surf_tr = xx_tr = hess_tr = kal_tr = refs_tr = None
    naive_tr = None
    if has_tr:
        surf_tr = {}
        for m in RAT_OX:
            surf_tr[m] = {g: {L: float(np.mean([by[(s, g)]["grid_train"][m]["1"][L]
                                                for s in seeds])) for L in Ls}
                          for g in RAT_GK}
        for m in RAT_OW:
            surf_tr[m] = {ce: {g: {L: float(np.mean([by[(s, g)]["grid_train"][m][ce][L]
                                                     for s in seeds])) for L in Ls}
                               for g in RAT_GK} for ce in RAT_CES}
        xx_tr = {m.replace("DoublyRobust", "DR"):
                 {L: float(np.mean([c["xx_train"][m][L] for c in g1])) for L in Ls}
                 for m in ("IPW-X-X", "DoublyRobust-X-X", "Direct-X-X")}
        hess_tr = {g: float(np.mean([by[(s, g)]["baselines_train"]["SharpHess"]
                                     for s in seeds])) for g in RAT_GK}
        kal_tr = {g: float(np.mean([by[(s, g)]["baselines_train"]["Kallus"]
                                    for s in seeds])) for g in RAT_GK}
        naive_tr = float(np.mean([c["naive_train"] for c in g1]))
        refs_tr = {k: float(np.mean([c["refs_train"][k] for c in g1]))
                   for k in ("oracle", "all", "never")}

    # ---- matched-Gamma table + bar chart (normalised; best cell per method) ----
    rows, stats = [], {}
    for m in RAT_OW:
        best = max((surf[m][ce][RAT_MATCH][L], ce, L) for ce in RAT_CES for L in Ls)
        per = [by[(s, RAT_MATCH)]["grid"][m][best[1]][best[2]] for s in seeds]
        sd = float(np.std([(v - by[(s, RAT_MATCH)]["bc"]) / by[(s, RAT_MATCH)]["sc"]
                           for v, s in zip(per, seeds)]))
        lbl = RAT_LBL[m]
        stats[(lbl, "rat")] = (nz(best[0]), sd)
        rows.append((nz(best[0]), "<tr class='hl'><td class='l'>%s</td><td>%.3f</td>"
                     "<td>%.3f</td><td>%.3f</td><td>%s, L=%s</td></tr>"
                     % (lbl, best[0], nz(best[0]), sd, "c_eps=" + best[1], best[2])))
    for m in RAT_OX:
        best = max((surf[m][RAT_MATCH][L], L) for L in Ls)
        per = [by[(s, RAT_MATCH)]["grid"][m]["1"][best[1]] for s in seeds]
        sd = float(np.std([(v - by[(s, RAT_MATCH)]["bc"]) / by[(s, RAT_MATCH)]["sc"]
                           for v, s in zip(per, seeds)]))
        lbl = RAT_LBL[m]
        stats[(lbl, "rat")] = (nz(best[0]), sd)
        rows.append((nz(best[0]), "<tr><td class='l'>%s</td><td>%.3f</td><td>%.3f</td>"
                     "<td>%.3f</td><td>&mdash;, L=%s</td></tr>"
                     % (lbl, best[0], nz(best[0]), sd, best[1])))
    for lbl, dd in xx.items():
        best = max((v, L) for L, v in dd.items())
        per = [c["xx"][lbl.replace("DR", "DoublyRobust")][best[1]] for c in g1]
        sd = float(np.std([(v - c["bc"]) / c["sc"] for v, c in zip(per, g1)]))
        stats[(lbl, "rat")] = (nz(best[0]), sd)
        rows.append((nz(best[0]), "<tr><td class='l'>%s <span class='hsub'>(&Gamma;-free)"
                     "</span></td><td>%.3f</td><td>%.3f</td><td>%.3f</td><td>&mdash;, L=%s"
                     "</td></tr>" % (lbl, best[0], nz(best[0]), sd, best[1])))
    for lbl, dd in (("Hess (paper)", hess), ("Kallus (paper)", kal)):
        per = [by[(s, RAT_MATCH)]["baselines"][lbl.split(" ")[0].replace("Hess", "SharpHess")]
               for s in seeds]
        sd = float(np.std([(v - by[(s, RAT_MATCH)]["bc"]) / by[(s, RAT_MATCH)]["sc"]
                           for v, s in zip(per, seeds)]))
        stats[(lbl, "rat")] = (nz(dd[RAT_MATCH]), sd)
        rows.append((nz(dd[RAT_MATCH]), "<tr><td class='l'>%s</td><td>%.3f</td><td>%.3f</td>"
                     "<td>%.3f</td><td>&mdash;</td></tr>"
                     % (lbl, dd[RAT_MATCH], nz(dd[RAT_MATCH]), sd)))
    sd_n = float(np.std([(c["naive"] - c["bc"]) / c["sc"] for c in g1]))
    stats[("naive", "rat")] = (nz(naive), sd_n)
    rows.append((nz(naive), "<tr><td class='l'>naive (DR plug-in) <span class='hsub'>"
                 "(&Gamma;-free)</span></td><td>%.3f</td><td>%.3f</td><td>%.3f</td>"
                 "<td>&mdash;</td></tr>" % (naive, nz(naive), sd_n)))
    rows.sort(key=lambda t: -t[0])
    main_tbl = ("<div class='scrollx'><table class='dt'><tr><th class='l'>method</th>"
                "<th>E[Y] at &Gamma;* = 5</th><th>normalised</th><th>sd (norm)</th>"
                "<th class='l'>best cell</th></tr>" + "".join(r[1] for r in rows)
                + "</table></div>")
    ordered = [k[0] for k in sorted(stats, key=lambda k: -stats[k][0])]
    chart = barchart(stats, "Rational-DM DGP at the matched Gamma* = 5, n=400 -- "
                     "normalised value",
                     methods=ordered, series=["rat"], colors={"rat": "#2d4a2d"},
                     labels=lambda k: "matched Gamma*",
                     note="0 = best constant policy, 1 = the x-measurable oracle; "
                          "whisker = across-seed sd (%d seeds)" % len(seeds))
    if has_tr:
        bctr = max(refs_tr["all"], refs_tr["never"]); hrtr = refs_tr["oracle"] - bctr
        nzt = lambda v: (v - bctr) / hrtr
        stats_tr = {}
        for m in RAT_OW:
            best = max((surf[m][ce][RAT_MATCH][L], ce, L) for ce in RAT_CES for L in Ls)
            pv = [nzt(by[(s, RAT_MATCH)]["grid_train"][m][best[1]][best[2]])
                  for s in seeds]
            stats_tr[(RAT_LBL[m], "rat")] = (float(np.mean(pv)), float(np.std(pv)))
        for m in RAT_OX:
            best = max((surf[m][RAT_MATCH][L], L) for L in Ls)
            pv = [nzt(by[(s, RAT_MATCH)]["grid_train"][m]["1"][best[1]]) for s in seeds]
            stats_tr[(RAT_LBL[m], "rat")] = (float(np.mean(pv)), float(np.std(pv)))
        for lbl, dd in xx.items():
            bl = max(dd, key=dd.get)
            pv = [nzt(c["xx_train"][lbl.replace("DR", "DoublyRobust")][bl]) for c in g1]
            stats_tr[(lbl, "rat")] = (float(np.mean(pv)), float(np.std(pv)))
        for name, lbl in (("SharpHess", "Hess (paper)"), ("Kallus", "Kallus (paper)")):
            pv = [nzt(by[(s, RAT_MATCH)]["baselines_train"][name]) for s in seeds]
            stats_tr[(lbl, "rat")] = (float(np.mean(pv)), float(np.std(pv)))
        pv = [nzt(c["naive_train"]) for c in g1]
        stats_tr[("naive", "rat")] = (float(np.mean(pv)), float(np.std(pv)))
        chart_tr = barchart(
            stats_tr, "Rational-DM DGP at the matched Gamma* = 5 -- TRAIN (in-sample) "
            "value, same cells",
            methods=[m for m in ordered if any(k[0] == m for k in stats_tr)],
            series=["rat"], colors={"rat": "#6b4a2d"}, labels=lambda k: "matched Gamma*",
            note="same policies as the test chart (test-best cell per method), valued on "
                 "their own training draws; 0 = best constant, 1 = oracle, both on train")
        _tg = ('onchange="document.getElementById(\'rbc_test\').style.display='
               "this.value=='test'?'':'none';document.getElementById('rbc_train')"
               ".style.display=this.value=='train'?'':'none';\"")
        chart = ('<div><div style="margin:4px 0 0 58px;font-size:13px">'
                 '<span class="hsub" style="margin-right:6px">evaluate on:</span>'
                 '<label style="cursor:pointer;margin-right:8px"><input type="radio" '
                 'name="rbcmode" value="test" checked %s> test</label>'
                 '<label style="cursor:pointer"><input type="radio" name="rbcmode" '
                 'value="train" %s> train (in-sample)</label></div>'
                 '<div id="rbc_test">%s</div>'
                 '<div id="rbc_train" style="display:none">%s</div></div>'
                 % (_tg, _tg, chart, chart_tr))

    # ---- transport margin, paired at identical (c_eps, L), matched Gamma ----
    tm = {}
    for a_, b_ in (("IPW-O-W", "IPW-O-X"), ("DoublyRobust-O-W", "DoublyRobust-O-X")):
        dd = [nz(surf[a_][ce][RAT_MATCH][L]) - nz(surf[b_][RAT_MATCH][L])
              for ce in RAT_CES for L in Ls]
        tm[RAT_LBL[a_]] = (float(np.mean(dd)), float(np.max(dd)))

    # ---- per-Gamma sweep table (best cell per (method, Gamma), normalised) ----
    sw = []
    for m in RAT_OW:
        cells_ = [max(surf[m][ce][g][L] for ce in RAT_CES for L in Ls) for g in RAT_GK]
        sw.append((cells_[RAT_GK.index(RAT_MATCH)],
                   "<tr class='hl'><td class='l'>%s</td>%s</tr>"
                   % (RAT_LBL[m], "".join("<td>%.3f</td>" % nz(v) for v in cells_))))
    for m in RAT_OX:
        cells_ = [max(surf[m][g][L] for L in Ls) for g in RAT_GK]
        sw.append((cells_[RAT_GK.index(RAT_MATCH)],
                   "<tr><td class='l'>%s</td>%s</tr>"
                   % (RAT_LBL[m], "".join("<td>%.3f</td>" % nz(v) for v in cells_))))
    for lbl, dd in (("Hess (paper)", hess), ("Kallus (paper)", kal)):
        sw.append((dd[RAT_MATCH], "<tr><td class='l'>%s</td>%s</tr>"
                   % (lbl, "".join("<td>%.3f</td>" % nz(dd[g]) for g in RAT_GK))))
    sw.sort(key=lambda t: -t[0])
    hdrs = "".join("<th>&Gamma; = %s%s</th>" % (g, " (matched)" if g == RAT_MATCH else "")
                   for g in RAT_GK)
    sweep_tbl = ("<div class='scrollx'><table class='dt'><tr><th class='l'>method "
                 "(best cell per &Gamma;)</th>" + hdrs + "</tr>"
                 + "".join(r[1] for r in sw) + "</table></div>")

    # ---- interactive per-Gamma line chart (JS prefix "rw") ----
    series = []
    for m in RAT_OW[:2] + RAT_OX + [RAT_OW[2]]:
        if m.endswith("O-W"):
            series.append({"lbl": RAT_LBL[m], "kind": "ow", "data": surf[m]})
        else:
            series.append({"lbl": RAT_LBL[m], "kind": "ox", "data": surf[m]})
    for lbl, dd in xx.items():
        series.append({"lbl": lbl, "kind": "xx", "data": dd})
    series.append({"lbl": "Hess (paper)", "kind": "fixed", "data": hess})
    series.append({"lbl": "Kallus (paper)", "kind": "fixed", "data": kal})
    for s in series:
        s["col"], s["dsh"], s["shape"] = DSHMAP.get(s["lbl"], ("#000", "", "circle"))
    refs3_train = None
    if has_tr:
        for s in series:
            m = {v: k for k, v in RAT_LBL.items()}.get(s["lbl"])
            if s["kind"] == "ow":      s["dtr"] = surf_tr[m]
            elif s["kind"] == "ox":    s["dtr"] = surf_tr[m]
            elif s["kind"] == "xx":    s["dtr"] = xx_tr[s["lbl"]]
            elif s["lbl"].startswith("Hess"):   s["dtr"] = hess_tr
            elif s["lbl"].startswith("Kallus"): s["dtr"] = kal_tr
        refs3_train = [("oracle", "#111", refs_tr["oracle"], "5 4"),
                       ("all-treat", "#555", refs_tr["all"], "3 3"),
                       ("never-treat", "#555", refs_tr["never"], "3 3")]
    allv = ([v for m in RAT_OX for g in RAT_GK for v in surf[m][g].values()]
            + [v for m in RAT_OW for ce in RAT_CES for g in RAT_GK
               for v in surf[m][ce][g].values()]
            + [v for dd in xx.values() for v in dd.values()]
            + list(hess.values()) + list(kal.values()) + [naive] + list(refs.values()))
    if has_tr:
        allv += ([v for m in RAT_OX for g in RAT_GK for v in surf_tr[m][g].values()]
                 + [v for m in RAT_OW for ce in RAT_CES for g in RAT_GK
                    for v in surf_tr[m][ce][g].values()]
                 + [v for dd in xx_tr.values() for v in dd.values()]
                 + list(hess_tr.values()) + list(kal_tr.values())
                 + [naive_tr] + list(refs_tr.values()))
    ylo, yhi = min(allv) - 0.06, max(allv) + 0.06
    lines = ilines(
        "rw", RAT_GK, RAT_MATCH, "5*", series,
        [("oracle", "#111", refs["oracle"], "5 4"), ("all-treat", "#555", refs["all"], "3 3"),
         ("never-treat", "#555", refs["never"], "3 3")],
        Ls, RAT_CEOPT,
        "Rational-DM DGP: all methods across the solver Gamma (n=400, %d seeds; "
        "pick L and c_eps below)" % len(seeds),
        "(X-X and naive are &Gamma;-free; X-X was solved on the FULL L grid here; "
        "c<sub>&epsilon;</sub> scales the Wasserstein radius, so O-W only; Hess/Kallus "
        "have neither knob%s)"
        % ("; train = value of the same policy on its own training draw" if has_tr else ""),
        ylo, yhi,
        "Gamma assumed by the solver (* = matched Gamma) -- ordinal spacing",
        refs3_train=refs3_train)

    # ---- interactive 3-D surface (JS prefix "r3") ----
    S3 = {}
    S3tr = {} if has_tr else None
    zlo, zhi = 1e9, -1e9
    for ce in RAT_CES:
        S3[ce] = {}
        if S3tr is not None: S3tr[ce] = {}
        for m in RAT_OW[:2] + RAT_OX + [RAT_OW[2]]:
            src = surf[m][ce] if m.endswith("O-W") else surf[m]
            z = [[src[g][L] for L in Ls] for g in RAT_GK]
            S3[ce][RAT_LBL[m]] = {"g": RAT_GK, "z": z}
            for row in z:
                zlo = min(zlo, min(row)); zhi = max(zhi, max(row))
            if S3tr is not None:
                tsrc = surf_tr[m][ce] if m.endswith("O-W") else surf_tr[m]
                tz = [[tsrc[g][L] for L in Ls] for g in RAT_GK]
                S3tr[ce][RAT_LBL[m]] = {"g": RAT_GK, "z": tz}
                for row in tz:
                    zlo = min(zlo, min(row)); zhi = max(zhi, max(row))
    zlo, zhi = zlo - 0.05, zhi + 0.05
    s3d = isurf(
        "r3", S3, Ls, zlo, zhi, RAT_CEOPT,
        [RAT_LBL[m] for m in RAT_OW[:2] + RAT_OX + [RAT_OW[2]]],
        RAT_MATCH, "5*",
        "drag the plot to rotate; colour = height; X-X / Hess / Kallus have no "
        "(&Gamma;, L) surface; every &Gamma; was solved at every c<sub>&epsilon;</sub> here",
        s3_train=S3tr)

    # ---- interactive policy plot (present only once the with-policies rerun landed) ----
    polfig = None
    c_any = by[(seeds[0], "1")]
    if "policies" in c_any:
        skeys = [str(s) for s in seeds]
        has_lrn = all("policies_learned" in by[(s, g)] for s in seeds for g in RAT_GK)
        methods = {}
        for m in RAT_OW:
            data = {}
            lrn = {}
            for ce in RAT_CES:
                data[ce] = {g: {L: {str(s): by[(s, g)]["policies"][m][ce][L]
                                    for s in seeds
                                    if L in by[(s, g)].get("policies", {})
                                                      .get(m, {}).get(ce, {})}
                                for L in Ls} for g in RAT_GK}
                if has_lrn:
                    lrn[ce] = {g: {L: {str(s): by[(s, g)]["policies_learned"][m][ce][L]
                                       for s in seeds
                                       if L in by[(s, g)]["policies_learned"]
                                                          .get(m, {}).get(ce, {})}
                                   for L in Ls} for g in RAT_GK}
            methods[RAT_LBL[m]] = {"kind": "ow", "data": data}
            if has_lrn: methods[RAT_LBL[m]]["lrn"] = lrn
        for m in RAT_OX:
            methods[RAT_LBL[m]] = {"kind": "ox", "data": {
                g: {L: {str(s): by[(s, g)]["policies"][m]["1"][L] for s in seeds
                        if L in by[(s, g)].get("policies", {}).get(m, {}).get("1", {})}
                    for L in Ls} for g in RAT_GK}}
            if has_lrn:
                methods[RAT_LBL[m]]["lrn"] = {
                    g: {L: {str(s): by[(s, g)]["policies_learned"][m]["1"][L]
                            for s in seeds
                            if L in by[(s, g)]["policies_learned"].get(m, {}).get("1", {})}
                        for L in Ls} for g in RAT_GK}
        for m in ("IPW-X-X", "DoublyRobust-X-X", "Direct-X-X"):
            e = {"kind": "xx", "data": {
                L: {str(c["seed"]): c["xx_policies"][m][L] for c in g1
                    if L in c.get("xx_policies", {}).get(m, {})} for L in Ls}}
            if has_lrn and all("xx_learned" in c for c in g1):
                e["lrn"] = {L: {str(c["seed"]): c["xx_learned"][m][L] for c in g1
                                if L in c["xx_learned"].get(m, {})} for L in Ls}
            methods[m.replace("DoublyRobust", "DR")] = e
        for name, lbl in (("SharpHess", "Hess (paper)"), ("Kallus", "Kallus (paper)")):
            methods[lbl] = {"kind": "fixed", "par": True, "data": {
                g: {str(s): by[(s, g)]["baseline_policies"][name] for s in seeds
                    if name in by[(s, g)].get("baseline_policies", {})} for g in RAT_GK}}
        methods["naive (DR)"] = {"kind": "flat",
                                 "data": {str(c["seed"]): c["naive_policy"] for c in g1
                                          if "naive_policy" in c}}
        if has_lrn and all("naive_learned" in c for c in g1):
            methods["naive (DR)"]["lrn"] = {str(c["seed"]): c["naive_learned"] for c in g1}
        for lbl in methods:
            methods[lbl]["col"] = DSHMAP.get(lbl, ("#334155",))[0]
        pdata = {"grid": c_any["policy_grid"], "xlo": -1.0, "xhi": 1.0,
                 "oracle": c_any["oracle_policy"], "methods": methods}
        if has_lrn:
            pdata["lx"] = {str(s): by[(s, "1")]["train_x"] for s in seeds}
        order = [m for m in ["IPW-O-W", "DR-O-W", "Hajek-O-W", "IPW-O-X", "DR-O-X",
                             "Hajek-O-X", "IPW-X-X", "DR-X-X", "Direct-X-X",
                             "Hess (paper)", "Kallus (paper)", "naive (DR)"]
                 if m in methods]
        polfig = ipol(
            "rp", pdata, order, RAT_GK, RAT_MATCH, Ls, RAT_CEOPT, skeys,
            "Rational-DM DGP: deployed policy pi(x) (n=400; oracle treats x > 0)",
            "&Gamma; applies to O-X/O-W/Hess/Kallus; L to the LP methods (X-X included "
            "-- full L grid); c<sub>&epsilon;</sub> moves O-W only; Hess/Kallus/naive "
            "ignore L and c<sub>&epsilon;</sub>. <b>At L = inf the LP is per-point "
            "separable and positive reweighting cannot flip a sign, so the IPW/Hajek "
            "family returns one identical policy at every &Gamma;</b> &mdash; pick "
            "L &le; 10 to see the methods separate.")

    return {"refs": refs, "chart": chart, "main_tbl": main_tbl, "tm": tm,
            "sweep_tbl": sweep_tbl, "lines": lines, "s3d": s3d,
            "seeds": len(seeds), "naive": naive, "polfig": polfig}


_RAT = _rat_tab()
if _RAT:
    RAT_CHART, RAT_MAIN, RAT_LINES, RAT_3DFIG = (_RAT["chart"], _RAT["main_tbl"],
                                                 _RAT["lines"], _RAT["s3d"])
    RAT_SWEEP = _RAT["sweep_tbl"]
    RAT_POL = _RAT["polfig"] or ("<p class='muted'>Policy curves for this DGP are being "
                                 "recomputed with curve storage; rebuild once the "
                                 "with-policies sweep lands.</p>")
    RAT_REFS = ("oracle %.3f, all-treat %.3f, never-treat %.3f"
                % (_RAT["refs"]["oracle"], _RAT["refs"]["all"], _RAT["refs"]["never"]))
    RAT_TM = ("Transport margin, paired at identical (c<sub>&epsilon;</sub>, L) at the "
              "matched &Gamma;: IPW <b>%+.3f</b> mean / %+.3f max; DR <b>%+.3f</b> mean / "
              "%+.3f max (normalised units)."
              % (_RAT["tm"]["IPW-O-W"][0], _RAT["tm"]["IPW-O-W"][1],
                 _RAT["tm"]["DR-O-W"][0], _RAT["tm"]["DR-O-W"][1]))
else:
    RAT_CHART = RAT_MAIN = RAT_LINES = RAT_3DFIG = RAT_SWEEP = RAT_POL = (
        "<p class='muted'>The &Gamma; sweep for this DGP is still running on SLURM; "
        "rebuild this report when assets/exp_rational/gsweep/ holds all 35 cells.</p>")
    RAT_REFS = ""; RAT_TM = ""


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
<div class="tb" id="tb-rational" onclick="showTab('rational')">rational</div>
<div class="tb" id="tb-readme" onclick="showTab('readme')">read me</div>
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
varies, the DGP stays fixed; X-X are &Gamma;-free flat lines; best L per point</span></h2>
{KMZ_SWEEPCH}
{KMZ_SWEEP}

<h3>&Gamma; &times; L &times; E[Y] <span class="hsub">&mdash; the same surface in 3-D;
drag to rotate</span></h3>
{KMZ_3D}

<h2>Policy curves <span class="hsub">&mdash; the deployed &pi;(x) itself, per
(method, &Gamma;, L, c<sub>&epsilon;</sub>, seed); X is one-dimensional here, so the whole
policy is visible</span></h2>
{KMZ_POL}
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

<div id="tab-rational" class="tabpane">
<h2>The rational-DM DGP <span class="hsub">&mdash; the synthetic benchmark whose decision
maker is defensible</span></h2>
<p>gstar's construction had a flaw a referee would find: the benefit of treatment rises in
<i>x</i> while the historical propensity falls in <i>x</i>, so the units who benefit most were
treated least &mdash; an <i>irrational</i> decision maker. Here the decision maker observes a
private prognostic signal <i>S</i> that the analyst does not, and acts rationally on
<b>both</b> <i>x</i> and <i>S</i>: treatment probability and treatment benefit both rise in
both. The unobserved confounder is now a reason the decision maker was <i>right</i>. This is
the recommended <span class="mono">nocouple</span> configuration (<i>S</i> a fair coin,
independent of <i>x</i>).</p>
{RAT_MATH}
<p class="muted">&Gamma;* = 5 is exact by algebra with no clipping (odds-ratio error
&lt; 10<sup>&minus;10</sup>, asserted by <span class="mono">dgp_rational.check()</span> in
every run), overlap stays in [0.14, 0.86], and corr(e, benefit) &gt; 0 is the programmatic
rationality check. n = 400 train / 4000 test, {_RAT["seeds"]} seeds ({RAT_REFS}). The Hess k-NN diagnostic
arm is excluded throughout; Hess and Kallus are the authors' own code.</p>

<h2>The DGP in pictures <span class="hsub">&mdash; assignment, outcomes, effect, and the
bias a naive analyst inherits</span></h2>
{RAT_DGPFIGS}

<h2>All methods at the matched &Gamma;* = 5</h2>
{RAT_CHART}
{RAT_MAIN}
<p class="muted">{RAT_TM} The O-W&ndash;vs&ndash;X-X comparison is at matched policy class:
X-X was swept over the same Lipschitz grid and still peaks below 0.1 normalised.</p>

<h2>Across &Gamma; <span class="hsub">&mdash; the solver's &Gamma; varies, the DGP stays
fixed at &Gamma;* = 5; X-X and naive are &Gamma;-free; best cell per point</span></h2>
{RAT_LINES}
{RAT_SWEEP}

<h3>&Gamma; &times; L &times; E[Y] <span class="hsub">&mdash; the same surface in 3-D;
drag to rotate</span></h3>
{RAT_3DFIG}

<h2>Policy curves <span class="hsub">&mdash; the deployed &pi;(x) itself, per
(method, &Gamma;, L, c<sub>&epsilon;</sub>, seed); the oracle treats x &gt; 0</span></h2>
{RAT_POL}
</div>

<div id="tab-readme" class="tabpane">
<h2>Read me <span class="hsub">&mdash; how the plots encode the methods</span></h2>
<p>Every method is named <b>Estimator&ndash;UncertaintySet</b>. The estimator says how the policy
value is estimated from the observational data; the uncertainty set says which set of inverse
propensity weights the adversary may choose from. In the line charts these two axes are encoded
independently: <b>colour = estimator</b>, <b>line style and marker shape = uncertainty set</b>.
So all IPW curves share one colour regardless of set, and all O-W curves share one line
style/marker regardless of estimator.</p>

<h2>Colour &mdash; the estimator</h2>
<table class="dt">
<tr><th class="l">estimator</th><th class="l">colour</th><th class="l">sample</th></tr>
<tr><td class="l">IPW</td><td class="l">blue</td>
<td class="l"><svg width="34" height="14" viewBox="0 0 34 14"><line x1="2" y1="7" x2="32" y2="7" stroke="#1f77b4" stroke-width="3"/></svg></td></tr>
<tr><td class="l">DoublyRobust (DR)</td><td class="l">red</td>
<td class="l"><svg width="34" height="14" viewBox="0 0 34 14"><line x1="2" y1="7" x2="32" y2="7" stroke="#d62728" stroke-width="3"/></svg></td></tr>
<tr><td class="l">Hajek</td><td class="l">green</td>
<td class="l"><svg width="34" height="14" viewBox="0 0 34 14"><line x1="2" y1="7" x2="32" y2="7" stroke="#0a7d33" stroke-width="3"/></svg></td></tr>
<tr><td class="l">Direct</td><td class="l">slate grey</td>
<td class="l"><svg width="34" height="14" viewBox="0 0 34 14"><line x1="2" y1="7" x2="32" y2="7" stroke="#64748b" stroke-width="3"/></svg></td></tr>
<tr><td class="l">Hess (published baseline, authors' code)</td><td class="l">purple</td>
<td class="l"><svg width="34" height="14" viewBox="0 0 34 14"><line x1="2" y1="7" x2="32" y2="7" stroke="#7d1f6a" stroke-width="3"/></svg></td></tr>
<tr><td class="l">Kallus (published baseline, authors' code)</td><td class="l">brown</td>
<td class="l"><svg width="34" height="14" viewBox="0 0 34 14"><line x1="2" y1="7" x2="32" y2="7" stroke="#8c564b" stroke-width="3"/></svg></td></tr>
</table>

<h2>Line style + marker &mdash; the uncertainty set</h2>
<table class="dt">
<tr><th class="l">set</th><th class="l">meaning</th><th class="l">style</th><th class="l">sample (shown here in IPW blue)</th></tr>
<tr><td class="l">O-W</td><td class="l">odds box &cap; Wasserstein ball (the headline method)</td>
<td class="l">solid line, filled circle</td>
<td class="l"><svg width="40" height="14" viewBox="0 0 40 14"><line x1="2" y1="7" x2="38" y2="7" stroke="#1f77b4" stroke-width="2.4"/><circle cx="20" cy="7" r="3" fill="#1f77b4"/></svg></td></tr>
<tr><td class="l">O-X</td><td class="l">odds box only (box twin of O-W)</td>
<td class="l">dashed line, open square</td>
<td class="l"><svg width="40" height="14" viewBox="0 0 40 14"><line x1="2" y1="7" x2="38" y2="7" stroke="#1f77b4" stroke-width="2.4" stroke-dasharray="6 4"/><rect x="17.2" y="4.2" width="5.6" height="5.6" fill="#fff" stroke="#1f77b4" stroke-width="1.6"/></svg></td></tr>
<tr><td class="l">X-X</td><td class="l">non-robust point estimate (&Gamma;-free)</td>
<td class="l">dotted line, open triangle</td>
<td class="l"><svg width="40" height="14" viewBox="0 0 40 14"><line x1="2" y1="7" x2="38" y2="7" stroke="#1f77b4" stroke-width="2.4" stroke-dasharray="2 3"/><path d="M 20 3.6 L 16.8 10 L 23.2 10 Z" fill="#fff" stroke="#1f77b4" stroke-width="1.6"/></svg></td></tr>
<tr><td class="l">&mdash; (baselines)</td><td class="l">Hess / Kallus have their own published sets</td>
<td class="l">solid line; filled diamond (Hess), filled triangle (Kallus)</td>
<td class="l"><svg width="64" height="14" viewBox="0 0 64 14"><line x1="2" y1="7" x2="28" y2="7" stroke="#7d1f6a" stroke-width="2.4"/><rect x="12.4" y="4.4" width="5.2" height="5.2" fill="#7d1f6a" transform="rotate(45 15 7)"/><line x1="36" y1="7" x2="62" y2="7" stroke="#8c564b" stroke-width="2.4"/><path d="M 49 3.6 L 45.8 9.6 L 52.2 9.6 Z" fill="#8c564b"/></svg></td></tr>
</table>

<p class="muted">This code is used in the per-&Gamma; line charts on the KMZ and rational
tabs (use their method menus to show or hide individual methods). Reference lines (oracle, all-treat, never-treat)
are thin black/grey dashes and the matched &Gamma; is marked by a grey vertical dashed line.
Bar charts elsewhere in the report list methods explicitly on their axes, so they keep their own
per-panel colours.</p>
</div>

<script>
function showTab(id){{
  for (const t of ['semi','rct','kmz','rational','readme']){{
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
