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
    return ('<div style="text-align:center;overflow-x:auto;margin:8px 0">%s</div>'
            % l2m(tex))

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
    M(r"T_i \sim \mathrm{Bernoulli}\big(1-\pi^{0}(X_i,U_i)\big),\qquad "
      r"Y_i \;=\; T_i Y_i^{1} + (1-T_i) Y_i^{0}"),
    M(r"\frac{\pi^{0}(x,+1)}{1-\pi^{0}(x,+1)} \Big/ \frac{\pi^{0}(x,-1)}{1-\pi^{0}(x,-1)}"
      r" \;=\; e^{2\gamma} \quad \forall x \qquad\Longrightarrow\qquad "
      r"\Gamma^{\!\star} = e^{2\gamma}"),
])

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

rows = []
for d in SEMI:
    rows.append("<tr class='dsrow'><td colspan='%d'>%s</td></tr>" % (len(COLS) + 3, esc(d)))
    for s in SEMI[d]:
        r, rr = s["rows"], s["rows_rat"]
        vals, sds = {}, {}
        for m in COLS:
            if m in r: vals[m] = r[m]["mean"]; sds[m] = r[m]["sd"]
        rows.append("<tr><td class='l'>&gamma; = %.1f</td><td class='ref'>%.2f</td>%s</tr>"
                    % (s["gamma"], s["Gamma"], cells(vals, COLS, sds)))
semi_tbl = ("<div class='scrollx'><table class='dt'><tr><th class='l'>&nbsp;</th><th>&Gamma;</th>"
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
    pool.append("<tr><td class='l'>&gamma; = %.1f</td><td class='ref'>%.2f</td>%s</tr>"
                % (g, float(np.exp(2 * g)), cells(agg, POOL)))
pool_tbl = ("<div class='scrollx'><table class='dt'><tr><th class='l'>&nbsp;</th><th>&Gamma;</th>"
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


def barchart(stats, title, W=1240, H=430):
    """stats[(method, gamma)] = (mean, sd). Grouped bars with sd whiskers.

    Geometry is derived from len(GAM) rather than assumed: with the gamma grid extended to five
    values the old fixed bar width (0.24 of a slot) spanned 1.2 slots and neighbouring method
    groups ran into each other. Bars now fill a fixed FRACTION of each slot and are centred on it,
    so adding or removing a gamma re-flows cleanly.
    """
    pL, pR, pT, pB = 58, 16, 34, 74
    vals = ([m for (m, sd) in stats.values()] + [m + sd for (m, sd) in stats.values()]
            + [m - sd for (m, sd) in stats.values()])
    lo, hi = min(vals + [0.0]), max(vals + [0.0])
    pad = 0.10 * (hi - lo + 1e-9); lo -= pad; hi += pad
    Y = lambda v: H - pB - (v - lo) / (hi - lo + 1e-12) * (H - pT - pB)
    n, ng = len(BAR_METHODS), len(GAM)
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
    for i, m in enumerate(BAR_METHODS):
        cx = pL + slot * (i + 0.5)
        for j, g in enumerate(GAM):
            if (m, g) not in stats: continue
            mu, sd = stats[(m, g)]
            x = cx - gw / 2 + j * bw
            yt, hgt = Y(max(mu, 0.0)), abs(y0 - Y(mu))
            p.append('<rect x="%.1f" y="%.1f" width="%.1f" height="%.1f" fill="%s" rx="1"/>'
                     % (x + 0.6, yt, max(bw - 1.2, 1.0), max(hgt, 0.8), GCOL[g]))
            if sd > 0:
                xm, cap = x + bw / 2, min(bw * 0.30, 3.2)
                p.append('<line x1="%.1f" y1="%.1f" x2="%.1f" y2="%.1f" class="whisk"/>'
                         % (xm, Y(mu - sd), xm, Y(mu + sd)))
                for yy in (Y(mu - sd), Y(mu + sd)):
                    p.append('<line x1="%.1f" y1="%.1f" x2="%.1f" y2="%.1f" class="whisk"/>'
                             % (xm - cap, yy, xm + cap, yy))
        p.append('<text x="%.1f" y="%d" class="tk" text-anchor="end" '
                 'transform="rotate(-35 %.1f %d)">%s</text>'
                 % (cx, H - pB + 17, cx, H - pB + 17, esc(BAR_SHORT.get(m, m))))
    p.append('<line x1="%d" y1="%.1f" x2="%d" y2="%.1f" class="ax"/>' % (pL, y0, W - pR, y0))
    p.append('<line x1="%d" y1="%d" x2="%d" y2="%d" class="ax"/>' % (pL, pT, pL, H - pB))
    ymid = (pT + H - pB) // 2
    p.append('<text x="15" y="%d" class="al" text-anchor="middle" transform="rotate(-90 15 %d)">'
             'normalised value</text>' % (ymid, ymid))
    p.append('</svg>')
    leg = "".join('<span class="li"><span class="sw" style="background:%s"></span>'
                  '&gamma; = %g</span>' % (GCOL[g], g) for g in GAM)
    return ('<figure class="fig"><div class="scrollx chartbox">%s</div><div class="leg">%s'
            '<span class="li">whisker = &plusmn;1 sd across the 5 DGP draws</span></div></figure>'
            % ("".join(p), leg))


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

CHARTS = barchart(_stats_for(list(SEMI)),
                  "ALL FIVE DATASETS POOLED -- normalised value by method and gamma")
CHARTS += "".join(barchart(_stats_for([d]), "%s -- normalised value by method and gamma" % d)
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
nseeds = SEMI[SEMI_DS[0]][0]["n_seeds"] if SEMI else 0
html = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Semi-synthetic policy-learning experiments</title>
<style>{CSS}{EXTRA}{NOTECSS}{STAMPCSS}{CHARTCSS}</style></head><body>
<div class="hero" style="background:linear-gradient(135deg,#3b0764,#7e22ce)">
<h1>Semi-synthetic policy-learning experiments</h1>
<p>Two constructions, both giving real covariates a synthetic confounded assignment with a KNOWN
sensitivity parameter. <b>Higher is better</b> throughout.</p></div>

<div class="tabbar">
<div class="tb on" id="tb-semi" onclick="showTab('semi')">semi</div>
<div class="tb" id="tb-rct" onclick="showTab('rct')">RCT</div>
</div>

<div id="tab-semi" class="tabpane on">
{STAMP}
<h2>The construction</h2>
<p>Follows the data-generating procedure of <i>"Learning Risk Scores Robust to Unobserved
Confounders"</i> exactly for <i>X</i>, <i>U</i>, <i>Y</i><sup>0</sup> and <i>T</i>, and adds the
one thing policy learning needs and a risk score does not: a second potential outcome.</p>
{SEMI_MATH}
<p class="muted">The last line is the identity that makes the matched &Gamma; <b>exact</b> rather
than the interval [e<sup>&gamma;</sup>, e<sup>2&gamma;</sup>] the paper settles for: because
&pi;<sup>0</sup> is never clipped, the odds ratio between the two hidden states is
e<sup>2&gamma;</sup> at <i>every</i> x. Verified to ~1e&minus;13 on all five datasets.
Only &tau; and <i>Y</i><sup>1</sup> are ours; everything else is theirs.</p>
{SEMI_FLOW}
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
SharpHess now uses the paper's Table-5 nuisances: every head is a {{64,64,32}} ReLU net trained with
Adam at lr&nbsp;1e&minus;3, 300 epochs, batch&nbsp;64, early-stopping patience&nbsp;10. Checked
against population ground truth, the net is the <i>most accurate</i> propensity estimator
(RMSE&nbsp;0.037 vs k-NN's 0.070 and 0.125) &mdash; but its <i>truncated-mean</i> regressions are
10&ndash;17&times; worse (0.708 vs 0.041 at &gamma;&nbsp;=&nbsp;2), because the target
Y&middot;1{{Y&nbsp;&le;&nbsp;q(x)}} is discontinuous and Y&sup0; here is two-valued, so a smooth
regression averages away what an empirical k-NN quantile reproduces exactly. Those terms carry the
c&plusmn; weights that grow with &Gamma;, which is why the neural nuisances help at
&gamma;&nbsp;=&nbsp;0 and hurt when confounding is strong. Both arms are reported below; neither is
tuned to the outcome.
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

<h2>Nuisance estimator for SharpHess
<span class="hsub">&mdash; the paper's neural instantiation vs the k-NN one used before</span></h2>
{hess_tbl}

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
<p>Three datasets that already carry both potential outcomes, given the same treatment of a
synthetic confounded assignment with a known &Lambda; = 4.</p>
{RCT_MATH}
<p class="muted">Here <i>z<sup>A</sup></i> and <i>z<sup>B</sup></i> are the standardised
principal-direction indices of the two covariate groups, and <i>q</i><sub>99</sub> is the 99th
percentile, so about 1% of the mass is clipped to the boundary. As above, not clipping
<i>e</i> makes &Gamma;<sup>&#9733;</sup> = &Lambda; exact; the realised value measured back out of
the data is 4.000 on all three benchmarks.</p>
{RCT_FLOW}
<div class="scrollx"><table class="dt">
<tr><th class="l">benchmark</th><th>N</th><th>outcome</th><th>corr(x,S)</th><th>&Lambda; realised</th>
<th>best const</th><th>oracle</th><th>headroom</th></tr>
{rct_cons}</table></div>

<h2>Average test outcome at the matched &Gamma; = 4
<span class="hsub">&mdash; best (L, c<sub>&epsilon;</sub>) cell, mean &plusmn; sd over 5 seeds</span></h2>
{rct_tbl}

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

<script>
function showTab(id){{
  for (const t of ['semi','rct']){{
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
