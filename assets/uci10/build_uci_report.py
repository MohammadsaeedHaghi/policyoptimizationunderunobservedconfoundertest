#!/usr/bin/env python3
"""Build the self-contained HTML report for the 10-dataset UCI semi-synthetic campaign.

Two tabs:
  Benchmarks -- how each of the 10 datasets was turned into a confounded semi-synthetic problem
                (which columns became X, which became the hidden U, the realised correlations,
                the exact Lambda, and the oracle / naive / never / all-treat reference values).
  Results    -- the full table at matched Gamma and at the best cell, the win ledger, the
                transport margin, and one interactive E[Y]-vs-Gamma widget per dataset.

Usage: python3 assets/uci10/build_uci_report.py
"""
import sys, json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT / "semi experiments"))
from report_common import CSS, esc

RES = HERE / "results"
MATCHED = "4"
OX = ["IPW-O-X", "DoublyRobust-O-X", "Hajek-O-X"]
OW = ["IPW-O-W", "DoublyRobust-O-W"]          # Hajek-O-W dropped from the report
XX = ["IPW-X-X", "DoublyRobust-X-X", "Direct-X-X"]
ORDER = OX + OW + XX + ["SharpHess", "Kallus"]
SHORT = {"DoublyRobust-O-X": "DR-O-X", "DoublyRobust-O-W": "DR-O-W",
         "DoublyRobust-X-X": "DR-X-X", "SharpHess": "Hess"}

SUM = json.loads((HERE / "summary.json").read_text())

TABLE_CSS = """
table.rt{border-collapse:collapse;font-size:12.5px;margin:12px 0;width:100%}
table.rt th,table.rt td{border:1px solid #e3e6ea;padding:4px 7px;text-align:right;
  white-space:nowrap;font-variant-numeric:tabular-nums}
table.rt th{background:#f4f6f8;font-weight:600}
table.rt td.l,table.rt th.l{text-align:left}
table.rt td.ref{color:#111}
table.rt .sd{font-size:10.5px;opacity:.65;font-weight:400}
table.rt tr.sumrow td{border-top:2px solid #b9c0c8;background:#f7f9fb}
table.rt td.pos{color:#0a7d33}table.rt td.neg{color:#b3261e}
.scrollx{overflow-x:auto;max-width:100%}
.hsub{font-weight:400;font-size:.82rem;color:#667085}
"""


def fmt(v, d=3): return "%.*f" % (d, v) if v == v else "&mdash;"


# ------------------------------------------------------------------ construction flowchart
# Hand-laid SVG rather than a diagram library: the page must stay self-contained and ASCII-only.
# Colour encodes the KIND of step, which is the point of the figure -- what came from the real
# data (blue), what we chose (amber), what became ground truth (green), and what was synthesised
# (purple). Amber marks where a design decision enters.
FW, FH = 980, 730
PAL = {"real": ("#dbeafe", "#1e40af"), "choice": ("#fef3c7", "#92400e"),
       "truth": ("#dcfce7", "#166534"), "synth": ("#ede9fe", "#5b21b6"),
       "out": ("#f1f5f9", "#334155")}
_f = []


def box(x, y, w, h, kind, title, lines, mono=False):
    bg, fg = PAL[kind]
    _f.append('<rect x="%d" y="%d" width="%d" height="%d" rx="8" fill="%s" stroke="%s" '
              'stroke-width="1.5"/>' % (x, y, w, h, bg, fg))
    _f.append('<text x="%d" y="%d" font-size="13.5" font-weight="700" fill="%s">%s</text>'
              % (x + 12, y + 21, fg, title))
    for i, ln in enumerate(lines):
        _f.append('<text x="%d" y="%d" font-size="11.5" fill="#334155"%s>%s</text>'
                  % (x + 12, y + 39 + 15 * i, ' font-family="monospace"' if mono else '', ln))


def arrow(x1, y1, x2, y2, lab=""):
    _f.append('<line x1="%g" y1="%g" x2="%g" y2="%g" stroke="#94a3b8" stroke-width="1.8" '
              'marker-end="url(#ah)"/>' % (x1, y1, x2, y2))
    if lab:
        _f.append('<text x="%g" y="%g" font-size="10.5" fill="#64748b" text-anchor="middle">%s</text>'
                  % ((x1 + x2) / 2, (y1 + y2) / 2 - 5, lab))


def stage(y, n):
    _f.append('<text x="14" y="%d" font-size="11" font-weight="700" fill="#94a3b8">%s</text>' % (y, n))


_f.append('<defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" '
          'markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="#94a3b8"/></marker></defs>')

# The dataset is ONE table; stage 1 assigns each of its columns to a role. Drawing that as a
# left-to-right chain (dataset -> outcome -> treatment) was misleading, because it implied each box
# transformed the one before it. It is a partition, so it is drawn as a fork: one source, four
# roles, every column used for exactly one of them.
box(40, 30, 900, 66, "real", "UCI dataset  --  one table",
    ["N rows  x  K columns;  categoricals ordinal-encoded, rows with missing values dropped",
     "each of the K columns is then assigned to EXACTLY ONE of the four roles below"])
arrow(490, 96, 490, 110)
_f.append('<line x1="250" y1="110" x2="730" y2="110" stroke="#94a3b8" stroke-width="1.8"/>')
arrow(250, 110, 250, 134); arrow(730, 110, 730, 134)

stage(128, "1 . COLUMN ROLES")
box(40, 138, 420, 92, "real", "Columns that define the OUTCOME",
    ["Y  =  the outcome column itself",
     "         binary -> kept 0/1 ;   continuous -> standardised",
     "A  =  one binary column, SEARCHED, as pseudo-treatment"])
box(520, 138, 420, 92, "choice", "Columns that define the COVARIATES",
    ["X-group  ->  becomes the OBSERVED covariate",
     "U-group  ->  becomes the HIDDEN confounder",
     "the split is SEARCHED to maximise |corr(x, u)|"])

stage(252, "2 . SCALARISE")
box(520, 262, 420, 76, "real", "Reduce each group to one number per row",
    ["x  =  X-group index, rescaled to [-1, 1]     (OBSERVED)",
     "u  =  U-group index                                     (HIDDEN)"])
arrow(730, 230, 730, 260)

stage(352, "3 . GROUND TRUTH")
box(40, 362, 420, 80, "truth", "Outcome surface for each arm",
    ["mu_t(x, u)  =  E[ Y | x, u, A = t ],   t = 0, 1",
     "gradient boosting, fit on ALL real rows"])
box(520, 362, 420, 80, "truth", "Hidden binary confounder",
    ["S  =  sign( u - median u )   in   {-1, +1}",
     "never shown to the learner"])
arrow(730, 338, 730, 360)
_f.append('<line x1="730" y1="350" x2="250" y2="350" stroke="#94a3b8" stroke-width="1.8"/>')
arrow(250, 230, 250, 360)

stage(456, "4 . SYNTHETIC ASSIGNMENT")
box(40, 466, 420, 76, "synth", "Potential outcomes",
    ["Y0  ~  mu_0(x, u)", "Y1  ~  mu_1(x, u)"], mono=True)
box(520, 466, 420, 76, "synth", "Propensity -- exact by construction",
    ["e(x,S) = sigma( 1.2 x + 0.5 ln(Lambda) S ),  NO clipping",
     "=>  odds ratio  =  Lambda  =  4  at EVERY x"], mono=True)
arrow(250, 442, 250, 464); arrow(730, 442, 730, 464)

stage(556, "5 . OBSERVE")
box(40, 566, 900, 52, "synth", "What actually gets recorded",
    ["T  ~  Bern( e(x, S) )          Y_obs  =  Y_T"], mono=True)
arrow(250, 542, 250, 564); arrow(730, 542, 730, 564)

stage(632, "6 . SPLIT")
box(40, 642, 420, 66, "out", "TRAIN  --  300 rows",
    ["learner sees only (x, T, Y_T)", "plus an ESTIMATED propensity"])
box(520, 642, 420, 66, "out", "TEST  --  up to 4000 rows",
    ["both Y0 and Y1 kept,", "for evaluation only"])
arrow(250, 618, 250, 640); arrow(730, 618, 730, 640)

FLOW = ('<figure class="fig"><svg viewBox="0 0 %d %d" class="chart" '
        'style="max-width:%dpx">%s</svg></figure>' % (FW, FH, FW, "".join(_f)))

FLOW_BLOCK = f"""
<h2>How each UCI dataset became a confounded benchmark</h2>
{FLOW}
"""

# ------------------------------------------------------------------ value colouring
# Every method cell is tinted on a red-to-green ramp scored WITHIN ITS OWN ROW, so the comparison
# is always "against the other methods on this dataset" -- absolute levels differ far too much
# between datasets (mushroom sits near 0.7, wine_quality near 0.26) for a global scale to mean
# anything. The oracle column is deliberately left black: it is a reference point,
# not competitors, and tinting them would imply they are in the ranking.
RAMP = [(0.0, (0xC0, 0x39, 0x2B)), (0.5, (0xB0, 0x71, 0x05)), (1.0, (0x0A, 0x7D, 0x33))]


def tint(t):
    t = 0.0 if t != t else min(max(t, 0.0), 1.0)
    for (a, ca), (b, cb) in zip(RAMP, RAMP[1:]):
        if t <= b:
            f = (t - a) / (b - a)
            return "#%02x%02x%02x" % tuple(int(round(ca[i] + f * (cb[i] - ca[i]))) for i in range(3))
    return "#0a7d33"


def cells(vals, order, sds=None):
    """Render one row of method cells, tinted by within-row rank; "mean +- sd" when sds given."""
    vs = [vals[m] for m in order if m in vals]
    lo, hi = (min(vs), max(vs)) if vs else (0.0, 1.0)
    out = []
    for m in order:
        if m not in vals: out.append("<td>&mdash;</td>"); continue
        v = vals[m]
        t = 1.0 if hi <= lo else (v - lo) / (hi - lo)
        bold = ";font-weight:700" if abs(v - hi) < 1e-9 else ""
        txt = "%.3f" % v
        if sds is not None and m in sds:
            txt += " <span class='sd'>&plusmn;%.3f</span>" % sds[m]
        out.append("<td style='color:%s%s'>%s</td>" % (tint(t), bold, txt))
    return "".join(out)


# ------------------------------------------------------------------ tab 2: results
def table(key, caption):
    head = ("<tr><th class='l'>dataset</th>"
            + "".join("<th>%s</th>" % esc(SHORT.get(m, m)) for m in ORDER)
            + "<th>oracle</th></tr>")
    body = []
    for s in SUM:
        vals = {m: s["rows"][m][key]["mean"] for m in ORDER
                if m in s["rows"] and s["rows"][m].get(key)}
        sds = {m: s["rows"][m][key]["sd"] for m in ORDER
               if m in s["rows"] and s["rows"][m].get(key) and "sd" in s["rows"][m][key]}
        rsd = s.get("refs_sd", {})
        body.append("<tr><td class='l'>%s</td>%s"
                    "<td class='ref'>%.3f <span class='sd'>&plusmn;%.3f</span></td></tr>"
                    % (esc(s["dataset"]), cells(vals, ORDER, sds),
                       s["refs"]["oracle"], rsd.get("oracle", 0.0)))

    # ---- cross-dataset summary rows -----------------------------------------------------
    # Three, because none is honest alone. MEAN is dominated by whichever datasets happen to
    # live on a bigger outcome scale. V/oracle is scale-free but NOT shift-free (wine_quality
    # and communities_crime have never-treat below zero, so a method can post a high ratio while
    # barely beating the free constant policy). The normalized score is both, so it is the one to
    # rank methods on.
    for lab, fn, orc_ref in (
            ("mean", lambda v, R: v, np.mean([x["refs"]["oracle"] for x in SUM])),
            ("mean V / oracle", lambda v, R: v / R["oracle"], 1.0),
            ("mean normalized", lambda v, R: (v - R["never_treat"]) / (R["oracle"] - R["never_treat"]),
             1.0)):
        agg = {}
        for m in ORDER:
            vs = [fn(x["rows"][m][key]["mean"], x["refs"]) for x in SUM
                  if m in x["rows"] and x["rows"][m].get(key)]
            if vs: agg[m] = float(np.mean(vs))
        body.append("<tr class='sumrow'><td class='l'><b>%s</b></td>%s"
                    "<td class='ref'>%.3f</td></tr>" % (esc(lab), cells(agg, ORDER), orc_ref))

    return "<div class='scrollx'><table class='rt'>%s%s</table></div>" % (head, "".join(body))


BODY = f"""
<h2>Average test outcome at the matched &Gamma; = 4 <span class="hsub">&mdash; mean &plusmn; sd over 10 seeds, <i>n</i><sub>train</sub> = 300</span></h2>
{table("matched_best_L", "")}

<h3>At each method's best cell <span class="hsub">&mdash; mean &plusmn; sd over 10 seeds</span></h3>
{table("best", "")}

"""

hero = """<div class="hero" style="background:linear-gradient(135deg,#0f4c5c,#1b7a8c)">
<h1>UCI semi-synthetic campaign &mdash; ten real datasets, known confounding</h1>
</div>"""

html = ('<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        f'<title>UCI 10-dataset semi-synthetic campaign</title>'
        f'<style>{CSS}{TABLE_CSS}</style></head><body>'
        f'{hero}{FLOW_BLOCK}{BODY}</body></html>')
html = html.encode("ascii", "xmlcharrefreplace").decode("ascii")
(HERE / "uci10_report.html").write_text(html)
print("wrote %s (%d KB)" % (HERE / "uci10_report.html", len(html) // 1024))
