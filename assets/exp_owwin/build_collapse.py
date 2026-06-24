"""Build the large-Γ collapse panel: value-vs-Γ and treat-fraction-vs-Γ charts (x = log10 Γ) from owwin_collapse.json,
splice var OWCOL_VAL / OWCOL_FRAC + render calls into the owwin tab."""
import json, re, math
from pathlib import Path
ROOT = Path("/Users/mohammadsaeedhaghi/Desktop/01-Research/paper - policy optimization under unobsorved confounder - code/code 1.1")
HERE = ROOT / "assets" / "exp_owwin"
R = json.loads((HERE / "owwin_collapse.json").read_text())
G = R["gammas"]; X = [round(math.log10(g), 4) for g in G]
FAM = {"IPW": "#1f77b4", "DoublyRobust": "#2ca02c", "Hajek": "#9467bd"}
def fam(m): return m.split("-")[0]
def mk(m): return "o" if m.endswith("-O-W") else "square" if m.endswith("-O-X") else None
METHODS = ["IPW-O-X", "DoublyRobust-O-X", "Hajek-O-X", "IPW-O-W", "DoublyRobust-O-W"]
def series(field):
    out = []
    for m in METHODS:
        marker = "circle" if m.endswith("-O-W") else "square"
        out.append({"id": m, "label": m, "color": FAM[fam(m)], "marker": marker, "y": [round(float(v), 4) for v in R[field][m]]})
    return out
gticks = "Γ: " + ", ".join(str(int(g)) for g in G)
VAL = {"x": X, "xmin": min(X), "xmax": max(X), "xlabel": "log₁₀ Γ   (" + gticks + ")", "ylabel": "realised E[Y] (3-seed mean)",
       "ymin": -0.03, "ymax": round(R["oracle"] + 0.03, 3), "series": series("value_mean"),
       "hlines": [{"y": R["oracle"], "color": "#94a3b8", "dash": "4,4", "label": "oracle"},
                  {"y": R["never_treat"], "color": "#cbd5e1", "dash": "2,3", "label": "never-treat"}],
       "note": "Realised value vs Γ (to 1000), capped, N_train=700. Hajek-O-X (regret, purple) collapses to the never-treat line; IPW-O-X/DoublyRobust-O-X (value, squares) and OW (circles) do NOT collapse — they plateau."}
FRAC = {"x": X, "xmin": min(X), "xmax": max(X), "xlabel": "log₁₀ Γ   (" + gticks + ")", "ylabel": "fraction treated",
        "ymin": -0.03, "ymax": 0.6, "series": series("frac_mean"),
        "hlines": [{"y": 0.5, "color": "#dc2626", "dash": "4,4", "label": "cap = 0.50"}],
        "note": "Fraction treated vs Γ. Hajek-O-X drops to 0 (never-treat) — the collapse; the value methods stay pinned at the 50% cap at every Γ."}
(HERE / "owwin_collapse_charts.json").write_text(json.dumps({"value": VAL, "frac": FRAC}))

idx = ROOT / "index.html"; h = idx.read_text()
for var, dat, cid in (("OWCOL_VAL", VAL, "ichart-ow-collapse-value"), ("OWCOL_FRAC", FRAC, "ichart-ow-collapse-frac")):
    line = "var %s = %s;" % (var, json.dumps(dat, separators=(",", ":")))
    if re.search(r'^var %s = .*;$' % var, h, flags=re.M):
        h = re.sub(r'^var %s = .*;$' % var, lambda m: line, h, count=1, flags=re.M)
    else:
        h = re.sub(r'^(var OWWIN_CHARTS = .*;)$', lambda m: m.group(1) + "\n" + line, h, count=1, flags=re.M)
    call = "renderChart('%s',%s);" % (cid, var)
    h = re.sub(r"\nrenderChart\('%s'[^\n]*" % cid, "", h)
    if call not in h:
        anchor = "renderChart('ichart-ow-value',OWWIN_CHARTS.value);"
        assert anchor in h; h = h.replace(anchor, anchor + "\n" + call, 1)
idx.write_text(h)
print("spliced collapse charts; Hajek frac=%s  DR-O-X val=%s" % (R["frac_mean"]["Hajek-O-X"], R["value_mean"]["DoublyRobust-O-X"]))
