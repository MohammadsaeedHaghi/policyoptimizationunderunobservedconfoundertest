"""Build the ε-sweep panel: DoublyRobust-O-W (and IPW-O-W) value vs Γ, one curve per c_eps, showing larger ε ->
collapse at large Γ. x = log10 Γ. Splice OWEPS_DR / OWEPS_IPW into the owwin tab."""
import json, re, math
from pathlib import Path
ROOT = Path("/Users/mohammadsaeedhaghi/Desktop/01-Research/paper - policy optimization under unobsorved confounder - code/code 1.1")
HERE = ROOT / "assets" / "exp_owwin"
R = json.loads((HERE / "owwin_epssweep.json").read_text())
G = R["gammas"]; X = [round(math.log10(g), 4) for g in G]; CEPS = R["c_eps"]
COL = {"0.5": "#22c55e", "1.0": "#1f77b4", "2.0": "#f59e0b", "4.0": "#dc2626"}  # green->blue->amber->red as ε grows
gticks = "Γ: " + ", ".join(str(int(g)) for g in G)
def chart(method, ylab):
    series = [{"id": "ceps%s" % ce, "label": "ε scale c=%s" % ce, "color": COL[ce], "marker": "circle",
               "y": [round(float(v), 4) for v in R["value"][method][ce]]} for ce in ["%.1f" % c for c in CEPS]]
    return {"x": X, "xmin": min(X), "xmax": max(X), "xlabel": "log₁₀ Γ   (" + gticks + ")", "ylabel": ylab,
            "ymin": -0.03, "ymax": round(R["oracle"] + 0.03, 3), "series": series,
            "hlines": [{"y": R["oracle"], "color": "#94a3b8", "dash": "4,4", "label": "oracle"},
                       {"y": R["never_treat"], "color": "#cbd5e1", "dash": "2,3", "label": "never-treat"}],
            "note": "%s value vs Γ (uncapped, N=700, 3-seed mean), one curve per Wasserstein-radius scale c_eps. Tight ε (green/blue) stays robust and good; large ε (amber/red) is over-conservative and collapses toward never-treat as Γ grows." % method}
DR = chart("DoublyRobust-O-W", "realised E[Y]  (DoublyRobust-O-W)")
IP = chart("IPW-O-W", "realised E[Y]  (IPW-O-W)")
(HERE / "owwin_epspanel_charts.json").write_text(json.dumps({"dr": DR, "ipw": IP}))
idx = ROOT / "index.html"; h = idx.read_text()
for var, dat, cid in (("OWEPS_DR", DR, "ichart-ow-eps-dr"), ("OWEPS_IPW", IP, "ichart-ow-eps-ipw")):
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
print("spliced ε-panel; DR-O-W c=4 val=%s" % R["value"]["DoublyRobust-O-W"]["4.0"])
