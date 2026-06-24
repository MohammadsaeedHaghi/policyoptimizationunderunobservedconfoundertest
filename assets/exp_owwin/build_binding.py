"""Build the binding-frontier chart Γ*(ε): below the curve the odds-box binds, above it the Wasserstein ball binds.
Splice var OWBIND + render call into the owwin tab."""
import json, re
from pathlib import Path
import numpy as np
ROOT = Path("/Users/mohammadsaeedhaghi/Desktop/01-Research/paper - policy optimization under unobsorved confounder - code/code 1.1")
HERE = ROOT / "assets" / "exp_owwin"
R = json.loads((HERE / "owwin_binding.json").read_text())
eps = [r["eps"] for r in R["rows"]]; gst = [r["gamma_star"] for r in R["rows"]]
tight = R["tight_eps"]
# linear fit for the note
A = np.polyfit(eps, gst, 1)
W = {"x": [float(e) for e in eps], "xmin": 0.0, "xmax": round(max(eps) * 1.03, 3),
     "xlabel": "Wasserstein radius ε  (treated arm)", "ylabel": "Γ* : box stops binding (W takes over)",
     "ymin": 0.0, "ymax": round(max(gst) * 1.08, 1),
     "series": [{"id": "frontier", "label": "Γ*(ε) binding frontier", "color": "#7c3aed", "marker": "circle", "width": 3.0, "y": [float(g) for g in gst]}],
     "hlines": [{"y": 1.0, "color": "#cbd5e1", "dash": "2,3", "label": "Γ=1 (box collapses to nominal)"}],
     "note": "Binding frontier (treated arm). BELOW the line (smaller Γ): the odds-box is tighter, so the BOX binds and the W-ball is slack. ABOVE the line (larger Γ): the box is looser than what the W-ball already implies, so the WASSERSTEIN ball binds. The relation is near-linear, Γ* ≈ %.1f + %.0f·ε; at the tight ε (c=1, green line) the box stops binding around Γ*≈%.0f." % (A[1], A[0], np.interp(tight, eps, gst))}
(HERE / "owwin_binding_chart.json").write_text(json.dumps(W))
idx = ROOT / "index.html"; h = idx.read_text()
line = "var OWBIND = %s;" % json.dumps(W, separators=(",", ":"))
if re.search(r'^var OWBIND = .*;$', h, flags=re.M):
    h = re.sub(r'^var OWBIND = .*;$', lambda m: line, h, count=1, flags=re.M)
else:
    h = re.sub(r'^(var OWWIN_CHARTS = .*;)$', lambda m: m.group(1) + "\n" + line, h, count=1, flags=re.M)
call = "renderChart('ichart-ow-binding',OWBIND);"
h = re.sub(r"\nrenderChart\('ichart-ow-binding'[^\n]*", "", h)
if call not in h:
    anchor = "renderChart('ichart-ow-eps-ipw',OWEPS_IPW);"
    assert anchor in h, "anchor missing"; h = h.replace(anchor, anchor + "\n" + call, 1)
idx.write_text(h)
print("spliced OWBIND; Γ*(ε): %s" % list(zip(eps, gst)))
print("fit Γ* ≈ %.2f + %.1f·ε ; tight ε=%.4f -> Γ*≈%.1f" % (A[1], A[0], tight, float(np.interp(tight, eps, gst))))
