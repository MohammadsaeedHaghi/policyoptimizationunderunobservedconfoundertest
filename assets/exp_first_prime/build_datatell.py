"""Build the 'What the data tell us' chart: observed average outcome per arm per X (raw cell means) from the
N=10000 first-prime sample. Splice var PRIME_DATATELL + render call (ichart-p10k-datatell) into index.html."""
import json, re
from pathlib import Path
ROOT = Path("/Users/mohammadsaeedhaghi/Desktop/01-Research/paper - policy optimization under unobsorved confounder - code/code 1.1")
HERE = ROOT / "assets" / "exp_first_prime"
D = json.loads((HERE / "prime_n10000_datatell.json").read_text())
X = D["X"]
W = {"x": [float(v) for v in X], "xmin": -1.0, "xmax": 1.0, "xlabel": "X  (employability score)",
     "ylabel": "observed mean outcome  (mean Y | X, T)", "ymin": -0.05, "ymax": 1.08,
     "series": [
        {"id": "treated", "label": "treated (T=1):  mean Y | X", "color": "#1f77b4", "marker": "square", "y": D["mean_treated"]},
        {"id": "control", "label": "control (T=0):  mean Y | X", "color": "#dc2626", "marker": "circle", "y": D["mean_control"]},
     ],
     "hlines": [{"y": 0.0, "color": "#cbd5e1", "dash": "3,3", "label": ""}],
     "note": "Raw observed arm means per X (N=10000). At X=-0.4 the lines nearly coincide (treated 0.630 vs control 0.613); for X<0 control is at least as high, for X>=0 treated jumps to ~1. Low-X treated cells are based on few units (low propensity), so they are noisier."}
(HERE / "prime_datatell_chart.json").write_text(json.dumps(W))

smax = max([v for v in (D["sum_treated"] + D["sum_control"]) if v is not None]) * 1.12
WS = {"x": [float(v) for v in X], "xmin": -1.0, "xmax": 1.0, "xlabel": "X  (employability score)",
      "ylabel": "sum of outcomes  (Σ Y | X, T)", "ymin": 0.0, "ymax": round(smax, 0),
      "series": [
         {"id": "treated", "label": "treated (T=1):  Σ Y | X", "color": "#1f77b4", "marker": "square", "y": D["sum_treated"]},
         {"id": "control", "label": "control (T=0):  Σ Y | X", "color": "#dc2626", "marker": "circle", "y": D["sum_control"]},
      ],
      "note": "Total observed successes (Y=1) per arm per X (N=10000). These raw sums are exactly what the IPW masses weight: at X=-0.4, treated ΣY=68 and control ΣY=222 give A1=68/e_hat=273.6 and A0=222/(1-e_hat)=295.4. Control sums dominate at low X (control is far more common there); treated sums overtake only at high X."}
(HERE / "prime_datatell_sum_chart.json").write_text(json.dumps(WS))

idx = ROOT / "index.html"; h = idx.read_text()
for var, dat, cid in (("PRIME_DATATELL", W, "ichart-p10k-datatell"), ("PRIME_DATATELL_SUM", WS, "ichart-p10k-datatell-sum")):
    line = "var %s = %s;" % (var, json.dumps(dat, separators=(",", ":")))
    if re.search(r'^var %s = .*;$' % var, h, flags=re.M):
        h = re.sub(r'^var %s = .*;$' % var, lambda m: line, h, count=1, flags=re.M)
    else:
        h = re.sub(r'^(var PRIME_N10K = .*;)$', lambda m: m.group(1) + "\n" + line, h, count=1, flags=re.M)
    call = "renderChart('%s',%s);" % (cid, var)
    h = re.sub(r"\nrenderChart\('%s'[^\n]*" % cid, "", h)
    if call not in h:
        anchor = "renderChart('ichart-p10k-uncap-policy',PRIME_N10K.uncap_policy);"
        assert anchor in h, "anchor missing"
        h = h.replace(anchor, anchor + "\n" + call, 1)
idx.write_text(h)
print("spliced PRIME_DATATELL + PRIME_DATATELL_SUM + render calls")
