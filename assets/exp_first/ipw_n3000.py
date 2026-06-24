"""First experiment (Case 4): run ONLY the IPW value-maximiser (uncapped) at N_train=3000 for 3 seeds,
capture the selected policy π(treat|X) over the 21-X grid per seed + the Average, build the interactive
renderChart var POLICY_FIRST_IPW3000 (seed dropdown) and splice it + the render call into index.html, then
regenerate _exp_first_view.html. No other method is run; nothing else in the tab is changed.
Usage: python3 ipw_n3000.py"""
import sys, json, re, importlib.util
import numpy as np
from pathlib import Path
NEW = Path("/Users/mohammadsaeedhaghi/Desktop/01-Research/paper - policy optimization under unobsorved confounder - code/code 1.1")
sys.path.insert(0, str(NEW)); import common
HERE = NEW / "assets" / "exp_first"
sp = importlib.util.spec_from_file_location("dgp_first", str(HERE / "dgp.py")); dgp = importlib.util.module_from_spec(sp); sys.modules["dgp_first"] = dgp; sp.loader.exec_module(dgp)
def load(n, rel):
    s = importlib.util.spec_from_file_location(n, str(NEW / rel)); m = importlib.util.module_from_spec(s); sys.modules[n] = m; s.loader.exec_module(m); return m
ipw = getattr(load("ipw", "methods/IPW-X-X/Uncapped/ipw_x_x_uncapped.py"), "solve_ipw_x_x_uncapped")
K = dgp.K; GRID = dgp.GRID; N = 3000; SEEDS = [0, 1, 2]

per_seed = {}
for s in SEEDS:
    rng = np.random.default_rng(s); tr = dgp.generate(N, rng)
    w, _ = common.ipw_weights_from_data(tr.X, tr.T, K)            # estimated propensity -> Hájek weights (never true)
    pi = ipw(tr.X, tr.T, tr.Y, w, n_arms=K, discretize=False).pi  # uncapped IPW value-maximiser
    xr = np.round(tr.X.ravel(), 6)
    g = np.array([pi[1, np.where(xr == round(float(gx), 6))[0][0]] if (xr == round(float(gx), 6)).any() else np.nan for gx in GRID])
    per_seed[str(s)] = [round(float(v), 4) for v in g]
    print("seed %d: IPW treat-frac=%.2f | X<0 treat-frac=%.2f | X>=0 treat-frac=%.2f"
          % (s, np.nanmean(g), np.nanmean(g[GRID < 0]), np.nanmean(g[GRID >= 0])))
avg = [round(float(v), 4) for v in np.nanmean(np.array([per_seed[str(s)] for s in SEEDS]), axis=0)]
print("AVG : IPW treat-frac=%.2f | X<0=%.2f | X>=0=%.2f" % (np.nanmean(avg), np.nanmean(np.array(avg)[GRID < 0]), np.nanmean(np.array(avg)[GRID >= 0])))

def series(y): return [{"id": "IPW", "label": "IPW", "color": "#2ca02c", "y": y}]
sbsg = {s: {"1": series(per_seed[s])} for s in (str(x) for x in SEEDS)}
sbsg["avg"] = {"1": series(avg)}
DATA = {"x": [float(x) for x in GRID], "xmin": -1.0, "xmax": 1.0, "xlabel": "X", "ylabel": "π(treat | x)",
        "ymin": 0.0, "ymax": 1.0, "gammas": [1.0], "defaultGamma": 1.0, "matchedGamma": 1.0,
        "seeds": [{"key": str(s), "label": "Seed %d" % s} for s in SEEDS] + [{"key": "avg", "label": "Average"}],
        "defaultSeed": "avg", "seriesBySeedGamma": sbsg,
        "note": "IPW π(treat|X), uncapped, N_train=3000 · pick a seed or Average"}
(HERE / "policy_ipw_n3000.json").write_text(json.dumps(DATA))

# ---- splice var + render call into index.html (idempotent) ----
idx = NEW / "index.html"; h = idx.read_text()
line = "var POLICY_FIRST_IPW3000 = %s;" % json.dumps(DATA, separators=(',', ':'))
if re.search(r'^var POLICY_FIRST_IPW3000 = .*;$', h, flags=re.M):
    h = re.sub(r'^var POLICY_FIRST_IPW3000 = .*;$', lambda m: line, h, count=1, flags=re.M)
else:
    h = re.sub(r'^(var POLICY_FIRST_CAP = .*;)$', lambda m: m.group(1) + "\n" + line, h, count=1, flags=re.M)
call = "renderChart('ichart-first-ipw3000',POLICY_FIRST_IPW3000);"
if call not in h:
    h = h.replace("renderChart('ichart-first-cap-policy',POLICY_FIRST_CAP);",
                  "renderChart('ichart-first-cap-policy',POLICY_FIRST_CAP);\n" + call, 1)
idx.write_text(h)
# ---- regenerate standalone viewer (auto-open sub-exp_first) ----
INJ = ('<script>window.addEventListener("load",function(){setTimeout(function(){'
       'document.querySelectorAll(".tab-panel").forEach(function(p){p.classList.remove("active")});'
       'var t=document.getElementById("tab-experiment");if(t)t.classList.add("active");'
       'document.querySelectorAll(".subpanel").forEach(function(p){p.classList.remove("active")});'
       'var s=document.getElementById("sub-exp_first");if(s)s.classList.add("active");'
       'document.querySelectorAll(".tab-btn,.subtab-btn").forEach(function(b){b.classList.remove("active")});'
       'if(window.MathJax&&MathJax.typesetPromise)MathJax.typesetPromise();},500)});</script>')
head, _, tail = h.rpartition('</body>')
(NEW / "_exp_first_view.html").write_text(head + INJ + '</body>' + tail)
print("spliced POLICY_FIRST_IPW3000 + render call; regenerated _exp_first_view.html")
