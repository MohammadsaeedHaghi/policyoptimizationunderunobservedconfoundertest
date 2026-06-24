"""First experiment (Case 4): run ONLY IPW (uncapped) at N_train=5000 for 3 seeds; capture the selected
policy π(treat|X) per seed + Average for the interactive seed-dropdown chart, AND a per-seed diagnostic of
every grid point where IPW declines treatment (counts, weighted sums, observed/true means). Splices the var
POLICY_FIRST_IPW5000 + render call into index.html, removes the old N=3000 var/call, regenerates the viewer.
Usage: python3 ipw_n5000.py"""
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
K = dgp.K; GRID = dgp.GRID; N = 5000; SEEDS = [0, 1, 2]; mu = dgp.mu_arm(GRID)

per_seed = {}; diag = {}
for s in SEEDS:
    rng = np.random.default_rng(s); tr = dgp.generate(N, rng)
    X = tr.X.ravel(); T = tr.T; Y = tr.Y; S = tr.S
    w, Pm = common.ipw_weights_from_data(tr.X, tr.T, K); ehat = Pm[:, 1]
    pi = ipw(tr.X, tr.T, tr.Y, w, n_arms=K, discretize=False).pi
    xr = np.round(X, 6); g = np.array([pi[1, np.where(xr == round(float(gx), 6))[0][0]] for gx in GRID])
    per_seed[str(s)] = [round(float(v), 4) for v in g]
    rows = []
    for i, gx in enumerate(GRID):
        if g[i] >= 0.5:
            continue
        m = xr == round(float(gx), 6); mt = m & (T == 1); mc = m & (T == 0)
        wt = float(w[mt].sum()); wc = float(w[mc].sum())                      # per-arm Hájek weight mass in cell
        sumT = float((w[mt] * Y[mt]).sum()); sumC = float((w[mc] * Y[mc]).sum())  # IPW value contributions (treat vs control)
        rows.append({"X": round(float(gx), 2), "nT": int(mt.sum()), "nC": int(mc.sum()),
                     "ybarT": round(float(Y[mt].mean()), 3), "ybarC": round(float(Y[mc].mean()), 3),
                     "ehat": round(float(ehat[m].mean()), 3), "S1frac_T": round(float(S[mt].mean()), 2),
                     "ipw_sum_treat": round(sumT, 2), "ipw_sum_control": round(sumC, 2),
                     "trueMu1": round(float(mu[i, 1]), 3), "trueMu0": round(float(mu[i, 0]), 3)})
    diag[str(s)] = rows
    print("seed %d: treat-frac=%.2f ; declines at X=%s" % (s, np.nanmean(g), [r["X"] for r in rows]))
    for r in rows:
        print("   X=%+.1f nT=%d nC=%d  ybarT=%.3f ybarC=%.3f  IPW-sum treat=%.2f vs control=%.2f (control %s)  | true mu1=%.3f<mu0=%.3f"
              % (r["X"], r["nT"], r["nC"], r["ybarT"], r["ybarC"], r["ipw_sum_treat"], r["ipw_sum_control"],
                 "wins" if r["ipw_sum_control"] >= r["ipw_sum_treat"] else "loses", r["trueMu1"], r["trueMu0"]))
avg = [round(float(v), 4) for v in np.nanmean(np.array([per_seed[str(s)] for s in SEEDS]), axis=0)]
print("AVG treat-frac=%.2f" % np.nanmean(avg))

def series(y): return [{"id": "IPW", "label": "IPW", "color": "#2ca02c", "y": y}]
sbsg = {str(s): {"1": series(per_seed[str(s)])} for s in SEEDS}; sbsg["avg"] = {"1": series(avg)}
DATA = {"x": [float(x) for x in GRID], "xmin": -1.0, "xmax": 1.0, "xlabel": "X", "ylabel": "π(treat | x)",
        "ymin": 0.0, "ymax": 1.0, "gammas": [1.0], "defaultGamma": 1.0, "matchedGamma": 1.0,
        "seeds": [{"key": str(s), "label": "Seed %d" % s} for s in SEEDS] + [{"key": "avg", "label": "Average"}],
        "defaultSeed": "avg", "seriesBySeedGamma": sbsg,
        "note": "IPW π(treat|X), uncapped, N_train=5000 · pick a seed or Average"}
(HERE / "policy_ipw_n5000.json").write_text(json.dumps(DATA))
(HERE / "ipw_n5000_declines.json").write_text(json.dumps(diag, indent=1))

idx = NEW / "index.html"; h = idx.read_text()
# remove old N=3000 var + render call (orphaned by the 5000 panel)
h = re.sub(r'^var POLICY_FIRST_IPW3000 = .*;\n', '', h, flags=re.M)
h = h.replace("renderChart('ichart-first-ipw3000',POLICY_FIRST_IPW3000);\n", "")
# splice the N=5000 var + render call (idempotent)
line = "var POLICY_FIRST_IPW5000 = %s;" % json.dumps(DATA, separators=(',', ':'))
if re.search(r'^var POLICY_FIRST_IPW5000 = .*;$', h, flags=re.M):
    h = re.sub(r'^var POLICY_FIRST_IPW5000 = .*;$', lambda m: line, h, count=1, flags=re.M)
else:
    h = re.sub(r'^(var POLICY_FIRST_CAP = .*;)$', lambda m: m.group(1) + "\n" + line, h, count=1, flags=re.M)
call = "renderChart('ichart-first-ipw5000',POLICY_FIRST_IPW5000);"
if call not in h:
    h = h.replace("renderChart('ichart-first-cap-policy',POLICY_FIRST_CAP);",
                  "renderChart('ichart-first-cap-policy',POLICY_FIRST_CAP);\n" + call, 1)
idx.write_text(h)
INJ = ('<script>window.addEventListener("load",function(){setTimeout(function(){'
       'document.querySelectorAll(".tab-panel").forEach(function(p){p.classList.remove("active")});'
       'var t=document.getElementById("tab-experiment");if(t)t.classList.add("active");'
       'document.querySelectorAll(".subpanel").forEach(function(p){p.classList.remove("active")});'
       'var s=document.getElementById("sub-exp_first");if(s)s.classList.add("active");'
       'document.querySelectorAll(".tab-btn,.subtab-btn").forEach(function(b){b.classList.remove("active")});'
       'if(window.MathJax&&MathJax.typesetPromise)MathJax.typesetPromise();},500)});</script>')
head, _, tail = h.rpartition('</body>')
(NEW / "_exp_first_view.html").write_text(head + INJ + '</body>' + tail)
print("spliced POLICY_FIRST_IPW5000 (+removed N=3000); regenerated viewer")
