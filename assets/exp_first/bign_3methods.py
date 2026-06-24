"""First experiment (Case 4), large sample N_train=3000, UNCAPPED, 3 seeds. Run THREE value methods —
IPW-X-X (Hájek IPW, Γ-free), Direct-X-X (Direct-X-X, unit weights, Γ-free) and IPW-O-X (R-O,
worst-case value over the odds box) SWEPT over the Γ grid. Capture each method's selected policy π(treat|X)
over the 21-X grid per seed + the Average, build the interactive chart var POLICY_FIRST_BIGN with a Γ
dropdown (only IPW-O-X moves with Γ; IPW-X-X & Direct-X-X are flat) + a seed dropdown + method toggle,
splice into index.html (replacing the old var/call), regenerate _exp_first_view.html. Usage: python3 bign_3methods.py"""
import sys, json, re, importlib.util
import numpy as np
from pathlib import Path
NEW = Path("/Users/mohammadsaeedhaghi/Desktop/01-Research/paper - policy optimization under unobsorved confounder - code/code 1.1")
sys.path.insert(0, str(NEW)); import common
HERE = NEW / "assets" / "exp_first"
sp = importlib.util.spec_from_file_location("dgp_first", str(HERE / "dgp.py")); dgp = importlib.util.module_from_spec(sp); sys.modules["dgp_first"] = dgp; sp.loader.exec_module(dgp)
def L(rel, fn):
    s = importlib.util.spec_from_file_location(fn, str(NEW / rel)); m = importlib.util.module_from_spec(s); sys.modules[fn] = m; s.loader.exec_module(m); return getattr(m, fn)
ipwxx = L("methods/IPW-X-X/Uncapped/ipw_x_x_uncapped.py", "solve_ipw_x_x_uncapped")
popt  = L("methods/Direct-X-X/Uncapped/direct_x_x_uncapped.py", "solve_direct_x_x_uncapped")
ipwox = L("methods/IPW-O-X/Uncapped/ipw_o_x_uncapped.py", "solve_ipw_o_x_uncapped")
ipwow = L("methods/IPW-O-W/Uncapped/ipw_o_w_uncapped.py", "solve_ipw_o_w_uncapped")
K = dgp.K; GRID = dgp.GRID; N = 3000; N_W = 600; SEEDS = [0, 1, 2]; mu = dgp.mu_arm(GRID)
# IPW-O-W is run at N_W=600 because its Wasserstein transport LP is O(n^2) — it OOMs at N=3000.
GAMMAS, mi = dgp.gammas_for(dgp.GAMMA_CONF); MG = GAMMAS[mi]          # [1,1.5,2,2.72,3,4.35,7.07], matched=2.72
optimal = (mu[:, 1] > mu[:, 0]).astype(float)
def gkey(g): g = float(g); return str(int(g)) if g == int(g) else str(g)
COL = {"IPW-X-X": "#2ca02c", "Direct-X-X": "#ff7f0e", "IPW-O-X": "#9467bd", "IPW-O-W": "#d62728"}

# per[seedkey] = {"IPW-X-X":[grid], "Direct-X-X":[grid], "IPW-O-X": {gkey:[grid]}, "IPW-O-W": {gkey:[grid]}}
per = {}
for s in SEEDS:
    rng = np.random.default_rng(s); tr = dgp.generate(N, rng)
    w, _ = common.ipw_weights_from_data(tr.X, tr.T, K); xr = np.round(tr.X.ravel(), 6)
    def grid_of(pi, xrr): return [round(float(pi[1, np.where(xrr == round(float(gx), 6))[0][0]]), 4) for gx in GRID]
    d = {"IPW-X-X": grid_of(ipwxx(tr.X, tr.T, tr.Y, w, n_arms=K, discretize=False).pi, xr),
         "Direct-X-X": grid_of(popt(tr.X, tr.T, tr.Y, n_arms=K, discretize=False).pi, xr),
         "IPW-O-X": {gkey(g): grid_of(ipwox(tr.X, tr.T, tr.Y, w, n_arms=K, Gamma=float(g), discretize=False).pi, xr) for g in GAMMAS}}
    # IPW-O-W on a separate n=600 sample (Wasserstein LP can't scale to n=3000)
    trw = dgp.generate(N_W, np.random.default_rng(1000 + s)); ww, _ = common.ipw_weights_from_data(trw.X, trw.T, K)
    xrw = np.round(trw.X.ravel(), 6)
    d["IPW-O-W"] = {gkey(g): grid_of(ipwow(trw.X, trw.T, trw.Y, ww, n_arms=K, Gamma=float(g), discretize=False, zscore=False).pi, xrw) for g in GAMMAS}
    per[str(s)] = d
    print("seed %d: IPW-X-X=%.2f Direct-X-X=%.2f | IPW-O-X@Γm=%.2f | IPW-O-W@Γm=%.2f (n=%d)" % (
        s, np.mean(d["IPW-X-X"]), np.mean(d["Direct-X-X"]), np.mean(d["IPW-O-X"][gkey(MG)]), np.mean(d["IPW-O-W"][gkey(MG)]), N_W), flush=True)
# average over seeds
avg = {"IPW-X-X": [round(float(v), 4) for v in np.mean([per[str(s)]["IPW-X-X"] for s in SEEDS], 0)],
       "Direct-X-X": [round(float(v), 4) for v in np.mean([per[str(s)]["Direct-X-X"] for s in SEEDS], 0)],
       "IPW-O-X": {gkey(g): [round(float(v), 4) for v in np.mean([per[str(s)]["IPW-O-X"][gkey(g)] for s in SEEDS], 0)] for g in GAMMAS},
       "IPW-O-W": {gkey(g): [round(float(v), 4) for v in np.mean([per[str(s)]["IPW-O-W"][gkey(g)] for s in SEEDS], 0)] for g in GAMMAS}}
per["avg"] = avg
for nm in ("IPW-O-X", "IPW-O-W"):
    a = np.array(avg[nm][gkey(MG)]); print("AVG %s@matchedΓ=%.2f treat-frac=%.2f (X<0=%.2f X>=0=%.2f)" % (nm, MG, a.mean(), a[GRID < 0].mean(), a[GRID >= 0].mean()))

def ser(seedkey, g):
    d = per[seedkey]
    return [{"id": "IPW-X-X", "label": "IPW-X-X", "color": COL["IPW-X-X"], "y": d["IPW-X-X"]},
            {"id": "Direct-X-X", "label": "Direct-X-X", "color": COL["Direct-X-X"], "y": d["Direct-X-X"]},
            {"id": "IPW-O-X", "label": "IPW-O-X", "color": COL["IPW-O-X"], "y": d["IPW-O-X"][gkey(g)]},
            {"id": "IPW-O-W", "label": "IPW-O-W (n=600)", "color": COL["IPW-O-W"], "y": d["IPW-O-W"][gkey(g)]}]
sbsg = {sk: {gkey(g): ser(sk, g) for g in GAMMAS} for sk in [str(s) for s in SEEDS] + ["avg"]}
DATA = {"x": [float(x) for x in GRID], "xmin": -1.0, "xmax": 1.0, "xlabel": "X", "ylabel": "π(treat | x)",
        "ymin": 0.0, "ymax": 1.0, "gammas": [float(g) for g in GAMMAS], "defaultGamma": float(MG), "matchedGamma": float(MG),
        "seeds": [{"key": str(s), "label": "Seed %d" % s} for s in SEEDS] + [{"key": "avg", "label": "Average"}],
        "defaultSeed": "avg", "seriesBySeedGamma": sbsg,
        "note": "π(treat|X), uncapped, N=3000 · only IPW-O-X moves with Γ (IPW-X-X & Direct-X-X are Γ-free) · matched Γ=%.2f" % MG}
(HERE / "policy_first_bigN_n3000.json").write_text(json.dumps(DATA))

idx = NEW / "index.html"; h = idx.read_text()
line = "var POLICY_FIRST_BIGN = %s;" % json.dumps(DATA, separators=(',', ':'))
if re.search(r'^var POLICY_FIRST_BIGN = .*;$', h, flags=re.M):
    h = re.sub(r'^var POLICY_FIRST_BIGN = .*;$', lambda m: line, h, count=1, flags=re.M)
else:
    h = re.sub(r'^(var POLICY_FIRST_CAP = .*;)$', lambda m: m.group(1) + "\n" + line, h, count=1, flags=re.M)
call = "renderChart('ichart-first-bigN',POLICY_FIRST_BIGN);"
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
print("spliced POLICY_FIRST_BIGN (Γ-swept) into index.html; regenerated viewer")
