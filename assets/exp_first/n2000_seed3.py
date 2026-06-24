"""First experiment (Case 4), UNCAPPED, single seed=3 at N_train=2000 — all FOUR value methods at the SAME n
(the companion to the n=3000 panel where IPW-O-W had to run at n=600). Methods:
  Direct-X-X (Direct-X-X, unit weights, Γ-free), IPW-X-X (Hájek IPW, Γ-free),
  IPW-O-X (worst-case IPW value over the odds box) and IPW-O-W (box ∩ Wasserstein balance ball).
Γ grid = {1, e≈2.72, 8}. At Γ=1 the box pins W=ŵ so BOTH robust methods collapse to IPW-X-X — we copy
that curve rather than re-solve (saves the slow Wasserstein LP). IPW-O-W is solved only at Γ=e and Γ=8
(~35 min each at n=2000). Builds var POLICY_FIRST_N2000_S3 (Γ dropdown, single seed), splices into index.html,
regenerates _exp_first_view.html.  Usage: python3 n2000_seed3.py"""
import sys, json, re, importlib.util, time
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
K = dgp.K; GRID = dgp.GRID; N = 2000; SEED = 3; mu = dgp.mu_arm(GRID)
_, mi = dgp.gammas_for(dgp.GAMMA_CONF); MG = round(float(np.exp(dgp.GAMMA_CONF / 2)), 4)   # matched Γ=e≈2.72
GAMMAS = [1.0, MG, 8.0]
def gkey(g): g = float(g); return str(int(g)) if g == int(g) else str(g)
COL = {"IPW-X-X": "#2ca02c", "Direct-X-X": "#ff7f0e", "IPW-O-X": "#9467bd", "IPW-O-W": "#d62728"}

rng = np.random.default_rng(SEED); tr = dgp.generate(N, rng)
w, _ = common.ipw_weights_from_data(tr.X, tr.T, K); xr = np.round(tr.X.ravel(), 6)
def grid_of(pi): return [round(float(pi[1, np.where(xr == round(float(gx), 6))[0][0]]), 4) for gx in GRID]

# Γ-free baselines (one solve each)
g_dopt = grid_of(popt(tr.X, tr.T, tr.Y, n_arms=K, discretize=False).pi)
g_ipw  = grid_of(ipwxx(tr.X, tr.T, tr.Y, w, n_arms=K, discretize=False).pi)
print("Direct-X-X treat-frac=%.3f | IPW-X-X treat-frac=%.3f" % (np.mean(g_dopt), np.mean(g_ipw)), flush=True)

# per Γ: IPW-O-X (fast) and IPW-O-W (slow, only Γ>1). At Γ=1 both robust methods == IPW-X-X (box pins W=ŵ).
ox, ow = {}, {}
for g in GAMMAS:
    if g == 1.0:
        ox[gkey(g)] = list(g_ipw); ow[gkey(g)] = list(g_ipw)
        print("Γ=1: IPW-O-X = IPW-O-W = IPW-X-X (copied, no solve)", flush=True)
        continue
    t0 = time.time(); ox[gkey(g)] = grid_of(ipwox(tr.X, tr.T, tr.Y, w, n_arms=K, Gamma=float(g), discretize=False).pi)
    print("IPW-O-X Γ=%s treat-frac=%.3f (%.1fs)" % (gkey(g), np.mean(ox[gkey(g)]), time.time() - t0), flush=True)
    t0 = time.time(); ow[gkey(g)] = grid_of(ipwow(tr.X, tr.T, tr.Y, w, n_arms=K, Gamma=float(g), discretize=False, zscore=False).pi)
    print("IPW-O-W Γ=%s treat-frac=%.3f (%.1fs)" % (gkey(g), np.mean(ow[gkey(g)]), time.time() - t0), flush=True)

def ser(g):
    return [{"id": "IPW-X-X", "label": "IPW-X-X", "color": COL["IPW-X-X"], "y": g_ipw},
            {"id": "Direct-X-X", "label": "Direct-X-X", "color": COL["Direct-X-X"], "y": g_dopt},
            {"id": "IPW-O-X", "label": "IPW-O-X", "color": COL["IPW-O-X"], "y": ox[gkey(g)]},
            {"id": "IPW-O-W", "label": "IPW-O-W", "color": COL["IPW-O-W"], "y": ow[gkey(g)]}]
sbsg = {"3": {gkey(g): ser(g) for g in GAMMAS}}
DATA = {"x": [float(x) for x in GRID], "xmin": -1.0, "xmax": 1.0, "xlabel": "X", "ylabel": "π(treat | x)",
        "ymin": 0.0, "ymax": 1.0, "gammas": [float(g) for g in GAMMAS], "defaultGamma": float(MG), "matchedGamma": float(MG),
        "seeds": [{"key": "3", "label": "Seed 3"}], "defaultSeed": "3", "seriesBySeedGamma": sbsg,
        "note": "π(treat|X), uncapped, N=2000, seed 3 · all four methods at the SAME n · Γ∈{1, e≈2.72, 8} · matched Γ=%.2f" % MG}
(HERE / "policy_first_n2000_seed3.json").write_text(json.dumps(DATA))
print("\nsaved policy_first_n2000_seed3.json", flush=True)

idx = NEW / "index.html"; h = idx.read_text()
line = "var POLICY_FIRST_N2000_S3 = %s;" % json.dumps(DATA, separators=(',', ':'))
if re.search(r'^var POLICY_FIRST_N2000_S3 = .*;$', h, flags=re.M):
    h = re.sub(r'^var POLICY_FIRST_N2000_S3 = .*;$', lambda m: line, h, count=1, flags=re.M)
else:
    h = re.sub(r'^(var POLICY_FIRST_BIGN = .*;)$', lambda m: m.group(1) + "\n" + line, h, count=1, flags=re.M)
call = "renderChart('ichart-first-n2000',POLICY_FIRST_N2000_S3);"
if call not in h:
    h = h.replace("renderChart('ichart-first-bigN',POLICY_FIRST_BIGN);",
                  "renderChart('ichart-first-bigN',POLICY_FIRST_BIGN);\n" + call, 1)
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
print("spliced POLICY_FIRST_N2000_S3 + renderChart call into index.html; regenerated viewer", flush=True)
