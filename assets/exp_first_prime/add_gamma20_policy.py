"""Add Γ=20 to the N=10000 first-prime policy charts (ichart-p10k-{uncap,cap}-policy): solve the OX methods' deployed
policies at Γ=20 (XX and Oracle are Γ-free, copied from an existing Γ key) and add a seriesByGamma['20'] entry +
the dropdown option. Patches PRIME_N10K in index.html."""
import sys, json, re, importlib.util, copy
from pathlib import Path
import numpy as np
ROOT = Path("/Users/mohammadsaeedhaghi/Desktop/01-Research/paper - policy optimization under unobsorved confounder - code/code 1.1")
HERE = ROOT / "assets" / "exp_first_prime"
sys.path.insert(0, str(ROOT)); import common
sp = importlib.util.spec_from_file_location("pdgp", str(HERE / "dgp.py")); dgp = importlib.util.module_from_spec(sp); sys.modules["pdgp"] = dgp; sp.loader.exec_module(dgp)
def Lf(rel, fn):
    s = importlib.util.spec_from_file_location(fn, str(ROOT / rel)); m = importlib.util.module_from_spec(s); sys.modules[fn] = m; s.loader.exec_module(m); return getattr(m, fn)
K = dgp.K; rng = np.random.default_rng(0); tr = dgp.generate(10000, rng)
w, _ = common.ipw_weights_from_data(tr.X, tr.T, K); muhat = common.outcome_means(tr.X, tr.T, tr.Y, n_arms=K, cross_fit=True)
by = {}
for j, xv in enumerate(np.round(tr.X.ravel(), 6)): by.setdefault(round(float(xv), 6), j)
GRIDcol = np.array([by[round(float(x), 6)] for x in dgp.GRID])
def polgrid(pi): return [round(float(pi[1, GRIDcol[i]]), 4) for i in range(len(dgp.GRID))]
SOLV = {"uncap": {"IPW-O-X": Lf("methods/IPW-O-X/Uncapped/ipw_o_x_uncapped.py", "solve_ipw_o_x_uncapped"),
                  "DoublyRobust-O-X": Lf("methods/DoublyRobust-O-X/Uncapped/doublyrobust_o_x_uncapped.py", "solve_doublyrobust_o_x_uncapped"),
                  "Hajek-O-X": Lf("methods/Hajek-O-X/Uncapped/hajek_o_x_uncapped.py", "solve_hajek_o_x_uncapped")},
        "cap": {"IPW-O-X": Lf("methods/IPW-O-X/Capped/ipw_o_x_capped.py", "solve_ipw_o_x_capped"),
                "DoublyRobust-O-X": Lf("methods/DoublyRobust-O-X/Capped/doublyrobust_o_x_capped.py", "solve_doublyrobust_o_x_capped"),
                "Hajek-O-X": Lf("methods/Hajek-O-X/Capped/hajek_o_x_capped.py", "solve_hajek_o_x_capped")}}
pol20 = {}
for rg, cap in (("uncap", dgp.CAP_UNCAP), ("cap", dgp.CAP_CAP)):
    kw = {} if rg == "uncap" else {"cap": cap}; S = SOLV[rg]
    ro = S["IPW-O-X"](tr.X, tr.T, tr.Y, w, n_arms=K, Gamma=20.0, discretize=False, **kw)
    rd = S["DoublyRobust-O-X"](tr.X, tr.T, tr.Y, w, muhat, n_arms=K, Gamma=20.0, discretize=False, **kw)
    rh = S["Hajek-O-X"](tr.X, tr.T, tr.Y, w, n_arms=K, Gamma=20.0, maximize=True, discretize=False, **kw)
    pol20[rg] = {"IPW-O-X": polgrid(ro.pi), "DoublyRobust-O-X": polgrid(rd.pi), "Hajek-O-X": polgrid(rh.pi)}
    print("[%s] Γ=20 treat-fractions:" % rg, {m: round(float(np.mean(v)), 3) for m, v in pol20[rg].items()})

h = (ROOT / "index.html").read_text()
m = re.search(r'var PRIME_N10K = (.*?);\n', h); P = json.loads(m.group(1))
for rg in ("uncap", "cap"):
    pol = P[rg + "_policy"]
    if 20.0 not in pol["gammas"]: pol["gammas"].append(20.0)
    base_key = "8" if "8" in pol["seriesByGamma"] else list(pol["seriesByGamma"])[0]
    new = copy.deepcopy(pol["seriesByGamma"][base_key])      # XX + Oracle are Γ-free; copy then overwrite OX
    for s in new:
        if s["id"] in pol20[rg]: s["y"] = pol20[rg][s["id"]]
    pol["seriesByGamma"]["20"] = new
h = h[:m.start()] + "var PRIME_N10K = " + json.dumps(P, separators=(",", ":")) + ";\n" + h[m.end():]
(ROOT / "index.html").write_text(h)
print("added Γ=20 to PRIME_N10K policy dropdown (both regimes)")
