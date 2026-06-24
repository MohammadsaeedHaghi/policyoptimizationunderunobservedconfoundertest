"""UNCAPPED Wasserstein (OW) methods on the first-prime DGP, N_train=1500 (same seed/train/test as run_prime.py).
IPW-O-W and DoublyRobust-O-W, EXACT per-unit transport (discretize=False, NO cell aggregation), no cap.
Solves over the same Γ grid as the XX/OX run (Γ=1 = the capped X-X parent, not re-solved).  ε computed once.
Patches PRIME_RES.uncap_value + PRIME_RES.uncap_policy in index.html to add the two OW series (blue/green circles)."""
import sys, json, re, importlib.util, time
from pathlib import Path
import numpy as np
ROOT = Path("/Users/mohammadsaeedhaghi/Desktop/01-Research/paper - policy optimization under unobsorved confounder - code/code 1.1")
HERE = ROOT / "assets" / "exp_first_prime"
sys.path.insert(0, str(ROOT)); import common
sp = importlib.util.spec_from_file_location("pdgp", str(HERE / "dgp.py")); dgp = importlib.util.module_from_spec(sp); sys.modules["pdgp"] = dgp; sp.loader.exec_module(dgp)
def Lf(rel, fn):
    s = importlib.util.spec_from_file_location(fn, str(ROOT / rel)); m = importlib.util.module_from_spec(s); sys.modules[fn] = m; s.loader.exec_module(m); return getattr(m, fn)
ipwow = Lf("methods/IPW-O-W/Uncapped/ipw_o_w_uncapped.py", "solve_ipw_o_w_uncapped")
drow  = Lf("methods/DoublyRobust-O-W/Uncapped/doublyrobust_o_w_uncapped.py", "solve_doublyrobust_o_w_uncapped")

K = dgp.K; CAP = dgp.CAP_UNCAP; NTR, NTE = 1500, 20000
R0 = json.loads((HERE / "prime_results.json").read_text())
GAMMAS = R0["gammas"]; G = np.array(R0["grid"])
def gk(g): g = float(g); return str(int(g)) if g == int(g) else str(g)
rng = np.random.default_rng(0); tr = dgp.generate(NTR, rng); te = dgp.generate(NTE, rng)
w, _ = common.ipw_weights_from_data(tr.X, tr.T, K)
muhat = common.outcome_means(tr.X, tr.T, tr.Y, n_arms=K, cross_fit=True)
by = {}
for j, xv in enumerate(np.round(tr.X.ravel(), 6)): by.setdefault(round(float(xv), 6), j)
GRIDcol = np.array([by[round(float(x), 6)] for x in dgp.GRID])
nn_te = np.array([by[round(float(x), 6)] for x in np.round(te.X.ravel(), 6)])
def metrics(pi):
    pi = np.asarray(pi, float); pte = pi[:, nn_te]
    return float((pte * te.Ypot.T).sum() / pte.shape[1]), float(pi[1].mean())
def polgrid(pi): return [round(float(pi[1, GRIDcol[i]]), 4) for i in range(len(dgp.GRID))]

Dm = common.pairwise_distance_matrix(tr.X)
eps = tuple(common.tight_epsilon(Dm, tr.T, w, K, is_distance=True, c_eps=1.0))
print("tight ε (per arm): %.5f , %.5f" % (eps[0], eps[1]), flush=True)

# Γ=1 anchors = capped X-X parents (from the XX/OX run)
anchor = {"IPW-O-W": R0["regimes"]["uncap"]["value"]["IPW-X-X"][0],
          "DoublyRobust-O-W": R0["regimes"]["uncap"]["value"]["DoublyRobust-X-X"][0]}
pol_anchor = {"IPW-O-W": R0["regimes"]["uncap"]["policy"]["IPW-X-X"]["_flat"],
              "DoublyRobust-O-W": R0["regimes"]["uncap"]["policy"]["DoublyRobust-X-X"]["_flat"]}

SOLVE = [4.0, float(R0["matched_gamma"])]   # TRIMMED grid: solve only these; others -> null on the 7-pt x-axis
val = {"IPW-O-W": [], "DoublyRobust-O-W": []}; pol = {"IPW-O-W": {}, "DoublyRobust-O-W": {}}
for g in GAMMAS:
    k = gk(g)
    if g == 1.0:
        for nm in val: val[nm].append(round(anchor[nm], 4)); pol[nm][k] = pol_anchor[nm]
        print("Γ=1 (copied parent): IPW-O-W=%.4f DR-O-W=%.4f" % (anchor["IPW-O-W"], anchor["DoublyRobust-O-W"]), flush=True)
        continue
    if g not in SOLVE:
        for nm in val: val[nm].append(None)
        print("Γ=%s skipped (trimmed grid)" % k, flush=True); continue
    t0 = time.time()
    ro = ipwow(tr.X, tr.T, tr.Y, w, n_arms=K, Gamma=float(g), discretize=False, zscore=False, epsilon=eps)
    rd = drow(tr.X, tr.T, tr.Y, w, muhat, n_arms=K, Gamma=float(g), discretize=False, zscore=False, epsilon=eps)
    vo, _ = metrics(ro.pi); vd, _ = metrics(rd.pi)
    val["IPW-O-W"].append(round(vo, 4)); val["DoublyRobust-O-W"].append(round(vd, 4))
    pol["IPW-O-W"][k] = polgrid(ro.pi); pol["DoublyRobust-O-W"][k] = polgrid(rd.pi)
    print("Γ=%-4s IPW-O-W=%.4f DR-O-W=%.4f  (%.1f min)" % (k, vo, vd, (time.time() - t0) / 60), flush=True)

out = {"N_train": NTR, "cap": list(CAP), "gammas": GAMMAS, "epsilon": list(eps), "value": val, "policy": pol}
(HERE / "prime_ow_uncap_results.json").write_text(json.dumps(out, indent=2))
print("\nsaved prime_ow_uncap_results.json", flush=True)
print("IPW-O-W byΓ=%s\nDR-O-W byΓ=%s" % (val["IPW-O-W"], val["DoublyRobust-O-W"]), flush=True)

# ---- patch PRIME_RES.cap_value + cap_policy ----
idx = ROOT / "index.html"; h = idx.read_text()
m = re.search(r'var PRIME_RES = (.*?);\n', h); P = json.loads(m.group(1))
def ser(idl, color, y): return {"id": idl, "label": idl, "color": color, "y": [None if v is None else round(float(v), 4) for v in y]}
COL = {"IPW-O-W": "#1f77b4", "DoublyRobust-O-W": "#2ca02c"}
# value: insert before the Oracle series (keep Oracle last)
cv = P["uncap_value"]["series"]; cv[:] = [s for s in cv if s["id"] not in COL]
oidx = next(i for i, s in enumerate(cv) if s["id"] == "Oracle")
for nm in ("IPW-O-W", "DoublyRobust-O-W"): cv.insert(oidx, ser(nm, COL[nm], val[nm])); oidx += 1
# policy: per Γ, insert before Oracle
for k, sl in P["uncap_policy"]["seriesByGamma"].items():
    sl[:] = [s for s in sl if s["id"] not in COL]
    oi = next(i for i, s in enumerate(sl) if s["id"] == "Oracle")
    for nm in ("IPW-O-W", "DoublyRobust-O-W"):
        yy = pol[nm].get(k, pol[nm].get(gk(float(k)), [None] * len(G)))
        sl.insert(oi, ser(nm, COL[nm], yy)); oi += 1
P["uncap_value"]["note"] = P["uncap_value"]["note"] + " Wasserstein (-O-W, circles) added."
h = h[:m.start()] + "var PRIME_RES = " + json.dumps(P, separators=(",", ":")) + ";\n" + h[m.end():]
idx.write_text(h)
print("patched PRIME_RES cap_value + cap_policy with IPW-O-W + DoublyRobust-O-W", flush=True)
