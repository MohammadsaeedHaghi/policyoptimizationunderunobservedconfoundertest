"""Add Direct-X-X's realised test E[Y] (flat, Γ-free) to the CAPPED value-vs-Γ chart of the fourth experiment.
IPW-X-X and DoublyRobust-X-X are already present; Direct-X-X was never run for the capped continuous DGP.
Same DGP/geometry/cap and the same true-means realised-value operator as run_wass_cap.py."""
import sys, json, re, importlib.util
from pathlib import Path
import numpy as np
ROOT = Path("/Users/mohammadsaeedhaghi/Desktop/01-Research/paper - policy optimization under unobsorved confounder - code/code 1.1")
HERE = ROOT / "assets" / "exp_wass"
sys.path.insert(0, str(ROOT)); import common
sp = importlib.util.spec_from_file_location("wdgp", str(HERE / "dgp.py")); d = importlib.util.module_from_spec(sp); sys.modules["wdgp"] = d; sp.loader.exec_module(d)
def L(rel, fn):
    s = importlib.util.spec_from_file_location(fn, str(ROOT / rel)); m = importlib.util.module_from_spec(s); sys.modules[fn] = m; s.loader.exec_module(m); return getattr(m, fn)
dirxx = L("methods/Direct-X-X/Capped/direct_x_x_capped.py", "solve_direct_x_x_capped")

K, N, MESH = 2, 1000, 11
CAP = (1.0, 0.5)
G = d.GRID
obs, full = d.generate(N, 0)
_, tf = d.generate(40000, 99); Xt = tf["X"].ravel(); St = tf["S"]
def to_grid(res):
    s = res.support_X.ravel(); lv = np.array(sorted(set(np.round(s, 6))))
    pol = np.array([float(res.pi[1, np.where(np.round(s, 6) == round(float(c), 6))[0][0]]) for c in lv])
    nn = lv[np.argmin(np.abs(np.asarray(G)[:, None] - lv[None, :]), axis=1)]
    idx = {round(float(c), 6): i for i, c in enumerate(lv)}
    return np.array([pol[idx[round(float(v), 6)]] for v in nn])
def realised(res):
    polG = to_grid(res)
    nn = G[np.argmin(np.abs(Xt[:, None] - G[None, :]), axis=1)]
    pos = {round(float(G[i]), 6): i for i in range(len(G))}
    pit = np.array([polG[pos[round(float(v), 6)]] for v in nn])
    return float(np.mean(pit * d.mu1(Xt, St) + (1 - pit) * d.mu0(Xt, St)))

r_dir = dirxx(obs["X"], obs["T"], obs["Y"], n_arms=K, cap=CAP, discretize=True, mesh=MESH)
v_dir = round(realised(r_dir), 4)
print("Direct-X-X capped realised test E[Y] = %.4f  (treat-frac %.3f)" % (v_dir, float(np.mean(to_grid(r_dir)))))

idx = ROOT / "index.html"; h = idx.read_text()
m = re.search(r'var WASSCAP_CHARTS = (.*?);\n', h); W = json.loads(m.group(1))
nG = len(W["value"]["x"])
W["value"]["series"] = [s for s in W["value"]["series"] if s["id"] != "Direct-X-X"]
oidx = next(i for i, s in enumerate(W["value"]["series"]) if s["id"] == "Oracle")
W["value"]["series"].insert(oidx, {"id": "Direct-X-X", "label": "Direct-X-X", "color": "#ff7f0e", "y": [v_dir] * nG})
h = h[:m.start()] + "var WASSCAP_CHARTS = " + json.dumps(W, separators=(",", ":")) + ";\n" + h[m.end():]
idx.write_text(h)
print("added Direct-X-X (flat) to capped value chart; series now:", [s["id"] for s in W["value"]["series"]])
