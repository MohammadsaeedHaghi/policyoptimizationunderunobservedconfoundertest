"""Add the deployed policies of the three XX methods (IPW-X-X, Direct-X-X, DoublyRobust-X-X) to the CAPPED policy
chart of the fourth (continuous) experiment.  These are Γ-free, so the same policy is shown at every Γ.  Same DGP /
geometry / cap as run_wass_cap.py.  Patches WASSCAP_CHARTS.policy.seriesByGamma in index.html."""
import sys, json, re, importlib.util
from pathlib import Path
import numpy as np
ROOT = Path("/Users/mohammadsaeedhaghi/Desktop/01-Research/paper - policy optimization under unobsorved confounder - code/code 1.1")
HERE = ROOT / "assets" / "exp_wass"
sys.path.insert(0, str(ROOT)); import common
sp = importlib.util.spec_from_file_location("wdgp", str(HERE / "dgp.py")); d = importlib.util.module_from_spec(sp); sys.modules["wdgp"] = d; sp.loader.exec_module(d)
def L(rel, fn):
    s = importlib.util.spec_from_file_location(fn, str(ROOT / rel)); m = importlib.util.module_from_spec(s); sys.modules[fn] = m; s.loader.exec_module(m); return getattr(m, fn)
ipwxx = L("methods/IPW-X-X/Capped/ipw_x_x_capped.py", "solve_ipw_x_x_capped")
drxx  = L("methods/DoublyRobust-X-X/Capped/doublyrobust_x_x_capped.py", "solve_doublyrobust_x_x_capped")
dirxx = L("methods/Direct-X-X/Capped/direct_x_x_capped.py", "solve_direct_x_x_capped")

K, N, MESH = 2, 1000, 11
CAP = (1.0, 0.5)
G = d.GRID
obs, full = d.generate(N, 0)
w, P = common.ipw_weights_from_data(obs["X"], obs["T"], K)
muhat = common.outcome_means(obs["X"], obs["T"], obs["Y"], K)

def to_grid(res):
    s = res.support_X.ravel(); lv = np.array(sorted(set(np.round(s, 6))))
    pol = np.array([float(res.pi[1, np.where(np.round(s, 6) == round(float(c), 6))[0][0]]) for c in lv])
    nn = lv[np.argmin(np.abs(np.asarray(G)[:, None] - lv[None, :]), axis=1)]
    idx = {round(float(c), 6): i for i, c in enumerate(lv)}
    return [round(float(pol[idx[round(float(v), 6)]]), 4) for v in nn]

r_ipw = ipwxx(obs["X"], obs["T"], obs["Y"], w, n_arms=K, cap=CAP, discretize=True, mesh=MESH)
r_dr  = drxx(obs["X"], obs["T"], obs["Y"], w, muhat, n_arms=K, cap=CAP, discretize=True, mesh=MESH)
r_dir = dirxx(obs["X"], obs["T"], obs["Y"], n_arms=K, cap=CAP, discretize=True, mesh=MESH)
pols = {"IPW-X-X": to_grid(r_ipw), "DoublyRobust-X-X": to_grid(r_dr), "Direct-X-X": to_grid(r_dir)}
print("capped XX treat-fractions:", {k: round(float(np.mean(v)), 3) for k, v in pols.items()})

COL = {"IPW-X-X": "#1f77b4", "DoublyRobust-X-X": "#2ca02c", "Direct-X-X": "#ff7f0e"}
idx = ROOT / "index.html"; h = idx.read_text()
m = re.search(r'var WASSCAP_CHARTS = (.*?);\n', h); W = json.loads(m.group(1))
for gkey, sl in W["policy"]["seriesByGamma"].items():
    sl[:] = [s for s in sl if s["id"] not in COL]          # idempotent
    for nm in ("IPW-X-X", "DoublyRobust-X-X", "Direct-X-X"):
        sl.insert(0, {"id": nm, "label": nm, "color": COL[nm], "y": pols[nm]})
W["policy"]["note"] = W["policy"]["note"] + " XX baselines (dashed) added: IPW-X-X, DoublyRobust-X-X, Direct-X-X."
h = h[:m.start()] + "var WASSCAP_CHARTS = " + json.dumps(W, separators=(",", ":")) + ";\n" + h[m.end():]
idx.write_text(h)
print("patched WASSCAP_CHARTS.policy with 3 XX policies (Γ-free)")
