"""IPW-O-W at TRAIN N=5000 on the HR x_varying DGP — the "hard task".

The library IPW-O-W LP and tight_epsilon are BOTH O(n²) (transport over n × |I_k|), so they OOM at n=5000.
But X is discrete (21 cells), so the transport collapses to the cell level EXACTLY:
  • tight ε  : the transport from the empirical X-law to the arm-reweighted law is a 21×21 (cell) transport.
  • IPW-O-W  : the transport-demand dual gd[k,i] is constant within an X-cell at the optimum (its only upper
               bounds are the metric constraints β_k D_ij − gd[k,i] − θ[k,j] ≥ 0, and D_ij depends only on the
               cells), so we index gd PER CELL and add metric constraints per (cell, factual-unit). Everything
               else (π tied per cell, θ/μ/ν per unit, objective) is byte-identical to the library dual.

We VALIDATE both reductions against the library solver at n=400 (where the library is feasible); if the policy,
objective and ε match, we trust the reduction and run at n=5000. Reports ε_t per arm. Usage: python3 run_ipwow_n5000.py"""
import sys, json, re, importlib.util
from pathlib import Path
import numpy as np
import gurobipy as gp
from gurobipy import GRB

ROOT = Path("/Users/mohammadsaeedhaghi/Desktop/01-Research/paper - policy optimization under unobsorved confounder - code/code 1.1")
HERE = ROOT / "assets" / "exp_hr"
sys.path.insert(0, str(ROOT)); import common
from common.sensitivity import marginal_sensitivity_box
from common.gurobi_env import configure_gurobi_license
sp = importlib.util.spec_from_file_location("hrdgp", str(HERE / "dgp.py")); d = importlib.util.module_from_spec(sp); sys.modules["hrdgp"] = d; sp.loader.exec_module(d)
def L(rel, fn):
    s = importlib.util.spec_from_file_location(fn, str(ROOT / rel)); m = importlib.util.module_from_spec(s); sys.modules[fn] = m; s.loader.exec_module(m); return getattr(m, fn)
solve_ipw_o_w = L("methods/IPW-O-W/Uncapped/ipw_o_w_uncapped.py", "solve_ipw_o_w_uncapped")

GRID = d.X_GRID
K = 2


def _cells(X):
    xr = np.round(np.asarray(X, float).ravel(), 2)
    cells = np.array(sorted(set(xr.tolist())))
    cidx = {round(float(c), 2): k for k, c in enumerate(cells)}
    idx = np.array([cidx[round(float(x), 2)] for x in xr])
    return cells, idx


def cell_tight_epsilon(X, T, w_hat, n_arms, c_eps=1.0):
    """ε_k by the cell-aggregated transport (exact equivalent of common.tight_epsilon on discrete X)."""
    cells, cidx = _cells(X); nc = len(cells); n = len(T); T = np.asarray(T).astype(int)
    out = []
    for k in range(n_arms):
        demand = np.array([(cidx == c).sum() for c in range(nc)], float) / n          # empirical law, mass 1/n each
        supply = np.array([w_hat[(cidx == c) & (T == k)].sum() for c in range(nc)]) / n  # arm-k reweighted law
        configure_gurobi_license()
        m = gp.Model("celleps"); m.Params.OutputFlag = 0
        z = m.addVars(nc, nc, lb=0.0)                                                  # ζ_{c->c'}
        m.setObjective(gp.quicksum(abs(float(cells[a] - cells[b])) * z[a, b] for a in range(nc) for b in range(nc)), GRB.MINIMIZE)
        for b in range(nc):
            m.addConstr(gp.quicksum(z[a, b] for a in range(nc)) == float(supply[b]))   # col marginal
        for a in range(nc):
            m.addConstr(gp.quicksum(z[a, b] for b in range(nc)) == float(demand[a]))   # row marginal
        m.optimize()
        out.append(c_eps * float(m.ObjVal))
    return tuple(out)


def solve_ipwow_reduced(X, T, Y, w_hat, n_arms, Gamma, epsilon):
    """IPW-O-W dual with gd indexed PER CELL (exact reduction for discrete X). Returns (pi_grid, objval)."""
    T = np.asarray(T).astype(int); Y = np.asarray(Y, float); w_hat = np.asarray(w_hat, float)
    n = len(T); cells, cidx = _cells(X); nc = len(cells)
    a_box, b_box = marginal_sensitivity_box(w_hat, Gamma)
    cnt_cell = np.array([(cidx == c).sum() for c in range(nc)], float)
    i_by_t = {k: np.where(T == k)[0] for k in range(n_arms)}
    configure_gurobi_license()
    m = gp.Model("ipwow_reduced"); m.Params.OutputFlag = 0; m.Params.DualReductions = 0
    pi = m.addVars(n_arms, nc, lb=0.0, ub=1.0)
    beta = m.addVars(n_arms, lb=0.0)
    gd = m.addVars(n_arms, nc, lb=-GRB.INFINITY)
    theta = {(k, int(j)): m.addVar(lb=-GRB.INFINITY) for k in range(n_arms) for j in i_by_t[k]}
    mu = m.addVars(n, lb=0.0); nu = m.addVars(n, lb=0.0)
    obj = gp.LinExpr()
    for k in range(n_arms):
        obj += -beta[k] * float(epsilon[k])
        obj += (1.0 / n) * gp.quicksum(cnt_cell[c] * gd[k, c] for c in range(nc))
    for i in range(n):
        obj += mu[i] * float(a_box[i]) - nu[i] * float(b_box[i])
    m.setObjective(obj, GRB.MAXIMIZE)
    for c in range(nc):
        m.addConstr(gp.quicksum(pi[k, c] for k in range(n_arms)) == 1.0)
    for k in range(n_arms):
        for j in i_by_t[k]:
            j = int(j)
            m.addConstr((1.0 / n) * pi[k, cidx[j]] * float(Y[j]) + (1.0 / n) * theta[k, j] - mu[j] + nu[j] >= 0.0)
    for k in range(n_arms):
        for c in range(nc):
            for j in i_by_t[k]:
                j = int(j)
                m.addConstr(beta[k] * abs(float(cells[c] - X.ravel()[j])) - gd[k, c] - theta[k, j] >= 0.0)
    m.optimize()
    if m.Status != GRB.OPTIMAL:
        raise RuntimeError("reduced IPW-O-W non-optimal status=%d Gamma=%s" % (m.Status, Gamma))
    pol = np.array([float(pi[1, cidx[round(float(g), 2)]].X) for g in GRID]) if False else \
          np.array([round(float(pi[1, c].X), 4) for c in range(nc)])
    # map cells -> GRID order (cells already sorted == GRID)
    return np.array([round(float(v), 4) for v in pol]), float(m.ObjVal)


# ============================== VALIDATION at n=400 ==============================
print("=== VALIDATION at n=400 (reduced vs library) ===", flush=True)
obs4, _ = d.generate_data(400, 7, "x_varying")
w4, P4 = common.count_ipw_weights_from_data(obs4["X"], obs4["T"], K)
D4, _, _ = common.distance_matrix(np.asarray(obs4["X"], float).reshape(-1, 1), zscore=False)
eps_lib = common.tight_epsilon(D4, obs4["T"], w4, K, is_distance=True, c_eps=1.0)
eps_red = cell_tight_epsilon(obs4["X"], obs4["T"], w4, K, c_eps=1.0)
print("  ε  library =", tuple(round(e, 5) for e in eps_lib))
print("  ε  reduced =", tuple(round(e, 5) for e in eps_red), " | max|Δ| = %.2e" % max(abs(a - b) for a, b in zip(eps_lib, eps_red)))
ok = True
for G in (2.0, 6.0):
    libres = solve_ipw_o_w(obs4["X"], obs4["T"], obs4["Y"], w4, n_arms=K, Gamma=G, discretize=False, zscore=False, epsilon=eps_lib)
    xr4 = np.round(obs4["X"].ravel(), 2)
    lib_grid = np.array([round(float(libres.pi[1, np.where(xr4 == round(float(g), 2))[0][0]]), 4) for g in GRID])
    red_grid, red_obj = solve_ipwow_reduced(obs4["X"], obs4["T"], obs4["Y"], w4, K, G, eps_red)
    dpi = float(np.max(np.abs(lib_grid - red_grid))); dobj = abs(float(libres.objective_value) - red_obj)
    print("  Γ=%g  policy max|Δ|=%.2e  obj lib=%.5f red=%.5f Δ=%.2e" % (G, dpi, libres.objective_value, red_obj, dobj))
    ok = ok and dpi < 1e-4 and dobj < 1e-4
print("  VALIDATION:", "PASS" if ok else "FAIL", flush=True)
if not ok:
    print("Aborting — reduction does not match the library."); sys.exit(1)

# ============================== RUN at n=5000 ==============================
print("\n=== IPW-O-W at TRAIN N=5000 (reduced, exact) ===", flush=True)
N_TR = 5000
obs5, _ = d.generate_data(N_TR, 0, "x_varying")
w5, P5 = common.count_ipw_weights_from_data(obs5["X"], obs5["T"], K)
eps5 = cell_tight_epsilon(obs5["X"], obs5["T"], w5, K, c_eps=1.0)
print("  ε_t used (per arm, c_eps=1.0):  ε_0(control)=%.5f   ε_1(treat)=%.5f" % (eps5[0], eps5[1]), flush=True)

# test set + realised-value operator (true potential outcomes)
_, tf = d.generate_data(5000, 1, "x_varying")
txr = np.round(tf["X"], 2); Y1t = tf["Y1"].astype(float); Y0t = tf["Y0"].astype(float)
_pos = {round(float(GRID[i]), 2): i for i in range(len(GRID))}
def test_realised(pol_grid):
    pit = np.array([pol_grid[_pos[round(float(x), 2)]] for x in txr])
    return float(np.mean(pit * Y1t + (1.0 - pit) * Y0t))

GAMMAS = [1, 2, 3, 4, 6, 8, 12]
pol_by_g, rv_by_g = {}, {}
for G in GAMMAS:
    pol, _ = solve_ipwow_reduced(obs5["X"], obs5["T"], obs5["Y"], w5, K, float(G), eps5)
    pol_by_g[G] = [float(v) for v in pol]; rv_by_g[G] = test_realised(pol)
    print("  Γ=%-2d  treat-frac=%.2f  realised test E[Y]=%.4f" % (G, float(np.mean(pol)), rv_by_g[G]), flush=True)

out = {"epsilon": {"control": eps5[0], "treat": eps5[1]}, "gammas": GAMMAS,
       "policy_by_gamma": pol_by_g, "realised_test_by_gamma": rv_by_g, "N_train": N_TR, "N_test": 5000}
(HERE / "hr_ipwow_n5000.json").write_text(json.dumps(out, indent=2))
print("\nsaved hr_ipwow_n5000.json", flush=True)

# patch the value-vs-Γ chart: replace IPW-O-W y-values with the N=5000 ones
idx = ROOT / "index.html"; h = idx.read_text()
m = re.search(r'var HR_VALUE_GAMMA = (.*?);\n', h)
HV = json.loads(m.group(1))
newy = [round(rv_by_g[g], 4) for g in GAMMAS]
for s in HV["series"]:
    if s["id"] == "IPW-O-W":
        s["y"] = newy; s["label"] = "IPW-O-W"
HV["note"] = "Realised test E[Y] vs Γ (N_test=5000). Flat: IPW-X-X/Direct-X-X/Oracle. IPW-O-X/Hajek-O-X & IPW-O-W now trained at N=5000 (IPW-O-W via the exact discrete cell-reduction); Hajek-O-W still N=600."
h = h[:m.start()] + "var HR_VALUE_GAMMA = " + json.dumps(HV, separators=(",", ":")) + ";\n" + h[m.end():]
idx.write_text(h)
print("patched HR_VALUE_GAMMA: IPW-O-W ->", newy)
