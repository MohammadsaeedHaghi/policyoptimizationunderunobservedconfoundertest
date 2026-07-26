#!/usr/bin/env python3
"""Mechanism diagnostics for the "why each method picks its policy" report section.

License-free (numpy + scipy.linprog; NO Gurobi): because X lives on 7 levels, the 1-D
Wasserstein-1 distance between reweighted level distributions is linear in the weights
(sum of |CDF differences| x level gaps), so the box AND box+W worst cases of a FIXED
policy are small LPs.

Everything is computed on the seed-0 draw (N=600) with the empirically fitted weights
w_hat_i = 1 / e_hat_{T_i}(X_i) and the standard MSM box a=1+(w-1)/G, b=1+G(w-1).
The box+W adversary is Hajek-normalized per arm (sum of w over arm = n) so the
reweighted arm measures are probability distributions and W1(q_arm, p_pooled) <= eps_arm
with eps_arm = the fitted weights' own distance (the "tightest" budget, c_eps=1).
This mirrors (not byte-replicates) the production LP; it is the mechanism, quantified.

Outputs assets/exp_owgap_v2/mechanism.json:
  levels, gamma_grid,
  scores: per-level treat-vs-control worst-case score at G=5 for naive / box / box+W
  dists:  treated-arm reweighted level distribution: fitted / box adversary / box+W adversary
  curves: worst-case value vs Gamma for fixed policies (oracle, naive-inf, never) x (box, box+W)
"""
import sys, json
from pathlib import Path
import numpy as np
from scipy.optimize import linprog
import importlib.util

HERE = Path(__file__).resolve().parent
sp = importlib.util.spec_from_file_location("v2dgp", str(HERE / "dgp.py"))
d = importlib.util.module_from_spec(sp); sp.loader.exec_module(d)

LV = np.asarray(d.LEVELS, float); NL = len(LV); GAP = np.diff(LV)
obs, full = d.generate(600, 0)
X, T, Y = obs["X"].ravel(), obs["T"], obs["Y"]
n = len(X)
lvl = np.searchsorted(LV, X - 1e-9)                       # level index per unit

# fitted per-level propensity and IPW weights
ehat = np.array([T[lvl == j].mean() for j in range(NL)])
ehat = np.clip(ehat, 0.02, 0.98)
w_hat = np.where(T == 1, 1.0 / ehat[lvl], 1.0 / (1.0 - ehat[lvl]))

def msm_box(w, G):
    a = 1.0 + (w - 1.0) / G; b = 1.0 + G * (w - 1.0)
    return np.minimum(a, b), np.maximum(a, b)

def contrib(pi_lvl):
    """c_i: per-unit factual contribution multiplier so V(w) = (1/n) sum_i c_i w_i Y_i."""
    pi_i = pi_lvl[lvl]
    return np.where(T == 1, pi_i, 1.0 - pi_i)

def box_worst(pi_lvl, G):
    """Separable box-only worst case (plain IPW, unnormalized) and its per-unit w."""
    a, b = msm_box(w_hat, G); c = contrib(pi_lvl) * Y
    w = np.where(c > 0, a, b)
    return float(np.mean(c * w)), w

def _cdf_rows(mask):
    """Rows M s.t. M @ w = CDF of the arm's (mass/n) measure at levels 0..NL-2."""
    rows = []
    for j in range(NL - 1):
        r = np.zeros(n); sel = mask & (lvl <= j); r[sel] = 1.0 / n
        rows.append(r)
    return np.asarray(rows)

def boxw_worst(pi_lvl, G, eps=None, ret_w=False):
    """Box + W1 worst case: min V(w) s.t. box, per-arm Hajek normalization, and
    W1(q_arm, pooled empirical) <= eps_arm; W1 linearized via CDF aux vars."""
    a, b = msm_box(w_hat, G)
    cobj = contrib(pi_lvl) * Y / n
    p_cdf = np.array([(lvl <= j).mean() for j in range(NL - 1)])
    arms = [(T == 1), (T == 0)]
    Ms = [_cdf_rows(m) for m in arms]
    if eps is None:                                        # tightest: fitted weights' own distance
        eps = []
        for m, M in zip(arms, Ms):
            q = M @ (w_hat * (n / w_hat[m].sum()) * m)     # normalized fitted arm CDF
            eps.append(float(np.sum(np.abs(q - p_cdf) * GAP)))
    nu = 2 * (NL - 1)                                      # aux u_{arm,j} >= |q_cdf - p_cdf|
    cvec = np.concatenate([cobj, np.zeros(nu)])
    A_ub, b_ub = [], []
    for k, (m, M) in enumerate(zip(arms, Ms)):
        for j in range(NL - 1):
            u = np.zeros(nu); u[k * (NL - 1) + j] = 1.0
            A_ub.append(np.concatenate([M[j] * m, -u])); b_ub.append(p_cdf[j])
            A_ub.append(np.concatenate([-M[j] * m, -u])); b_ub.append(-p_cdf[j])
        grow = np.zeros(n + nu)
        grow[n + k * (NL - 1): n + (k + 1) * (NL - 1)] = GAP
        A_ub.append(grow); b_ub.append(eps[k])
    A_eq = [np.concatenate([m.astype(float), np.zeros(nu)]) for m in arms]
    b_eq = [n, n]                                          # Hajek: sum_{arm} w = n
    bounds = [(a[i], b[i]) for i in range(n)] + [(0, None)] * nu
    r = linprog(cvec, A_ub=np.asarray(A_ub), b_ub=np.asarray(b_ub),
                A_eq=np.asarray(A_eq), b_eq=np.asarray(b_eq), bounds=bounds, method="highs")
    if not r.success:
        return (float("nan"), None) if ret_w else float("nan")
    return (float(r.fun), r.x[:n]) if ret_w else float(r.fun)

def lvl_dist(w, mask):
    q = np.array([w[mask & (lvl == j)].sum() for j in range(NL)])
    return (q / q.sum()).tolist()

gt = d.grid_truth()
oracle_pi = (gt["cate"] > 0).astype(float)
naive_pi = np.array([0, 0, 0, 1, 1, 1, 1], float)          # infinite-data naive plug-in
never_pi = np.zeros(NL)
G0 = 5.0

# --- B1: per-level treat-vs-control worst-case scores at matched Gamma ---
scores = {"naive": [], "box": [], "boxw": []}
a5, b5 = msm_box(w_hat, G0)
for j in range(NL):
    tr = (lvl == j) & (T == 1); ct = (lvl == j) & (T == 0)
    m1 = np.sum(np.where(Y[tr] > 0, a5[tr], b5[tr]) * Y[tr]) / n
    m0 = np.sum(np.where(Y[ct] > 0, a5[ct], b5[ct]) * Y[ct]) / n
    scores["box"].append(round(float(NL * (m1 - m0)), 3))  # per-capita scale
    e1 = np.mean(Y[tr]) if tr.any() else 0.0; e0 = np.mean(Y[ct]) if ct.any() else 0.0
    scores["naive"].append(round(float(e1 - e0), 3))       # empirical within-level plug-in
for j in range(NL):
    hi = oracle_pi.copy(); hi[j] = 1.0
    lo = oracle_pi.copy(); lo[j] = 0.0
    scores["boxw"].append(round(NL * (boxw_worst(hi, G0) - boxw_worst(lo, G0)), 3))

# --- B2: treated-arm reweighted distributions at oracle policy, Gamma=5 ---
_, w_box = box_worst(oracle_pi, G0)
_, w_bw = boxw_worst(oracle_pi, G0, ret_w=True)
dists = {"empirical": [float(np.mean(lvl == j)) for j in range(NL)],
         "fitted": lvl_dist(w_hat, T == 1),
         "box_adversary": lvl_dist(w_box, T == 1),
         "boxw_adversary": lvl_dist(w_bw, T == 1) if w_bw is not None else None}

# --- B3: worst-case value vs Gamma for fixed policies ---
GG = [1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0]
curves = {}
for name, pi in [("oracle", oracle_pi), ("naive", naive_pi), ("never", never_pi)]:
    curves[name] = {"box": [round(box_worst(pi, g)[0], 3) for g in GG],
                    "boxw": [round(boxw_worst(pi, g), 3) for g in GG]}

out = {"levels": LV.tolist(), "gamma_grid": GG, "gamma_matched": G0, "N": n,
       "true_cate": [round(float(v), 3) for v in gt["cate"]],
       "scores": scores, "dists": dists, "curves": curves,
       "exact": {"oracle": round(d.exact_value(oracle_pi), 3),
                 "naive": round(d.exact_value(naive_pi), 3), "never": 0.0}}
(HERE / "mechanism.json").write_text(json.dumps(out, indent=1))
print("saved mechanism.json")
print("scores naive:", scores["naive"]); print("scores box  :", scores["box"])
print("scores box+W:", scores["boxw"])
print("curves oracle box:", curves["oracle"]["box"]); print("curves oracle boxw:", curves["oracle"]["boxw"])
print("curves naive boxw:", curves["naive"]["boxw"]); print("never:", curves["never"]["box"])
