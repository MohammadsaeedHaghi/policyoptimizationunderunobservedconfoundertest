"""Diagnostics for exp_owgap_v2 (no Gurobi needed) -> v2_diag.json.

Computes, exactly on the level grid and by Monte Carlo at N=600:
  - true CATE per level vs the infinite-data naive plug-in CATE-hat (the ranking-inversion plot)
  - values of never / all-treat / oracle / naive plug-in, uncapped and capped(0.3)
  - the true selection odds ratio Lambda (matched Gamma)
Run:  python3 assets/exp_owgap_v2/make_diagnostics.py
"""
import json, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
import sys; sys.path.insert(0, HERE)
import dgp as d

X = d.LEVELS
t = d.grid_truth()
cate, eY0, eY1 = t["cate"], t["eY0"], t["eY1"]
p = t["p_s1"]; ep, em = t["e_plus"], t["e_minus"]

# infinite-data naive plug-in: E[Y|X,T=t] mixes S by within-level selection
pS1_T1 = p * ep / (p * ep + (1 - p) * em)
pS1_T0 = p * (1 - ep) / (p * (1 - ep) + (1 - p) * (1 - em))
catehat_inf = (d.D1 * (2 * pS1_T1 - 1) + d.B1 * X - d.THETA) - (d.D0 * (2 * pS1_T0 - 1) + d.B0 * X)

val = lambda pi: float(np.mean(np.asarray(pi) * eY1 + (1 - np.asarray(pi)) * eY0))
CAPF = d.CAP[1]

def capped_greedy(scores, cap=CAPF):
    order = np.argsort(-np.asarray(scores)); pi = np.zeros(len(X)); b = cap * len(X)
    for i in order:
        if scores[i] <= 0 or b <= 0: break
        pi[i] = min(1.0, b); b -= pi[i]
    return pi

Lam = float(np.max(np.maximum(ep / (1 - ep) * (1 - em) / em, em / (1 - em) * (1 - ep) / ep)))

# finite-sample MC of the naive per-level plug-in policy
vu, vc = [], []
N_MC, SEEDS_MC = 600, 20
for seed in range(SEEDS_MC):
    obs, full = d.generate(N_MC, seed)
    Xi, T, Y = obs["X"].ravel(), obs["T"], obs["Y"]
    ch = np.zeros(len(X))
    for i, x in enumerate(X):
        m = np.isclose(Xi, x)
        y1, y0 = Y[m & (T == 1)], Y[m & (T == 0)]
        ch[i] = (y1.mean() if len(y1) else -99) - (y0.mean() if len(y0) else 99)
    vu.append(val((ch > 0).astype(float))); vc.append(val(capped_greedy(ch)))

out = dict(
    levels=X.tolist(), cate=cate.tolist(), catehat_naive_inf=catehat_inf.tolist(),
    p_s1=p.tolist(), eY0=eY0.tolist(), eY1=eY1.tolist(),
    lambda_true=Lam, cap=CAPF, theta=d.THETA, b1=d.B1,
    values=dict(
        never=val(np.zeros(len(X))), all_treat=val(np.ones(len(X))),
        oracle_uncap=val((cate > 0).astype(float)), oracle_cap=val(capped_greedy(cate)),
        naive_inf_uncap=val((catehat_inf > 0).astype(float)), naive_inf_cap=val(capped_greedy(catehat_inf)),
        naive_mc_uncap=[float(np.mean(vu)), float(np.std(vu))],
        naive_mc_cap=[float(np.mean(vc)), float(np.std(vc))],
        n_mc=N_MC, seeds_mc=SEEDS_MC,
    ),
)
path = os.path.join(HERE, "v2_diag.json")
json.dump(out, open(path, "w"), indent=1)
print(f"wrote {path}")
print(f"Lambda={Lam:.2f}  cap={CAPF}")
v = out["values"]
print(f"never={v['never']:.3f} all={v['all_treat']:.3f} | oracle uncap={v['oracle_uncap']:.3f} cap={v['oracle_cap']:.3f}")
print(f"naive inf: uncap={v['naive_inf_uncap']:.3f} cap={v['naive_inf_cap']:.3f}")
print(f"naive MC : uncap={v['naive_mc_uncap'][0]:.3f}+-{v['naive_mc_uncap'][1]:.3f} cap={v['naive_mc_cap'][0]:.3f}+-{v['naive_mc_cap'][1]:.3f}")
