#!/usr/bin/env python3
"""exp_owgap_alpha diagnostics: how the data LOOKS at each coupling strength ALPHA.

For each ALPHA in {1,2,4,6,10} this computes (no solver, no Gurobi):
  - the P(S=+1|X) coupling curve (fine grid + the 7 levels),
  - CATE(X) = (D1-D0)(2 sigma(ALPHA X) - 1) + B1 X  (zero-crossing at X=0 for every ALPHA),
  - Monte-Carlo (n=200k) corr(X,S), corr(T,S), P(T=1), P(S=+1),
  - the TRUE selection odds ratio Lambda = odds(e(X,+1))/odds(e(X,-1)) per level (post-clip);
    the sweep leaves it untouched -- only the X-S coupling varies,
  - oracle / never-treat / all-treat exact values.

Writes alpha_diag.json + three PNGs here, and copies the JSON to
"selected experiment/exp_owgap_alpha/" for the report build.
"""
import json, os, sys, importlib.util
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
SEL = os.path.join(os.path.dirname(os.path.dirname(HERE)), "selected experiment", "exp_owgap_alpha")
ALPHAS = [1, 2, 4, 6, 10]
# sequential single-hue ramp, light -> dark = weak -> strong coupling
SEQ = {1: '#bdd7e7', 2: '#6baed6', 4: '#3182bd', 6: '#08519c', 10: '#08306b'}


def _load(a):
    p = os.path.join(HERE, "dgp_a%d.py" % a)
    sp = importlib.util.spec_from_file_location("diag_a%d" % a, p)
    m = importlib.util.module_from_spec(sp); sp.loader.exec_module(m)
    return m


def main():
    xf = np.linspace(-1, 1, 201)
    diag = {"alphas": ALPHAS, "x_fine": [round(float(v), 4) for v in xf], "per": {}}
    for a in ALPHAS:
        d = _load(a); t = d.grid_truth()
        obs, full = d.generate(200000, seed=0)
        X = obs["X"].ravel(); S = full["S"]; T = obs["T"]
        ep, em = t["e_plus"], t["e_minus"]
        lam = (ep / (1 - ep)) / (em / (1 - em))
        diag["per"][str(a)] = {
            "ps1_fine": [round(float(v), 4) for v in d.p_s1(xf)],
            "cate_fine": [round(float(v), 4) for v in
                          ((d._m.D1 - d._m.D0) * (2 * d.p_s1(xf) - 1) + d._m.B1 * xf)] if hasattr(d, "_m") else None,
            "levels": [float(v) for v in d.LEVELS],
            "ps1_levels": [round(float(v), 4) for v in t["p_s1"]],
            "cate_levels": [round(float(v), 4) for v in t["cate"]],
            "e_plus": [round(float(v), 4) for v in ep], "e_minus": [round(float(v), 4) for v in em],
            "e_marg": [round(float(v), 4) for v in t["e_marg"]],
            "lambda_true": [round(float(v), 3) for v in lam],
            "corr_XS": round(float(np.corrcoef(X, S)[0, 1]), 4),
            "corr_TS": round(float(np.corrcoef(T, S)[0, 1]), 4),
            "p_T1": round(float(T.mean()), 4), "p_S1": round(float((S > 0).mean()), 4),
            "oracle": round(d.exact_value(t["oracle"]), 4),
            "never": round(d.exact_value(np.zeros(len(d.LEVELS))), 4),
            "all": round(d.exact_value(np.ones(len(d.LEVELS))), 4),
        }
    # cate_fine fallback (wrapper always has _m, but keep the script standalone-safe)
    for a in ALPHAS:
        pa = diag["per"][str(a)]
        if pa["cate_fine"] is None:
            d = _load(a)
            pa["cate_fine"] = [round(float(v), 4) for v in (2 * d.p_s1(xf) - 1) + 1.5 * xf]

    out = os.path.join(HERE, "alpha_diag.json")
    with open(out, "w") as f:
        json.dump(diag, f, indent=1)
    os.makedirs(SEL, exist_ok=True)
    with open(os.path.join(SEL, "alpha_diag.json"), "w") as f:
        json.dump(diag, f, indent=1)

    # ---- P(S=+1|X)
    fig, ax = plt.subplots(figsize=(7, 4.3))
    for a in ALPHAS:
        pa = diag["per"][str(a)]
        ax.plot(xf, pa["ps1_fine"], color=SEQ[a], lw=2, label='α=%d  (corr=%.2f)' % (a, pa["corr_XS"]))
        ax.plot(pa["levels"], pa["ps1_levels"], 'o', color=SEQ[a], ms=4)
    ax.axhline(0.5, color='#bbb', ls=':')
    ax.set_xlabel('X'); ax.set_ylabel('P(S=+1 | X)'); ax.grid(alpha=.25); ax.legend(fontsize=8)
    ax.set_title('Hidden-confounder coupling P(S=+1|X)=σ(αX) across the sweep', fontsize=10)
    fig.tight_layout(); fig.savefig(os.path.join(HERE, 'alpha_ps1.png'), dpi=120); plt.close(fig)

    # ---- CATE(X)
    fig, ax = plt.subplots(figsize=(7, 4.3))
    for a in ALPHAS:
        pa = diag["per"][str(a)]
        ax.plot(xf, pa["cate_fine"], color=SEQ[a], lw=2, label='α=%d' % a)
        ax.plot(pa["levels"], pa["cate_levels"], 'o', color=SEQ[a], ms=4)
    ax.axhline(0, color='#bbb', ls=':'); ax.axvline(0, color='#bbb', ls=':')
    ax.set_xlabel('X'); ax.set_ylabel('CATE(X)'); ax.grid(alpha=.25); ax.legend(fontsize=8)
    ax.set_title('CATE(X) = (2σ(αX)−1) + 1.5X — zero-crossing at X=0 for every α', fontsize=10)
    fig.tight_layout(); fig.savefig(os.path.join(HERE, 'alpha_cate.png'), dpi=120); plt.close(fig)

    # ---- summary vs alpha
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.6, 4.0))
    cx = [diag["per"][str(a)]["corr_XS"] for a in ALPHAS]
    ct = [diag["per"][str(a)]["corr_TS"] for a in ALPHAS]
    ax1.plot(ALPHAS, cx, 'o-', color='#08519c', lw=2, label='corr(X, S)')
    ax1.plot(ALPHAS, ct, 's--', color='#6baed6', lw=2, label='corr(T, S)')
    ax1.axhline(0.20, color='#b45309', ls=':', lw=1.5, label='diabetes real corr(X,S)≈0.20')
    ax1.set_xlabel('α'); ax1.set_ylabel('correlation'); ax1.grid(alpha=.25); ax1.legend(fontsize=8)
    ax1.set_title('How observable the hidden S is', fontsize=10)
    orc = [diag["per"][str(a)]["oracle"] for a in ALPHAS]
    ax2.plot(ALPHAS, orc, 'o-', color='#444', lw=2, label='oracle E[Y]')
    ax2.axhline(0.0, color='#bbb', ls=':', label='never-treat = all-treat = 0')
    ax2.set_xlabel('α'); ax2.set_ylabel('exact E[Y]'); ax2.grid(alpha=.25); ax2.legend(fontsize=8)
    ax2.set_title('Value ceilings vs α', fontsize=10)
    fig.tight_layout(); fig.savefig(os.path.join(HERE, 'alpha_summary.png'), dpi=120); plt.close(fig)

    for a in ALPHAS:
        pa = diag["per"][str(a)]
        print('α=%-2d corr(X,S)=%.3f corr(T,S)=%.3f P(T=1)=%.3f Λ∈[%.2f,%.2f] oracle=%.4f' %
              (a, pa["corr_XS"], pa["corr_TS"], pa["p_T1"], min(pa["lambda_true"]), max(pa["lambda_true"]), pa["oracle"]))
    print('wrote alpha_diag.json + 3 PNGs')


if __name__ == "__main__":
    main()
