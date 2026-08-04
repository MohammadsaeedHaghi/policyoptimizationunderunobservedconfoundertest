#!/usr/bin/env python3
"""Turn one screened UCI dataset into population arrays the policy-learning runner consumes.

Emits one file per gamma: prepared/{dataset}_g{gamma}_pop.npz with
    x   scalar covariate the SOLVERS see, in [-1, 1]
    e   true P(T=1 | x_full, u)   (oracle, used only to draw T and to measure operative Gamma)
    Y0  {-1,+1}                    (oracle)
    Y1  Y0 + tau + eps             (oracle)
    u, tau, Xfull                  (oracle / for the later d-dimensional arm)

THE SCALAR INDEX.  Our LP solvers' Lipschitz block is exact only for a one-dimensional covariate.
We use x = b'X, the CATE index, standardised and clipped to [-1, 1]. This is deliberate and it is
the mildest possible projection here: tau(X) = a*(sigmoid(b'X - c) - 0.5) depends on X ONLY through
b'X, so b'X is a SUFFICIENT STATISTIC FOR THE OPTIMAL POLICY -- projecting to it loses nothing
about which decision is correct. The experiment therefore isolates confounding-robustness rather
than dimension reduction, which is what we want to test first.

One consequence to keep in mind, and to measure rather than assume: the assignment depends on X
through lam'X, which is NOT a function of b'X. So the nominal propensity estimated from x alone
carries extra unexplained variation on top of the hidden u, and the OPERATIVE Gamma a fitted
pipeline needs can exceed the declared e^{2 gamma}. check_gamma_semisynth.py measures it.
"""
import argparse, json, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from semisynthetic_dgp import load_uci_dataset, generate_semisynthetic

OUT = HERE / "prepared"; OUT.mkdir(exist_ok=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="bank_marketing")
    ap.add_argument("--gammas", default="0.0,1.0,1.5,2.0")
    ap.add_argument("--pool", type=int, default=20000,
                    help="population size drawn once per gamma (train+test come from it)")
    ap.add_argument("--sigma", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--uncentred-tau", action="store_true")
    a = ap.parse_args()

    X, y, info = load_uci_dataset(a.dataset, enforce_screen=True)
    print("loaded %s: n=%d d=%d CV log-loss %.4f (band [0.35, %.4f])"
          % (a.dataset, info["n"], info["d"], info["cv_logloss"], np.log(2)))
    recs = []
    for g in [float(t) for t in a.gammas.split(",")]:
        pool = min(a.pool, info["n"])
        # one draw per gamma; the runner then re-draws only T and the train/test split per seed
        D, O, meta = generate_semisynthetic(X, y, gamma=g, seed=a.seed, sigma=a.sigma,
                                            n_train=pool, n_test=0,
                                            centre_tau=not a.uncentred_tau)
        Xf = D["X"]
        b = np.asarray(meta.tau_params["b"], float)
        idx = Xf @ b                                   # CATE index: sufficient for the optimal policy
        idx = (idx - idx.mean()) / (idx.std() + 1e-9)
        x = np.clip(idx / (np.percentile(np.abs(idx), 99) + 1e-9), -1, 1)
        Y0, Y1, tau, e, u = O["Y0"].astype(float), O["Y1"], O["tau"], O["e_star"], O["U"]
        orc = (Y1 > Y0)
        rec = {"dataset": a.dataset, "gamma": g, "Gamma_MSM": float(np.exp(2 * g)),
               "n_pool": int(len(x)), "d": int(Xf.shape[1]),
               "cv_logloss": info["cv_logloss"],
               "odds_ratio_max_abs_err": meta.diagnostics["odds_ratio_max_abs_err"],
               "e_min": float(e.min()), "e_max": float(e.max()), "P_T1": float(e.mean()),
               "corr_x_u": float(np.corrcoef(x, u)[0, 1]),
               "frac_tau_pos": float((tau > 0).mean()),
               "oracle": float(np.mean(np.where(orc, Y1, Y0))),
               "never": float(np.mean(Y0)), "all": float(np.mean(Y1)),
               "tau_params_a": meta.tau_params["a"], "tau_params_c": meta.tau_params["c"],
               "noise_sigma": a.sigma, "centre_tau": not a.uncentred_tau,
               "seed": a.seed, "dgp_seed": a.seed}
        rec["headroom"] = rec["oracle"] - max(rec["never"], rec["all"])
        np.savez_compressed(OUT / ("%s_g%g_d%d_pop.npz" % (a.dataset, g, a.seed)),
                            x=x, e=e, Y0=Y0, Y1=Y1, u=u.astype(float), tau=tau, Xfull=Xf)
        recs.append(rec)
        print("  d%d g=%-4g Gamma*=%8.3f  headroom %+0.4f  corr(x,u) %+0.3f  e=[%.3f,%.3f]  "
              "frac tau>0 %.3f  ORerr %.1e"
              % (a.seed, g, rec["Gamma_MSM"], rec["headroom"], rec["corr_x_u"], rec["e_min"],
                 rec["e_max"], rec["frac_tau_pos"], rec["odds_ratio_max_abs_err"]))
    # MERGE, do not overwrite: this script is called once per dataset, and the runner looks up
    # its metadata by (dataset, gamma) in this one shared index.
    ipath = OUT / "_index.json"
    prev = json.loads(ipath.read_text()) if ipath.exists() else []
    keep = [r for r in prev
            if not (r["dataset"] == a.dataset and r.get("dgp_seed", 0) == a.seed
                    and any(abs(r["gamma"] - n["gamma"]) < 1e-12 for n in recs))]
    json.dump(keep + recs, open(ipath, "w"), indent=1)
    print("\nwrote %d population files -> %s" % (len(recs), OUT))


if __name__ == "__main__":
    main()
