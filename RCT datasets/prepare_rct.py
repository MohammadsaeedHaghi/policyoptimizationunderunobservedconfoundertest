#!/usr/bin/env python3
"""Turn IHDP, Twins and IST into confounded policy-learning benchmarks with a KNOWN Gamma*.

Why anything has to be constructed at all: these three are randomised (or, for Twins, a matched
pair), so there is NO hidden confounding in them. Running a marginal-sensitivity-model method on
clean randomised data is uninformative -- the adversary has nothing to fight, Gamma = 1 is
correct, and every robust method collapses onto its X-X twin. So the treatment assignment is
REPLACED by a confounded one with a known odds ratio, exactly as in the UCI campaign.

What is real here and what is not:
  REAL   the covariates, and BOTH potential outcomes Y0, Y1 taken from the dataset itself
         (IHDP: yf + ycf, simulated by the NPCI surface; Twins: mort_0 / mort_1, genuinely
         observed; IST: the T-learner imputation, the only one of the four with real predictive
         signal at AUC 0.79/0.80).
  SYNTHETIC  only the assignment: S is a hidden index built from held-out columns, and
         e(x,S) = sigma(1.2 x + 0.5 ln(LAM) S) with NO clipping, so the odds ratio between
         S = +1 and S = -1 is LAM at every x. LAM = 4.

The covariates are split into an X-group (observed, reduced to a scalar index in [-1,1]) and a
U-group (hidden), with the split searched to make |corr(x, u)| large -- the solvers' Lipschitz
block is exact only in one dimension, and a trackable confounder is the regime where a transport
term can help at all.
"""
import json, warnings
from pathlib import Path
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
OUT = HERE / "prepared"; OUT.mkdir(exist_ok=True)
LAM, CX = 4.0, 1.2
rng_global = np.random.default_rng(0)


def _sig(z): return 1.0 / (1.0 + np.exp(-np.clip(z, -40, 40)))


def split_xu(M, rng, tries=40):
    """Split columns into two groups; keep the split whose scalar indices correlate most."""
    M = np.nan_to_num(np.asarray(M, float), nan=0.0)
    Z = (M - M.mean(0)) / (M.std(0) + 1e-9)
    _, _, Vt = np.linalg.svd(Z - Z.mean(0), full_matrices=False)
    load = np.abs(Vt[0]); order = np.argsort(-load); k = max(1, len(order) // 2)

    def idx(cols):
        w = load[cols] + 1e-9
        s = Z[:, cols] @ (w / w.sum())
        return (s - s.mean()) / (s.std() + 1e-9)

    cands = [(list(order[0::2]), list(order[1::2])), (list(order[:k]), list(order[k:]))]
    for _ in range(tries):
        pm = rng.permutation(len(order)); cands.append((list(pm[:k]), list(pm[k:])))
    best = None
    for a, b in cands:
        if not a or not b: continue
        c = abs(float(np.corrcoef(idx(a), idx(b))[0, 1]))
        if best is None or c > best[0]: best = (c, a, b)
    return best[1], best[2], idx(best[1]), idx(best[2])


def build(name, Xmat, Y0, Y1, kind, note):
    rng = np.random.default_rng(0)
    ok = np.isfinite(Y0) & np.isfinite(Y1)
    Xmat, Y0, Y1 = Xmat[ok], Y0[ok], Y1[ok]
    gA, gB, xi, ui = split_xu(Xmat, rng)
    x = np.clip(xi / (np.percentile(np.abs(xi), 99) + 1e-9), -1, 1)
    u = ui
    S = np.where(u > np.median(u), 1.0, -1.0)
    e = _sig(CX * x + 0.5 * np.log(LAM) * S)          # no clipping -> odds ratio == LAM exactly
    e1, e0 = _sig(CX * x + 0.5 * np.log(LAM)), _sig(CX * x - 0.5 * np.log(LAM))
    lam_real = float(np.mean((e1 / (1 - e1)) / (e0 / (1 - e0))))
    cate = Y1 - Y0
    orc = (cate > 0).astype(float)
    rec = {"name": name, "n_total": int(len(x)), "outcome_kind": kind,
           "n_cols_X": len(gA), "n_cols_U": len(gB),
           "LAM_declared": LAM, "LAM_realised": lam_real, "CX": CX,
           "corr_x_u": float(np.corrcoef(x, u)[0, 1]),
           "corr_x_S": float(np.corrcoef(x, S)[0, 1]),
           "e_min": float(e.min()), "e_max": float(e.max()),
           "oracle": float(np.mean(orc * Y1 + (1 - orc) * Y0)),
           "never": float(np.mean(Y0)), "all": float(np.mean(Y1)),
           "cate_sd": float(cate.std()), "outcome_sd": float(np.std(np.r_[Y0, Y1])),
           "note": note}
    np.savez_compressed(OUT / (name + "_pop.npz"), x=x, u=u, S=S, e=e, Y0=Y0, Y1=Y1,
                        kind=np.array([1.0 if kind == "binary" else 0.0]))
    print("%-7s n=%-6d corrxu=%+.2f corrxS=%+.2f Lam=%.3f e=[%.2f,%.2f] "
          "oracle %.3f never %.3f all %.3f" %
          (name, len(x), rec["corr_x_u"], rec["corr_x_S"], lam_real, rec["e_min"], rec["e_max"],
           rec["oracle"], rec["never"], rec["all"]), flush=True)
    return rec


R = []

# ---------------------------------------------------------------- IHDP (replication 1)
z = np.load(HERE / "ihdp/ihdp_npci_1-100.train.npz")
r = 0
X = z["x"][:, :, r]; t = z["t"][:, r].astype(int)
yf, ycf = z["yf"][:, r], z["ycf"][:, r]
Y1 = np.where(t == 1, yf, ycf)          # yf is the arm actually received
Y0 = np.where(t == 1, ycf, yf)
R.append(build("ihdp", X, Y0, Y1, "continuous",
               "Y0/Y1 from the benchmark's own yf + ycf (NPCI surface A), replication 1."))

# ---------------------------------------------------------------- Twins
tx = pd.read_csv(HERE / "twins/twin_pairs_X_3years_samesex.csv", low_memory=False)
ty = pd.read_csv(HERE / "twins/twin_pairs_Y_3years_samesex.csv")
drop = [c for c in tx.columns if c.startswith("Unnamed") or c.startswith("infant_id")]
XT = tx.drop(columns=drop).apply(pd.to_numeric, errors="coerce")
XT = XT.loc[:, XT.notna().mean() > 0.9].fillna(XT.median())
# mortality is a BAD outcome and every solver here MAXIMISES, so it is flipped to survival.
R.append(build("twins", XT.values, 1.0 - ty["mort_0"].values.astype(float),
               1.0 - ty["mort_1"].values.astype(float), "binary",
               "Y0/Y1 are the GENUINELY OBSERVED 1-year SURVIVAL (1 - mortality) of the lighter "
               "and heavier twin."))

# ---------------------------------------------------------------- IST (T-learner imputation)
cf = pd.read_csv(HERE / "counterfactuals/ist_T-learner.csv")
XC = [c for c in cf.columns if c not in ("T", "Y_factual", "Y0", "Y1", "which_imputed")]
# same flip: the IST outcome imputed above is death at 6 months.
R.append(build("ist", cf[XC].values, 1.0 - cf["Y0"].values, 1.0 - cf["Y1"].values, "binary",
               "Y0/Y1 from the T-learner imputation (AUC 0.79/0.80 on the factual arm); the "
               "imputed arm is a model output, not an observation. Flipped to 6-month SURVIVAL."))

json.dump(R, open(OUT / "_index.json", "w"), indent=1)
print("\nprepared %d -> %s" % (len(R), OUT))
