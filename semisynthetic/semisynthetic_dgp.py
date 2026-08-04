#!/usr/bin/env python3
"""Semi-synthetic observational data from UCI classification datasets.

Based on the data-generation procedure of *"Learning Risk Scores Robust to Unobserved
Confounders"* (anonymous submission, 42350_MainPaper.pdf), extended with a synthetic
heterogeneous CATE tau(X) to yield both potential outcomes Y^0 and Y^1 for off-policy learning
experiments.

WHAT IS THEIRS (followed exactly)
  X            real UCI covariates, standardised
  U = Y^0      the UCI label itself, relabelled to {-1, +1}, playing BOTH roles
  T            their Eq. (6):  logit pi0(x,u) = lam^T x + gamma*u,  lam_j ~ U(-0.1, 0.1),
               pi0 = P(NO treatment), T ~ Bern(1 - pi0), NO clipping of pi0
  screen       a dataset is admissible only if its cross-validated logistic log-loss lies in
               [0.35, log 2] -- not trivially separable, not pure noise

  The odds ratio between the two hidden states is exactly e^{2 gamma} at every x (verified to
  ~1e-15 by example_generate_and_check.py), so the matched MSM parameter is Gamma = e^{2 gamma}.
  This is EXACT, not an approximation, because pi0 is never clipped.

WHAT IS OURS (one deliberate, documented deviation)
  tau(x)       a * (sigmoid(b^T x - c) - 0.5)          <-- CENTRED
  Y^1          Y^0 + tau(x) + eps,   eps ~ N(0, sigma^2)

  The spec this was written from used tau(x) = a * sigmoid(b^T x - c), which is STRICTLY
  POSITIVE (a > 0 and sigmoid in (0,1)). Under that form the true CATE never changes sign, so
  the optimal policy is "treat everyone", the oracle value equals the all-treat value, and no
  policy learner can be separated from a constant. Measured over d in {5,10,20,50}:
  P(tau < 0) = 0.0000 in every case. Subtracting 0.5 makes the sign flip on about half the
  population and gives the oracle a real advantage over the best constant policy. Set
  centre_tau=False to reproduce the uncentred (degenerate) version as a negative control.

ORACLE / OBSERVED SEPARATION is a hard invariant: D_obs carries only (X, T, Y). U, Y0, Y1, tau,
pi0, e_star and the raw label live in D_oracle and must never reach an estimator.
"""
from __future__ import annotations

import argparse, json, warnings
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

warnings.filterwarnings("ignore")

LOG2 = float(np.log(2.0))
SCREEN_LO, SCREEN_HI = 0.35, LOG2          # the paper's admissibility band


def _sig(z):
    return 1.0 / (1.0 + np.exp(-np.clip(np.asarray(z, float), -60, 60)))


# --------------------------------------------------------------------------- dataset loading
#: name -> (ucimlrepo id, positive-class predicate applied to the factorised target)
UCI_CANDIDATES = {
    "adult": 2, "bank_marketing": 222, "credit_default": 350, "online_shoppers": 468,
    "mushroom": 73, "spambase": 94, "german_credit": 144, "contraceptive": 30,
    "breast_cancer": 17, "heart_disease": 45, "support2": 880, "aids_clinical": 890,
    "student_performance": 320, "wine_quality": 186, "communities_crime": 183,
}


def _numeric(df):
    import pandas as pd
    out = df.copy()
    for c in out.columns:
        if out[c].dtype == object or str(out[c].dtype) == "category":
            out[c] = pd.factorize(out[c])[0].astype(float)
        else:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def cv_logloss(X, y, folds=5, seed=0):
    """Cross-validated logistic log-loss, the paper's dataset-admissibility statistic."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import log_loss
    yb = (np.asarray(y) > 0).astype(int)
    if len(np.unique(yb)) < 2:
        return float("nan")
    p = np.zeros(len(yb))
    for tr, te in StratifiedKFold(folds, shuffle=True, random_state=seed).split(X, yb):
        p[te] = LogisticRegression(max_iter=2000).fit(X[tr], yb[tr]).predict_proba(X[te])[:, 1]
    return float(log_loss(yb, np.clip(p, 1e-9, 1 - 1e-9)))


def load_uci_dataset(name: str, enforce_screen: bool = True, seed: int = 0):
    """-> (X standardised (n,d) float64, y in {-1,+1} int8, info dict).

    Multiclass targets are merged to binary by splitting at the majority class, per the paper's
    "merge similar classes on a per-domain basis"; rows with missing values are dropped.
    """
    from ucimlrepo import fetch_ucirepo
    import pandas as pd
    if name not in UCI_CANDIDATES:
        raise KeyError("unknown dataset %r; known: %s" % (name, sorted(UCI_CANDIDATES)))
    d = fetch_ucirepo(id=UCI_CANDIDATES[name])
    Xdf, Ydf = d.data.features, d.data.targets
    ycol = Ydf.columns[0]
    df = pd.concat([Xdf, Ydf[ycol].rename("__y__")], axis=1).dropna()
    Xn = _numeric(df.drop(columns=["__y__"]))
    Xn = Xn.loc[:, Xn.notna().mean() > 0.9]
    Xn = Xn.fillna(Xn.median()).loc[:, Xn.std() > 0]
    yv = _numeric(df[["__y__"]])["__y__"].values
    uq = np.unique(yv)
    if len(uq) == 2:
        y = np.where(yv == uq.max(), 1, -1)
    else:                                       # merge to binary at the majority class
        maj = uq[np.argmax([(yv == v).sum() for v in uq])]
        y = np.where(yv == maj, 1, -1)
    X = Xn.values.astype(np.float64)
    X = (X - X.mean(0)) / (X.std(0) + 1e-9)     # standardise: the treatment logistic needs it
    y = y.astype(np.int8)
    ll = cv_logloss(X, y, seed=seed)
    info = {"dataset": name, "uci_id": UCI_CANDIDATES[name], "n": int(len(y)),
            "d": int(X.shape[1]), "cv_logloss": ll, "screen_lo": SCREEN_LO,
            "screen_hi": SCREEN_HI, "admissible": bool(SCREEN_LO <= ll <= SCREEN_HI),
            "pos_frac": float((y > 0).mean()), "target_col": str(ycol)}
    if enforce_screen and not info["admissible"]:
        raise ValueError("%s fails the paper's screen: CV log-loss %.4f not in [%.2f, %.4f]"
                         % (name, ll, SCREEN_LO, SCREEN_HI))
    return X, y, info


# --------------------------------------------------------------------------- the DGP
@dataclass
class Metadata:
    dataset_name: str
    gamma: float
    Gamma_MSM: float
    lambda_coefs: list
    tau_params: dict
    noise_sigma: float
    centre_tau: bool
    binary_Y1: bool
    seed: int
    n_train: int
    n_test: int
    d: int
    diagnostics: dict = field(default_factory=dict)
    notes: str = ""


def generate_semisynthetic(X, y, gamma: float, seed: int, sigma: float = 0.1,
                           binary_Y1: bool = False, n_train: int = 1500, n_test: int = 500,
                           centre_tau: bool = True):
    """Steps 2-7 of the construction. -> (D_obs, D_oracle, Metadata)."""
    rng = np.random.default_rng(seed)
    X = np.asarray(X, np.float64)
    y = np.asarray(y).astype(np.int8)
    N, d = X.shape

    # --- sample the replication's units (with replacement only if the pool is too small) ---
    need = n_train + n_test
    idx = rng.choice(N, size=need, replace=need > N)
    Xs, ys = X[idx], y[idx]

    # --- Step 2: the label plays BOTH roles (paper's construction, verbatim) ---
    U = ys.astype(np.int8)
    Y0 = ys.astype(np.int8)

    # --- Step 3: draw the CATE function (our extension) ---
    a = float(rng.uniform(0.5, 2.0))
    b = rng.standard_normal(d)
    c = float(rng.standard_normal())
    s = _sig(Xs @ b - c)
    tau = (a * (s - 0.5) if centre_tau else a * s).astype(np.float64)

    # --- Step 4: realise Y^1 ---
    eps1 = rng.normal(0.0, sigma, size=need)
    Y1 = Y0.astype(np.float64) + tau + eps1
    if binary_Y1:
        Y1 = np.where(Y1 > 0, 1.0, -1.0)

    # --- Step 5: the confounded assignment, paper's Eq. (6). pi0 = P(NO treatment) ---
    lam = rng.uniform(-0.1, 0.1, size=d)
    pi0 = _sig(Xs @ lam + gamma * U)                       # NO clipping, per the paper
    e_star = 1.0 - pi0                                     # P(T = 1 | x, u)
    T = (rng.uniform(size=need) < e_star).astype(np.int8)

    # --- Step 6: the observed outcome ---
    Yobs = np.where(T == 1, Y1, Y0.astype(np.float64))

    # exact odds ratio between the two hidden states at each x -- must equal e^{2 gamma}
    p_p, p_m = _sig(Xs @ lam + gamma), _sig(Xs @ lam - gamma)
    orat = (p_p / (1 - p_p)) / (p_m / (1 - p_m))

    tr = np.arange(n_train)
    te = np.arange(n_train, need)
    orc = (tau > 0)
    diag = {
        "odds_ratio_mean": float(orat.mean()),
        "odds_ratio_max_abs_err": float(np.max(np.abs(orat - np.exp(2 * gamma)))),
        "e_min": float(e_star.min()), "e_max": float(e_star.max()),
        "P_T1": float(T.mean()),
        "frac_tau_pos": float(orc.mean()),
        "oracle": float(np.mean(np.where(orc, Y1, Y0)[te])),
        "never": float(np.mean(Y0[te])), "all": float(np.mean(Y1[te])),
        "corr_x_U": float(np.corrcoef((Xs @ b), U)[0, 1]),
        "pos_frac": float((ys > 0).mean()),
    }
    diag["headroom"] = diag["oracle"] - max(diag["never"], diag["all"])

    D_obs = {"X": Xs.astype(np.float64), "T": T.astype(np.int8),
             "Y": Yobs.astype(np.float64), "train_idx": tr, "test_idx": te}
    D_oracle = {"U": U, "Y0": Y0, "Y1": Y1.astype(np.float64), "tau": tau,
                "pi0": pi0.astype(np.float64), "e_star": e_star.astype(np.float64)}
    meta = Metadata(dataset_name="", gamma=float(gamma), Gamma_MSM=float(np.exp(2 * gamma)),
                    lambda_coefs=[float(v) for v in lam],
                    tau_params={"a": a, "b": [float(v) for v in b], "c": c},
                    noise_sigma=float(sigma), centre_tau=bool(centre_tau),
                    binary_Y1=bool(binary_Y1), seed=int(seed), n_train=int(n_train),
                    n_test=int(n_test), d=int(d), diagnostics=diag,
                    notes=("tau CENTRED (a*(sigmoid-0.5)); the uncentred form of the source spec "
                           "is strictly positive and makes the oracle policy 'treat everyone'."
                           if centre_tau else
                           "tau UNCENTRED (a*sigmoid) -- strictly positive, oracle == all-treat. "
                           "Negative control only."))

    # ---- hard invariants (the spec's assert block) ----
    assert np.array_equal(D_oracle["U"], D_oracle["Y0"]), "U must equal Y0"
    assert set(np.unique(D_oracle["U"]).tolist()) <= {-1, 1}, "U must be in {-1,+1}"
    c0 = D_obs["T"] == 0
    assert np.allclose(D_obs["Y"][c0], D_oracle["Y0"][c0]), "Y must equal Y0 where T=0"
    assert np.allclose(D_obs["Y"][~c0], D_oracle["Y1"][~c0]), "Y must equal Y1 where T=1"
    assert "y" not in D_obs and "U" not in D_obs, "the raw label must not leak into D_obs"
    return D_obs, D_oracle, meta


def run_replications(dataset_name: str, gamma: float, n_reps: int, base_seed: int = 0,
                     **kw):
    X, y, info = load_uci_dataset(dataset_name, enforce_screen=kw.pop("enforce_screen", True))
    out = []
    for r in range(n_reps):
        o, orc, meta = generate_semisynthetic(X, y, gamma, base_seed + r, **kw)
        meta.dataset_name = dataset_name
        meta.diagnostics["cv_logloss"] = info["cv_logloss"]
        out.append((o, orc, meta))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--gamma", type=float, required=True)
    ap.add_argument("--n-reps", type=int, default=3)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-train", type=int, default=1500)
    ap.add_argument("--n-test", type=int, default=500)
    ap.add_argument("--sigma", type=float, default=0.1)
    ap.add_argument("--binary-Y1", action="store_true")
    ap.add_argument("--uncentred-tau", action="store_true",
                    help="reproduce the degenerate strictly-positive CATE (negative control)")
    ap.add_argument("--no-screen", action="store_true")
    a = ap.parse_args()

    od = Path(a.out_dir); od.mkdir(parents=True, exist_ok=True)
    reps = run_replications(a.dataset, a.gamma, a.n_reps, a.seed, sigma=a.sigma,
                            binary_Y1=a.binary_Y1, n_train=a.n_train, n_test=a.n_test,
                            centre_tau=not a.uncentred_tau, enforce_screen=not a.no_screen)
    for r, (o, orc, meta) in enumerate(reps):
        tag = "%s_g%g_s%d" % (a.dataset, a.gamma, a.seed + r)
        np.savez_compressed(od / (tag + ".npz"), **{k: v for k, v in o.items()},
                            **{("oracle_" + k): v for k, v in orc.items()})
        (od / (tag + ".json")).write_text(json.dumps(asdict(meta), indent=1))
        d = meta.diagnostics
        print("%-28s Gamma=%8.3f  headroom %+0.4f  P(T=1)=%.3f  e=[%.3f,%.3f]  ORerr=%.2e"
              % (tag, meta.Gamma_MSM, d["headroom"], d["P_T1"], d["e_min"], d["e_max"],
                 d["odds_ratio_max_abs_err"]))
    print("\nwrote %d replications -> %s" % (len(reps), od))


if __name__ == "__main__":
    main()
