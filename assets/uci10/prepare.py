#!/usr/bin/env python3
"""Build semi-synthetic confounded benchmarks from 10 UCI datasets.

RECIPE (per dataset), the standard semi-synthetic design: real covariates and a real outcome
surface, synthetic treatment assignment so the confounding strength is KNOWN.

  1. Load, encode, drop NAs. Pick the outcome column Y (binary kept 0/1; continuous standardised).
  2. Pick a binary column as the PSEUDO-TREATMENT A_real, and fit the outcome surface separately
     for each arm on the FULL real data:  mu_t(x, u) = E[Y | x, u, A_real = t].
     Binary Y -> gradient-boosted classifier probability; continuous Y -> gradient-boosted
     regressor. Fitting on all rows (thousands) is what makes these a credible ground truth; the
     300-row sample below is only the policy learner's training set.
  3. Split the remaining columns into two disjoint groups and reduce each to a SCALAR:
        x = standardised index of group A, rescaled to [-1, 1]   (OBSERVED covariate)
        u = standardised index of group B                        (HIDDEN confounder)
     Groups are chosen to make corr(x, u) LARGE (greedy pairing on the correlation matrix), per
     the request: the transport term needs the hidden confounder to be x-trackable to help.
     A scalar x is required because the solvers are univariate.
  4. S = sign(u - median(u)) in {-1, +1}: the hidden binary confounder.
  5. SYNTHETIC treatment with exact, known confounding:
        e(x, S) = sigma(CX * x + (1/2) ln(LAM) * S),   NO clipping
     so odds(T=1|x,S=+1) / odds(T=1|x,S=-1) = LAM EXACTLY at every x. Matched Gamma = LAM = 4.
  6. Y_0, Y_1 drawn from mu_0, mu_1 (Bernoulli for binary outcomes, + noise for continuous);
     T ~ Bern(e(x,S)); Y_obs = Y_T.
  7. 300 random rows -> TRAIN (the learner sees only x, T, Y_obs); the rest -> TEST with both
     potential outcomes retained for evaluation.

Everything is saved: the prepared arrays, the construction metadata (which columns went to X vs
U, the realised corr(x,S), the achieved Lambda, oracle/naive/never-treat references).

Usage: python3 assets/uci10/prepare.py            # all 10
       python3 assets/uci10/prepare.py adult      # one
"""
import sys, json, warnings
from pathlib import Path
import numpy as np
warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve().parent
OUT = HERE / "data"; OUT.mkdir(parents=True, exist_ok=True)

LAM = 4.0                      # Gamma* -- known by construction, flagged in every table
CX = 1.2                       # propensity's dependence on the observed index
N_TRAIN = 300
N_TEST = 4000
NOISE_C = 0.3                  # noise sd for continuous outcomes (post-standardisation)

# (uci id, name, outcome column, pseudo-treatment column, outcome kind)
DATASETS = [
    (2,   "adult",             "income",              "sex",              "binary"),
    (222, "bank_marketing",    "y",                   "housing",          "binary"),
    (350, "credit_default",    "Y",                   "SEX",              "binary"),
    (468, "online_shoppers",   "Revenue",             "Weekend",          "binary"),
    (73,  "mushroom",          "poisonous",           "bruises",          "binary"),
    (186, "wine_quality",      "quality",             "color",            "continuous"),
    (94,  "spambase",          "Class",               "__median__",       "binary"),
    (880, "support2",          "death",               "dnr",              "binary"),
    (890, "aids_clinical",     "cid",                 "trt",              "binary"),
    (183, "communities_crime", "ViolentCrimesPerPop", "__median__",       "continuous"),
]


def _numeric(df):
    """Ordinal-encode object columns; coerce everything to float."""
    import pandas as pd
    out = df.copy()
    for c in out.columns:
        if out[c].dtype == object or str(out[c].dtype) == "category":
            out[c] = pd.factorize(out[c])[0].astype(float)
        else:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def _score_treatment(Xn, Y, kind, col_mask, F_builder):
    """Oracle-minus-never headroom this pseudo-treatment would give. A benchmark whose optimal
    policy is 'never treat' teaches nothing about policy learning, so the column is CHOSEN to
    make the optimal policy non-trivial -- a disclosed design step, not a tuned result."""
    from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
    import numpy as _np
    F = F_builder()
    mu = {}
    for t in (0, 1):
        m = col_mask == t
        if m.sum() < 50: return -1.0, None
        if kind == "binary" and len(_np.unique(Y[m])) < 2: return -1.0, None
        if kind == "binary":
            g = GradientBoostingClassifier(n_estimators=40, max_depth=3, random_state=0).fit(F[m], Y[m])
            mu[t] = _np.clip(g.predict_proba(F)[:, 1], 0.01, 0.99)
        else:
            g = GradientBoostingRegressor(n_estimators=40, max_depth=3, random_state=0).fit(F[m], Y[m])
            mu[t] = g.predict(F)
    cate = mu[1] - mu[0]
    orc = (cate > 0).astype(float)
    return float(_np.mean(orc * mu[1] + (1 - orc) * mu[0]) - _np.mean(mu[0])), mu


def _pick_treatment(Xn, col):
    """Return a 0/1 treatment vector and the column used."""
    if col == "__median__" or col not in Xn.columns:
        # no natural binary column: use the most balanced column, split at its median
        best, bc = None, None
        for c in Xn.columns:
            v = Xn[c].values
            if np.nanstd(v) <= 0: continue
            frac = float(np.nanmean(v > np.nanmedian(v)))
            if best is None or abs(frac - 0.5) < abs(best - 0.5): best, bc = frac, c
        col = bc
    v = Xn[col].values.astype(float)
    uq = np.unique(v[~np.isnan(v)])
    a = (v == uq.max()).astype(int) if len(uq) == 2 else (v > np.nanmedian(v)).astype(int)
    return a, col


def _split_xu(Xn, exclude, rng):
    """Split columns into an X group and a U group so their scalar indices are CORRELATED.

    Greedy: rank columns by |corr| with the first principal direction, then deal them
    alternately into the two groups. Alternating (rather than top-half / bottom-half) keeps both
    indices loaded on the same dominant factor, which is what makes corr(x, u) large.
    """
    cols = [c for c in Xn.columns if c not in exclude]
    M = Xn[cols].values.astype(float)
    M = np.nan_to_num(M, nan=np.nanmean(M, axis=0))
    Z = (M - M.mean(0)) / (M.std(0) + 1e-9)
    _, _, Vt = np.linalg.svd(Z - Z.mean(0), full_matrices=False)
    load = np.abs(Vt[0])
    order = np.argsort(-load)
    def idx_of(g):
        A = Xn[g].values.astype(float)
        A = np.nan_to_num(A, nan=np.nanmean(A, axis=0))
        Zg = (A - A.mean(0)) / (A.std(0) + 1e-9)
        w = np.abs(Vt[0][[cols.index(c) for c in g]]) + 1e-9
        s_ = Zg @ (w / w.sum())
        return (s_ - s_.mean()) / (s_.std() + 1e-9)
    # search a few splits and keep the one with the LARGEST |corr(x, u)| -- the request was for a
    # hidden confounder the observed covariate can track, which is what the W-term needs.
    best = None
    cands = [(list(order[0::2]), list(order[1::2]))]                      # alternating
    k = max(1, len(order) // 2)
    cands.append((list(order[:k]), list(order[k:])))                      # top / bottom by loading
    for r in range(40):
        pm = rng.permutation(len(order)); cands.append((list(pm[:k]), list(pm[k:])))
    for ia, ib in cands:
        if not ia or not ib: continue
        ga = [cols[i] for i in ia]; gb = [cols[i] for i in ib]
        c = abs(float(np.corrcoef(idx_of(ga), idx_of(gb))[0, 1]))
        if best is None or c > best[0]: best = (c, ga, gb)
    gA, gB = best[1], best[2]
    def idx(g):
        A = Xn[g].values.astype(float)
        A = np.nan_to_num(A, nan=np.nanmean(A, axis=0))
        Zg = (A - A.mean(0)) / (A.std(0) + 1e-9)
        w = np.abs(Vt[0][[cols.index(c) for c in g]]) + 1e-9
        s = Zg @ (w / w.sum())
        return (s - s.mean()) / (s.std() + 1e-9)
    return gA, gB, idx(gA), idx(gB)


def _sig(z): return 1.0 / (1.0 + np.exp(-np.clip(z, -40, 40)))


def build(uid, name, ycol, tcol, kind, seed=0):
    from ucimlrepo import fetch_ucirepo
    from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
    import pandas as pd
    rng = np.random.default_rng(seed)
    d = fetch_ucirepo(id=uid)
    Xdf, Ydf = d.data.features, d.data.targets
    y_col = ycol if ycol in Ydf.columns else Ydf.columns[0]
    y_raw = Ydf[y_col]
    df = pd.concat([Xdf, y_raw.rename("__Y__")], axis=1).dropna()
    if len(df) < 1200:
        df = pd.concat([Xdf, y_raw.rename("__Y__")], axis=1)
        df = df.dropna(axis=1, thresh=int(0.8 * len(df))).dropna()
    Xn = _numeric(df.drop(columns=["__Y__"]))
    yv = df["__Y__"]
    if kind == "binary":
        yv = _numeric(yv.to_frame("y"))["y"].values
        uq = np.unique(yv); Y = (yv == uq.max()).astype(float) if len(uq) == 2 else (yv > np.median(yv)).astype(float)
    else:
        yv = pd.to_numeric(yv, errors="coerce").values.astype(float)
        Y = (yv - np.nanmean(yv)) / (np.nanstd(yv) + 1e-9)
    ok = ~np.isnan(Y); Xn, Y = Xn[ok], Y[ok]

    # candidate pseudo-treatments: the requested column plus the most balanced binary-isable ones
    cands = []
    if tcol in Xn.columns: cands.append(tcol)
    for c in Xn.columns:
        v = Xn[c].values.astype(float)
        if np.nanstd(v) <= 0: continue
        frac = float(np.nanmean(v > np.nanmedian(v)))
        if 0.2 < frac < 0.8: cands.append(c)
    cands = list(dict.fromkeys(cands))[:14]
    gA0, gB0, xi0, ui0 = _split_xu(Xn, exclude=set(), rng=rng)
    x0 = np.clip(xi0 / (np.percentile(np.abs(xi0), 99) + 1e-9), -1, 1)
    F0 = np.column_stack([x0, ui0])
    best_t = (-1.0, None)
    for c in cands:
        a, _ = _pick_treatment(Xn, c)
        if kind == "binary" and min(len(np.unique(Y[a == 0])), len(np.unique(Y[a == 1]))) < 2:
            continue
        sc, _mu = _score_treatment(Xn, Y, kind, a, lambda: F0)
        if sc > best_t[0]: best_t = (sc, c)
    tcol_eff = best_t[1] or tcol
    A_real, tcol_used = _pick_treatment(Xn, tcol_eff)
    gA, gB, xi, ui = _split_xu(Xn, exclude={tcol_used}, rng=rng)
    x = np.clip(xi / (np.percentile(np.abs(xi), 99) + 1e-9), -1, 1)   # observed, in [-1, 1]
    u = ui                                                            # hidden

    # outcome surface per arm, fit on ALL real rows
    F = np.column_stack([x, u])
    mu = {}
    for t in (0, 1):
        m = A_real == t
        if m.sum() < 50 or (kind == "binary" and len(np.unique(Y[m])) < 2):
            m = np.ones(len(x), bool)          # degenerate arm -> pooled fit
        if kind == "binary":
            g = GradientBoostingClassifier(n_estimators=120, max_depth=3, random_state=0).fit(F[m], Y[m])
            mu[t] = np.clip(g.predict_proba(F)[:, 1], 0.01, 0.99)
        else:
            g = GradientBoostingRegressor(n_estimators=120, max_depth=3, random_state=0).fit(F[m], Y[m])
            mu[t] = g.predict(F)

    S = np.where(u > np.median(u), 1.0, -1.0)                # hidden binary confounder
    e = _sig(CX * x + 0.5 * np.log(LAM) * S)                 # NO clip => odds ratio == LAM exactly
    T = (rng.uniform(size=len(x)) < e).astype(int)
    if kind == "binary":
        Y0 = (rng.uniform(size=len(x)) < mu[0]).astype(float)
        Y1 = (rng.uniform(size=len(x)) < mu[1]).astype(float)
    else:
        Y0 = mu[0] + rng.normal(0, NOISE_C, len(x))
        Y1 = mu[1] + rng.normal(0, NOISE_C, len(x))
    Yobs = np.where(T == 1, Y1, Y0)

    n = len(x); perm = rng.permutation(n)
    tr = perm[:N_TRAIN]; te = perm[N_TRAIN:N_TRAIN + N_TEST]
    cate = mu[1] - mu[0]
    orc = (cate > 0).astype(float)
    e1 = _sig(CX * x + 0.5 * np.log(LAM)); e0 = _sig(CX * x - 0.5 * np.log(LAM))
    lam_real = float(np.mean((e1 / (1 - e1)) / (e0 / (1 - e0))))

    rec = {
        "name": name, "uci_id": uid, "outcome": y_col, "outcome_kind": kind,
        "pseudo_treatment_col": str(tcol_used), "treatment_search_headroom": round(float(best_t[0]), 4), "n_total": int(n),
        "cols_X": [str(c) for c in gA], "cols_U": [str(c) for c in gB],
        "LAM_declared": LAM, "LAM_realised": lam_real, "CX": CX,
        "corr_x_u": float(np.corrcoef(x, u)[0, 1]),
        "corr_x_S": float(np.corrcoef(x, S)[0, 1]),
        "P_T1": float(T.mean()), "e_min": float(e.min()), "e_max": float(e.max()),
        "oracle_test": float(np.mean(orc[te] * Y1[te] + (1 - orc[te]) * Y0[te])),
        "never_test": float(np.mean(Y0[te])), "all_test": float(np.mean(Y1[te])),
        "cate_sd": float(cate.std()), "outcome_sd": float(Yobs.std()),
        "n_train": len(tr), "n_test": len(te),
    }
    # POPULATION arrays: the runner redraws T and the 300-row sample per seed from these, so the
    # 5 seeds vary in both treatment assignment and training sample -- the split saved below is
    # only a reference copy of seed 0.
    np.savez_compressed(OUT / f"{name}_pop.npz",
                        x=x, u=u, S=S, e=e, mu0=mu[0], mu1=mu[1],
                        kind=np.array([1.0 if kind == "binary" else 0.0]))
    np.savez_compressed(OUT / f"{name}.npz",
                        x_tr=x[tr], T_tr=T[tr], Y_tr=Yobs[tr], u_tr=u[tr], S_tr=S[tr],
                        Y0_tr=Y0[tr], Y1_tr=Y1[tr],
                        x_te=x[te], T_te=T[te], Y_te=Yobs[te], u_te=u[te], S_te=S[te],
                        Y0_te=Y0[te], Y1_te=Y1[te], cate_te=cate[te])
    (OUT / f"{name}.json").write_text(json.dumps(rec, indent=1))
    print("%-18s n=%-6d corr(x,u)=%+.2f corr(x,S)=%+.2f Lam=%.3f e=[%.2f,%.2f] "
          "oracle %.3f never %.3f | X:%d cols U:%d cols" %
          (name, n, rec["corr_x_u"], rec["corr_x_S"], lam_real, rec["e_min"], rec["e_max"],
           rec["oracle_test"], rec["never_test"], len(gA), len(gB)), flush=True)
    return rec


if __name__ == "__main__":
    want = sys.argv[1:] or [d[1] for d in DATASETS]
    recs = []
    for uid, name, ycol, tcol, kind in DATASETS:
        if name not in want: continue
        try:
            recs.append(build(uid, name, ycol, tcol, kind))
        except Exception as ex:
            print("%-18s FAILED: %s" % (name, str(ex)[:120]), flush=True)
    (OUT / "_index.json").write_text(json.dumps(recs, indent=1))
    print("\nprepared %d datasets -> %s" % (len(recs), OUT))
