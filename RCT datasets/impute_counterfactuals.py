#!/usr/bin/env python3
"""Impute the missing potential outcome for LaLonde, NHEFS, IST and STAR by three methods.

These four record only the factual arm, so Y(1-T) has to be estimated. Three methods are used,
chosen to fail differently rather than to be three flavours of the same idea:

  T-learner   two separate outcome regressions, mu_0 and mu_1, each fit on its own arm only.
              Flexible, but each model extrapolates into covariate regions its arm barely covers.
  X-learner   Kunzel et al. Fit the T-learner, impute each unit's individual effect from the
              OTHER arm's model, regress those imputed effects on X per arm, then blend the two
              effect models by the propensity score. Designed for arms of very unequal size --
              which is exactly LaLonde (185 vs 260) and NHEFS (403 vs 1163).
  1-NN match  non-parametric: the counterfactual is the OBSERVED outcome of the nearest unit in
              the opposite arm, on standardised covariates. Never extrapolates and invents no
              value, but is noisy because it uses a single neighbour.

VALIDATION. Three of the four are randomised, so difference-in-means is an unbiased estimate of
the true ATE. That gives a real target: after imputation the completed table implies its own ATE
(mean of Y1 - Y0 over ALL units), and it should land near the randomised benchmark. A method that
misses it is imputing badly, and this is checkable without ever seeing a true counterfactual.
The second check is factual: 5-fold cross-validated fit on the OBSERVED arm, which measures
whether the outcome model can predict at all before it is trusted to extrapolate.

Everything is written to counterfactuals/<name>_<method>.csv with Y0, Y1, which_imputed.
"""
import json, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import r2_score, roc_auc_score

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
OUT = HERE / "counterfactuals"; OUT.mkdir(exist_ok=True)
SEED = 0


# ------------------------------------------------------------------ model helpers
def _reg(binary):
    return (GradientBoostingClassifier(n_estimators=200, max_depth=3, random_state=SEED)
            if binary else
            GradientBoostingRegressor(n_estimators=200, max_depth=3, random_state=SEED))


def _fit_predict(Xtr, ytr, Xev, binary):
    m = _reg(binary).fit(Xtr, ytr)
    return m.predict_proba(Xev)[:, 1] if binary else m.predict(Xev)


def factual_cv(X, T, Y, binary, folds=5):
    """5-fold CV fit on the FACTUAL outcome, per arm. Can the model predict at all?"""
    out = {}
    for t in (0, 1):
        m = T == t
        Xa, ya = X[m], Y[m]
        if len(ya) < 5 * folds: out["arm%d" % t] = None; continue
        pred = np.zeros(len(ya))
        for tr, te in KFold(folds, shuffle=True, random_state=SEED).split(Xa):
            pred[te] = _fit_predict(Xa[tr], ya[tr], Xa[te], binary)
        try:
            out["arm%d" % t] = (float(roc_auc_score(ya, pred)) if binary
                                else float(r2_score(ya, pred)))
        except Exception:
            out["arm%d" % t] = None
    return out


# ------------------------------------------------------------------ the three methods
def m_tlearner(X, T, Y, binary):
    mu0 = _fit_predict(X[T == 0], Y[T == 0], X, binary)
    mu1 = _fit_predict(X[T == 1], Y[T == 1], X, binary)
    return mu0, mu1


def m_xlearner(X, T, Y, binary):
    mu0, mu1 = m_tlearner(X, T, Y, binary)
    # imputed individual effects, each using the OTHER arm's model
    d1 = Y[T == 1] - mu0[T == 1]            # treated: observed minus predicted-if-control
    d0 = mu1[T == 0] - Y[T == 0]            # control: predicted-if-treated minus observed
    tau1 = GradientBoostingRegressor(n_estimators=200, max_depth=3,
                                     random_state=SEED).fit(X[T == 1], d1).predict(X)
    tau0 = GradientBoostingRegressor(n_estimators=200, max_depth=3,
                                     random_state=SEED).fit(X[T == 0], d0).predict(X)
    e = LogisticRegression(max_iter=2000).fit(X, T).predict_proba(X)[:, 1]
    tau = e * tau0 + (1 - e) * tau1         # Kunzel's propensity blend
    # turn the effect back into two potential outcomes anchored on the factual one
    Y0 = np.where(T == 1, Y - tau, Y)
    Y1 = np.where(T == 1, Y, Y + tau)
    return Y0, Y1


def m_match(X, T, Y, binary):
    """1-NN in the opposite arm on standardised covariates: the counterfactual is a real outcome."""
    Z = (X - X.mean(0)) / (X.std(0) + 1e-9)
    Y0 = np.where(T == 0, Y, np.nan); Y1 = np.where(T == 1, Y, np.nan)
    for t in (0, 1):
        src = T == (1 - t)                                  # donors: the opposite arm
        need = T == t
        nn = NearestNeighbors(n_neighbors=1).fit(Z[src])
        idx = nn.kneighbors(Z[need], return_distance=False).ravel()
        donor = Y[src][idx]
        if t == 0: Y1[need] = donor
        else:      Y0[need] = donor
    return Y0, Y1


METHODS = [("T-learner", m_tlearner), ("X-learner", m_xlearner), ("1NN-match", m_match)]


def run(name, X, T, Y, binary, bench_note, cols):
    X = np.asarray(X, float); T = np.asarray(T).astype(int); Y = np.asarray(Y, float)
    ate_rand = float(Y[T == 1].mean() - Y[T == 0].mean())
    rec = {"dataset": name, "n": int(len(Y)), "n_treated": int(T.sum()),
           "n_control": int((1 - T).sum()), "outcome_binary": bool(binary),
           "benchmark_ate": ate_rand, "benchmark_note": bench_note,
           "factual_cv": factual_cv(X, T, Y, binary), "methods": {}}
    print("\n" + "=" * 84)
    print("%s   n=%d (%d treated / %d control)   outcome=%s"
          % (name, len(Y), T.sum(), (1 - T).sum(), "binary" if binary else "continuous"))
    print("  benchmark ATE (difference in means) = %+.4f   [%s]" % (ate_rand, bench_note))
    print("  factual 5-fold CV (%s): %s" % ("AUC" if binary else "R2",
          {k: (None if v is None else round(v, 3)) for k, v in rec["factual_cv"].items()}))
    print("  %-11s %12s %12s %10s" % ("method", "implied ATE", "vs benchmark", "|error|"))
    for mname, fn in METHODS:
        a, b = fn(X, T, Y, binary)
        if mname == "T-learner":
            Y0 = np.where(T == 0, Y, a); Y1 = np.where(T == 1, Y, b)   # keep the factual value
        else:
            Y0, Y1 = a, b
        ate = float(np.nanmean(Y1 - Y0))
        rec["methods"][mname] = {"implied_ate": ate, "abs_error_vs_benchmark": abs(ate - ate_rand),
                                 "cate_sd": float(np.nanstd(Y1 - Y0))}
        print("  %-11s %+12.4f %+12.4f %10.4f" % (mname, ate, ate - ate_rand, abs(ate - ate_rand)))
        df = pd.DataFrame(X, columns=cols)
        df["T"] = T; df["Y_factual"] = Y; df["Y0"] = Y0; df["Y1"] = Y1
        df["which_imputed"] = np.where(T == 1, "Y0", "Y1")
        df.to_csv(OUT / ("%s_%s.csv" % (name.lower().replace("/", "_").replace(" ", ""), mname)),
                  index=False)
    return rec


# ------------------------------------------------------------------ 3. LaLonde / NSW
cols = ["treat", "age", "education", "black", "hispanic", "married", "nodegree",
        "re74", "re75", "re78"]
tr = pd.read_csv(HERE / "lalonde/nswre74_treated.txt", sep=r"\s+", header=None, names=cols)
co = pd.read_csv(HERE / "lalonde/nswre74_control.txt", sep=r"\s+", header=None, names=cols)
nsw = pd.concat([tr, co], ignore_index=True)
XC = ["age", "education", "black", "hispanic", "married", "nodegree", "re74", "re75"]
R = [run("LaLonde", nsw[XC].values, nsw.treat.values, nsw.re78.values, False,
         "RANDOMISED -- this is the true ATE", XC)]

# ------------------------------------------------------------------ 4. NHEFS
nh = pd.read_csv(HERE / "nhefs/nhefs.csv").dropna(subset=["wt82_71", "qsmk"])
XC = ["sex", "age", "race", "education", "smokeintensity", "smokeyrs", "exercise",
      "active", "wt71"]
nh = nh.dropna(subset=XC)
R.append(run("NHEFS", nh[XC].values, nh.qsmk.values, nh.wt82_71.values, False,
             "OBSERVATIONAL -- NOT a valid target, shown only for reference", XC))

# ------------------------------------------------------------------ 5. IST
ist = pd.read_csv(HERE / "ist/IST_corrected.csv", low_memory=False, encoding="latin-1")
ist["T"] = ist.RXASP.astype(str).str.strip().str.upper().eq("Y").astype(int)
ist["Ydead"] = ist.OCCODE.eq(1).astype(float)          # dead at 6 months
XC = ["AGE", "RSBP", "RDELAY"] + ["RDEF%d" % i for i in range(1, 9)]
for c in ["SEX", "RCONSC", "RATRIAL", "RVISINF", "RSLEEP", "STYPE"]:
    ist[c + "_n"] = pd.factorize(ist[c])[0]; XC.append(c + "_n")
for c in ["RDEF%d" % i for i in range(1, 9)]:
    ist[c] = pd.factorize(ist[c])[0]
ist = ist.dropna(subset=XC + ["Ydead", "T"])
R.append(run("IST", ist[XC].values, ist["T"].values, ist["Ydead"].values, True,
             "RANDOMISED (aspirin) -- this is the true ATE", XC))

# ------------------------------------------------------------------ 6. STAR
st = pd.read_csv(HERE / "star/STAR.csv", low_memory=False)
st = st.dropna(subset=["stark", "readk", "mathk"]).copy()
st["T"] = st.stark.eq("small").astype(int)             # small vs (regular + regular+aide)
st["Ytot"] = st.readk + st.mathk
XC = []
for c in ["gender", "ethnicity", "birth", "lunchk", "schoolk", "degreek", "ladderk",
          "experiencek", "tethnicityk"]:
    if c not in st.columns: continue
    st[c + "_n"] = pd.factorize(st[c])[0] if st[c].dtype == object else st[c]
    XC.append(c + "_n")
st = st.dropna(subset=XC + ["Ytot"])
R.append(run("STAR", st[XC].values, st["T"].values, st["Ytot"].values, False,
             "RANDOMISED (within school) -- this is the true ATE", XC))

json.dump(R, open(HERE / "counterfactuals.json", "w"), indent=1)
print("\n\nwrote counterfactuals.json and %d CSVs in %s" % (len(list(OUT.glob("*.csv"))), OUT))
