#!/usr/bin/env python3
"""SUPPORT2 preparation (experiment R2 pilot): real ICU cohort, hidden APS severity.

Source: UCI 880 (cached as /scratch1/haghim/uci_cache/support2_{X,y}.pkl by the survey's
smoke_tests.py). 9,105 seriously ill hospitalized adults from the SUPPORT study (same cohort
family as the RHC dataset used throughout the MSM-Gamma sensitivity literature).

Defines, and saves to support2_prepared.npz:
  x      1-D observed clinical risk composite: logistic P(death | X_obs) on
         age/sex/dzgroup/num.co/edu/income/race (NO physiology), rank-uniform to [-1, 1].
  S      +-1: APS physiology score > median -- HIDDEN severity.
  T      real DNR order (P ~ 0.35).
  Yr     real death indicator (for calibration + reporting only; experiment outcomes are
         synthetic per recipe B).
  ps1_ab logistic fit of P(S=+1 | x) on the real coupling (used by dgp_b oracle).
  Also prints: corr(x,S), matched Lambda on the 1-D composite, real death rates by (S,T).
"""
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression

HERE = Path(__file__).resolve().parent
CACHE = Path("/scratch1/haghim/uci_cache")

X = pd.read_pickle(CACHE / "support2_X.pkl")
y = pd.read_pickle(CACHE / "support2_y.pkl")
df = pd.concat([X, y], axis=1)
aps = pd.to_numeric(df["aps"], errors="coerce")
ok = aps.notna() & df["dnr"].notna() & df["death"].notna() & df["age"].notna()
d = df[ok].reset_index(drop=True); aps = aps[ok.values].reset_index(drop=True)
n = len(d)
S01 = (aps > aps.median()).astype(int).values
T = (d["dnr"].astype(str).str.strip().str.lower() != "no dnr").astype(int).values
Yr = pd.to_numeric(d["death"], errors="coerce").fillna(0).astype(int).values
print(f"n={n}  P(S)={S01.mean():.3f}  P(T=DNR)={T.mean():.3f}  P(death)={Yr.mean():.3f}")

feats = d[["age", "sex", "dzgroup", "num.co", "edu", "income", "race"]].copy()
num = feats[["age", "num.co", "edu"]].apply(pd.to_numeric, errors="coerce")
num = num.fillna(num.median())
cat = pd.get_dummies(feats[["sex", "dzgroup", "income", "race"]].astype(str), drop_first=True, dtype=float)
Z = pd.concat([num, cat], axis=1)
Z = ((Z - Z.mean()) / (Z.std() + 1e-9)).values

# 1-D composite = APPARENT-SEVERITY score: the observable projection most informative about S
# (cross-validated so the coupling is honest, not overfit). NOTE (disclosed design choice): a
# death-risk composite was tried first and measured corr(x,S) = -0.07 -- sociodemographic
# mortality risk is nearly orthogonal to APS physiology -- which would silently turn R2 into
# a deep out-of-regime experiment. The S-proxy direction preserves the survey's multivariate
# coupling (0.47) in the 1-D pipeline and stands in for a clinician's observable severity
# impression; the learner still never sees S itself.
from sklearn.model_selection import cross_val_predict
lrS = LogisticRegression(max_iter=3000, C=1.0)
score = cross_val_predict(lrS, Z, S01, cv=3, method="predict_proba")[:, 1]
x = (2 * pd.Series(score).rank(method="average").values / n - 1).astype(float)
print(f"composite (S-proxy, CV): corr(x, S)={np.corrcoef(x, S01)[0,1]:.3f}  corr(x, death)={np.corrcoef(x, Yr)[0,1]:.3f}")

Spm = 2 * S01 - 1.0
lam_fit = LogisticRegression(max_iter=3000, C=1e6).fit(np.column_stack([x, Spm]), T)
lam = float(np.exp(2 * abs(lam_fit.coef_[0][1])))
print(f"matched Lambda on 1-D composite: {lam:.2f}")

cp = LogisticRegression(max_iter=3000, C=1e6).fit(x.reshape(-1, 1), S01)
a_, b_ = float(cp.intercept_[0]), float(cp.coef_[0][0])
print(f"real coupling fit P(S=+1|x) = sigma({a_:.3f} + {b_:.3f} x)")
for s in (0, 1):
    for t in (0, 1):
        m = (S01 == s) & (T == t)
        print(f"  P(death | S={s}, T={t}) = {Yr[m].mean():.3f}   (n={m.sum()})")

np.savez(HERE / "support2_prepared.npz", x=x, S=Spm, T=T, Yr=Yr,
         ps1_ab=np.array([a_, b_]), matched_lambda=lam,
         t_coefs=np.array([float(lam_fit.intercept_[0]), float(lam_fit.coef_[0][0]),
                           float(lam_fit.coef_[0][1])]))
print("saved support2_prepared.npz")
