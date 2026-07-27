#!/usr/bin/env python3
"""IST recipe-D preparation: real RCT outcomes + INJECTED confounding (experiment R1 pilot).

Source: IST_corrected.csv (Edinburgh DataShare, open licence; cached in /scratch1/haghim/uci_cache;
latin-1 encoded). n=19,435 stroke patients randomized to aspirin (P=0.5 exactly).

Construction:
  Y  = REAL favorable 6-month outcome, 1{OCCODE >= 3} (1 dead, 2 dependent, 3 not recovered,
       4 recovered). No outcome synthesis anywhere.
  T  = randomized aspirin (RXASP).
  S  = fully alert at baseline (RCONSC == 'F'), HIDDEN from the learner. P(S)=0.77,
       S->Y = +0.34, corr(X_obs, S) = 0.57 (AUC 0.86) -- upper-boundary coupling.
  X  = 1-D prognostic composite: logistic P(favorable | X_obs) fitted on the CONTROL arm of the
       TRAIN split only (no treatment leakage), then rank-transformed to Uniform[-1, 1].
  Confounded logging (train pool only): keep patient with probability q(T,S) where
       q(1,S=1)=q(0,S=0)=a, q(1,S=0)=q(0,S=1)=b, a/b = sqrt(Lambda) -> conditional-on-X
       selection odds ratio EXACTLY Lambda (keep prob depends on X only through S).
       Lambda = 4.95 -> matched Gamma = 5, same protocol as owgap. Direction: alert patients
       enriched in the treated arm -> aspirin looks far better than it is (over-treatment).
  Split: 60% train pool (subsampled with bias) / 40% test pool (UNTOUCHED RCT, for evaluation
       via Horvitz-Thompson pseudo-outcomes Y1=2*T*Y, Y0=2*(1-T)*Y, unbiased since P(T)=1/2).
  "Oracle": binned CATE estimated on the FULL train RCT (unconfounded, n~11.6k) -- the best
       available ground-truth benchmark; with aspirin's small real effect this is intentionally
       a hard, honest test (see pilot notes).

Writes ist_prepared.npz + prints all verification numbers.
"""
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression

HERE = Path(__file__).resolve().parent
CSV = Path("/scratch1/haghim/uci_cache/IST_corrected.csv")
LAMBDA = 4.95
KEEP_B = 0.35                              # a/b = sqrt(LAMBDA)
KEEP_A = KEEP_B * np.sqrt(LAMBDA)

df = pd.read_csv(CSV, low_memory=False, encoding="latin-1")
occ = pd.to_numeric(df["OCCODE"], errors="coerce")
ok = df["RCONSC"].notna() & occ.isin([1, 2, 3, 4]) & df["AGE"].notna() & df["RSBP"].notna()
d = df[ok].reset_index(drop=True); occ = occ[ok.values].reset_index(drop=True)
Y = (occ >= 3).astype(int).values
S = (d["RCONSC"].astype(str).str.upper().str.startswith("F")).astype(int).values
T = (d["RXASP"].astype(str).str.upper() == "Y").astype(int).values
n = len(d)
print(f"usable n={n}  P(T)={T.mean():.3f}  P(S)={S.mean():.3f}  P(Y=fav)={Y.mean():.3f}")
print(f"REAL aspirin effect (full RCT): {Y[T==1].mean() - Y[T==0].mean():+.4f}")
print(f"S->Y: {Y[S==1].mean() - Y[S==0].mean():+.3f}")

# ---- observed features -> one-hot design (consciousness EXCLUDED) ----
feats = d[["AGE", "SEX", "RSBP", "STYPE", "RDEF1", "RDEF2", "RDEF3", "RDEF4",
           "RDEF5", "RDEF6", "RDEF7", "RDEF8"]].copy()
num = feats[["AGE", "RSBP"]].astype(float)
cat = pd.get_dummies(feats.drop(columns=["AGE", "RSBP"]).astype(str), drop_first=True, dtype=float)
Z = pd.concat([num, cat], axis=1)
Z = ((Z - Z.mean()) / (Z.std() + 1e-9)).values

# ---- train / test split (fixed) ----
rng = np.random.default_rng(0)
test_mask = rng.uniform(size=n) < 0.40
tr, te = ~test_mask, test_mask
print(f"train RCT pool {tr.sum()}  test RCT pool {te.sum()}")

# ---- 1-D prognostic composite fitted on TRAIN CONTROL arm only ----
fit = tr & (T == 0)
lr = LogisticRegression(max_iter=3000, C=1.0).fit(Z[fit], Y[fit])
score = lr.predict_proba(Z)[:, 1]
ranks = pd.Series(score).rank(method="average").values / n
x = (2 * ranks - 1).astype(float)                      # Uniform[-1, 1] marginal
print(f"composite: corr(x, S) = {np.corrcoef(x, S)[0,1]:.3f}  corr(x, Y) = {np.corrcoef(x, Y)[0,1]:.3f}")

# ---- confounded logging on the train pool ----
keep_p = np.where(T == S, KEEP_A, KEEP_B)              # q(1,1)=q(0,0)=a; q(1,0)=q(0,1)=b
keep = tr & (rng.uniform(size=n) < keep_p)
print(f"kept (biased) train pool: {keep.sum()}  P(T|kept)={T[keep].mean():.3f}")
print(f"P(S=1|T=1,kept)={S[keep & (T==1)].mean():.3f}  P(S=1|T=0,kept)={S[keep & (T==0)].mean():.3f}")
print(f"NAIVE apparent aspirin effect in kept pool: {Y[keep & (T==1)].mean() - Y[keep & (T==0)].mean():+.4f}")
# verify conditional-on-x selection odds ratio == LAMBDA
zk = np.column_stack([x[keep], 2 * S[keep] - 1.0])
lam_fit = LogisticRegression(max_iter=3000, C=1e6).fit(zk, T[keep])
print(f"matched-Gamma check: Lambda_fit = {np.exp(2 * abs(lam_fit.coef_[0][1])):.2f} (target {LAMBDA})")

# ---- "oracle" benchmark: binned CATE on the FULL (unconfounded) train RCT ----
BINS = np.linspace(-1, 1, 21)
bi = np.clip(np.digitize(x, BINS) - 1, 0, 19)
cate_bin = np.full(20, 0.0)
for b in range(20):
    m1 = tr & (bi == b) & (T == 1); m0 = tr & (bi == b) & (T == 0)
    if m1.sum() > 20 and m0.sum() > 20:
        cate_bin[b] = Y[m1].mean() - Y[m0].mean()
cate_sm = np.convolve(np.pad(cate_bin, 1, mode="edge"), np.ones(3) / 3, mode="valid")   # light smooth
print("binned RCT CATE (smoothed):", np.round(cate_sm, 3))
print(f"oracle treats {np.mean(cate_sm > 0):.0%} of bins; value stakes: max|CATE| = {np.abs(cate_sm).max():.3f}")

np.savez(HERE / "ist_prepared.npz",
         x=x, S=S, T=T, Y=Y, keep=keep, test=te,
         bins=BINS, cate_sm=cate_sm, lam=LAMBDA)
print("saved ist_prepared.npz")
