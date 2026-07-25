"""Prepare the UCI Diabetes 130 dataset into the (X, S, T) arrays the semi-synthetic DGP bootstraps.

Real ingredients (from the data):
  X  observed covariate  = composite frailty score -> [-1, 1]   (age, num_medications,
                           time_in_hospital, num_lab_procedures, num_procedures; z-scored, summed)
  S  hidden confounder   = acuity in {-1, +1}  (number_inpatient + number_emergency +
                           number_diagnoses + 0.8*facility-discharge; median split)
  T  treatment           = 1{insulin != No}    (the REAL insulin decision)
  Yr readmit reference    = 1{not readmitted within 30 days}  (real; kept only for calibration ref)

Also fits and stores the smooth REAL couplings used by the DGP:
  p_s1:  P(S=+1 | X)      = sigma(a + b X)              (logistic fit)
  eprop: P(T=1 | X, S)    = sigma(c0 + cx X + cs S)     (logistic fit; matched Gamma = exp(2 cs))

Output: assets/exp_diabetes/diabetes_prepared.npz
Usage:  python3 assets/exp_diabetes/prepare_data.py
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
CSV = HERE / "diabetic_data.csv"
OUT = HERE / "diabetes_prepared.npz"

EXPIRED_HOSPICE = {11, 13, 14, 19, 20, 21}   # discharge dispositions where readmission is undefined
FACILITY_DISCH = {3, 4, 5, 22, 23, 24}       # transfer to SNF / other facility (acuity signal)
AGE_MID = {f"[{i}-{i+10})": i + 5 for i in range(0, 100, 10)}


def _z(s):
    s = np.asarray(s, float)
    return (s - s.mean()) / (s.std() + 1e-9)


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -40, 40)))


def _logfit(Xmat, y, iters=500, lr=0.3, l2=1e-4):
    """Tiny logistic regression (batch gradient descent) -> weights. Xmat already has an intercept col."""
    Xmat = np.asarray(Xmat, float); y = np.asarray(y, float)
    w = np.zeros(Xmat.shape[1])
    n = len(y)
    for _ in range(iters):
        p = _sigmoid(Xmat @ w)
        g = Xmat.T @ (p - y) / n + l2 * w
        w -= lr * g
    return w


def main():
    if not CSV.exists():
        sys.exit(f"missing {CSV} -- download UCI #296 first (see data_explainer).")
    df = pd.read_csv(CSV, na_values=["?"], low_memory=False)
    n0 = len(df)
    df = df[~df.discharge_disposition_id.isin(EXPIRED_HOSPICE)]
    df = df.sort_values("encounter_id").drop_duplicates("patient_nbr", keep="first").reset_index(drop=True)
    n1 = len(df)

    # --- observed covariate X: composite frailty score, robust-rescaled to [-1, 1] ---
    age = df.age.map(AGE_MID).astype(float)
    Xraw = (_z(age) + _z(df.num_medications) + _z(df.time_in_hospital)
            + _z(df.num_lab_procedures) + _z(df.num_procedures))
    lo, hi = np.percentile(Xraw, 1), np.percentile(Xraw, 99)
    X = np.clip(2 * (Xraw - lo) / (hi - lo) - 1.0, -1.0, 1.0)

    # --- hidden confounder S: acuity, median split to {-1, +1} ---
    facility = df.discharge_disposition_id.isin(FACILITY_DISCH).astype(float).values
    sev = _z(df.number_inpatient) + _z(df.number_emergency) + _z(df.number_diagnoses) + 0.8 * facility
    S = np.where(sev > np.median(sev), 1.0, -1.0)

    # --- real treatment T = 1{insulin != No}; real readmit reference ---
    T = (df.insulin.values != "No").astype(int)
    Yr = (df.readmitted.values == "<30").astype(int)   # 1 = readmitted <30d; we model no-readmit benefit
    Yr = 1 - Yr                                          # 1 = NOT readmitted within 30 days (higher better)

    # --- fit smooth real couplings ---
    a_b = _logfit(np.column_stack([np.ones_like(X), X]), (S > 0).astype(float))            # P(S=+1|X)
    prop = _logfit(np.column_stack([np.ones_like(X), X, S]), T.astype(float))              # P(T=1|X,S)
    matched_gamma = float(np.exp(2 * prop[2]))   # odds ratio of e at S=+1 vs S=-1

    # --- diagnostics ---
    corrXS = float(np.corrcoef(X, S)[0, 1])
    print(f"raw={n0}  cleaned+deduped={n1}")
    print(f"X in [{X.min():.2f},{X.max():.2f}]  S(+1) frac={np.mean(S>0):.3f}  corr(X,S)={corrXS:.3f}")
    print(f"T=1 (insulin) frac={T.mean():.3f}")
    print(f"real no-readmit rate: S=-1 {Yr[S<0].mean():.3f}  S=+1 {Yr[S>0].mean():.3f}  (severity lowers it)")
    print(f"p_s1 fit  sigma({a_b[0]:+.3f} {a_b[1]:+.3f} X)")
    print(f"prop fit  sigma({prop[0]:+.3f} {prop[1]:+.3f} X {prop[2]:+.3f} S)  => matched Gamma={matched_gamma:.3f}")

    np.savez(OUT, X=X.astype(np.float64), S=S.astype(np.float64), T=T.astype(np.int64),
             Yr=Yr.astype(np.int64), ps1_ab=a_b.astype(np.float64), prop_c=prop.astype(np.float64),
             matched_gamma=np.float64(matched_gamma), corr_xs=np.float64(corrXS))
    print(f"saved {OUT}  ({n1} rows)")


if __name__ == "__main__":
    main()
