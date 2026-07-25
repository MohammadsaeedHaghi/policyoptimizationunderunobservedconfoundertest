"""exp_diabetes DGP: SEMI-SYNTHETIC real-data experiment on UCI Diabetes 130.

Hybrid design (decided 2026-07-25): REAL (X, S, T) + SYNTHETIC potential outcomes Y(0), Y(1).
Keeping synthetic outcomes preserves ground-truth counterfactuals -> an oracle and a true realized
E[Y], while the covariate X, the hidden confounder S, the X-S coupling, and the treatment T are all
real (see prepare_data.py / diabetes_prepared.npz).

  X   observed covariate  = real composite frailty score in [-1, 1]        (bootstrapped from data)
  S   hidden confounder   = real acuity in {-1, +1}                        (bootstrapped, paired with X)
  T   treatment           = real insulin decision, 1{insulin != No}        (bootstrapped, paired)
  p_s1(X) = sigma(a + b X)   REAL fitted coupling  (NOT owgap's sigma(10 X))
  mu0(X,S) = -KSEV*S                    (sicker S=+1 -> lower benefit; S dominates the scale)
  mu1(X,S) = -KSEV*S + GAM*S + DEL*X    => CATE_ind = mu1-mu0 = GAM*S + DEL*X
  Y(t) = mu_t(X,S) + N(0, NOISE^2);   observed Y = Y(T) with the REAL T.
  CATE(X) = E[mu1-mu0 | X] = GAM*(2 p_s1(X) - 1) + DEL*X   => oracle treats iff CATE(X) > 0 (a subset).

Same generate(n, seed) -> (observed, full) contract as assets/exp_owgap_cont/dgp.py, so it flows
through run_owgap_continuous.py / the mesh / Lipschitz / KNN-Shapley machinery unchanged (via --dgp).
"""
from pathlib import Path
import numpy as np

_NPZ = Path(__file__).resolve().parent / "diabetes_prepared.npz"
_D = np.load(_NPZ)
_X_ALL = _D["X"].astype(float)          # (N,) real composite frailty in [-1,1]
_S_ALL = _D["S"].astype(float)          # (N,) real acuity in {-1,+1}
_T_ALL = _D["T"].astype(int)            # (N,) real insulin decision {0,1}
_PS1_AB = _D["ps1_ab"].astype(float)    # (2,) logistic (a,b) for P(S=+1|X)
_PROP_C = _D["prop_c"].astype(float)    # (3,) logistic (c0,cx,cs) for P(T=1|X,S)
MATCHED_GAMMA = float(_D["matched_gamma"])  # ~1.26 (real, weak)

X_LO, X_HI = -1.0, 1.0
LEVELS = np.round(np.linspace(-1.0, 1.0, 7), 6)   # vestigial (grid diagnostics only)
# --- synthetic-outcome calibration constants (S-dominant; CATE crosses zero over X) ---
KSEV, GAM, DEL, NOISE = 2.5, 0.6, 0.8, 0.6
D0, D1, B0, B1 = -KSEV, -(KSEV - GAM), 0.0, DEL   # owgap-style aliases: mu0=D0 S, mu1=D1 S + B1 X
CAP = (1.0, 0.5)
K = 2


def _sig(z): return 1.0 / (1.0 + np.exp(-np.clip(np.asarray(z, float), -40, 40)))
def p_s1(X): X = np.asarray(X, float); return _sig(_PS1_AB[0] + _PS1_AB[1] * X)
def propensity(X, S):
    X = np.asarray(X, float); S = np.asarray(S, float)
    return np.clip(_sig(_PROP_C[0] + _PROP_C[1] * X + _PROP_C[2] * S), 0.02, 0.98)
def mu0(X, S): return D0 * np.asarray(S, float) + B0 * np.asarray(X, float)
def mu1(X, S): return D1 * np.asarray(S, float) + B1 * np.asarray(X, float)


def oracle_policy(X):
    # observable optimal policy: treat iff E[CATE | X] > 0, using the REAL fitted E[S|X]
    X = np.asarray(X, float); ES = 2 * p_s1(X) - 1.0
    return (((D1 - D0) * ES + (B1 - B0) * X) > 0).astype(float)


def grid_truth():
    X = LEVELS; ps1 = p_s1(X); ES = 2 * ps1 - 1.0
    eY0 = D0 * ES + B0 * X; eY1 = D1 * ES + B1 * X; cate = eY1 - eY0
    return dict(X=X, p_s1=ps1, m0p=mu0(X, 1.0), m0m=mu0(X, -1.0), m1p=mu1(X, 1.0), m1m=mu1(X, -1.0),
                eY0=eY0, eY1=eY1, e_plus=propensity(X, 1.0), e_minus=propensity(X, -1.0),
                e_marg=ps1 * propensity(X, 1.0) + (1 - ps1) * propensity(X, -1.0),
                cate=cate, oracle=(cate > 0).astype(float))


def generate(n, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(_X_ALL), size=n)               # bootstrap real (X,S,T) triples
    X, S, T = _X_ALL[idx], _S_ALL[idx], _T_ALL[idx].astype(int)
    m0, m1 = mu0(X, S), mu1(X, S)
    Y0 = m0 + rng.normal(0, NOISE, size=n); Y1 = m1 + rng.normal(0, NOISE, size=n)
    Y = np.where(T == 1, Y1, Y0)                              # observed outcome under the REAL T
    e = propensity(X, S)
    observed = {"X": X.reshape(-1, 1), "T": T, "Y": Y}
    full = {**observed, "S": S, "Y0": Y0, "Y1": Y1, "e_true": e, "mu0": m0, "mu1": m1}
    return observed, full


def exact_value(pi_grid):
    pi = np.asarray(pi_grid, float); t = grid_truth()
    return float(np.mean(pi * t["eY1"] + (1 - pi) * t["eY0"]))
