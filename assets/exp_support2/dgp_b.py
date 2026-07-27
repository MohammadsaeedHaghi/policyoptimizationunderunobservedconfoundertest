"""SUPPORT2 recipe-B DGP (experiment R2-B): fully real (x, S, T), synthetic outcomes only.

Real: x = observable apparent-severity composite (rank-uniform on [-1,1], corr(x,S)=0.46),
S = APS > median (hidden, coded +-1), T = real DNR order (P=0.35, matched Lambda on the
composite = 2.20 -- the PREDICTED-BOUNDARY test of the coupling diagnostic).
Synthetic: only Y(0)/Y(1) = mu_t(x,S) + N(0, 0.6^2), severity-dominant, with the showcase
constants (K, G, D, TH = 3.0, 1.5, 1.0, 0.3):
    mu0 = -K S,   mu1 = -(K - G) S + D x - TH   =>   CATE = G S + D x - TH
Under the REAL fitted coupling P(S=+1|x) = sigma(a + b x):
    E[CATE | x] = G (2 sigma(a+bx) - 1) + D x - TH   =>  oracle treats x > x* ~ 0.13.
Failure mode (mirror of Diabetes-A): DNR patients are severity-enriched, so the naive
within-x comparison makes DNR look harmful -> under-treatment.
Observed Y = Y(T) with the REAL T. Test draws (seed >= anything) use the same generator --
potential outcomes are known, so the runner's standard evaluation applies.
"""
import numpy as np
from pathlib import Path

HERE = Path(__file__).resolve().parent
_D = np.load(HERE / "support2_prepared.npz")
_x, _S, _T = _D["x"], _D["S"], _D["T"].astype(int)
_A, _B = [float(v) for v in _D["ps1_ab"]]

LEVELS = np.round(np.linspace(-1.0, 1.0, 7), 6)   # vestigial (grid contract only)
K = 2
CAP = (1.0, 0.3)
KSEV, GAM, DEL, TH, NOISE = 3.0, 1.5, 1.0, 0.3, 0.6
MATCHED_LAMBDA = float(_D["matched_lambda"])


def p_s1(X):
    X = np.asarray(X, float)
    return 1.0 / (1.0 + np.exp(-np.clip(_A + _B * X, -40, 40)))


def mu0(X, S): return -KSEV * np.asarray(S, float)
def mu1(X, S): return -(KSEV - GAM) * np.asarray(S, float) + DEL * np.asarray(X, float) - TH


def oracle_policy(X):
    X = np.asarray(X, float).ravel()
    ES = 2 * p_s1(X) - 1.0
    return ((GAM * ES + DEL * X - TH) > 0).astype(float)


def generate(n, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(_x), size=n, replace=True)      # bootstrap real (x, S, T) triples
    X = _x[idx].reshape(-1, 1); S = _S[idx]; T = _T[idx]
    m0, m1 = mu0(X.ravel(), S), mu1(X.ravel(), S)
    Y0 = m0 + rng.normal(0, NOISE, size=n)
    Y1 = m1 + rng.normal(0, NOISE, size=n)
    Y = np.where(T == 1, Y1, Y0)
    observed = {"X": X, "T": T, "Y": Y}
    full = {**observed, "S": S, "Y1": Y1, "Y0": Y0}
    return observed, full


def grid_truth():
    X = LEVELS; ES = 2 * p_s1(X) - 1.0
    cate = GAM * ES + DEL * X - TH
    return dict(X=X, cate=cate, oracle=(cate > 0).astype(float))
