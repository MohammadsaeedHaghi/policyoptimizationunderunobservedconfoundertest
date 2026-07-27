"""SUPPORT2 recipe-A DGP (experiment R2-A): real covariates, AMPLIFIED synthetic confounding.

Real: x = the same observable apparent-severity composite (bootstrap of the 9,075 patients --
real covariate geometry). Synthetic: S ~ Bern(sigma(BP (x - x0))) with strong coupling BP=12
(owgap alpha=10 regime), x0 calibrated so P(S=+1) matches the real severity rate 0.499;
T ~ Bern(clip(sigma(C0 + CX x + CS S), .02, .98)) with CS = 0.8 -> Lambda = e^{1.6} = 4.95
(the SAME matched-Gamma=5 protocol as owgap/diabetes-A), CX = the real T ~ x coefficient,
C0 calibrated to the real DNR rate 0.352. Outcomes identical to dgp_b (K,G,D,TH = 3,1.5,1,0.3).
Role: the in-regime showcase on real ICU covariate geometry.
"""
import numpy as np
from pathlib import Path

HERE = Path(__file__).resolve().parent
_D = np.load(HERE / "support2_prepared.npz")
_x = _D["x"]
_C0r, _CXr, _CSr = [float(v) for v in _D["t_coefs"]]
P_S_REAL = 0.499
P_T_REAL = 0.352

LEVELS = np.round(np.linspace(-1.0, 1.0, 7), 6)   # vestigial (grid contract only)
K = 2
CAP = (1.0, 0.3)
BP = 12.0
CS = 0.8                                          # -> Lambda = e^{2*0.8} = 4.95, matched Gamma = 5
CX = _CXr                                         # realism anchor: real x-coefficient of the DNR fit
KSEV, GAM, DEL, TH, NOISE = 3.0, 1.5, 1.0, 0.3, 0.6


def _sig(z): return 1.0 / (1.0 + np.exp(-np.clip(np.asarray(z, float), -40, 40)))

def _bisect(f, lo, hi, tol=1e-6):
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if f(mid) > 0: hi = mid
        else: lo = mid
        if hi - lo < tol: break
    return 0.5 * (lo + hi)

X0 = _bisect(lambda x0: P_S_REAL - _sig(BP * (_x - x0)).mean(), -1.0, 1.0)   # increasing in x0

def _pt(c0):
    ps = _sig(BP * (_x - X0))
    e = np.clip(_sig(c0 + CX * _x[:, None] + CS * np.array([1.0, -1.0])[None, :]), 0.02, 0.98)
    return (e * np.column_stack([ps, 1 - ps])).sum(1).mean()

C0 = _bisect(lambda c0: _pt(c0) - P_T_REAL, -8.0, 8.0)                         # increasing in c0


def p_s1(X): return _sig(BP * (np.asarray(X, float) - X0))
def propensity(X, S): return np.clip(_sig(C0 + CX * np.asarray(X, float) + CS * np.asarray(S, float)), 0.02, 0.98)
def mu0(X, S): return -KSEV * np.asarray(S, float)
def mu1(X, S): return -(KSEV - GAM) * np.asarray(S, float) + DEL * np.asarray(X, float) - TH


def oracle_policy(X):
    X = np.asarray(X, float).ravel()
    ES = 2 * p_s1(X) - 1.0
    return ((GAM * ES + DEL * X - TH) > 0).astype(float)


def generate(n, seed=0):
    rng = np.random.default_rng(seed)
    X = _x[rng.choice(len(_x), size=n, replace=True)]        # real covariate geometry
    S = np.where(rng.uniform(size=n) < p_s1(X), 1.0, -1.0)
    e = propensity(X, S)
    T = (rng.uniform(size=n) < e).astype(int)
    m0, m1 = mu0(X, S), mu1(X, S)
    Y0 = m0 + rng.normal(0, NOISE, size=n)
    Y1 = m1 + rng.normal(0, NOISE, size=n)
    Y = np.where(T == 1, Y1, Y0)
    observed = {"X": X.reshape(-1, 1), "T": T, "Y": Y}
    full = {**observed, "S": S, "Y1": Y1, "Y0": Y0, "e_true": e}
    return observed, full


def grid_truth():
    X = LEVELS; ES = 2 * p_s1(X) - 1.0
    cate = GAM * ES + DEL * X - TH
    return dict(X=X, cate=cate, oracle=(cate > 0).astype(float))
