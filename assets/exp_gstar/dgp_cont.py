"""exp_gstar DGP (continuous): X ~ Uniform(-1, 1); otherwise identical constants to dgp.py.

Same Gamma-star-by-inspection property: e(x, S) = sigma(-1.5 x + (1/2) ln(5) S), no clipping,
so the S-odds ratio is exactly Gamma* = 5 at every x. CATE(x) = (2 sigma(10x) - 1) + 3x - 1
crosses zero at x* ~ 0.137 (oracle treats ~43% of mass); the L x Gamma pipeline (Lipschitz
policy class + Shapley off-support deployment) evaluates realized value on fresh test draws.
grid_truth/exact_value/LEVELS kept only for grid diagnostics and the Kallus grid contract.
"""
import numpy as np

LEVELS = np.round(np.linspace(-1.0, 1.0, 7), 6)   # vestigial (grid diagnostics only)
X_LO, X_HI = -1.0, 1.0
ALPHA, CX, D0, D1, B0, B1, NOISE = 10.0, 1.5, 8.0, 9.0, 0.0, 3.0, 0.6
LAM = 5.0
CS = 0.5 * np.log(LAM)
THETA = 1.0
CAP = (1.0, 0.3)
K = 2


def _sig(z): return 1.0 / (1.0 + np.exp(-np.clip(np.asarray(z, float), -40, 40)))
def p_s1(X): return _sig(ALPHA * np.asarray(X, float))
def propensity(X, S): return _sig(CS * np.asarray(S, float) - CX * np.asarray(X, float))   # NO clip
def mu0(X, S): return D0 * np.asarray(S, float) + B0 * np.asarray(X, float)
def mu1(X, S): return D1 * np.asarray(S, float) + B1 * np.asarray(X, float) - THETA


def grid_truth():
    X = LEVELS; ps1 = p_s1(X); ES = 2 * ps1 - 1.0
    eY0 = D0 * ES + B0 * X; eY1 = D1 * ES + B1 * X - THETA; cate = eY1 - eY0
    return dict(X=X, p_s1=ps1, eY0=eY0, eY1=eY1, e_plus=propensity(X, 1.0),
                e_minus=propensity(X, -1.0), cate=cate, oracle=(cate > 0).astype(float))


def oracle_policy(X):
    X = np.asarray(X, float); ES = 2 * p_s1(X) - 1.0
    return (((D1 - D0) * ES + (B1 - B0) * X - THETA) > 0).astype(float)


def generate(n, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.uniform(X_LO, X_HI, size=n)                       # CONTINUOUS covariate
    S = np.where(rng.uniform(size=n) < p_s1(X), 1.0, -1.0)
    e = propensity(X, S); T = (rng.uniform(size=n) < e).astype(int)
    m0, m1 = mu0(X, S), mu1(X, S)
    Y0 = m0 + rng.normal(0, NOISE, size=n); Y1 = m1 + rng.normal(0, NOISE, size=n)
    Y = np.where(T == 1, Y1, Y0)
    observed = {"X": X.reshape(-1, 1), "T": T, "Y": Y}
    full = {**observed, "S": S, "Y0": Y0, "Y1": Y1, "e_true": e}
    return observed, full


def exact_value(pi_grid):
    pi = np.asarray(pi_grid, float); t = grid_truth()
    return float(np.mean(pi * t["eY1"] + (1 - pi) * t["eY0"]))
