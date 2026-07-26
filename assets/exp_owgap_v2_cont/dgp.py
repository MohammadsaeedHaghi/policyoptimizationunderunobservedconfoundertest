"""exp_owgap_v2_cont DGP: CONTINUOUS-X version of exp_owgap_v2 (the showcase variant).

Same hidden-vitality clinical DGP as assets/exp_owgap_v2/dgp.py -- B1=3.0 and a fixed treatment
burden THETA=1.0 -- except the covariate X is drawn CONTINUOUSLY, uniform on [-1, 1]:

  X  ~ Uniform(-1, 1)                          (observed, CONTINUOUS)
  S in {-1,+1} UNOBSERVED, P(S=+1|X)=sigma(10 X)
  e(X,S)=clip(sigma(0.8 S - 2 X),0.02,0.98)
  mu0(X,S)= 8 S,   mu1(X,S)= 9 S + 3 X - 1
  Y(t)=mu_t(X,S)+N(0,0.6^2)
  CATE(X) = (2 sigma(10X)-1) + 3 X - 1   => treat iff X > x* where x* ~ 0.145 (~43% treated mass,
  matching the discrete variant's 3-of-7 oracle levels).

Why this is the continuous showcase: per-unit (L=inf) policies overfit (each unit is its own
support point, Gamma inert), so the Lipschitz constraint L is the variance dial and Gamma the bias
dial -- the L x Gamma surface shows both cures and their interaction on ONE DGP shared with the
discrete experiment. The burden THETA makes over-treating the confused mid-X region genuinely
costly (CATE(0) = -1), so bias failures are visible in test value, not just in the policy shape.

grid_truth/exact_value/LEVELS are kept ONLY for optional discrete-grid diagnostics; the continuous
runners evaluate on test draws' known potential outcomes Y0/Y1.
"""
import numpy as np

LEVELS = np.round(np.linspace(-1.0, 1.0, 7), 6)   # vestigial (grid diagnostics only)
X_LO, X_HI = -1.0, 1.0
ALPHA, CS, CX, D0, D1, B0, B1, NOISE = 10.0, 0.8, 2.0, 8.0, 9.0, 0.0, 3.0, 0.6
THETA = 1.0        # fixed burden of treatment (subtracted from arm-1 outcomes)
CAP = (1.0, 0.3)   # kept for contract parity; the continuous L x Gamma study runs uncapped
K = 2


def _sig(z): return 1.0 / (1.0 + np.exp(-np.clip(np.asarray(z, float), -40, 40)))
def p_s1(X): return _sig(ALPHA * np.asarray(X, float))
def propensity(X, S): return np.clip(_sig(CS * np.asarray(S, float) - CX * np.asarray(X, float)), 0.02, 0.98)
def mu0(X, S): return D0 * np.asarray(S, float) + B0 * np.asarray(X, float)
def mu1(X, S): return D1 * np.asarray(S, float) + B1 * np.asarray(X, float) - THETA


def grid_truth():
    X = LEVELS; ps1 = p_s1(X); ES = 2 * ps1 - 1.0
    eY0 = D0 * ES + B0 * X; eY1 = D1 * ES + B1 * X - THETA; cate = eY1 - eY0
    return dict(X=X, p_s1=ps1, m0p=mu0(X, 1.0), m0m=mu0(X, -1.0), m1p=mu1(X, 1.0), m1m=mu1(X, -1.0),
                eY0=eY0, eY1=eY1, e_plus=propensity(X, 1.0), e_minus=propensity(X, -1.0),
                e_marg=ps1 * propensity(X, 1.0) + (1 - ps1) * propensity(X, -1.0),
                cate=cate, oracle=(cate > 0).astype(float))


def oracle_policy(X):
    # observable optimal policy: treat iff E[CATE|X] > 0
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
    full = {**observed, "S": S, "Y0": Y0, "Y1": Y1, "e_true": e, "mu0": m0, "mu1": m1}
    return observed, full


def exact_value(pi_grid):
    # discrete-grid diagnostic only (NOT used by the continuous runners).
    pi = np.asarray(pi_grid, float); t = grid_truth()
    return float(np.mean(pi * t["eY1"] + (1 - pi) * t["eY0"]))
