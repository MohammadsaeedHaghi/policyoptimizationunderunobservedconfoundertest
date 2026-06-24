"""Discrete-X version of the continuous "Wasserstein wins" DGP.

Same structural recipe as assets/exp_wass/dgp.py, but X lives on a FINITE grid of levels
(LEVELS) instead of a continuum. Everything else is identical:

  X ~ Uniform over LEVELS = {-1, -0.75, ..., 1}   (9 discrete levels; "seniority bands")
  S | X : P(S=+1|X) = sigma(4X)                    (UNOBSERVED confounder, correlated with X)
  e(X,S) = sigma(S - 1.8 X)                         (propensity: treats low X, S=+1 more)
  mu0(X,S) = S                                      (control mean; S inflates baseline = confounding)
  mu1(X,S) = S + X                                  (treated mean) => CATE(X,S) = X
  Y(t) = mu_t(X,S) + N(0, 0.4^2)

The point of this discrete port: even with discrete X the tight Wasserstein radius is > 0 when the
propensity is a (mis-specified) parametric/logistic model, because the IPW-reweighted covariate law
does NOT exactly match the empirical law. epsilon = 0 only happens with the exact per-cell counting
propensity (the HR setup). So this isolates the cause: ε>0 comes from a mis-specified propensity, not
from X being continuous.

True policy: CATE(X) = X => treat iff X > 0.
"""
import numpy as np

NOISE = 0.4
LEVELS = np.round(np.linspace(-1.0, 1.0, 9), 4)    # 9 discrete X levels
GRID = LEVELS                                       # policy display grid = the support itself


def _sig(z):
    return 1.0 / (1.0 + np.exp(-np.asarray(z, float)))


def p_s1(X):
    """P(S=+1 | X) = sigma(4X)."""
    return _sig(4.0 * np.asarray(X, float))


def propensity(X, S):
    """e(X,S) = sigma(S - 1.8 X)  (S in {-1,+1})."""
    return _sig(np.asarray(S, float) - 1.8 * np.asarray(X, float))


def mu0(X, S): return np.asarray(S, float) + 0.0 * np.asarray(X, float)
def mu1(X, S): return np.asarray(S, float) + np.asarray(X, float)


def oracle_policy(X): return (np.asarray(X, float) > 0).astype(float)
def cate(X): return np.asarray(X, float)


def generate(n, seed=0):
    """Sample n units with X drawn uniformly from the discrete LEVELS."""
    rng = np.random.default_rng(seed)
    X = rng.choice(LEVELS, size=n)
    S = np.where(rng.uniform(size=n) < p_s1(X), 1.0, -1.0)
    e = propensity(X, S)
    T = (rng.uniform(size=n) < e).astype(int)
    m0, m1 = mu0(X, S), mu1(X, S)
    Y0 = m0 + rng.normal(0.0, NOISE, size=n)
    Y1 = m1 + rng.normal(0.0, NOISE, size=n)
    Y = np.where(T == 1, Y1, Y0)
    observed = {"X": X.reshape(-1, 1), "T": T, "Y": Y}
    full = {**observed, "S": S, "Y0": Y0, "Y1": Y1, "e_true": e, "mu0": m0, "mu1": m1}
    return observed, full


def grid_truth():
    """Ground-truth curves on the discrete LEVELS (S marginalised via P(S=+1|X)=sigma(4X))."""
    X = GRID
    ps1 = p_s1(X)
    e_plus, e_minus = propensity(X, 1.0), propensity(X, -1.0)
    e_marg = ps1 * e_plus + (1 - ps1) * e_minus
    eY0 = ps1 * mu0(X, 1.0) + (1 - ps1) * mu0(X, -1.0)
    eY1 = ps1 * mu1(X, 1.0) + (1 - ps1) * mu1(X, -1.0)
    return dict(X=X, p_s1=ps1, e_plus=e_plus, e_minus=e_minus, e_marg=e_marg,
                eY0=eY0, eY1=eY1, cate=cate(X), oracle=oracle_policy(X))


def realized_value_means(pi_grid, Xtest, Stest):
    """Realised E[Y] of a per-X policy deployed on a test set, using the TRUE means mu_t."""
    pi = np.asarray(pi_grid, float)
    return float(np.mean(pi * mu1(Xtest, Stest) + (1 - pi) * mu0(Xtest, Stest)))
