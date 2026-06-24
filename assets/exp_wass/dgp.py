"""Continuous-X "Wasserstein wins" DGP — the regime where the covariate-balance term genuinely helps.

X is CONTINUOUS (so exact per-cell balancing is impossible → the tight Wasserstein radius ε>0), and the
unobserved confounder S is CORRELATED with X, so balancing the covariate X also balances the hidden S.
Assignment is mis-targeted (treats low X) and S-confounded, creating real covariate imbalance between arms.

  X ~ Unif(-1,1)                          (continuous covariate; "seniority")
  S | X : P(S=+1|X) = σ(4X)               (UNOBSERVED; degree-holders concentrate at high X)   s := 2S-1∈{-1,+1}... here S∈{-1,+1}
  e(X,S) = σ(S - 1.8 X)                    (propensity: treats low X, and S=+1 more) → mis-targeted + S-confounded
  μ0(X,S) = S                              (control mean; S inflates the baseline = confounding)
  μ1(X,S) = S + X                          (treated mean) ⇒ CATE(X,S) = X  (independent of S)
  Y(t) = μ_t(X,S) + N(0, 0.4²)

True policy: CATE(X) = X ⇒ treat iff X>0.  The logistic P(T|X) propensity is mis-specified (the marginal is a
mixture over S, not logit-linear), so ε>0 and IPW-O-W / DoublyRobust-O-W beat their box-only counterparts.
"""
import numpy as np

NOISE = 0.4
GRID = np.round(np.linspace(-1.0, 1.0, 41), 4)     # fine grid for ground-truth curves + policy display


def _sig(z):
    return 1.0 / (1.0 + np.exp(-np.asarray(z, float)))


def p_s1(X):
    """P(S=+1 | X) = σ(4X) — the unobserved confounder correlates with the covariate."""
    return _sig(4.0 * np.asarray(X, float))


def propensity(X, S):
    """e(X,S) = σ(S - 1.8 X)  (S ∈ {-1,+1})."""
    return _sig(np.asarray(S, float) - 1.8 * np.asarray(X, float))


def mu0(X, S): return np.asarray(S, float) + 0.0 * np.asarray(X, float)      # control mean (S inflates baseline)
def mu1(X, S): return np.asarray(S, float) + np.asarray(X, float)            # treated mean ⇒ CATE = X


def oracle_policy(X): return (np.asarray(X, float) > 0).astype(float)        # treat iff CATE=X > 0
def cate(X): return np.asarray(X, float)                                     # marginal CATE(X) = X


def generate(n, seed=0):
    """Sample n units. Returns (observed{X,T,Y}, full{...,S,Y0,Y1,e_true,mu0,mu1})."""
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1.0, 1.0, size=n)
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
    """Ground-truth curves on GRID (S marginalised via P(S=+1|X)=σ(4X))."""
    X = GRID
    ps1 = p_s1(X)
    e_plus, e_minus = propensity(X, 1.0), propensity(X, -1.0)
    e_marg = ps1 * e_plus + (1 - ps1) * e_minus                  # marginal ẽ(X) — VARIES with X
    m0_plus, m0_minus = mu0(X, 1.0), mu0(X, -1.0)
    m1_plus, m1_minus = mu1(X, 1.0), mu1(X, -1.0)
    eY0 = ps1 * m0_plus + (1 - ps1) * m0_minus                   # E[Y(0)|X]
    eY1 = ps1 * m1_plus + (1 - ps1) * m1_minus                   # E[Y(1)|X]
    return dict(X=X, p_s1=ps1, e_plus=e_plus, e_minus=e_minus, e_marg=e_marg,
                m0_plus=m0_plus, m0_minus=m0_minus, m1_plus=m1_plus, m1_minus=m1_minus,
                eY0=eY0, eY1=eY1, cate=cate(X), oracle=oracle_policy(X))


def realized_value_means(pi_grid, Xtest, Stest):
    """Realised E[Y] of a per-X policy deployed on a test set, using the TRUE means μ_t (low-variance)."""
    pi = np.asarray(pi_grid, float)
    m0 = mu0(Xtest, Stest); m1 = mu1(Xtest, Stest)
    return float(np.mean(pi * m1 + (1 - pi) * m0))
