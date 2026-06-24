"""DGP for the "first prime" experiment = the first experiment (Case 4, discrete-X job-training, K=2) but with the
sigmoid pieces replaced by PIECEWISE-LINEAR functions, per the user's spec:

  X ~ Uniform on a 21-pt grid in [-1,1] (employability score); S ~ Bernoulli(1/2) UNOBSERVED (perseverance), S in {0,1}.
  Outcomes (binary):
    P(Y^0 = 1 | X, S) = min{ 0.5 + 0.231 X + 0.5 S , 1 }          (control; S lifts the baseline)
    P(Y^1 = 1 | X, S) = min{ 1[X>=0] + S , 1 }                    (training: a step at X=0, plus S)
  Propensity (piecewise-linear in X, one line per S):
    pi^1(X, S) = 0.158 + 0.111 X     if S = 0
               = 0.500 + 0.231 X     if S = 1
  Observed Y = Y^0 if T=0 else Y^1,  T ~ Bernoulli(pi^1).

These are the endpoint-linearisations of the original sigmoid DGP: 0.5+0.231X is the line through sigma(X) at
X=+-1; 0.158+0.111X is the line through sigma(X-2); 1[X>=0] is the limit of sigma(X+-5).  The effect is still
NON-MONOTONE and the optimal rule is "treat iff X>=0".  S boosts BOTH arms (positive outcome confounding) and also
raises the propensity (so high-S units are over-represented among the treated): unobserved-confounding by design.
"""
import numpy as np
from types import SimpleNamespace

GRID = np.round(np.linspace(-1.0, 1.0, 21), 6)
n_tr, n_te = 500, 2000
K = 2
CAP_UNCAP = (1.0, 1.0)
CAP_CAP   = (1.0, 0.5)


def pa(X, S, k):
    """P(Y^k = 1 | X, S), piecewise-linear, clipped to [0,1]."""
    X = np.asarray(X, float); S = np.asarray(S, float)
    if k == 0:
        return np.minimum(0.5 + 0.231 * X + 0.5 * S, 1.0)
    return np.minimum((X >= 0.0).astype(float) + S, 1.0)


def propensity(X, S):
    """pi^1(X,S): 0.158+0.111X for S=0, 0.500+0.231X for S=1.  Clipped to [0,1]."""
    X = np.asarray(X, float); S = np.asarray(S, float)
    p = np.where(S >= 0.5, 0.500 + 0.231 * X, 0.158 + 0.111 * X)
    return np.clip(p, 0.0, 1.0)


def mu_arm(X):
    """mu_k(X) = E_S[ P(Y^k=1|X,S) ] averaged over S~Bern(1/2)  ->  (n,2) array."""
    X = np.asarray(X, float)
    return np.column_stack([0.5 * pa(X, 0, k) + 0.5 * pa(X, 1, k) for k in range(K)])


def cate(X):
    """CATE(X) = mu_1(X) - mu_0(X)."""
    m = mu_arm(X); return m[:, 1] - m[:, 0]


def oracle_policy(X):
    """Optimal rule: treat iff CATE(X) > 0  ==  treat iff X >= 0."""
    return (cate(X) > 0).astype(float)


def generate(nn, rng):
    """Sample nn units.  rng is a numpy Generator."""
    X = rng.choice(GRID, size=nn)
    S = (rng.uniform(size=nn) < 0.5).astype(int)
    Yp = np.column_stack([(rng.uniform(size=nn) < pa(X, S, 0)).astype(float),
                          (rng.uniform(size=nn) < pa(X, S, 1)).astype(float)])
    p1 = propensity(X, S); T = (rng.uniform(size=nn) < p1).astype(int)
    mu = mu_arm(X)
    return SimpleNamespace(X=X.reshape(-1, 1), S=S, T=T, Y=Yp[np.arange(nn), T], Ypot=Yp, mu=mu, e_true=p1)


def grid_truth():
    """Ground-truth curves on GRID for the data-generation explanation plots."""
    X = GRID
    return dict(
        X=X,
        # control outcome P(Y0=1|X,S)
        y0_s0=pa(X, 0, 0), y0_s1=pa(X, 1, 0), y0_avg=0.5 * pa(X, 0, 0) + 0.5 * pa(X, 1, 0),
        # treated outcome P(Y1=1|X,S)
        y1_s0=pa(X, 0, 1), y1_s1=pa(X, 1, 1), y1_avg=0.5 * pa(X, 0, 1) + 0.5 * pa(X, 1, 1),
        # propensity
        e_s0=propensity(X, 0), e_s1=propensity(X, 1), e_avg=0.5 * propensity(X, 0) + 0.5 * propensity(X, 1),
        # effect + optimal policy
        cate=cate(X), oracle=oracle_policy(X),
        mu0=mu_arm(X)[:, 0], mu1=mu_arm(X)[:, 1],
    )
