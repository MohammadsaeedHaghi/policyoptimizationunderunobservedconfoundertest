"""E3 DGP: discrete-X where XX<OX<OW AND the robust methods collapse at large Γ, WITHOUT needing a capacity cap
(works both capped and uncapped). Strong, steep confounding makes naive XX over-treat the harmful region on its own.

  X ~ Uniform on 7 discrete levels in [-1,1]
  S in {-1,+1} UNOBSERVED, P(S=+1|X)=sigma(7X)        (strong S-X correlation -> Wasserstein reaches S)
  e(X,S)=sigma(4S - 2X), clipped [0.02,0.98]           (STRONG selection on S = large confounding bias)
  mu0(X,S)=5 S,  mu1(X,S)=5 S + 3 X                     => CATE(X)=3X, oracle = treat iff X>0
  Y(t)=mu_t(X,S)+N(0,0.9^2)

Methods FIT a logistic P(T|X) (never the true e). Oracle approx 0.857.
"""
import numpy as np

LEVELS = np.round(np.linspace(-1.0, 1.0, 7), 6)
ALPHA, CS, CX, D, B, NOISE = 7.0, 4.0, 2.0, 5.0, 3.0, 0.9
CAP = (1.0, 0.5)
K = 2


def _sig(z): return 1.0 / (1.0 + np.exp(-np.clip(np.asarray(z, float), -40, 40)))
def p_s1(X): return _sig(ALPHA * np.asarray(X, float))
def propensity(X, S): return np.clip(_sig(CS * np.asarray(S, float) - CX * np.asarray(X, float)), 0.02, 0.98)
def mu0(X, S): return D * np.asarray(S, float) + 0.0 * np.asarray(X, float)
def mu1(X, S): return D * np.asarray(S, float) + B * np.asarray(X, float)
def cate(X): return B * np.asarray(X, float)
def oracle_policy(X): return (np.asarray(X, float) > 0).astype(float)


def generate(n, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.choice(LEVELS, size=n)
    S = np.where(rng.uniform(size=n) < p_s1(X), 1.0, -1.0)
    e = propensity(X, S); T = (rng.uniform(size=n) < e).astype(int)
    m0, m1 = mu0(X, S), mu1(X, S)
    Y0 = m0 + rng.normal(0, NOISE, size=n); Y1 = m1 + rng.normal(0, NOISE, size=n)
    Y = np.where(T == 1, Y1, Y0)
    observed = {"X": X.reshape(-1, 1), "T": T, "Y": Y}
    full = {**observed, "S": S, "Y0": Y0, "Y1": Y1, "e_true": e, "mu0": m0, "mu1": m1}
    return observed, full


def grid_truth():
    X = LEVELS; ps1 = p_s1(X); ES = 2 * ps1 - 1.0
    return dict(X=X, p_s1=ps1, m0p=mu0(X, 1.0), m0m=mu0(X, -1.0), m1p=mu1(X, 1.0), m1m=mu1(X, -1.0),
                eY0=D * ES, eY1=D * ES + B * X, e_plus=propensity(X, 1.0), e_minus=propensity(X, -1.0),
                e_marg=ps1 * propensity(X, 1.0) + (1 - ps1) * propensity(X, -1.0), cate=cate(X), oracle=oracle_policy(X))


def exact_value(pi_grid):
    pi = np.asarray(pi_grid, float); t = grid_truth()
    return float(np.mean(pi * t["eY1"] + (1 - pi) * t["eY0"]))
