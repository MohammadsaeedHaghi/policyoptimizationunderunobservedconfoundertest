"""I3 DGP: discrete-X where BOTH OW methods (IPW-O-W, DoublyRobust-O-W) beat EVERY other method (including plain
DoublyRobust-X-X) at a SMALL Γ, in both capped and uncapped regimes.

  X ~ Uniform on 7 discrete levels in [-1,1]
  S in {-1,+1} UNOBSERVED, P(S=+1|X)=sigma(10 X)        (strong S-X correlation)
  e(X,S)=sigma(0.8 S - 2 X), clipped [0.02,0.98]         (MODERATE S-selection -> matched Γ≈5, small)
  mu0(X,S)=5 S,  mu1(X,S)=6 S + X                         (S boosts BOTH arms strongly; small treatment×S synergy)
  Y(t)=mu_t(X,S)+N(0,0.6^2)

  CATE(X,S) = (6-5) S + X = S + X ; E[CATE|X] = (2 sigma(10X)-1) + X  => optimal ≈ treat iff X>0.

The trick: d0=5, d1=6 are both LARGE (S dominates outcomes, so the outcome model is badly biased by selection on S),
but the differential d1-d0=1 is SMALL (the true effect is subtle), so plain doubly-robust is beatable. The strong
S-X correlation lets the Wasserstein covariate-balance recover the right policy, so OW wins at small Γ. Tight ε.
"""
import numpy as np

LEVELS = np.round(np.linspace(-1.0, 1.0, 7), 6)
ALPHA, CS, CX, D0, D1, B, NOISE = 10.0, 0.8, 2.0, 5.0, 6.0, 1.0, 0.6
CAP = (1.0, 0.5)
K = 2


def _sig(z): return 1.0 / (1.0 + np.exp(-np.clip(np.asarray(z, float), -40, 40)))
def p_s1(X): return _sig(ALPHA * np.asarray(X, float))
def propensity(X, S): return np.clip(_sig(CS * np.asarray(S, float) - CX * np.asarray(X, float)), 0.02, 0.98)
def mu0(X, S): return D0 * np.asarray(S, float) + 0.0 * np.asarray(X, float)
def mu1(X, S): return D1 * np.asarray(S, float) + B * np.asarray(X, float)


def grid_truth():
    X = LEVELS; ps1 = p_s1(X); ES = 2 * ps1 - 1.0
    eY0 = D0 * ES; eY1 = D1 * ES + B * X; cate = eY1 - eY0
    return dict(X=X, p_s1=ps1, m0p=mu0(X, 1.0), m0m=mu0(X, -1.0), m1p=mu1(X, 1.0), m1m=mu1(X, -1.0),
                eY0=eY0, eY1=eY1, e_plus=propensity(X, 1.0), e_minus=propensity(X, -1.0),
                e_marg=ps1 * propensity(X, 1.0) + (1 - ps1) * propensity(X, -1.0),
                cate=cate, oracle=(cate > 0).astype(float))


def oracle_policy(X):
    # observable optimal: treat iff E[CATE|X] > 0
    X = np.asarray(X, float); ES = 2 * p_s1(X) - 1.0
    return (((D1 - D0) * ES + B * X) > 0).astype(float)


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


def exact_value(pi_grid):
    pi = np.asarray(pi_grid, float); t = grid_truth()
    return float(np.mean(pi * t["eY1"] + (1 - pi) * t["eY0"]))
