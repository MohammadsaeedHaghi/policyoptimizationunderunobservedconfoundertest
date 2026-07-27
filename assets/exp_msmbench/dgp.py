"""exp_msmbench: the community-standard MSM synthetic DGP (Kallus-Mao-Zhou 2019), as used by
Dorn-Guo and by Hess/Frauen et al. (ICLR 2026, arXiv 2502.13022) -- adapted to our pipeline.

Original form: X ~ Unif[-2,2], U ~ Bern(1/2) INDEPENDENT of X,
  Y(a) = (2a-1)X + (2a-1) - 2 sin(2(2a-1)X) - 2(2U-1)(1 + 0.5X) + N(0,1),
  nominal e(1,x) = sigma(0.75 X + 0.5),
  true propensity = the exact MSM extremal:
      e(1|x,u) = u / rho(x, 1/G*) + (1-u) / rho(x, G*),   rho(x,g) = 1 + (1/e(1,x) - 1) g,
  so the marginal sensitivity model holds EXACTLY with odds ratio G*.

Our adaptation (pure relabeling + protocol):
  - x = X/2 in [-1,1] (pipeline convention; Lipschitz constants refer to x).
  - S = 2U - 1 in {+-1}.
  - G* = 4.95 => matched Gamma = 5, identical protocol to the rest of the paper.
  - CAP = 30% for the capped variant (novel vs the literature's uncapped regret setups).

Structural facts that make this the corr(X,S) = 0 anchor of the diagnostic map:
  - U is INDEPENDENT of X: the observable covariate carries zero information about the
    confounder, so the Wasserstein balance constraint has provably nothing to grab.
  - The MSM is correctly specified with KNOWN Gamma*: box-only methods' home turf.
  - CATE(X) = 2X + 2 - 4 sin(2X): heterogeneous; oracle treats all x except an interior band.
Declared predictions: naive UNDER-treats (treated arm enriched in low-outcome U=1 patients);
O-X at matched Gamma performs well; O-W ~ O-X (no W leverage). Verifying these IS the
experiment: the coupling diagnostic must explain the literature's benchmark too.
"""
import numpy as np

GSTAR = 4.95
LEVELS = np.round(np.linspace(-1.0, 1.0, 7), 6)   # vestigial (grid contract only)
K = 2
CAP = (1.0, 0.3)
NOISE = 1.0


def _sig(z): return 1.0 / (1.0 + np.exp(-np.clip(np.asarray(z, float), -40, 40)))

def e_nom(x):                                     # nominal propensity, in x = X/2 units
    return _sig(0.75 * (2.0 * np.asarray(x, float)) + 0.5)

def _rho(x, g): return 1.0 + (1.0 / e_nom(x) - 1.0) * g

def propensity(x, S):
    u = (np.asarray(S, float) + 1.0) / 2.0
    return u / _rho(x, 1.0 / GSTAR) + (1.0 - u) / _rho(x, GSTAR)

def _mu(x, a, S):
    X = 2.0 * np.asarray(x, float); s = 2.0 * a - 1.0
    return s * X + s - 2.0 * np.sin(2.0 * s * X) - 2.0 * np.asarray(S, float) * (1.0 + 0.5 * X)

def mu0(x, S): return _mu(x, 0, S)
def mu1(x, S): return _mu(x, 1, S)

def cate(x):
    X = 2.0 * np.asarray(x, float)
    return 2.0 * X + 2.0 - 4.0 * np.sin(2.0 * X)   # U-term cancels: CATE does not depend on U

def p_s1(x):                                      # diagnostic contract: U independent of X
    return np.full_like(np.asarray(x, float), 0.5)

def oracle_policy(x):
    return (cate(x) > 0).astype(float)


def generate(n, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1.0, 1.0, size=n)
    S = np.where(rng.uniform(size=n) < 0.5, 1.0, -1.0)
    e = propensity(x, S)
    T = (rng.uniform(size=n) < e).astype(int)
    Y0 = mu0(x, S) + rng.normal(0, NOISE, size=n)
    Y1 = mu1(x, S) + rng.normal(0, NOISE, size=n)
    Y = np.where(T == 1, Y1, Y0)
    observed = {"X": x.reshape(-1, 1), "T": T, "Y": Y}
    full = {**observed, "S": S, "Y1": Y1, "Y0": Y0, "e_true": e}
    return observed, full


def grid_truth():
    X = LEVELS
    return dict(X=X, cate=cate(X), oracle=oracle_policy(X))
