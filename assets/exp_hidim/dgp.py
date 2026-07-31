"""exp_hidim: a HIGH-DIMENSIONAL DGP designed against the CELL-ORACLE CEILING.

THE DESIGN PRINCIPLE (this is what the previous two attempts got wrong)
----------------------------------------------------------------------
A partition-based bound like Sharp-O-X can never beat the best policy that is CONSTANT WITHIN
EACH CELL. Call that the cell-oracle ceiling. It is a property of the DGP and the partition
alone -- no estimation involved -- so it can be computed exactly and DESIGNED AGAINST, which is
what this file does.

In 1-D with 15 quantile bins the ceiling is essentially the true oracle: bin width is ~7% of the
range, so a cell-constant policy loses nothing. That is why sharp wins every 1-D benchmark, and
why exp_owin v1/v2 failed -- their CATE lived in a 2-D subspace, so k-means cells tracked it.

In d dimensions with K cells, cells have diameter ~ (1/K)^(1/d) of the range. MEASURED ceiling
(assets/exp_hidim/design_check.py), oracle minus best cell-constant policy:
    d=1, K=15 -> +0.004   (sharp is unbeatable in 1-D BY CONSTRUCTION -- this is why five
                           separate attempts to beat it on 1-D benchmarks all failed)
    d=4, K=15 -> +0.146
    d=6, K=15 -> +0.209
    d=8, K=15 -> +0.241,  K=30 -> +0.215,  K=60 -> +0.165 (and n=200 leaves ~3 units/cell there)
Chosen: D=8, LAM=10 -> oracle 1.042, cell-oracle ceiling 0.664, infinite-data naive 0.804,
never-treat 0.001, signal/noise 0.86, overlap e in [0.063, 0.937].
So the gap opens through DIMENSION, not through wiggliness. Crucially the CATE here stays SIMPLE
(linear plus one sigmoid ridge) so the problem remains learnable at n=200 -- the previous
attempts drowned every method below never-treat by making the target hard AND noisy at once.

WHAT EACH PIECE IS FOR
  * CATE is nearly LINEAR in x -> the oracle policy is a smooth halfspace. Cheap to learn with a
    metric/Lipschitz policy; badly approximated by isotropic cell blobs in high d.
  * The confounder enters the TREATED ARM ONLY (delta * S), not as a level shift in both arms.
    exp_owin used mu0 = 8S, which put the confounder's variance into BOTH outcomes and crushed
    signal/noise to 0.18. Here the control arm is clean, so S/N stays ~0.6.
  * P(S=+1 | x) = sigma(ALPHA * v'x) with v NOT aligned with the CATE direction a. The confounding
    bias is therefore a NONLINEAR ridge in a different direction than the CATE, so a linear naive
    cannot absorb it (the failure mode that made gstar's capped rows unwinnable).
  * Propensity sigma(-c'x + (1/2)ln(LAM) S), NO clipping -> S-odds ratio == LAM exactly at every
    x, so matched Gamma = LAM = 5 is readable off the DGP and nothing is tuned.

Contract matches every other DGP (K, CAP, generate, oracle_policy, p_s1, propensity, mu0, mu1,
cate) so all runners and baselines consume it unchanged.
"""
import numpy as np

D = 8                                              # covariate dimension (see design_check.py)
LAM = 10.0                                         # Gamma* -- exact selection odds ratio
CS = 0.5 * np.log(LAM)

_e = np.ones(D) / np.sqrt(D)
A_CATE = 1.2 * _e                                  # CATE direction (all coordinates active)
V_CONF = np.array([(-1.0) ** j for j in range(D)]) / np.sqrt(D)   # confounder dir, orthogonal to a
C_PROP = 0.8 * _e                                  # propensity's x-dependence (overlap-friendly)
B0 = 0.5 * np.array([(-1.0) ** (j // 2) for j in range(D)])       # baseline outcome slope
ALPHA = 3.0                                        # coupling strength (S depends on x)
DELTA = 4.0                                        # confounder's effect, TREATED ARM ONLY
THETA = 0.0                                        # treatment burden (0 => boundary near centre)
NOISE = 0.5                                        # small: keeps signal/noise ~0.6
K = 2
CAP = (1.0, 0.3)
LEVELS = np.round(np.linspace(-1.0, 1.0, 7), 6)    # vestigial (grid contract only)


def _sig(z): return 1.0 / (1.0 + np.exp(-np.clip(np.asarray(z, float), -40, 40)))


def _as2d(X):
    X = np.asarray(X, float)
    return X.reshape(1, -1) if X.ndim == 1 else X


def p_s1(X):
    """P(S = +1 | x): a sigmoid ridge along V_CONF, deliberately not aligned with the CATE."""
    return _sig(ALPHA * (_as2d(X) @ V_CONF))


def propensity(X, S):
    return _sig(CS * np.asarray(S, float) - (_as2d(X) @ C_PROP))     # no clip => odds ratio == LAM


def mu0(X, S):
    return _as2d(X) @ B0                                              # control arm is CLEAN of S


def mu1(X, S):
    return _as2d(X) @ B0 + _as2d(X) @ A_CATE + THETA + DELTA * np.asarray(S, float)


def cate(X):
    """E_S[mu1 - mu0 | x] = a'x + theta + delta * (2 P(S=1|x) - 1)."""
    X = _as2d(X)
    return X @ A_CATE + THETA + DELTA * (2.0 * p_s1(X) - 1.0)


def oracle_policy(X):
    return (cate(X) > 0).astype(float)


def generate(n, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1.0, 1.0, size=(n, D))
    S = np.where(rng.uniform(size=n) < p_s1(X), 1.0, -1.0)
    e = propensity(X, S)
    T = (rng.uniform(size=n) < e).astype(int)
    Y0 = mu0(X, S) + rng.normal(0, NOISE, size=n)
    Y1 = mu1(X, S) + rng.normal(0, NOISE, size=n)
    Y = np.where(T == 1, Y1, Y0)
    observed = {"X": X, "T": T, "Y": Y}
    full = {**observed, "S": S, "Y1": Y1, "Y0": Y0, "e_true": e}
    return observed, full


def grid_truth():
    rng = np.random.default_rng(0)
    G = rng.uniform(-1.0, 1.0, size=(4096, D))
    return dict(X=G, cate=cate(G), oracle=oracle_policy(G))
