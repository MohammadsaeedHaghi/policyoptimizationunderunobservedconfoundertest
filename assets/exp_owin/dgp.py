"""exp_owin: a MULTIVARIATE DGP built to be the regime where the transport (O-W) term is the
right tool and a partition-based sharp bound is not.

WHY THIS DGP EXISTS -- the argument, stated so a referee can attack it
---------------------------------------------------------------------
The sharp MSM bound and O-W restrict the adversary in ORTHOGONAL, non-nested ways:

  * sharpness = box INTERSECT per-(arm, cell) weight-total equalities. It blocks the adversary
    from concentrating weight on the low-outcome units WITHIN a cell. It says nothing about how
    weight is arranged ACROSS cells.
  * O-W       = box INTERSECT a Wasserstein ball on the reweighted covariate law. It blocks the
    adversary from shifting mass ACROSS the covariate space. It barely restricts reshuffling
    WITHIN a neighbourhood, because nearby points cost almost nothing to transport.

Neither bound dominates. In 1-D with a smooth CATE and 15 well-populated bins, sharpness wins --
that is exactly what the gstar / KMZ / KZ18 campaigns measured, and we report it. The regime
where it must lose is DIMENSION: sharpness needs per-(cell, arm) totals ESTIMATED from data, and
a partition fine enough to track a CATE that varies in several directions needs ~b^d cells. At
d = 4 and b = 6 that is 1296 cells for n = 500 -- most empty, and a cell holding one treated unit
pins W = w_hat exactly, destroying the robustness the bound was for. Coarsening restores the
counts but throws away the sharpness. O-W never partitions: it uses the ground-cost metric
directly, so it degrades gracefully as d grows.

This is a claim about a REGIME, not a rigged instance. The honest framing for the paper is the
crossover -- sharpness dominates in low dimension with populated cells, transport dominates as
dimension grows -- and this DGP is the far end of it.

THE DGP
-------
  X ~ Unif[-1, 1]^D                                    (D = 4)
  S = +-1 hidden,  P(S=+1 | x) = sigma(ALPHA * (x1 + x2) / sqrt(2))     <- X-trackable confounder
  e(x, S) = sigma(-CX' x + (1/2) ln(LAM) * S)          <- NO clipping => S-odds ratio == LAM
                                                          EXACTLY at every x, by construction
  mu0(x, S) = D0 * S + b0' x
  mu1(x, S) = D1 * S + b1' x + AMP * sin(FREQ * x1) * cos(FREQ * x2)
  Y_t = mu_t + N(0, NOISE)

Design choices and what each is for:
  * CX is LARGE (2.0 per active coordinate): the propensity depends strongly on x, so the
    reweighted covariate law is far from the empirical one and the transport constraint carries
    real information. This is the lever that helps O-W.
  * The sin*cos interaction term is the lever that hurts partitioning: the CATE oscillates in a
    2-D subspace, so a cellwise bound needs a fine 2-D grid to track it, while a Lipschitz policy
    represents it with a smoothness constraint instead of parameters.
  * ALPHA = 6 gives strong coupling corr(x, S), the regime our diagnostic says the W-term needs.
  * D1 - D0 = 1 with D0 = 8: the confounder dominates outcome LEVELS, so a naive contrast is
    badly biased -- robustness is genuinely needed, not decorative.
  * LAM = 5 is readable off the propensity: matched Gamma = 5, flagged, nothing tuned.

Contract: same interface as the other DGPs (K, CAP, generate, oracle_policy, p_s1, propensity,
mu0, mu1, cate) so every runner and baseline consumes it unchanged.
"""
import numpy as np

D = 4                                             # covariate dimension
LAM = 5.0                                         # Gamma* -- exact selection odds ratio
CS = 0.5 * np.log(LAM)
CX = np.array([1.0, 1.0, 1.0, 1.0])               # v2: weaker + spread over ALL dims.
#   v1 used [2,2,0,0]: overlap was too extreme (weights blew up) and only 2 dims were active,
#   so k-means cells tracked the CATE easily -- the dimension squeeze never bit.
ALPHA = 6.0                                       # confounder coupling to x
B0 = np.array([1.0, 0.0, 0.5, 0.0])
B1 = B0 + 0.5                                     # v2: CATE slope in EVERY coordinate
D0, D1 = 8.0, 9.0                                 # confounder's outcome levels (D1 - D0 = 1)
AMP, FREQ = 2.5, 3.0                              # the oscillating interaction (hurts partitions)
NOISE = 0.6
THETA = 1.0                                       # treatment burden
K = 2
CAP = (1.0, 0.3)
LEVELS = np.round(np.linspace(-1.0, 1.0, 7), 6)   # vestigial (grid contract only)


def _sig(z): return 1.0 / (1.0 + np.exp(-np.clip(np.asarray(z, float), -40, 40)))


def _as2d(X):
    X = np.asarray(X, float)
    return X.reshape(1, -1) if X.ndim == 1 else X


def p_s1(X):
    X = _as2d(X)
    return _sig(ALPHA * X.sum(axis=1) / 2.0)          # v2: coupling through all coordinates


def propensity(X, S):
    X = _as2d(X)
    return _sig(CS * np.asarray(S, float) - X @ CX)      # NO clip => odds ratio == LAM exactly


def _osc(X):
    """v2: oscillation in TWO orthogonal 2-D planes, so the CATE genuinely varies in all four
    directions. With v1's single plane the effective dimension was 2 and 30 k-means cells tracked
    it comfortably -- which is why the partition squeeze failed to appear."""
    X = _as2d(X)
    return 0.5 * AMP * (np.sin(FREQ * X[:, 0]) * np.cos(FREQ * X[:, 1])
                        + np.sin(FREQ * X[:, 2]) * np.cos(FREQ * X[:, 3]))


def mu0(X, S):
    X = _as2d(X)
    return D0 * np.asarray(S, float) + X @ B0


def mu1(X, S):
    X = _as2d(X)
    return D1 * np.asarray(S, float) + X @ B1 + _osc(X) - THETA


def cate(X):
    """Marginal CATE: E_S[mu1 - mu0 | x], the object the oracle thresholds."""
    X = _as2d(X)
    p = p_s1(X)
    hi = (D1 - D0) * 1.0 + X @ (B1 - B0) + _osc(X) - THETA
    lo = (D1 - D0) * -1.0 + X @ (B1 - B0) + _osc(X) - THETA
    return p * hi + (1.0 - p) * lo


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
    G = np.stack(np.meshgrid(*[LEVELS] * D, indexing="ij"), axis=-1).reshape(-1, D)
    return dict(X=G, cate=cate(G), oracle=oracle_policy(G))
