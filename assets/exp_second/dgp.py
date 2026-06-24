"""DGP for the 'second experiment' = 2-arm, 2-D discrete-X (the "2-arm 2-D, R-OW wins" setting), presented in the
exp_first structure. Binary treatment (arm0 = control, arm1 = treatment), 2-D covariates X=(X1,X2), unobserved S.

  X=(X1,X2) ~ Unif on the 11x11 grid in [-1,1]^2 ;  S ~ Bernoulli(1/2) UNOBSERVED.
  proj(X) = X1 + 0.6 X2.
  P(Y^0=1|X,S) = σ(0.5)               (control: FLAT, clean — no X, no S)
  P(Y^1=1|X,S) = σ(-1.2 + 4·proj + 5·S)   (treatment: X-benefit + S-inflated)
  logit P(T=1|X,S) = -proj + γ(S-1/2)     (MIS-targeted: treats where proj is LOW; + S channel)
  Y = Y^T.  γ=5 → matched Γ=e^{γ/2}=e^{2.5}≈12.18.  Optimal: treat where μ1>μ0=σ(0.5) (≈ treat proj>0.04, ~47% of cells).

The Wasserstein covariate-balance term has 2-D imbalance to correct, and the loose cap (treat≤50%) lets it bind ⇒ R-OW wins.
Two regimes reported (exp_first style): UNCAPPED cap=(1,1) and CAPPED cap=(1,0.5). n_train=500, n_test=2000, 5 seeds."""
import numpy as np
from types import SimpleNamespace

GAMMA_CONF = 5.0
AX = np.round(np.linspace(-1, 1, 11), 6)                 # 11-pt axis
_GX1, _GX2 = np.meshgrid(AX, AX)                          # 11x11
GRID = np.column_stack([_GX1.ravel(), _GX2.ravel()])     # (121, 2) row-major: X2 outer, X1 inner
n_tr, n_te = 500, 2000
K = 2
CAP_UNCAP = (1.0, 1.0)
CAP_CAP = (1.0, 0.5)
GAMMAS = [1.0, 3.0, 6.0, 9.0, round(float(np.exp(GAMMA_CONF / 2)), 4), 16.0]   # {1,3,6,9,12.18,16}
mi = 4                                                    # matched-Γ index

def sig(z):
    z = np.asarray(z, float); return 1.0 / (1.0 + np.exp(-np.clip(z, -40, 40)))

def proj(X):
    X = np.atleast_2d(np.asarray(X, float)); return X[:, 0] + 0.6 * X[:, 1]

def pa(X, S, k):
    """P(Y^k=1 | X, S)."""
    X = np.atleast_2d(np.asarray(X, float)); S = np.asarray(S, float)
    if k == 0:
        return np.full(X.shape[0], sig(0.5))             # flat clean control
    return sig(-1.2 + 4.0 * proj(X) + 5.0 * S)

def propensity(X, S, gamma=None):
    g = GAMMA_CONF if gamma is None else gamma
    return sig(-proj(X) + g * (np.asarray(S, float) - 0.5))

def mu_arm(X):
    """μ_k(X) = E_S[ P(Y^k=1|X,S) ] over S~Bern(1/2)."""
    X = np.atleast_2d(np.asarray(X, float))
    return np.column_stack([0.5 * pa(X, 0, k) + 0.5 * pa(X, 1, k) for k in range(K)])

def generate(nn, rng, gamma=None):
    idx = rng.integers(0, len(GRID), size=nn)            # uniform over the 121 grid points
    X = GRID[idx]
    S = (rng.uniform(size=nn) < 0.5).astype(int)
    Yp = np.column_stack([(rng.uniform(size=nn) < pa(X, S, 0)).astype(float),
                          (rng.uniform(size=nn) < pa(X, S, 1)).astype(float)])
    p1 = propensity(X, S, gamma); T = (rng.uniform(size=nn) < p1).astype(int)
    return SimpleNamespace(X=X, S=S, T=T, Y=Yp[np.arange(nn), T], Ypot=Yp, mu=mu_arm(X))
