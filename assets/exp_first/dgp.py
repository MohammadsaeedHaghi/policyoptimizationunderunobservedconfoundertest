"""DGP for the "first experiment" = Case 4 (case 1, but discrete-X). Job-training story, binary treatment (K=2):
  arm 0 = no training (control), arm 1 = training.
  X ~ Uniform on a 21-pt grid in [-1,1] (employability score); S ~ Bernoulli(1/2) UNOBSERVED (perseverance).
  Outcomes (binary):
    P(Y^0=1|X,S) = min{ σ(X) + S/2 , 1 }
    P(Y^1=1|X,S) = min{ σ(X+5) + S , 1 }   if X>=0   (training HELPS high X: score shifted up)
                 = min{ σ(X-5) + S , 1 }   if X< 0   (training HURTS low  X: score shifted down)
  => NON-MONOTONE effect; the optimal rule is "treat iff X>=0".  S boosts BOTH arms (positive outcome confounding).
  Assignment:  π^1(X,S) = clip( σ(X + γ·S - 2), 0.05, 0.95 );  T ~ Bernoulli(π^1).  γ = confounding strength.
  Observed Y = Y^0 if T=0 else Y^1.

GAMMA_CONF (γ) chosen by screen so (a) R-OW > IPW and (b) X plays a real role in the propensity (not S-dominated).
matched Γ = e^{γ/2}.  Two deployment regimes reported: UNCAPPED cap=(1,1) and CAPPED cap=(1,0.5) (treat <=50%)."""
import numpy as np
from types import SimpleNamespace

GAMMA_CONF = 2.0                                   # γ chosen by screen: R-OW>IPW in BOTH regimes + X plays a real role (spread 0.46); γ≥2.5 fails (R-OW treats-all, loses to IPW uncapped)
GRID = np.round(np.linspace(-1, 1, 21), 6)
n_tr, n_te = 500, 2000
K = 2
CAP_UNCAP = (1.0, 1.0)                              # arm0 control uncapped, arm1 unconstrained
CAP_CAP   = (1.0, 0.5)                              # arm0 control uncapped, arm1 treat <= 50%

def sig(z):
    z = np.asarray(z, float); return 1.0 / (1.0 + np.exp(-np.clip(z, -40, 40)))

def pa(X, S, k):
    """P(Y^k = 1 | X, S), faithful to the spec (min{.,1})."""
    X = np.asarray(X, float); S = np.asarray(S, float)
    if k == 0:
        return np.minimum(sig(X) + S / 2.0, 1.0)
    shift = np.where(X >= 0.0, 5.0, -5.0)           # +5 for X>=0, -5 for X<0
    return np.minimum(sig(X + shift) + S, 1.0)

def propensity(X, S, gamma=None):
    """π^1(X,S) = clip( σ(X + γS - 2), 0.05, 0.95 )."""
    g = GAMMA_CONF if gamma is None else gamma
    return np.clip(sig(np.asarray(X, float) + g * np.asarray(S, float) - 2.0), 0.05, 0.95)

def mu_arm(X):
    """μ_k(X) = E_S[ P(Y^k=1|X,S) ] averaged over S~Bern(1/2)."""
    X = np.asarray(X, float)
    return np.column_stack([0.5 * pa(X, 0, k) + 0.5 * pa(X, 1, k) for k in range(K)])

def generate(nn, rng, gamma=None):
    X = rng.choice(GRID, size=nn)
    S = (rng.uniform(size=nn) < 0.5).astype(int)
    Yp = np.column_stack([(rng.uniform(size=nn) < pa(X, S, 0)).astype(float),
                          (rng.uniform(size=nn) < pa(X, S, 1)).astype(float)])
    p1 = propensity(X, S, gamma); T = (rng.uniform(size=nn) < p1).astype(int)
    mu = np.column_stack([0.5 * pa(X, 0, 0) + 0.5 * pa(X, 1, 0),
                          0.5 * pa(X, 0, 1) + 0.5 * pa(X, 1, 1)])
    return SimpleNamespace(X=X.reshape(-1, 1), S=S, T=T, Y=Yp[np.arange(nn), T], Ypot=Yp, mu=mu)

# Γ sweeps keyed by γ (bracket the matched Γ=e^{γ/2})
def gammas_for(gamma):
    mg = round(float(np.exp(gamma / 2.0)), 4)
    base = sorted(set([1.0, 1.5, 2.0, 3.0, mg, round(mg * 1.6, 2), round(mg * 2.6, 2)]))
    return base, base.index(mg)
