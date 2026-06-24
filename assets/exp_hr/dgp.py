"""HR leadership-training DGP (two variants) — see the user's spec.

X = job seniority on a 21-pt grid in [-1,1]; S = college degree (±1, UNOBSERVED); T = enrolled in training;
Y = promoted (±1). Training helps degree-holders when junior (X<0) and non-degree-holders when senior (X>0):
the marginal CATE crosses 0 at X=0, so the TRUE policy is "train iff X>0".

Two DGPs share the outcome model; they differ only in the propensity (hence the confounding strength Γ):
  • "uniform"   — e(X,S=+1)=0.75, e(X,S=-1)=0.25  → constant Γ_true = 3 everywhere ("Stale HR Rule").
  • "x_varying" — e decreases/increases with X so HR is strict on juniors, loose on seniors → Γ_local from
    ~12 at X=-1 down to ~1.35 at X=+1 ("Seniority-Adaptive HR Rule"). Marginal ẽ(X)=0.5 in BOTH.
"""
import numpy as np

X_GRID = np.round(np.linspace(-1.0, 1.0, 21), 2)


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.asarray(z, float)))


# ---- outcome model (promotion probability P(Y(t)=+1 | X,S)); shared by both DGPs ----
def p_y0(X, S):
    X = np.asarray(X, float); S = np.asarray(S, float)
    return np.where(S == +1, 0.4, 0.6 - 0.2 * X)


def p_y1(X, S):
    X = np.asarray(X, float); S = np.asarray(S, float)
    p_plus = np.where(X <= 0, 0.95 + 0.05 * X, 0.05 + 0.05 * X)
    p_minus = np.where(X <= 0, 0.05, np.minimum(0.95 + 0.05 * X, 1.0))
    return np.where(S == +1, p_plus, p_minus)


# ---- propensity e(X,S) per DGP ----
def propensity(X, S, dgp):
    X = np.asarray(X, float); S = np.asarray(S, float)
    if dgp == "uniform":
        return np.where(S == +1, 0.75, 0.25)
    if dgp == "x_varying":
        # logistic: e(X,S=+1)=σ(+γ(X)), e(X,S=-1)=σ(-γ(X)) with γ(X)=1.40-1.10X  (so e+ + e- = 1, marginal ẽ=0.5)
        g = 1.40 - 1.10 * X
        return np.where(S == +1, _sigmoid(g), _sigmoid(-g))
    raise ValueError(dgp)


def gamma_local(X, dgp):
    """Per-X true Tan-box sensitivity parameter Γ_local(X)."""
    X = np.asarray(X, float)
    if dgp == "uniform":
        return np.full_like(X, 3.0)
    if dgp == "x_varying":
        g = 1.40 - 1.10 * X
        ep, em = _sigmoid(g), _sigmoid(-g)
        return np.maximum(ep / (1 - ep), (1 - em) / em)      # = exp(γ(X)) = exp(1.40-1.10X)
    raise ValueError(dgp)


def oracle_policy(X):
    """Train iff true CATE > 0, i.e. X > 0 (enroll seniors)."""
    return (np.asarray(X, float) > 0).astype(float)


def generate_data(n, seed=0, dgp="uniform"):
    """Sample n units. Returns (observed{X,T,Y}, full{...,S,Y0,Y1,e_true}). Y,S,Y0,Y1 ∈ {-1,+1}."""
    rng = np.random.default_rng(seed)
    X = X_GRID[rng.integers(0, 21, size=n)]
    S = 2 * rng.integers(0, 2, size=n) - 1
    e_true = propensity(X, S, dgp)
    T = (rng.uniform(0, 1, size=n) < e_true).astype(int)
    P0 = p_y0(X, S); P1 = p_y1(X, S)
    Y0 = 2 * (rng.uniform(0, 1, size=n) < P0).astype(int) - 1
    Y1 = 2 * (rng.uniform(0, 1, size=n) < P1).astype(int) - 1
    Y = np.where(T == 1, Y1, Y0)
    observed = {"X": X, "T": T, "Y": Y}
    full = {**observed, "S": S, "Y0": Y0, "Y1": Y1, "e_true": e_true}
    return observed, full


# ---- ground-truth E[Y(t)|X] marginalised over S~Unif{±1}; CATE; population "what HR sees" ----
def grid_truth(dgp):
    """All ground-truth curves on X_GRID (S marginalised 50/50). E[Y]=2p-1 ∈ [-1,1]."""
    X = X_GRID
    pp1, pp0 = p_y1(X, +1), p_y0(X, +1)         # S=+1
    pm1, pm0 = p_y1(X, -1), p_y0(X, -1)         # S=-1
    pp1, pp0, pm1, pm0 = [np.broadcast_to(a, X.shape).astype(float) for a in (pp1, pp0, pm1, pm0)]
    # E[Y(t)|X] marginal over S
    eY1 = 0.5 * ((2 * pp1 - 1) + (2 * pm1 - 1))
    eY0 = 0.5 * ((2 * pp0 - 1) + (2 * pm0 - 1))
    cate = eY1 - eY0                              # marginal CATE
    cate_splus = (2 * pp1 - 1) - (2 * pp0 - 1)
    cate_sminus = (2 * pm1 - 1) - (2 * pm0 - 1)
    # propensity per S + selection P(S=+1 | X, T)
    eplus = np.broadcast_to(propensity(X, +1, dgp), X.shape).astype(float)
    eminus = np.broadcast_to(propensity(X, -1, dgp), X.shape).astype(float)
    pS1_T1 = eplus / (eplus + eminus)            # 0.5*e+/(0.5*e+ + 0.5*e-)
    pS1_T0 = (1 - eplus) / ((1 - eplus) + (1 - eminus))
    # population "what HR sees": E[Y|X,T] = sum_S P(S|X,T) E[Y(T)|X,S]
    obsY_T1 = pS1_T1 * (2 * pp1 - 1) + (1 - pS1_T1) * (2 * pm1 - 1)
    obsY_T0 = pS1_T0 * (2 * pp0 - 1) + (1 - pS1_T0) * (2 * pm0 - 1)
    naive = obsY_T1 - obsY_T0                     # confounded "observed effect"
    return dict(X=X, p_y1_splus=pp1, p_y1_sminus=pm1, p_y0_splus=pp0, p_y0_sminus=pm0,
                eY1=eY1, eY0=eY0, cate=cate, cate_splus=cate_splus, cate_sminus=cate_sminus,
                eplus=eplus, eminus=eminus, gamma=gamma_local(X, dgp),
                pS1_T1=pS1_T1, pS1_T0=pS1_T0, obsY_T1=obsY_T1, obsY_T0=obsY_T0, naive=naive,
                oracle=oracle_policy(X))


def realized_value(pi_grid, dgp=None):
    """Population realised E[Y] of a per-X policy π(X) on the grid (S marginalised 50/50)."""
    X = X_GRID
    eY1 = 0.5 * ((2 * p_y1(X, +1) - 1) + (2 * p_y1(X, -1) - 1))
    eY0 = 0.5 * ((2 * p_y0(X, +1) - 1) + (2 * p_y0(X, -1) - 1))
    pi = np.asarray(pi_grid, float)
    return float(np.mean(pi * eY1 + (1 - pi) * eY0))
