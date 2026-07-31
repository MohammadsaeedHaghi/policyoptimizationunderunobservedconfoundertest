"""exp_kz18: the Kallus-Zhou binary-treatment synthetic (Minimax-Optimal Policy Learning
Under Unobserved Confounding, Management Science / NeurIPS'18 CRPI) -- adapted to our pipeline.

Source of truth is the AUTHORS' CODE (github.com/CausalML/confounding-robust-policy-improvement,
data_scenarios.py), which resolves three places where the paper text is loose:
  1. Gamma convention: REAL_PROP_LOG passes Gamma=1.5 into get_bnds, which EXPONENTIATES it.
     The true propensity realizes the MSM bounds at odds ratio Lambda* = e^1.5 ~ 4.4817
     (the paper text's "Gamma = 1.5" is log-scale) => matched Gamma = 4.4817, flagged.
  2. Outcome noise is 2 * N(0,1) (code), not N(0,1) (paper caption), common to both arms.
  3. The xi level-shift coefficient is w = 1.5 (code), not omega = 1 (paper caption).

The DGP (loss scale, authors' notation; xi in {0,1} is the hidden group):
  xi ~ Bern(1/2),  X5 ~ N(mu_x * (2 xi - 1), I_5),  mu_x = [-1, .5, -1, 0, -1]
  loss(t) = 2.5 t + beta_x' X5 + beta_xT' X5 * t + (-2) xi (2t - 1) + 1.5 xi + 2 eps
      beta_x = [0, .5, -.5, 0, 0],  beta_xT = [-1.5, 1, -1.5, 1, .5],  eps ~ N(0,1) shared
  U = I[loss(1) < loss(0)] (noiseless),  nominal e~(x) = sigma(theta' X5), theta = [0,.75,-.5,0,-1]
  e(X5, U) = MSM-extremal around e~ at Lambda* = e^1.5: the HIGH bound when treatment is
  better (U = 1), the LOW bound otherwise ("doctors treat when beneficial").

Our adaptation (pure relabeling + disclosed reductions):
  - REWARD scale: R = -loss (pipeline maximizes); oracle treats iff loss(1) < loss(0).
  - Scalar covariate (our estimators are univariate; every method sees the SAME scalar):
      INDEX = "prop" (MAIN): x = theta' X5 / 4    -- the nominal-propensity direction
      INDEX = "cate" (ablation, dgp_cidx.py): x = beta_xT' X5 / 8  -- the CATE direction
  - S = 2U - 1 in {+-1} is the propensity-relevant confounder for the coupling diagnostic.
  - CAP = 30% for the capped variant (novel vs their uncapped regret setup).

Why "prop" is the main arm (measured, disclosed -- not tuned): the paper's phenomenon is that
methods assuming unconfoundedness DO HARM. That harm survives the scalar reduction only on the
propensity index: infinite-data naive OVER-treats (62.5% vs the oracle's 38.6%) and gives back
0.126 of the 0.310 gain available between never-treat and oracle. On the CATE index the naive
policy is already within 0.021 of the oracle -- no harm to undo -- so that arm is reported as a
DO-NO-HARM check (a robust method must not destroy an already-good naive policy), not as the
headline. Both arms are run and both are reported.

Structural facts: corr(x, S) = -0.37 (prop) / -0.71 (cate) -- U is x-TRACKABLE here, the
opposite regime from exp_msmbench's zero-coupling anchor, so the Wasserstein term has real
signal to use. Doctors treat the treatment-favorable, so the treated arm is enriched in
patients whose treatment gain is large => the naive contrast is biased UP => naive OVER-treats
(their Fig. 1's IPW/GRF harm).
"""
import numpy as np

LOGG = 1.5
GSTAR = float(np.exp(LOGG))                       # 4.481689 -- matched Gamma, by construction
MU_X = np.array([-1.0, 0.5, -1.0, 0.0, -1.0])
BETA_X = np.array([0.0, 0.5, -0.5, 0.0, 0.0])
BETA_XT = np.array([-1.5, 1.0, -1.5, 1.0, 0.5])
THETA = np.array([0.0, 0.75, -0.5, 0.0, -1.0])
ALPHA_TE = 2.5                                    # constant treatment effect (loss scale)
ETA_CONF = -2.0                                   # xi coefficient on (2t-1) (code: alpha)
W_CONF = 1.5                                      # xi level shift (code: w; paper says 1)
NOISE = 2.0                                       # code: 2*N(0,1); paper caption says N(0,1)
INDEX = "prop"                                    # scalar covariate: "prop" (main) or "cate"
SCALE = {"cate": 8.0, "prop": 4.0}

K = 2
CAP = (1.0, 0.3)
LEVELS = np.round(np.linspace(-1.0, 1.0, 7), 6)   # vestigial (grid contract only)

MC_ = float(BETA_XT @ MU_X)                       # 3.0    E[beta_xT'X5 | xi] = +-MC_
S2C = float(BETA_XT @ BETA_XT)                    # 6.75   Var(beta_xT'X5 | xi)
MT_ = float(THETA @ MU_X)                         # 1.875
S2T = float(THETA @ THETA)                        # 1.8125
CT_ = float(BETA_XT @ THETA)                      # 1.0    Cov(beta_xT'X5, theta'X5 | xi)


def _sig(z): return 1.0 / (1.0 + np.exp(-np.clip(np.asarray(z, float), -40, 40)))

def _ncdf(z):
    try:
        from scipy.special import ndtr
        return ndtr(z)
    except Exception:                             # logistic approximation, diagnostic-only
        return _sig(1.702 * np.asarray(z, float))

def _pxi1(v, m, s2):                              # P(xi=1 | raw index v), v|xi ~ N(+-m, s2)
    return _sig(2.0 * m / s2 * np.asarray(v, float))

def _dY(v_c, xi):                                 # loss(1) - loss(0)
    return ALPHA_TE + np.asarray(v_c, float) + 2.0 * ETA_CONF * np.asarray(xi, float)

def _EdY(v):
    """E[loss(1) - loss(0) | raw index value v] under the current INDEX."""
    v = np.asarray(v, float)
    if INDEX == "cate":
        return ALPHA_TE + v + 2.0 * ETA_CONF * _pxi1(v, MC_, S2C)
    p1 = _pxi1(v, MT_, S2T); s = 2.0 * p1 - 1.0
    ev_c = MC_ * s + (CT_ / S2T) * (v - MT_ * s)
    return ALPHA_TE + ev_c + 2.0 * ETA_CONF * p1

def cate(x):                                      # marginal reward-scale CATE on the scalar
    return -_EdY(SCALE[INDEX] * np.asarray(x, float))

def oracle_policy(x):                             # best policy measurable w.r.t. the scalar
    return (cate(np.asarray(x, float)) > 0).astype(float)

def p_s1(x):                                      # P(S=+1 | x): coupling diagnostic contract
    v = SCALE[INDEX] * np.asarray(x, float)
    thr1 = -(ALPHA_TE + 2.0 * ETA_CONF)           # dY<0 with xi=1  <=>  v_c < 1.5
    thr0 = -ALPHA_TE                              # dY<0 with xi=0  <=>  v_c < -2.5
    if INDEX == "cate":
        p1 = _pxi1(v, MC_, S2C)
        return p1 * (v < thr1) + (1.0 - p1) * (v < thr0)
    p1 = _pxi1(v, MT_, S2T)
    s2c = S2C - CT_ ** 2 / S2T; sd = np.sqrt(s2c)
    out = 0.0
    for xi, pxi in ((1.0, p1), (0.0, 1.0 - p1)):
        s = 2.0 * xi - 1.0
        mc = MC_ * s + (CT_ / S2T) * (v - MT_ * s)
        thr = thr1 if xi == 1.0 else thr0
        out = out + pxi * _ncdf((thr - mc) / sd)
    return out


def generate(n, seed=0):
    rng = np.random.default_rng(seed)
    xi = (rng.uniform(size=n) < 0.5).astype(float)
    X5 = MU_X * (2.0 * xi - 1.0)[:, None] + rng.standard_normal((n, 5))
    v_c = X5 @ BETA_XT; v_t = X5 @ THETA
    en = _sig(v_t)
    dY = _dY(v_c, xi)
    opt = dY < 0.0                                # U: treatment (t=1) is better
    odds = en / (1.0 - en)
    p_hi = GSTAR * odds / (1.0 + GSTAR * odds)
    p_lo = (odds / GSTAR) / (1.0 + odds / GSTAR)
    e = np.where(opt, p_hi, p_lo)
    T = (rng.uniform(size=n) < e).astype(int)
    loss0 = X5 @ BETA_X + (W_CONF - ETA_CONF) * xi
    loss1 = loss0 + dY
    eps = NOISE * rng.standard_normal(n)          # SHARED across arms (paper: "the same")
    R1 = -(loss1 + eps); R0 = -(loss0 + eps)
    x = (v_c if INDEX == "cate" else v_t) / SCALE[INDEX]
    Y = np.where(T == 1, R1, R0)
    observed = {"X": x.reshape(-1, 1), "T": T, "Y": Y}
    full = {**observed, "S": np.where(opt, 1.0, -1.0), "XI": xi,
            "Y1": R1, "Y0": R0, "e_true": e, "e_nom": en, "X5": X5}
    return observed, full


def grid_truth():
    X = LEVELS
    return dict(X=X, cate=cate(X), oracle=oracle_policy(X))
