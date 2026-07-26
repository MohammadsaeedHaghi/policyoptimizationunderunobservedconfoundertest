"""exp_diabetes IN-REGIME semi-synthetic DGP ("Experiment A"): real covariates, amplified confounding.

Companion to assets/exp_diabetes/dgp.py (the fully-real "Experiment B"). B keeps real (X, S, T) and
shows the honest out-of-regime story: with the real weak coupling (corr(X,S)=0.196, matched
Gamma=1.26) the naive plug-in is already near-oracle (linear-naive gap ~0.001) and robustness has
nothing to fix. THIS file is the ACIC-style in-regime variant: the REAL frailty covariate X
(bootstrapped from the 69,990 cleaned Diabetes-130 patients, so the covariate geometry is real)
with SIMULATED hidden acuity S and insulin decision T placed inside the operating regime that the
owgap alpha sweep mapped out:

  S ~ Bern(sigma(BP (X - X0)))  BP=12, X0 calibrated so P(S=+1) = the real acuity rate 0.249.
      E[Var(S|X)] = 0.23 -- comparable to owgap alpha=10 (0.18); balancing X controls S.
  T ~ Bern(clip(sigma(C0 + CX X + CS S), .02, .98))  CS=0.8 -> true selection Lambda = 4.95, the
      SAME matched Gamma=5 protocol as exp_owgap_v2; CX = the real fitted X-coefficient (0.54);
      C0 calibrated so P(T=1) = the real insulin rate 0.51.
  mu0 = -K S,  mu1 = -(K-G) S + D X - TH   with K,G,D,TH = 3.0, 1.5, 1.0, 0.3
      (S=+1 = sicker = more readmission = lower Y in both arms; insulin helps more the sicker and
       the frailer, minus a fixed burden TH). CATE(X) = G (2 p_s1(X) - 1) + D X - TH crosses zero
      just above X0 -> oracle treats the frailest ~22%.

THE FAILURE MODE IS THE MIRROR IMAGE OF OWGAP: sicker patients get insulin AND do worse, so the
naive plug-in concludes insulin is HARMFUL (its linear CATE-hat threshold leaves [-1,1] entirely)
and treats NOBODY -- naive value == never-treat (1.51), oracle 1.78, linear-naive gap 0.27. The
paper thus shows both directions: over-treatment (owgap, vitality flatters the drug) and
under-treatment (here, acuity damns it), with the O-W correction recovering each.

Same generate(n, seed) -> (observed, full) contract as exp_owgap_cont; continuous X -> run via
assets/run_owgap_lip_gamma_2d.py --dgp.
"""
from pathlib import Path
import numpy as np

_NPZ = Path(__file__).resolve().parent / "diabetes_prepared.npz"
_D = np.load(_NPZ)
_X_ALL = _D["X"].astype(float)                # real composite frailty, 69,990 patients, in [-1,1]
_P_S_REAL = float(np.mean(_D["S"] > 0))       # 0.249  (real acuity rate, kept as the S marginal)
_P_T_REAL = 0.51                              # real insulin rate target
_CX_REAL = float(_D["prop_c"][1])             # 0.54   (real fitted X-coefficient in the propensity)

X_LO, X_HI = -1.0, 1.0
LEVELS = np.round(np.linspace(-1.0, 1.0, 7), 6)   # vestigial (grid diagnostics only)
BP = 12.0                                      # amplified coupling steepness (in-regime)
CS = 0.8                                       # selection on S -> Lambda = exp(2*.8) = 4.95, Gamma*=5
K_SEV, G_SYN, D_FR, TH = 3.0, 1.5, 1.0, 0.3    # outcome constants (see docstring)
NOISE = 0.6
D0, D1, B0, B1 = -K_SEV, -(K_SEV - G_SYN), 0.0, D_FR
CAP = (1.0, 0.3)
K = 2


def _sig(z): return 1.0 / (1.0 + np.exp(-np.clip(np.asarray(z, float), -40, 40)))


def _calibrate():
    lo, hi = -3.0, 3.0
    for _ in range(60):
        x0 = (lo + hi) / 2
        if np.mean(_sig(BP * (_X_ALL - x0))) > _P_S_REAL: lo = x0
        else: hi = x0
    ps = _sig(BP * (_X_ALL - x0))
    rng = np.random.default_rng(12345)
    S = np.where(rng.uniform(size=len(_X_ALL)) < ps, 1.0, -1.0)
    lo2, hi2 = -4.0, 4.0
    for _ in range(60):
        c0 = (lo2 + hi2) / 2
        if np.mean(np.clip(_sig(c0 + _CX_REAL * _X_ALL + CS * S), .02, .98)) > _P_T_REAL: hi2 = c0
        else: lo2 = c0
    return x0, c0

X0, C0 = _calibrate()


def p_s1(X): return _sig(BP * (np.asarray(X, float) - X0))
def propensity(X, S):
    return np.clip(_sig(C0 + _CX_REAL * np.asarray(X, float) + CS * np.asarray(S, float)), 0.02, 0.98)
def mu0(X, S): return D0 * np.asarray(S, float) + B0 * np.asarray(X, float)
def mu1(X, S): return D1 * np.asarray(S, float) + B1 * np.asarray(X, float) - TH


def oracle_policy(X):
    X = np.asarray(X, float); ES = 2 * p_s1(X) - 1.0
    return ((G_SYN * ES + D_FR * X - TH) > 0).astype(float)


def grid_truth():
    X = LEVELS; ps1 = p_s1(X); ES = 2 * ps1 - 1.0
    eY0 = D0 * ES; eY1 = D1 * ES + B1 * X - TH; cate = eY1 - eY0
    return dict(X=X, p_s1=ps1, m0p=mu0(X, 1.0), m0m=mu0(X, -1.0), m1p=mu1(X, 1.0), m1m=mu1(X, -1.0),
                eY0=eY0, eY1=eY1, e_plus=propensity(X, 1.0), e_minus=propensity(X, -1.0),
                e_marg=ps1 * propensity(X, 1.0) + (1 - ps1) * propensity(X, -1.0),
                cate=cate, oracle=(cate > 0).astype(float))


def generate(n, seed=0):
    rng = np.random.default_rng(seed)
    X = _X_ALL[rng.integers(0, len(_X_ALL), size=n)]           # bootstrap REAL covariates
    S = np.where(rng.uniform(size=n) < p_s1(X), 1.0, -1.0)     # synthetic in-regime confounder
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
