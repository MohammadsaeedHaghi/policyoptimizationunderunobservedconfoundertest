"""exp_owgap_v2 DGP: the "showcase" variant of exp_owgap, redesigned so the O-W win is LARGE in
BOTH the uncapped and the capped regime.

Two changes vs assets/exp_owgap/dgp.py (everything else -- levels, coupling alpha=10, propensity,
D0/D1, noise -- is byte-identical, so oracle policy, Lambda=4.95 and the matched-Gamma=5 story
carry over unchanged):

  1. THETA = 1.0 -- a FIXED treatment burden (toxicity / cost of aggressive therapy):
         mu1(X,S) = 9 S + 3 X - 1
     and B1 raised 1.5 -> 3.0 so the fitness-dependent benefit stays strong. Consequence:
         CATE(X) = (2 sigma(10X) - 1) + 3 X - 1   =>  still treat iff X > 0,
     but now CATE(0) = -1 (was 0). This is the whole point: the naive plug-in's signature
     mistake is treating the X=0 level, where within-level selection on hidden vitality S is
     maximal (Var(S|X) peaks at p_s1=0.5) and inflates the estimated CATE by ~ +6.5. In the
     base DGP that mistake was FREE (true CATE(0)=0), which is why capping even *rescued* the
     naive methods (harmless regularization). Here the same mistake costs real value.

  2. CAP = (1.0, 0.3) -- a genuinely SCARCE treatment budget (30% < the 43% oracle-treat mass;
     the base 50% cap barely bound). Under scarcity the ranking is everything: the naive
     methods rank X=0 FIRST (inflated CATE-hat ~ +5.5 vs true -1) and burn a third of the
     budget harming the patients who look like super-responders, while missing X=1 (+3).

Exact (infinite-data) values on the level grid, uniform X-marginal:
  never-treat 0.000, all-treat -1.000, oracle uncapped 0.847 (same as base), oracle capped(0.3)
  0.727; naive plug-in uncapped 0.704, capped 0.314. Finite-sample (N=600, per-level arm means,
  20 seeds): naive uncapped 0.697+-0.03, capped 0.325+-0.06 -- and the real pipeline's cross-fit
  mu-hat smooths the X=0 bias into neighbouring levels, so realized naive is typically lower
  still. True selection odds ratio Lambda = 4.95 (propensity unchanged) => Gamma=5 = matched.
  Diagnostics: make_diagnostics.py -> v2_diag.json.

Natural story (aggressive therapy under hidden vitality, with therapy burden):
  X  ~ Uniform on 7 discrete fitness/biomarker levels in [-1,1]   (observed)
  S in {-1,+1} UNOBSERVED patient "vitality", P(S=+1|X)=sigma(10 X)
  e(X,S)=clip(sigma(0.8 S - 2 X),0.02,0.98)
  mu0(X,S)= 8 S,       mu1(X,S)= 9 S + 3 X - 1     (burden 1; helps the fit, harms the unfit)
  Y(t)=mu_t(X,S)+N(0,0.6^2);  capped regime: at most 30% of patients may be treated.
"""
import numpy as np

LEVELS = np.round(np.linspace(-1.0, 1.0, 7), 6)
ALPHA, CS, CX, D0, D1, B0, B1, NOISE = 10.0, 0.8, 2.0, 8.0, 9.0, 0.0, 3.0, 0.6
THETA = 1.0        # fixed burden of treatment (subtracted from arm-1 outcomes)
CAP = (1.0, 0.3)   # arm 0 (control) uncapped; arm 1 (treat) <= 30% under the capped regime
K = 2


def _sig(z): return 1.0 / (1.0 + np.exp(-np.clip(np.asarray(z, float), -40, 40)))
def p_s1(X): return _sig(ALPHA * np.asarray(X, float))
def propensity(X, S): return np.clip(_sig(CS * np.asarray(S, float) - CX * np.asarray(X, float)), 0.02, 0.98)
def mu0(X, S): return D0 * np.asarray(S, float) + B0 * np.asarray(X, float)
def mu1(X, S): return D1 * np.asarray(S, float) + B1 * np.asarray(X, float) - THETA


def grid_truth():
    X = LEVELS; ps1 = p_s1(X); ES = 2 * ps1 - 1.0
    eY0 = D0 * ES + B0 * X; eY1 = D1 * ES + B1 * X - THETA; cate = eY1 - eY0
    return dict(X=X, p_s1=ps1, m0p=mu0(X, 1.0), m0m=mu0(X, -1.0), m1p=mu1(X, 1.0), m1m=mu1(X, -1.0),
                eY0=eY0, eY1=eY1, e_plus=propensity(X, 1.0), e_minus=propensity(X, -1.0),
                e_marg=ps1 * propensity(X, 1.0) + (1 - ps1) * propensity(X, -1.0),
                cate=cate, oracle=(cate > 0).astype(float))


def oracle_policy(X):
    # observable optimal policy: treat iff E[CATE|X] > 0
    X = np.asarray(X, float); ES = 2 * p_s1(X) - 1.0
    return (((D1 - D0) * ES + (B1 - B0) * X - THETA) > 0).astype(float)


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
    # noise-free expected outcome of a policy pi(treat|X) over the uniform X-grid (X-marginal eY integrates S out).
    pi = np.asarray(pi_grid, float); t = grid_truth()
    return float(np.mean(pi * t["eY1"] + (1 - pi) * t["eY0"]))
