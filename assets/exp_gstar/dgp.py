"""exp_gstar DGP (discrete): the Gamma-star-by-inspection showcase.

Identical to the proven exp_owgap_v2 family (hidden-vitality clinical story, treatment burden,
scarce cap) with ONE functional change -- the propensity is logistic in the hidden S with NO
clipping, so the true sensitivity parameter is readable off the DGP:

    e(x, S) = sigma( -CX x + (1/2) ln(LAM) S ),   CX = 1.5, LAM = 5
    =>  odds(T=1 | x, S=+1) / odds(T=1 | x, S=-1) = e^{ln LAM} = LAM = 5   EXACTLY, at every x
    (range sigma(+-(1.5 + 0.8047)) = [0.091, 0.909] -- the clip that perturbed Lambda in
     exp_owgap_v2 at extreme x is never needed).

Hence **Gamma* = 5 by inspection**: the matched-Gamma protocol runs every method at the single
flagged value Gamma = Gamma* = 5. Because the fitted marginal propensity e-hat(x) lies between
the two S-extremals, every unit's true inverse weight is inside the Gamma* odds-box around
e-hat -- the box is correctly specified with a known parameter, exactly as in the MSM synthetic
tradition (Kallus-Mao-Zhou 2019; Hess/Frauen et al. ICLR 2026), whose logistic-in-U propensity
and level-shift confounder this design mirrors.

Everything else is exp_owgap_v2 verbatim:
  X  ~ Uniform on 7 discrete fitness levels in [-1,1]      (observed)
  S in {-1,+1} UNOBSERVED vitality, P(S=+1|X) = sigma(ALPHA X), ALPHA = 10
  mu0(X,S) = 8 S,     mu1(X,S) = 9 S + 3 X - 1             (burden THETA = 1)
  Y(t) = mu_t + N(0, 0.6^2)
  CATE(X) = (2 sigma(10X) - 1) + 3X - 1  =>  oracle treats iff X > 0;  CATE(0) = -1
  capped regime: at most 30% treated (CAP < 43% oracle mass -- genuinely scarce)
Reference values (exact, uniform X-marginal): never-treat 0, all-treat -1,
oracle uncapped 0.847, oracle capped(0.3) 0.727.
"""
import numpy as np

LEVELS = np.round(np.linspace(-1.0, 1.0, 7), 6)
ALPHA, CX, D0, D1, B0, B1, NOISE = 10.0, 1.5, 8.0, 9.0, 0.0, 3.0, 0.6
LAM = 5.0          # Gamma* -- the true selection odds ratio, BY CONSTRUCTION (see docstring)
CS = 0.5 * np.log(LAM)
THETA = 1.0        # fixed burden of treatment (subtracted from arm-1 outcomes)
CAP = (1.0, 0.3)   # arm 0 (control) uncapped; arm 1 (treat) <= 30% under the capped regime
K = 2


def _sig(z): return 1.0 / (1.0 + np.exp(-np.clip(np.asarray(z, float), -40, 40)))
def p_s1(X): return _sig(ALPHA * np.asarray(X, float))
def propensity(X, S): return _sig(CS * np.asarray(S, float) - CX * np.asarray(X, float))   # NO clip
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
    # noise-free expected outcome of a policy pi(treat|X) over the uniform X-grid (S integrated out)
    pi = np.asarray(pi_grid, float); t = grid_truth()
    return float(np.mean(pi * t["eY1"] + (1 - pi) * t["eY0"]))
