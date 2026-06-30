"""exp_owgap DGP: a discrete-X, binary-treatment clinical DGP where BOTH O-W methods (IPW-O-W and
DoublyRobust-O-W) beat EVERY other method (incl. plain DoublyRobust-X-X) by a LARGE gap, in BOTH the
capped and uncapped regimes, peaking at a SMALL sensitivity Γ (2-3).

Natural story (aggressive therapy under hidden vitality):
  X  ~ Uniform on 7 discrete fitness/biomarker levels in [-1,1]   (observed)
  S in {-1,+1} UNOBSERVED patient "vitality", P(S=+1|X)=sigma(10 X)   (STRONG vitality-fitness coupling)
  e(X,S)=clip(sigma(0.8 S - 2 X),0.02,0.98)                          (clinicians escalate the vital S=+1
                                                                      and the low-fitness; MODERATE selection
                                                                      on S -> matched Γ stays SMALL)
  mu0(X,S)= 8.0 S,            mu1(X,S)= 9.0 S + 1.5 X                 (vitality DOMINATES outcomes in both
                                                                      arms (d0=8, d1=9); therapy adds a
                                                                      small vitality synergy d1-d0=1.0 AND a
                                                                      fitness-dependent term +1.5X: it helps
                                                                      the fit, harms the unfit)
  Y(t)=mu_t(X,S)+N(0,0.6^2)

  CATE(X) = E[Y(1)-Y(0) | X] = (d1-d0) E[S|X] + 1.5 X = 1.0 (2 sigma(10X)-1) + 1.5 X   => treat iff X>0.

Why O-W wins by a LARGE gap, and why plain DoublyRobust-X-X fails:
  Vitality S shifts BOTH arm outcomes hugely (d0=8, d1=9.5), so the outcome model is BADLY biased by the
  S-selection (treated patients look great because they are vital, not because of the therapy). The true
  treatment differential is SUBTLE (d1-d0=1.5) and partly HARMFUL for the unfit (+2X with X<0), so the
  naive doubly-robust estimate over-credits the therapy and OVER-TREATS into the harmful low-fitness region.
  Because fitness X strongly tracks the hidden vitality S, enforcing covariate (Wasserstein) balance on the
  observed X also balances the hidden S, so the O-W methods recover the right "treat only the fit" policy.
  Large common confounding magnitude (=large model bias) + small non-cancelling differential (d1!=d0) +
  strong U-X correlation is what makes the gap big while keeping the operating Γ small.
"""
import numpy as np

LEVELS = np.round(np.linspace(-1.0, 1.0, 7), 6)
ALPHA, CS, CX, D0, D1, B0, B1, NOISE = 10.0, 0.8, 2.0, 8.0, 9.0, 0.0, 1.5, 0.6
CAP = (1.0, 0.5)   # arm 0 (control) uncapped; arm 1 (treat) <= 50% under the capped regime
K = 2


def _sig(z): return 1.0 / (1.0 + np.exp(-np.clip(np.asarray(z, float), -40, 40)))
def p_s1(X): return _sig(ALPHA * np.asarray(X, float))
def propensity(X, S): return np.clip(_sig(CS * np.asarray(S, float) - CX * np.asarray(X, float)), 0.02, 0.98)
def mu0(X, S): return D0 * np.asarray(S, float) + B0 * np.asarray(X, float)
def mu1(X, S): return D1 * np.asarray(S, float) + B1 * np.asarray(X, float)


def grid_truth():
    X = LEVELS; ps1 = p_s1(X); ES = 2 * ps1 - 1.0
    eY0 = D0 * ES + B0 * X; eY1 = D1 * ES + B1 * X; cate = eY1 - eY0
    return dict(X=X, p_s1=ps1, m0p=mu0(X, 1.0), m0m=mu0(X, -1.0), m1p=mu1(X, 1.0), m1m=mu1(X, -1.0),
                eY0=eY0, eY1=eY1, e_plus=propensity(X, 1.0), e_minus=propensity(X, -1.0),
                e_marg=ps1 * propensity(X, 1.0) + (1 - ps1) * propensity(X, -1.0),
                cate=cate, oracle=(cate > 0).astype(float))


def oracle_policy(X):
    # observable optimal policy: treat iff E[CATE|X] > 0
    X = np.asarray(X, float); ES = 2 * p_s1(X) - 1.0
    return (((D1 - D0) * ES + (B1 - B0) * X) > 0).astype(float)


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
