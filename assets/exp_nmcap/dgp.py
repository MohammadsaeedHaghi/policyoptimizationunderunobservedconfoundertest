"""exp_nmcap DGP: a NON-MONOTONE discrete-X, binary-treatment DGP where the naive doubly-robust baseline
(DoublyRobust-X-X / AIPW) CATASTROPHICALLY FAILS under a capacity cap, while the robust O / O-W methods recover
the right policy. Companion to exp_owgap: owgap defeats naive IPW (unobserved-S confounding); this defeats naive
AIPW (a fooled outcome model on a non-monotone effect). Together they show no single naive baseline is reliable.

Natural story ("treat the responders, not the sickest"):
  X  ~ Uniform on 7 discrete severity levels in [-1,1]
  S in {-1,+1} UNOBSERVED robustness, P(S=+1)=1/2 (independent of X)
  e(X,S)=clip(sigma(1.2 S + 3 X^2),0.02,0.98)    (clinicians escalate the robust S=+1 AND the EXTREME-severity
                                                  edges |X|->1 -> the treated at the edges are a high-S elite)
  mu0(X,S)= 3 S,         mu1(X,S)= 4 S + 3 (0.3 - X^2)   (S shifts both arms; the therapy HELPS a MIDDLE band
                                                          X^2<0.3 and HARMS the severe edges)
  Y(t)=mu_t(X,S)+N(0,0.6^2)

  CATE(X)=E[Y(1)-Y(0)|X]= 3(0.3 - X^2)  (S independent of X -> the S terms cancel in the X-marginal)
  => oracle treats the MIDDLE band |X|<sqrt(0.3)~0.55, i.e. X in {-1/3,0,1/3}; sparing the edges.

Why naive AIPW fails under the cap, and robust wins:
  At the severe edges the treated are a high-S elite whose OBSERVED outcomes look great, so the outcome model
  mu1_hat is biased UP there; the naive doubly-robust estimate over-credits treating the edges. Under cap<=50%
  it then spends its budget treating the (harmful) edges instead of the (beneficial) middle. The MSM odds-box
  (O) / box+Wasserstein (O-W) hedge against the edge-imbalanced treated mass and recover the middle band.
  (Naive IPW happens to self-correct here because the edge over-selection shows up in the X-propensity; it is
  the S-confounded exp_owgap DGP that defeats IPW. The pair of DGPs is the point.)
"""
import numpy as np

LEVELS = np.round(np.linspace(-1.0, 1.0, 7), 6)
CS, CE, D0, D1, B, C, NOISE = 1.2, 3.0, 3.0, 4.0, 3.0, 0.3, 0.6
CAP = (1.0, 0.5)
K = 2


def _sig(z): return 1.0 / (1.0 + np.exp(-np.clip(np.asarray(z, float), -40, 40)))
def p_s1(X): return np.full_like(np.asarray(X, float), 0.5)   # S independent of X
def propensity(X, S): return np.clip(_sig(CS * np.asarray(S, float) + CE * np.asarray(X, float) ** 2), 0.02, 0.98)
def mu0(X, S): return D0 * np.asarray(S, float) + 0.0 * np.asarray(X, float)   # +0*X broadcasts to X's shape
def mu1(X, S): return D1 * np.asarray(S, float) + B * (C - np.asarray(X, float) ** 2)


def grid_truth():
    X = LEVELS; ES = np.zeros_like(X)            # E[S|X]=0
    eY0 = D0 * ES; eY1 = D1 * ES + B * (C - X ** 2); cate = eY1 - eY0
    return dict(X=X, p_s1=p_s1(X), m0p=mu0(X, 1.0), m0m=mu0(X, -1.0), m1p=mu1(X, 1.0), m1m=mu1(X, -1.0),
                eY0=eY0, eY1=eY1, e_plus=propensity(X, 1.0), e_minus=propensity(X, -1.0),
                e_marg=0.5 * propensity(X, 1.0) + 0.5 * propensity(X, -1.0),
                cate=cate, oracle=(cate > 0).astype(float))


def oracle_policy(X):
    X = np.asarray(X, float)
    return ((B * (C - X ** 2)) > 0).astype(float)   # treat the middle band X^2<C


def generate(n, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.choice(LEVELS, size=n)
    S = np.where(rng.uniform(size=n) < 0.5, 1.0, -1.0)
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
