"""IDEA 01 - Binary hidden type S, confounding on BOTH selection and outcome (the I3 baseline).

Story (clinical): patients have a hidden vitality type S in {-1,+1}, more vital at higher fitness X
(P(S=+1|X)=sigma(8X)). Vitality dominates outcomes in BOTH arms (large coefficients 5 and 6) but the
treatment helps the vital slightly more (differential 6-5=1, a subtle CATE). Clinicians escalate the
vital, so selection is on the hidden S. Because S tracks observed X, Wasserstein covariate-balance can
recover the right policy -> O-W is expected to win at small Gamma.

  S in {-1,+1}, P(S=+1|X)=sigma(8X);  e=clip(sigma(0.8 S - 2 X));  mu0=5 S,  mu1=6 S + X.
This is the known OW-favorable reference framework. CATE = S + X ; E[CATE|X] = (2 sigma(8X)-1) + X.
"""
import numpy as np
LEVELS = np.round(np.linspace(-1.0, 1.0, 7), 6)
CAP = (1.0, 0.5); K = 2; NOISE = 0.6
_Z = np.linspace(-5.0, 5.0, 121); _W = np.exp(-_Z**2 / 2); _W = _W / _W.sum()
def _sig(z): return 1.0 / (1.0 + np.exp(-np.clip(np.asarray(z, float), -40, 40)))

ALPHA, CS, CX, D0, D1, B = 8.0, 0.8, 2.0, 5.0, 6.0, 1.0

def _components(x):
    x = np.asarray(x, float); p = _sig(ALPHA * x)
    Sv = np.array([-1.0, 1.0]); w = np.stack([1 - p, p], axis=1)
    Sm = np.broadcast_to(Sv, (x.size, 2)); xb = x[:, None]
    e = np.clip(_sig(CS * Sm - CX * xb), 0.02, 0.98)
    return w, e, D0 * Sm + 0.0 * xb, D1 * Sm + B * xb

def _sample(X, rng):
    X = np.asarray(X, float); p = _sig(ALPHA * X)
    S = np.where(rng.uniform(size=X.size) < p, 1.0, -1.0)
    e = np.clip(_sig(CS * S - CX * X), 0.02, 0.98)
    return S, e, D0 * S, D1 * S + B * X

# ---- generic interface (identical across all idea modules) ----
def _eY(x):
    w, e, m0, m1 = _components(np.asarray(x, float)); return (w * m0).sum(1), (w * m1).sum(1)
def _obs(x):
    w, e, m0, m1 = _components(np.asarray(x, float)); s1 = (w * e).sum(1); s0 = (w * (1 - e)).sum(1)
    return (w * e * m1).sum(1) / s1, (w * (1 - e) * m0).sum(1) / s0, (w * e).sum(1)
def grid_truth():
    X = LEVELS; eY0, eY1 = _eY(X); o1, o0, em = _obs(X); cate = eY1 - eY0
    return dict(X=X, eY0=eY0, eY1=eY1, cate=cate, oracle=(cate > 0).astype(float),
                obs1=o1, obs0=o0, obs_contrast=o1 - o0, e_marg=em, naive=((o1 - o0) > 0).astype(float))
def oracle_policy(X):
    eY0, eY1 = _eY(X); return (eY1 > eY0).astype(float)
def naive_policy_grid(): return grid_truth()["naive"]
def exact_value(pi_grid):
    g = grid_truth(); pi = np.asarray(pi_grid, float); return float(np.mean(pi * g["eY1"] + (1 - pi) * g["eY0"]))
def generate(n, seed=0):
    rng = np.random.default_rng(seed); X = rng.choice(LEVELS, size=n)
    hidden, e, mu0, mu1 = _sample(X, rng); T = (rng.uniform(size=n) < e).astype(int)
    Y0 = mu0 + rng.normal(0, NOISE, n); Y1 = mu1 + rng.normal(0, NOISE, n); Y = np.where(T == 1, Y1, Y0)
    obs = {"X": X.reshape(-1, 1), "T": T, "Y": Y}
    full = {**obs, "U": np.asarray(hidden, float), "Y0": Y0, "Y1": Y1, "e_true": e, "mu0": mu0, "mu1": mu1}
    return obs, full
