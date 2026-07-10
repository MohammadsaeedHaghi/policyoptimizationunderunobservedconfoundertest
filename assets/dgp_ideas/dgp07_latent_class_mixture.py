"""IDEA 07 - LATENT-CLASS mixture: hidden responders vs non-responders.

Story: patients belong to a hidden class R in {0,1}: responders (R=1, more likely at high X,
P(R=1|X)=sigma(4X)) get a large benefit (TB), non-responders get a small/negative one (TS). Responders
also have a better baseline (coeff CR), and the better-baseline patients are treated more often
(e=sigma(1.2 R - 0 X)), so class membership confounds both the outcome and the assignment. The optimal
policy treats where the expected benefit pR*TB+(1-pR)*TS is positive; the naive contrast is biased by the
responders' baseline advantage.

  R in {0,1}, P(R=1|X)=sigma(ETA X);  e=clip(sigma(C R - D X));
  mu0 = CR R,  mu1 = mu0 + (R? TB : TS).  CATE = pR TB + (1-pR) TS.
"""
import numpy as np
LEVELS = np.round(np.linspace(-1.0, 1.0, 7), 6)
CAP = (1.0, 0.5); K = 2; NOISE = 0.6
_Z = np.linspace(-5.0, 5.0, 121); _W = np.exp(-_Z**2 / 2); _W = _W / _W.sum()
def _sig(z): return 1.0 / (1.0 + np.exp(-np.clip(np.asarray(z, float), -40, 40)))

ETA, CR, TB, TS, C, D = 4.0, 2.0, 2.5, -0.5, 1.2, 0.0

def _components(x):
    x = np.asarray(x, float); pr = _sig(ETA * x)
    Rv = np.array([0.0, 1.0]); w = np.stack([1 - pr, pr], axis=1)
    Rm = np.broadcast_to(Rv, (x.size, 2)); xb = x[:, None]
    e = np.clip(_sig(C * Rm - D * xb), 0.02, 0.98)
    m0 = CR * Rm + 0.0 * xb; eff = np.where(Rm > 0.5, TB, TS)
    return w, e, m0, m0 + eff

def _sample(X, rng):
    X = np.asarray(X, float); pr = _sig(ETA * X)
    R = (rng.uniform(size=X.size) < pr).astype(float)
    e = np.clip(_sig(C * R - D * X), 0.02, 0.98)
    m0 = CR * R; eff = np.where(R > 0.5, TB, TS)
    return R, e, m0, m0 + eff

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
