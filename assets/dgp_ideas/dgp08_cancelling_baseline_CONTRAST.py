"""IDEA 08 - CANCELLING baseline confounding (a CONTRAST / boundary case, expected NOT to favor O-W).

Story: a continuous hidden U shifts the baseline equally in BOTH arms (same coeff LAM), while the
treatment effect depends only on the OBSERVED X (effect = A + B X, no U). U drives selection. Because U
enters mu0 and mu1 identically, it CANCELS in the treat-minus-control contrast, so the true CATE is a
clean function of X. This is the regime where a plain doubly-robust estimator should already do well and
O-W has little to add. Included on purpose as a negative control / boundary, to show WHERE the
Wasserstein advantage does and does not appear.

  U|X ~ N(C_UX X, 1);  e=clip(sigma(G U - D X));
  mu0 = M0 X + LAM U,  mu1 = mu0 + (A + B X).  CATE = A + B X (U cancels).
"""
import numpy as np
LEVELS = np.round(np.linspace(-1.0, 1.0, 7), 6)
CAP = (1.0, 0.5); K = 2; NOISE = 0.6
_Z = np.linspace(-5.0, 5.0, 121); _W = np.exp(-_Z**2 / 2); _W = _W / _W.sum()
def _sig(z): return 1.0 / (1.0 + np.exp(-np.clip(np.asarray(z, float), -40, 40)))

C_UX, LAM, M0, A, B, G, D = 1.0, 4.0, 0.0, 0.0, 0.8, 1.0, 1.0

def _components(x):
    x = np.asarray(x, float); U = (C_UX * x)[:, None] + _Z[None, :]; xb = x[:, None]
    w = np.tile(_W, (x.size, 1)); e = np.clip(_sig(G * U - D * xb), 0.02, 0.98)
    m0 = M0 * xb + LAM * U; return w, e, m0, m0 + (A + B * xb)

def _sample(X, rng):
    X = np.asarray(X, float); U = C_UX * X + rng.normal(0, 1, X.size)
    e = np.clip(_sig(G * U - D * X), 0.02, 0.98); m0 = M0 * X + LAM * U
    return U, e, m0, m0 + (A + B * X)

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
