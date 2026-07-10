"""IDEA 10 - INTERACTION confounding in the TREATED arm only (clean control arm).

Story: the control outcome depends only on observed X (no hidden driver), but the treatment's effect
depends on a hidden "fitness" U that ALSO drives who gets treated. So the control arm is clean and easy
to model, while the treated arm is confounded: the treated are selected for high U, which inflates their
realized Y1 (coeff C), making the naive/plain-DR treated model over-optimistic. Because U is correlated
with X, balancing X corrects the treated-arm bias -> O-W is expected to help, while a plain outcome model
(DR-X-X) is biased only on the treated side.

  U|X ~ N(RHO X, 1);  e=clip(sigma(G U - D X));
  mu0 = M0 X (no U),  mu1 = M0 X + (A + B X) + C U.  E[CATE|X] = (A + B X) + C RHO X.
"""
import numpy as np
LEVELS = np.round(np.linspace(-1.0, 1.0, 7), 6)
CAP = (1.0, 0.5); K = 2; NOISE = 0.6
_Z = np.linspace(-5.0, 5.0, 121); _W = np.exp(-_Z**2 / 2); _W = _W / _W.sum()
def _sig(z): return 1.0 / (1.0 + np.exp(-np.clip(np.asarray(z, float), -40, 40)))

RHO, M0, A, B, C, G, D = 0.5, 0.0, -0.3, 0.3, 4.0, 1.0, 1.0

def _components(x):
    x = np.asarray(x, float); U = (RHO * x)[:, None] + _Z[None, :]; xb = x[:, None]
    w = np.tile(_W, (x.size, 1)); e = np.clip(_sig(G * U - D * xb), 0.02, 0.98)
    m0 = M0 * xb + 0.0 * U; return w, e, m0, M0 * xb + (A + B * xb) + C * U

def _sample(X, rng):
    X = np.asarray(X, float); U = RHO * X + rng.normal(0, 1, X.size)
    e = np.clip(_sig(G * U - D * X), 0.02, 0.98)
    return U, e, M0 * X, M0 * X + (A + B * X) + C * U

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
