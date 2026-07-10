"""IDEA 05 - MULTIPLICATIVE confounding: hidden U scales the treatment effect.

Story: a hidden positive "vitality" U (lognormal) multiplies the treatment benefit: the same intervention
helps a high-U patient much more than a low-U one (effect = (a + b X) * U). U also confounds the baseline
(coeff LAM) and selection is on log U. So the effect is heterogeneous through a hidden multiplicative
factor that is correlated with X (log U | X ~ N(rho X, sigma^2)). Balancing X balances the U distribution.

  log U | X ~ N(RHO X, SU^2);  e=clip(sigma(G logU - D X));
  mu0 = LAM U,  mu1 = mu0 + (A + B X) U.  CATE = (A + B X) E[U|X], sign of (A + B X).
"""
import numpy as np
LEVELS = np.round(np.linspace(-1.0, 1.0, 7), 6)
CAP = (1.0, 0.5); K = 2; NOISE = 0.6
_Z = np.linspace(-5.0, 5.0, 121); _W = np.exp(-_Z**2 / 2); _W = _W / _W.sum()
def _sig(z): return 1.0 / (1.0 + np.exp(-np.clip(np.asarray(z, float), -40, 40)))

RHO, SU, LAM, A, B, G, D = 0.8, 0.5, 1.0, -0.2, 1.0, 1.0, 1.0

def _components(x):
    x = np.asarray(x, float); lz = (RHO * x)[:, None] + SU * _Z[None, :]; U = np.exp(lz); xb = x[:, None]
    w = np.tile(_W, (x.size, 1)); e = np.clip(_sig(G * lz - D * xb), 0.02, 0.98)
    m0 = LAM * U; return w, e, m0, m0 + (A + B * xb) * U

def _sample(X, rng):
    X = np.asarray(X, float); lz = RHO * X + SU * rng.normal(0, 1, X.size); U = np.exp(lz)
    e = np.clip(_sig(G * lz - D * X), 0.02, 0.98)
    m0 = LAM * U; return U, e, m0, m0 + (A + B * X) * U

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
