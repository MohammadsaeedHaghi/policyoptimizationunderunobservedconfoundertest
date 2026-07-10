"""IDEA 06 - SATURATING dose-response on a latent index, with a treatment cost.

Story: outcomes are a saturating (logistic) function of a latent health index eta = X + U, where U is an
unobserved component correlated with X. Treatment shifts the index by DEL, but it carries a fixed cost
TAU (side effects / price). The net benefit M[sigma(k(eta+DEL)) - sigma(k eta)] - TAU is large only on
the STEEP part of the response curve and negative once the patient is already saturated or barely
responsive. So whether to treat depends on a nonlinear interaction of X and hidden U.

  U|X ~ N(C_UX X, 1);  eta = X + LAM U;  mu0 = M sigma(k eta),  mu1 = M sigma(k(eta+DEL)) - TAU.
  e=clip(sigma(A U - D X)).  Net CATE crosses zero -> nontrivial policy.
"""
import numpy as np
LEVELS = np.round(np.linspace(-1.0, 1.0, 7), 6)
CAP = (1.0, 0.5); K = 2; NOISE = 0.3
_Z = np.linspace(-5.0, 5.0, 121); _W = np.exp(-_Z**2 / 2); _W = _W / _W.sum()
def _sig(z): return 1.0 / (1.0 + np.exp(-np.clip(np.asarray(z, float), -40, 40)))

C_UX, M, KK, LAM, DEL, TAU, A, D = 1.0, 4.0, 2.0, 1.0, 1.2, 0.5, 1.0, 0.5

def _components(x):
    x = np.asarray(x, float); U = (C_UX * x)[:, None] + _Z[None, :]; xb = x[:, None]
    w = np.tile(_W, (x.size, 1)); e = np.clip(_sig(A * U - D * xb), 0.02, 0.98)
    eta = xb + LAM * U
    return w, e, M * _sig(KK * eta), M * _sig(KK * (eta + DEL)) - TAU

def _sample(X, rng):
    X = np.asarray(X, float); U = C_UX * X + rng.normal(0, 1, X.size)
    e = np.clip(_sig(A * U - D * X), 0.02, 0.98); eta = X + LAM * U
    return U, e, M * _sig(KK * eta), M * _sig(KK * (eta + DEL)) - TAU

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
