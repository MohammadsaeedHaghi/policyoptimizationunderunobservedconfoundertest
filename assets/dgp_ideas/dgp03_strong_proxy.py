"""IDEA 03 - X is a NEAR-PERFECT PROXY of a continuous hidden U (tight U|X).

Story: the observed covariate X is a high-quality but noisy measurement of the true latent driver U
(U|X ~ N(X, 0.4^2), small residual). U strongly confounds both arms and drives treatment. This is the
extreme of the proxy regime: because X almost pins U, covariate balance on X should almost fully balance
U, so Wasserstein (O-W) is expected to be at its strongest here. Contrast with idea02 (looser proxy).

  U|X ~ N(K X, 0.4^2);  e=clip(sigma(A U - CX X));  mu0=A0 X + L0 U,  mu1=A1 X + L1 U.
"""
import numpy as np
LEVELS = np.round(np.linspace(-1.0, 1.0, 7), 6)
CAP = (1.0, 0.5); K = 2; NOISE = 0.6
_Z = np.linspace(-5.0, 5.0, 121); _W = np.exp(-_Z**2 / 2); _W = _W / _W.sum()
def _sig(z): return 1.0 / (1.0 + np.exp(-np.clip(np.asarray(z, float), -40, 40)))

KK, SU, A, CX, L0, L1, A0, A1 = 1.0, 0.4, 1.2, 1.0, 4.0, 4.6, 0.0, 0.2

def _components(x):
    x = np.asarray(x, float); U = (KK * x)[:, None] + SU * _Z[None, :]; xb = x[:, None]
    w = np.tile(_W, (x.size, 1)); e = np.clip(_sig(A * U - CX * xb), 0.02, 0.98)
    return w, e, A0 * xb + L0 * U, A1 * xb + L1 * U

def _sample(X, rng):
    X = np.asarray(X, float); U = KK * X + SU * rng.normal(0, 1, X.size)
    e = np.clip(_sig(A * U - CX * X), 0.02, 0.98)
    return U, e, A0 * X + L0 * U, A1 * X + L1 * U

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
